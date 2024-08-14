import os
os.environ['CURL_CA_BUNDLE'] = ''
import numpy as np
import torch
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from einops import rearrange, reduce, repeat
from horovod import torch as hvd
import sys

from src.modeling.timesformer.vit import TimeSformer
from src.modeling.xbert import (BertForMaskedLM,
                            BertLMPredictionHead, BertModel, BertPreTrainedModel)
from src.utils.logger import LOGGER, RunningMeter, add_log_to_file
from torch import nn
from torch.nn import CrossEntropyLoss


#被嵌套使用
class AlproBaseModel(nn.Module):
    def __init__(self, config=None, input_format='RGB', video_enc_cfg=None, temp=0.07):
        super().__init__()

        #定义/转化为可训练的参数
        self.temp = nn.Parameter(torch.ones([]) * temp)   

        self.bert_config = config

        visual_model_cls = eval(video_enc_cfg['cls'])#video_enc_cfg['cls']：TimeSformer


        self.visual_encoder = visual_model_cls(model_cfg=video_enc_cfg, input_format=input_format, cross_attention_config=config)

        self.text_encoder = BertForMaskedLM.from_pretrained('bert-base-uncased', config=self.bert_config)


        # FIXME make them configurable
        embed_dim = 256
        vision_width = 768

        text_width = self.bert_config.hidden_size

        #定义线性映射层
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.itc_token_type = self.bert_config.itc_token_type
        self.itm_head = nn.Linear(text_width, 2)     


    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)

        # if bert_weights_path:
        #     load_multimodal_encoder_state_dict_with_mismatch(self.cross_encoder, bert_weights_path)
        #     load_mlm_head_state_dict_with_mismatch(self.mlm_head, bert_weights_path)

    # def freeze_cnn_backbone(self):
    #     for n, p in self.visual_encoder.feature.named_parameters():
    #         p.requires_grad = False

'''
config:model_cfg
video_enc_cfg:config_release/timesformer_divst_8x32_224_k600.json
'''
class AlproForPretrain(AlproBaseModel):
    def __init__(self, config, video_enc_cfg, input_format='RGB'):
        super(AlproForPretrain, self).__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg)

        # model for generating pseudo labels
        self.prompter = Prompter(config, video_enc_cfg)

        self.use_mask_prob = 0
        #定义了一个多层感知机mpm_head来处理伪标签，线性层
        self.mpm_head = nn.Sequential(
            nn.Linear(config.hidden_size,
                    config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, self.prompter.entity_num)
        )

    def build_text_prompts(self, prompts):
        self.prompter.build_text_prompts(prompts)

    def get_pseudo_labels(self, batch):
        return self.prompter.get_pseudo_labels(batch)

    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)#将temp参数固定，不训练它并将它的参数限制在(0.001到0.5)之间

        visual_inputs = batch['visual_inputs']

        #根据batch中是否含有“mpm_mask键判断是否使用mpm”
        use_mpm = 'mpm_mask' in batch
        if use_mpm:
            context_visual_inputs = batch['context_visual_inputs']

        #获取设备信息和尺寸，b批次，t时间维度，c通道数，hw高宽
        device = visual_inputs.device
        b, t, c, h, w = visual_inputs.shape

        # forward image and text features
        # feats are normalized embeds
        #获得视频的embed，决定是否使用掩码预测模型，如果use_mpm为True且随机概率小于0，同时处理原始视频和上下文视频输入，否则只处理原视频
        if use_mpm and np.random.uniform() < self.use_mask_prob:
            video_embeds_total = self._forward_visual_embeds(torch.cat([visual_inputs, context_visual_inputs], dim=0))
            # split for unmasked and masked
            video_embeds, context_video_embeds = video_embeds_total[:b], video_embeds_total[b:]
        else:
            video_embeds = self._forward_visual_embeds(visual_inputs)
            context_video_embeds = video_embeds

        # we compute normalized feats for unmasked visual inputs only, used for ITC
        #获得视频的特征和注意力表示（通过投影和标准化嵌入响亮的第一个时间步来获取视频特征）
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  
        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)
        
        # text embeddings and features
        #获得文本的embed和特征表示
        text_embeds, text_feat = self._forward_text_feats(batch)

        # ========== (in-batch) ITC loss ==========
        # ITC loss就是VTC loss（I-image 几个imgae就是一个Video）
        gathered_video_feats = hvd.allgather(video_feat)
        gathered_text_feats = hvd.allgather(text_feat)

        assert self.itc_token_type == 'cls', 'Support CLS tokens for ITC only, find {}.'.format(self.itc_token_type)
        #计算视频特征和文本特征的点积，除以温度参数进行缩放
        sim_v2t = video_feat @ gathered_text_feats.t() / self.temp 
        sim_t2v = text_feat @ gathered_video_feats.t() / self.temp 
                             
        # [IMPORTANT] be very careful when initializing the GT sim_v2t 
        # allgather return the concatenated features in the order of local_rank()
        sim_targets = torch.zeros_like(sim_v2t)

        local_rank = hvd.local_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.eye(b)

        #计算两个方向上的损失
        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean() 

        vtc_loss = (loss_v2t+loss_t2v) / 2

        # ========= VTM损失 ==========
        text_atts = batch['text_input_mask']

        # non-masked text and non-masked image 
        vtm_loss, vtm_logits, vtm_labels, encoder_outputs_pos = self.compute_vtm(text_embeds=text_embeds, 
                                                                                 text_atts=text_atts, 
                                                                                 video_embeds=video_embeds, 
                                                                                 video_atts=video_atts, 
                                                                                 sim_v2t=sim_v2t.clone(), # for hard mining
                                                                                 sim_t2v=sim_t2v.clone(), # for hard mining
                                                                                 return_encoder_out=True
                                                                                )

        # ========= MLM损失 ==========
        # masked text and non-masked image
        if 'mlm_labels' in batch: 
            mlm_labels = batch['mlm_labels']
            mlm_text_input_ids = batch['mlm_text_input_ids']

            mlm_loss, mlm_logits, mlm_labels = self.compute_mlm(input_ids=mlm_text_input_ids,
                                                                text_input_mask=text_atts,
                                                                video_embeds=video_embeds, 
                                                                video_atts=video_atts,
                                                                mlm_labels=mlm_labels
                                                                )
        else:
            mlm_logits = mlm_loss = mlm_labels = None

        # ========= MPM（掩码预测模型损失） ==========
        if use_mpm: 
            mpm_labels, ignore_masks = self.get_pseudo_labels(batch)

            mpm_loss, mpm_logits = self.compute_mpm_with_encoder_out(encoder_outputs=encoder_outputs_pos, 
                                                                     text_atts=text_atts, 
                                                                     soft_labels=mpm_labels, 
                                                                     ignore_masks=ignore_masks, 
                                                                     patch_masks=batch['mpm_mask']
                                                                    )

        else:
            mpm_loss = mpm_logits = mpm_labels =  None

        #返回损失和得分，用于在训练中更新权重
        return dict(
            itc_loss=vtc_loss,
            mlm_scores=mlm_logits,  # (B, Lt, vocab_size),  only text part
            mlm_loss=mlm_loss,  # (BxLt)
            mlm_labels=mlm_labels,  # (1, Lt), with -100 indicates ignored positions
            itm_scores=vtm_logits,  # (B, 2)
            itm_loss=vtm_loss,  # (1, )
            itm_labels=vtm_labels,  # (B, )
            mpm_loss=mpm_loss,
            mpm_logits=mpm_logits,
            mpm_labels=mpm_labels
        )


    def _forward_visual_embeds(self, visual_inputs):
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # image features
        #timeSformer要求(b, c, t, h, w) 的输入通道，需要交换维度位置
        visual_inputs = visual_inputs.transpose(1, 2)

        #用timesformer里面的forward_features计算返回的视觉特征向量
        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)

        return video_embeds

    def _forward_text_feats(self, batch):
        # text features
        text_output = self.text_encoder.bert(batch['text_input_ids'], 
                                             attention_mask=batch['text_input_mask'],                      
                                             return_dict = True, 
                                             mode = 'text'
                                            )
        #最后一层的隐藏状态可被看做文本的深层表示，作为文本嵌入
        text_embeds = text_output.last_hidden_state # b, Lt, fsz=768
        #通过线性层和归一化操作计算特征向量
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1) #self.text_proj = nn.Linear(text_width, embed_dim)

        return text_embeds, text_feat

    #与soft_label有关，但是只在最后计算交叉熵损失上
    def compute_mpm_with_encoder_out(self, encoder_outputs, text_atts, soft_labels, ignore_masks, patch_masks):
        txt_len = text_atts.shape[1]
        # adding one to ignore visual cls tokens
        visual_output = encoder_outputs.last_hidden_state[:, txt_len+1:]

        #掩码区域展平并反转，非掩码为1，掩码为0
        bsz, h, w = patch_masks.shape
        patch_masks_flatten_inverted = (1 - patch_masks.view(bsz, -1)).unsqueeze(-1)

        # mean embeds of masked visual regions
        #计算每个样本中的被掩码的补丁数量，翻转掩码对视觉输出加权，只保留被掩码的区域嵌入，求和得到总embed，后除以数量得到平均嵌入
        num_masked_patches = torch.sum(patch_masks_flatten_inverted.squeeze(-1), dim=-1, keepdim=True)

        masked_visual_embeds = patch_masks_flatten_inverted * visual_output
        masked_visual_embeds = torch.sum(masked_visual_embeds, dim=1)
        masked_visual_embeds /= num_masked_patches

        # loss通过一个全连接层得到预测logits
        mpm_logits = self.mpm_head(masked_visual_embeds)

        #计算预测与soft_labels之间的交叉熵损失，将ignore_masks的地方指定为0，这些为0的位置的损失不会被计算
        cross_entropy = -torch.sum(F.log_softmax(mpm_logits, dim=1) * soft_labels, dim=1)
        cross_entropy[ignore_masks] = 0.

        #通过总损失除以非忽略样本的数量计算mpm_loss损失
        mpm_loss = torch.sum(cross_entropy) / (bsz - torch.sum(ignore_masks))

        return mpm_loss, mpm_logits 

    def compute_mpm(self, text_embeds, text_atts, image_embeds, image_atts, soft_labels, ignore_masks, patch_masks):
        # forward cross-encoder
        attention_mask = torch.cat([text_atts, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs = self.text_encoder.bert(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                )

        txt_len = text_atts.shape[1]
        # adding one to ignore visual cls tokens
        visual_output = encoder_outputs.last_hidden_state[:, txt_len+1:]

        bsz, h, w = patch_masks.shape
        patch_masks_flatten_inverted = (1 - patch_masks.view(bsz, -1)).unsqueeze(-1)

        # mean embeds of masked visual regions
        num_masked_patches = torch.sum(patch_masks_flatten_inverted.squeeze(-1), dim=-1, keepdim=True)

        masked_visual_embeds = patch_masks_flatten_inverted * visual_output
        masked_visual_embeds = torch.sum(masked_visual_embeds, dim=1)
        masked_visual_embeds /= num_masked_patches

        # loss
        mpm_logits = self.mpm_head(masked_visual_embeds)

        cross_entropy = -torch.sum(F.log_softmax(mpm_logits, dim=1) * soft_labels, dim=1)
        cross_entropy[ignore_masks] = 0.

        mpm_loss = torch.sum(cross_entropy) / (bsz - torch.sum(ignore_masks))

        return mpm_loss, mpm_logits 

    def compute_vtm(self, text_embeds, text_atts, video_embeds, video_atts, sim_v2t, sim_t2v, return_encoder_out=False):
        device = text_embeds.device

        # ====== positive pairs =======
        #拼接文本embed和视觉embed，利用注意力判断有效输入，送入Bert获取正样本对
        attention_mask = torch.cat([text_atts, video_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_outputs_pos = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        # ====== negative pairs =======
        bs = text_embeds.shape[0] 

        local_rank = hvd.local_rank()
        b_start, b_end = bs * local_rank, bs * (local_rank + 1)

        with torch.no_grad():
            weights_i2t = sim_v2t[:,b_start:b_end]
            weights_t2i = sim_t2v[:,b_start:b_end]
   
            # never select self as negative
            weights_i2t.fill_diagonal_(-np.Inf)
            weights_t2i.fill_diagonal_(-np.Inf)

            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)

        # select a negative image for each text
        #对相似矩阵进行计算，为每一个正样本找到一个负样本
        # FIXME to optimize using indexing operations
        video_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            video_embeds_neg.append(video_embeds[neg_idx])
        video_embeds_neg = torch.stack(video_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text_atts, text_atts_neg],dim=0)     

        video_embeds_all = torch.cat([video_embeds_neg,video_embeds],dim=0)
        video_atts_all = torch.cat([video_atts,video_atts],dim=0)

        attention_mask_all = torch.cat([text_atts_all, video_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, video_embeds_all], dim=1)

        # forward negative pairs via cross encoder
        #对负样本也进行Bert编码，以获得他们的表示
        encoder_outputs_neg = self.text_encoder.bert(encoder_embeds=embedding_output_all,
                                                     attention_mask=attention_mask_all,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        #将正负样本对的编码器的输出拼接，通过线性层itm_head来产生每个样本对时正样本还是负样本的预测
        vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                   encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)
        vtm_logits = self.itm_head(vl_embeddings)         # self.itm_head = nn.Linear(text_width, 2)

        #创建正样本为1，负样本为0的标签张量，与vtm_logits做交叉熵，得VTM损失
        vtm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)], dim=0).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)     

        if return_encoder_out:
            return vtm_loss, vtm_logits, vtm_labels, encoder_outputs_pos 
        else:
            return vtm_loss, vtm_logits, vtm_labels, None

    #计算Mask language Modeling损失的函数
    def compute_mlm(self, input_ids, text_input_mask, video_embeds, video_atts, mlm_labels):
        # forward text features with masked_input_ids
        #用Bert处理得到文本编码，通过last_hidden_state得到最后一层输出
        text_output = self.text_encoder.bert(input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state

        # forward cross-encoder
        #交叉编码器前向传播，encoder_outputs融合了文本和视频信息的表示
        attention_mask = torch.cat([text_input_mask, video_atts], dim=1)
        embedding_output = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_outputs = self.text_encoder.bert(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                )

        #提取交叉编码器输出的文本部分
        txt_len = text_input_mask.shape[1]
        txt_output = encoder_outputs.last_hidden_state[:, :txt_len]

        #对可能遮掩单词进行分类，输出的mlm_logits包含了每个单词的预测分数
        mlm_logits = self.text_encoder.cls(txt_output)

        #计算损失，得到所有预测平均损失mlm_loss
        loss_fct = CrossEntropyLoss()
        mlm_loss = loss_fct(mlm_logits.view(-1, self.bert_config.vocab_size), mlm_labels.view(-1))

        return mlm_loss, mlm_logits, mlm_labels
        
    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None, prompter_weights_path=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)

        # [NOTE] BERT is initialized from huggingface pre-trained weights. 
        # if bert_weights_path:
        #     load_multimodal_encoder_state_dict_with_mismatch(self.cross_encoder, bert_weights_path)
        #     load_mlm_head_state_dict_with_mismatch(self.mlm_head, bert_weights_path)

        # TODO make path configurable
        if prompter_weights_path is not None:
            #加载预训练的权重但不包括与prompt相关的权重，把权重变为字典表示后，检查键当中的字符串是否包含“prompt_feat”，如果包含，就剔除，将新的权重加载进来
            self.prompter.load_pretrained_weights_without_prompts(prompter_weights_path)#cfg.teacher_weights_path


class Prompter(AlproBaseModel):
    def __init__(self, config, video_enc_cfg, input_format='RGB'):
        super(Prompter, self).__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg)

        # self.entity_num = 1000
        self.entity_num = config.num_entities

        self.register_buffer("video_prompt_feat", torch.rand(self.entity_num, 256)) 
        self.register_buffer("image_prompt_feat", torch.rand(self.entity_num, 256)) 

        self.prompt_initialized = False
        # if the prob for the most likely entity is < 0.2, we just ignore it
        self.ignore_threshold = 0.2


    def load_pretrained_weights_without_prompts(self, ckpt_path):
        LOGGER.info("Loading weights for teacher model.")
        loaded_state_dict = torch.load(ckpt_path, map_location='cpu')

        loaded_keys = loaded_state_dict.keys()
        model_keys = self.state_dict().keys()

        load_not_in_model = [k for k in loaded_keys if k not in model_keys]
        model_not_in_load = [k for k in model_keys if k not in loaded_keys]

        if hvd.rank() == 0:
            LOGGER.info("Keys in loaded but not in model:")
            LOGGER.info(f"In total {len(load_not_in_model)}, {sorted(load_not_in_model)}")
            LOGGER.info("Keys in model but not in loaded:")
            LOGGER.info(f"In total {len(model_not_in_load)}, {sorted(model_not_in_load)}")

        # FIXME a quick hack to avoid loading prompts
        new_loaded_state_dict = dict()
        for k in loaded_state_dict:
            if not 'prompt_feat' in k:
                new_loaded_state_dict[k] = loaded_state_dict[k]

        loaded_state_dict = new_loaded_state_dict

        self.load_state_dict(loaded_state_dict, strict=False)

    def build_text_prompts(self, prompts):
        """
        This function will be called, if no e2e.weights is provided.
        In that case, 
        """
        assert not self.prompt_initialized, "Repetitively building prompts?"

        if self.training:
            self.eval()

        video_prompt_feat_all = []
        image_prompt_feat_all = []

        with torch.no_grad():
            # this configurable depending on the GPU memory limit
            step_size = 10000

            # ====== initializing video prompting ======
            b_video, _ = prompts['batch_enc_video_prompts'].input_ids.shape

            start = 0
            end = start + step_size

            while start < b_video:
                video_prompt_output = self.text_encoder.bert(prompts['batch_enc_video_prompts'].input_ids[start:end].cuda(), 
                                                            attention_mask=prompts['batch_enc_video_prompts'].attention_mask[start:end].cuda(),                      
                                                            return_dict=True, 
                                                            mode='text'
                                                            )

                video_prompt_embeds = video_prompt_output.last_hidden_state # b, Lt, fsz=768
                video_prompt_feat = F.normalize(self.text_proj(video_prompt_embeds[:,0,:]),dim=-1)                 

                # collecting
                video_prompt_feat_all.append(video_prompt_feat)
            
                start += step_size
                end += step_size

            # average ensembling
            video_prompt_feat = torch.cat(video_prompt_feat_all, dim=0)
            video_num_templates = int(video_prompt_feat.shape[0] / self.entity_num)

            video_prompt_feat = torch.stack(video_prompt_feat.chunk(video_num_templates), dim=1)
            video_prompt_feat = torch.mean(video_prompt_feat, dim=1)
            self.video_prompt_feat = video_prompt_feat

            # ====== initializing image prompting ======
            b_image, _ = prompts['batch_enc_image_prompts'].input_ids.shape

            start = 0
            end = start + step_size

            while start < b_image:
                # image prompts
                image_prompt_output = self.text_encoder.bert(prompts['batch_enc_image_prompts'].input_ids[start:end].cuda(), 
                                                            attention_mask=prompts['batch_enc_image_prompts'].attention_mask[start:end].cuda(),                      
                                                            return_dict = True, 
                                                            mode = 'text'
                                                            )

                image_prompt_embeds = image_prompt_output.last_hidden_state # b, Lt, fsz=768
                image_prompt_feat = F.normalize(self.text_proj(image_prompt_embeds[:,0,:]),dim=-1)                 

                # collecting
                image_prompt_feat_all.append(image_prompt_feat)

                start += step_size
                end += step_size

            image_prompt_feat = torch.cat(image_prompt_feat_all, dim=0)
            image_num_templates = int(image_prompt_feat.shape[0] / self.entity_num)

            image_prompt_feat = torch.stack(image_prompt_feat.chunk(image_num_templates), dim=1)
            image_prompt_feat = torch.mean(image_prompt_feat, dim=1)
            self.image_prompt_feat = image_prompt_feat

        self.prompt_initialized = True

    def _forward_visual_embeds(self, visual_inputs):
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # image features
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)

        assert self.itc_token_type == 'cls', 'Expecting CLS token for ITC, found {}'.format(self.itc_token_type)
        if self.itc_token_type == 'cls':
            video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  
        else:
            raise NotImplementedError("itc_type_type must be one of ['mean', 'cls', 'mil'], found {}".format(self.itc_token_type))
        
        return video_embeds, video_feat

    def _compute_soft_labels(self, sim_vp_masked):
        soft_labels = nn.Softmax(dim=1)(sim_vp_masked)
        ignore_masks = torch.max(sim_vp_masked, dim=1)[1] < self.ignore_threshold

        return soft_labels, ignore_masks

    def get_pseudo_labels(self, batch):
        if self.training:
            self.eval()

        with torch.no_grad():
            masked_visual_inputs = batch['crop_visual_inputs']

            _, masked_image_feat = self._forward_visual_embeds(masked_visual_inputs)

            if batch['type'] == 'video':
                prompt_feat = self.video_prompt_feat
            else:
                prompt_feat = self.image_prompt_feat

            # visual feat to video prompts
            # masked visual feat to video prompts
            sim_masked = masked_image_feat @ prompt_feat.t() / self.temp 

            pseudo_labels, ignore_masks = self._compute_soft_labels(sim_masked)

        return pseudo_labels, ignore_masks

    def forward(self, batch):
        visual_inputs = batch['visual_inputs']

        device = visual_inputs.device
        b, t, c, h, w = visual_inputs.shape

        # forward image and text features
        # feats are normalized embeds
        video_embeds, video_feat, text_embeds, text_feat = self.forward_feats(batch)
        image_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)

        # ========== (in-batch) ITC loss ==========
        gathered_image_feats = hvd.allgather(video_feat)
        gathered_text_feats = hvd.allgather(text_feat)

        assert self.itc_token_type == 'cls', 'Expecting CLS token for ITC, found {}'.format(self.itc_token_type)

        sim_v2t = video_feat @ gathered_text_feats.t() / self.temp 
        sim_t2v = text_feat @ gathered_image_feats.t() / self.temp 
                             
        # [IMPORTANT] be very careful when initializing the GT sim_i2t 
        # allgather return the concatenated features in the order of local_rank()
        sim_targets = torch.zeros_like(sim_v2t)

        local_rank = hvd.local_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.eye(b)#生成对角线全1，其余部分全0的二维数组

        sim_v2t_scores = F.log_softmax(sim_v2t, dim=1)
        sim_t2v_scores = F.log_softmax(sim_t2v, dim=1)

        loss_v2t = -torch.sum(sim_v2t_scores * sim_targets,dim=1).mean()
        loss_t2v = -torch.sum(sim_t2v_scores * sim_targets,dim=1).mean() 

        vtc_loss = (loss_v2t+loss_t2v) / 2

        return dict(
            itc_loss=vtc_loss,
            itc_labels=torch.max(sim_targets, dim=1)[1],
            i2t_scores=sim_v2t_scores,
            t2i_scores=sim_t2v_scores
        )


    def forward_feats(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        visual_inputs = batch['visual_inputs']

        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # image features
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)

        assert self.itc_token_type == 'cls', 'Expecting CLS token for ITC, found {}'.format(self.itc_token_type)
        if self.itc_token_type == 'cls':
            video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)
        else:
            raise NotImplementedError("itc_type_type must be one of ['mean', 'cls', 'mil'], found {}".format(self.itc_token_type))

        # text features
        text_output = self.text_encoder.bert(batch['text_input_ids'], 
                                             attention_mask=batch['text_input_mask'],                      
                                             return_dict = True, 
                                             mode = 'text'
                                            )

        text_embeds = text_output.last_hidden_state # b, Lt, fsz=768

        if self.itc_token_type == 'cls':
            text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
        else:
            raise NotImplementedError("itc_token_type must be one of ['mean', 'cls', 'mil'], found {}".format(self.itc_token_type))

        return video_embeds, video_feat, text_embeds, text_feat


class AlproForSequenceClassification(AlproBaseModel):
    def __init__(self, config, video_enc_cfg, input_format='RGB'):
        super(AlproForSequenceClassification, self).__init__(config, video_enc_cfg=video_enc_cfg)

        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config, add_pooling_layer=False)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )

    # def forward(self, image, text, targets, alpha=0, train=True):
    def forward(self, batch):
        visual_inputs = batch['visual_inputs']
        targets = batch['labels']

        device = visual_inputs.device

        # forward text
        text_input_mask = batch['text_input_mask']
        text_output = self.text_encoder(batch['text_input_ids'],
                                        attention_mask=text_input_mask,
                                        return_dict=True,
                                        mode='text'
                                        )
        text_embeds = text_output.last_hidden_state

        # forward visual
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        visual_inputs = visual_inputs.transpose(1, 2)

        image_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

        # forward cross-encoder
        attention_mask = torch.cat([text_input_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        output = self.text_encoder(encoder_embeds=embedding_output,
                                attention_mask=attention_mask,
                                return_dict=True,
                                mode='fusion'
                                )

        prediction = self.classifier(output.last_hidden_state[:,0,:])                
        if targets is not None:
            loss = F.cross_entropy(prediction, targets)                
        else: # evaluation mode
            loss = 0

        return dict(loss=loss,
                    logits=prediction
                    )
            

    def forward_inference(self, batch):
        visual_inputs = batch['visual_inputs']
        device = visual_inputs.device

        # forward text
        text_input_mask = batch['text_input_mask']
        text_output = self.text_encoder.bert(batch['text_input_ids'],
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state

        # forward visual
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        visual_inputs = visual_inputs.transpose(1, 2)

        image_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

        # forward cross-encoder
        attention_mask = torch.cat([text_input_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        output = self.text_encoder.bert(encoder_embeds=embedding_output,
                                        attention_mask=attention_mask,
                                        return_dict=True,
                                        mode='fusion'
                                    )

        prediction = self.classifier(output.last_hidden_state[:,0,:])                

        return prediction


class AlproForVideoTextRetrieval(AlproBaseModel):
    """
    """
    def __init__(self, config, video_enc_cfg, input_format='RGB'):
        super(AlproForVideoTextRetrieval, self).__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg)

    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        visual_inputs = batch['visual_inputs']
        text_input_mask = batch['text_input_mask']
        text_input_ids = batch['text_input_ids']

        device = visual_inputs.device

        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # visual embeddings

        #交换矩阵维度
        visual_inputs = visual_inputs.transpose(1, 2)
        print("TEST1!!!")

        #视频数据编码
        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)
        # image_embeds = image_embeds.repeat(text_input_mask.shape[0], 1, 1)
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)

        # text embeddings
        text_output = self.text_encoder.bert(text_input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

        # ========== (in-batch) ITC loss ==========
        #allgather聚合每个进程上的数据
        gathered_video_feats = hvd.allgather(video_feat)
        gathered_text_feats = hvd.allgather(text_feat)

        sim_v2t = video_feat @ gathered_text_feats.t() / self.temp 
        sim_t2v = text_feat @ gathered_video_feats.t() / self.temp 

        sim_targets = torch.zeros_like(sim_v2t)

        local_rank = hvd.local_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.eye(b)

        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean() 

        vtc_loss = (loss_v2t+loss_t2v) / 2

        # ========= ITM ==========
        text_atts = batch['text_input_mask']

        # non-masked text and non-masked image 
        vtm_loss, vtm_logits, vtm_labels = self.compute_vtm(text_embeds=text_embeds, 
                                                            text_atts=text_atts, 
                                                            image_embeds=video_embeds, 
                                                            image_atts=video_atts, 
                                                            sim_i2t=sim_v2t.clone(), # for hard mining
                                                            sim_t2i=sim_t2v.clone()  # for hard mining
                                                           )

        return dict(
            itm_scores=vtm_logits,
            itm_loss=vtm_loss,
            itm_labels=vtm_labels,
            itc_loss=vtc_loss
        )
    
    def compute_vtm(self, text_embeds, text_atts, image_embeds, image_atts, sim_i2t, sim_t2i):
        device = text_embeds.device

        # ====== positive pairs =======
        attention_mask = torch.cat([text_atts, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs_pos = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        # ====== negative pairs =======
        bs = text_embeds.shape[0] 

        local_rank = hvd.local_rank()
        b_start, b_end = bs * local_rank, bs * (local_rank + 1)

        with torch.no_grad():
            weights_v2t = sim_i2t[:,b_start:b_end]
            weights_t2v = sim_t2i[:,b_start:b_end]
   
            # never select self as negative
            weights_v2t.fill_diagonal_(-np.Inf)
            weights_t2v.fill_diagonal_(-np.Inf)

            weights_v2t = F.softmax(weights_v2t, dim=1)
            weights_t2v = F.softmax(weights_t2v, dim=1)

        # select a negative image for each text
        # FIXME to optimize using indexing operations
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2v[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_v2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text_atts, text_atts_neg],dim=0)     

        video_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        video_atts_all = torch.cat([image_atts,image_atts],dim=0)

        attention_mask_all = torch.cat([text_atts_all, video_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, video_embeds_all], dim=1)

        # forward negative pairs via cross encoder
        encoder_outputs_neg = self.text_encoder.bert(encoder_embeds=embedding_output_all,
                                                     attention_mask=attention_mask_all,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                   encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)
        vtm_logits = self.itm_head(vl_embeddings)            

        vtm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)], dim=0).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)     

        return vtm_loss, vtm_logits, vtm_labels 

    def forward_inference(self, batch):
        visual_inputs = batch['visual_inputs']#1,1,8,3,224,224
        text_input_mask = batch['text_input_mask']
        text_input_ids = batch['text_input_ids']

        device = visual_inputs.device

        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        visual_inputs = visual_inputs.transpose(1, 2)#1,3,8,224,224

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)#video embedding
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  #video feature

        video_embeds = video_embeds.repeat(text_input_mask.shape[0], 1, 1)#重复64次  torch.Size([64, 197, 768])
        # image_feat = image_feat.repeat(text_input_mask.shape[0], 1)

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)
        text_output = self.text_encoder.bert(text_input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state#shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)     #text feature

        vtc_sim_scores = video_feat @ text_feat.t() / self.temp #@矩阵乘法

        attention_mask = torch.cat([text_input_mask, video_atts], dim=1)
        embedding_output = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_outputs = self.text_encoder.bert(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                )

        vl_embeddings = encoder_outputs.last_hidden_state[:,0,:]
        logits = self.itm_head(vl_embeddings)

        return dict(logits=logits, itc_scores=vtc_sim_scores)

