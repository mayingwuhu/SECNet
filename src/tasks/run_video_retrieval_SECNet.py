import math
import math
import os
import random
import sys
import time
from collections import defaultdict
from os.path import exists, join

import horovod.torch as hvd
import numpy as np
import torch
import torch.nn.functional as F
from apex import amp
from easydict import EasyDict as edict

from src.configs.config import shared_configs
from src.datasets.data_utils import ImageNorm, mk_input_group
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader_my, PrefetchLoader
from src.datasets.dataset_video_retrieval_my_re_all_wwcap import (
    AlproVideoRetrievalDataset, AlproVideoRetrievalEvalDataset,
    VideoRetrievalCollator,VideoRetrievalCollator_my_re)
from src.modeling.alpro_models_my_re_all_wwcap import AlproForVideoTextRetrieval, AlproForPretrain_my_1, AlproForPretrain_my_2, AlproForPretrain_my_3
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer, setup_e2e_optimizer_pre
from src.utils.basic_utils import (load_json,
                               load_jsonl, save_json)
from src.utils.distributed import all_gather_list
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.utils.load_save import (ModelSaver,
                                 load_state_dict_with_pos_embed_resizing,
                                 save_training_meta)
from src.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from src.utils.misc import NoOp, set_random_seed, zero_none_grad
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import BertConfig, BertTokenizerFast


def mk_video_ret_datalist(raw_datalist, cfg):
    """
    Args:
        raw_datalist: list(dict):raw_datalist中的一条数据{"caption": "a person is connecting something to system", "clip_name": "video9770", "retrieval_key": "ret0"}
        目标格式：Each data point is {id: int, txt: str, vid_id: str}

    Returns:

    """
    LOGGER.info(f"Loaded data size {len(raw_datalist)}")
    if cfg.data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * cfg.data_ratio)]
        LOGGER.info(f"Use {100 * cfg.data_ratio}% of the loaded data: {len(raw_datalist)}")

    datalist = []
    qid = 0
    for raw_d in raw_datalist:
        d = dict(
            id=qid,
            txt=raw_d["items"][0]["caption"],
            vid_id=raw_d["video_name"],
            items=raw_d["items"]
        )
        qid += 1
        datalist.append(d)
    LOGGER.info(f"datalist {len(datalist)}")
    '''分析OK 输出形式[{'id': 0, 'txt': 'a person is connecting something to system', 'vid_id': 'video9770'}, {'id': 1, 'txt': 'a little girl does gymnastics', 'vid_id': 'video9771'}, {'id': 2, 'txt': 'a woman creating a fondant baby and flower', 'vid_id': 'video7020'}, {'id': 3, 'txt': 'a boy plays grand theft auto 5', 'vid_id': 'video9773'}]'''
    return datalist


def mk_video_ret_dataloader(anno_path, lmdb_dir, cfg, tokenizer, is_train=True):
    """"""
    raw_datalist = load_json(anno_path)
    datalist = mk_video_ret_datalist(raw_datalist, cfg)
    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["vid_id"]].append(d)
    LOGGER.info(f"grouped {len(grouped)}")

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group=cfg.max_n_example_per_group if is_train else 1,  # force 1 in eval,
        is_train=is_train
    )
    LOGGER.info(f"group_datalist {len(group_datalist)}")

    dataset = AlproVideoRetrievalDataset(
        datalist=group_datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=lmdb_dir,
        max_img_size=cfg.crop_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        itm_neg_size=0,
        is_train=is_train,
        img_db_type='rawvideo',
        crop_size=cfg.crop_img_size,
        resize_size=cfg.resize_size,
        wh1_num_frm=cfg.wh1_num_frm,
        wh2_num_frm=cfg.wh2_num_frm,
        where_num_frm=cfg.where_num_frm
    )
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    vqa_collator = VideoRetrievalCollator_my_re(
        tokenizer=tokenizer, max_length=cfg.max_txt_len, mlm=cfg.use_mlm,
                                    mlm_probability=0.15,
                                    mpm=cfg.use_mpm,
                                    is_train=True)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vqa_collator.collate_batch)
    return dataloader


def mk_video_ret_eval_dataloader(anno_path, lmdb_dir, cfg, tokenizer):
    """
    anno_path:text.jsonl  eg:{"caption": "a person is connecting something to system", "clip_name": "video9770", "retrieval_key": "ret0"}
    lmdb_dir:视频存储文件夹
    tokenizer:下载的tokenizer所在文件夹
    eval_retrieval: bool, will sample one video per batch paired with multiple text.
    Returns:

    """
    raw_datalist = load_json(anno_path)#读取jsonl文件，以一句一句的格式存储为列表
    datalist = mk_video_ret_datalist(raw_datalist, cfg)
    frm_sampling_strategy = cfg.frm_sampling_strategy#default:rand
    if frm_sampling_strategy == "rand":
        frm_sampling_strategy = "uniform"

    if 'msvd' in cfg.train_datasets[0]['name']:
        video_fmt = '.avi'
    else:
        video_fmt = '.mp4'

    dataset = AlproVideoRetrievalEvalDataset(
        datalist=datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=lmdb_dir,
        max_img_size=cfg.crop_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        video_fmt=video_fmt,
        img_db_type='rawvideo'
    )
    #dataset[0] =  {"vid_id": 'video9770', "examples": [], 'n_examples' = 1000, 'ids' = [0, 1, ..., 999],'vid':[采样帧图像数据]}

    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=False)#分布式采样器，sampler提供数据集中元素的索引
    retrieval_collator = VideoRetrievalCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=1,  # already batched in dataset
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=retrieval_collator.collate_batch)
    #整合多个样本到一个batch时需要调用的函数，当 __getitem__ 返回的不是tensor而是字典之类时，需要进行 collate_fn的重载，同时可以进行数据的进一步处理以满足pytorch的输入要求
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    dataloader = PrefetchLoader(dataloader, img_norm)
    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")
    train_loader = mk_video_ret_dataloader(
        anno_path=cfg.train_datasets[0].ann,
        lmdb_dir=cfg.train_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=True
    )
    val_loader = mk_video_ret_dataloader(
        anno_path=cfg.val_datasets[0].ann,
        lmdb_dir=cfg.val_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=False
    )
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    train_loader = PrefetchLoader_my(train_loader, img_norm)
    val_loader = PrefetchLoader(val_loader, img_norm)
    return train_loader, val_loader


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)#一样的
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    add_attr_list = []
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    '''这个是re的'''
    LOGGER.info("setup e2e model")
    video_enc_cfg = load_json(cfg.visual_model_cfg)
    video_enc_cfg['num_frm'] = cfg.num_frm
    video_enc_cfg['img_size'] = cfg.crop_img_size

    '''这个是pre的'''
    WH1_enc_cfg = load_json(cfg.visual_model_cfg)#config_release/timesformer_divst_8x32_224_k600.json
    WH1_enc_cfg["wh1_num_frm"] = 2#后续需要在cfg里给成wh_num_frm = 2，但表示的是同一张图片中的两部分内容what/who-how
    WH1_enc_cfg["img_size"] = cfg.crop_img_size
    WH2_enc_cfg = load_json(cfg.visual_model_cfg)
    WH2_enc_cfg["wh2_num_frm"] = 2  # 后续需要在cfg里给成wh_num_frm = 2，但表示的是同一张图片中的两部分内容what/who-how
    WH2_enc_cfg["img_size"] = cfg.crop_img_size
    #Where的图像编码器
    Where_enc_cfg = load_json(cfg.visual_model_cfg)
    Where_enc_cfg["where_num_frm"] = 2  # 后续需要在cfg里给成where_num_frm = 1
    Where_enc_cfg["img_size"] = cfg.crop_img_size

    '''问题定位'''
    '''re'''
    model = AlproForVideoTextRetrieval(
        model_cfg, 
        input_format=cfg.img_input_format,
        video_enc_cfg=video_enc_cfg
        )
    '''pre_1'''
    model_1 = AlproForPretrain_my_1(
        model_cfg,
        input_format=cfg.img_input_format,
        WH1_enc_cfg=WH1_enc_cfg,
        WH2_enc_cfg=WH2_enc_cfg,
        Where_enc_cfg=Where_enc_cfg
    )
    '''pre_2'''
    model_2 = AlproForPretrain_my_2(
        model_cfg,
        input_format=cfg.img_input_format,
        WH1_enc_cfg=WH1_enc_cfg,
        WH2_enc_cfg=WH2_enc_cfg,
        Where_enc_cfg=Where_enc_cfg
    )
    '''pre_3'''
    model_3 = AlproForPretrain_my_3(
        model_cfg,
        input_format=cfg.img_input_format,
        WH1_enc_cfg=WH1_enc_cfg,
        WH2_enc_cfg=WH2_enc_cfg,
        Where_enc_cfg=Where_enc_cfg
    )


    '''re模型加载'''
    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        num_patches = (cfg.crop_img_size // video_enc_cfg['patch_size']) ** 2
        # NOTE strict if False if loaded from ALBEF ckpt
        load_state_dict_with_pos_embed_resizing(model,
                                                cfg.e2e_weights_path, 
                                                num_patches=num_patches, 
                                                num_frames=cfg.num_frm, 
                                                strict=False,
                                                )
    else:
        LOGGER.info(f"Loading visual weights from {cfg.visual_weights_path}")
        LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
        model.load_separate_ckpt(
            visual_weights_path=cfg.visual_weights_path,
            bert_weights_path=cfg.bert_weights_path
        )

    '''pre模型加载,cfg文件里的pre_weights_path是pre的模型参数'''
    '''321为训练不带初始的 给注释掉'''
    state_dict = torch.load(cfg.pre_weights_path)
    LOGGER.info(f"Loading pre weights from {cfg.pre_weights_path}")
    try:
        model_1.load_state_dict(state_dict)
        model_2.load_state_dict(state_dict)
        model_3.load_state_dict(state_dict)
    except:
        LOGGER.info(f"!!!!!!!!!!!!!!!!")
    # model_4.load_state_dict(state_dict)
    LOGGER.info(f"Loading pre finish")

    # model_1.load_separate_ckpt(
    #     visual_weights_path=cfg.visual_weights_path,
    #     prompter_weights_path=cfg.teacher_weights_path
    # )
    # model_2.load_separate_ckpt(
    #     visual_weights_path=cfg.visual_weights_path,
    #     prompter_weights_path=cfg.teacher_weights_path
    # )
    # model_3.load_separate_ckpt(
    #     visual_weights_path=cfg.visual_weights_path,
    #     prompter_weights_path=cfg.teacher_weights_path
    # )

    model.to(device)
    model_1.to(device)
    model_2.to(device)
    model_3.to(device)
    LOGGER.info("Setup model done!")
    return model,model_1,model_2,model_3


def forward_step(model, batch):
    """shared for training and validation"""
    '''re的model输出'''
    outputs = model(batch)  # dict
    return outputs


def forward_step_pre(model_1, model_2, model_3,batch):
    outputs_1 = model_1(batch)
    outputs_2 = model_2(batch)
    outputs_3 = model_3(batch)
    # outputs_4 = model_4(batch)
    return outputs_1, outputs_2, outputs_3

def forward_inference_step(model, batch):
    outputs = model.forward_inference(batch)
    return outputs

@torch.no_grad()
def validate(model, val_loader, eval_loader, cfg, train_global_step, eval_filepath):
    """use eval_score=False when doing inference on test sets where answers are not available"""
    model.eval()

    loss = 0.
    n_ex = 0
    n_corrects = 0
    st = time.time()
    debug_step = 10

    for val_step, batch in enumerate(val_loader):
        # forward pass
        del batch["caption_ids"]#删除batch中的“caption_ids"
        outputs = forward_step(model, batch)
        targets = batch['labels']

        batch_loss = outputs['itm_loss'] + outputs['itc_loss']

        if isinstance(batch_loss, torch.Tensor):
            loss += batch_loss.sum().item()
        else:
            raise NotImplementedError('Expecting loss as Tensor, found: {}'.format(type(loss)))

        # n_ex += len(targets)
        n_ex += len(targets)

        if cfg.debug and val_step >= debug_step:
            break

    loss = sum(all_gather_list(loss))
    n_ex = sum(all_gather_list(n_ex))
    n_corrects = sum(all_gather_list(n_corrects))

    _, retrieval_metrics = inference_retrieval(model, eval_loader, eval_filepath, cfg)

    model.train()

    if hvd.rank() == 0:
        # average loss for each example
        acc = float(n_corrects / n_ex)
        val_log = {'valid/loss': float(loss / n_ex), 'valid/acc': acc}
        for ret_type, ret_m in retrieval_metrics.items():
            val_log.update({f"valid/{ret_type}_{k}": round(v, 4) for k, v in ret_m.items()})

        TB_LOGGER.log_scalar_dict(val_log)
        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds."
                    f"itm_acc: {acc}. Retrieval res {retrieval_metrics}")


def start_training(cfg):
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    cfg.n_gpu = n_gpu
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    '''加载五个模型'''
    # model, model_1, model_2, model_3, model_4 = setup_model(cfg, device=device)
    model, model_1, model_2, model_3 = setup_model(cfg, device=device)
    LOGGER.info("Start .train")
    model.train()
    model_1.train()
    model_2.train()
    model_3.train()
    # model_4.train()
    '''定义五个模型的优化器'''
    LOGGER.info("Start setup optimizer")
    optimizer = setup_e2e_optimizer(model, cfg)
    optimizer_1 = setup_e2e_optimizer_pre(model_1, cfg)
    optimizer_2 = setup_e2e_optimizer_pre(model_2, cfg)
    optimizer_3 = setup_e2e_optimizer_pre(model_3, cfg)
    # optimizer_4 = setup_e2e_optimizer_pre(model_4, cfg)

    # Horovod: (optional) compression algorithm.compressin
    LOGGER.info("Start hvd.DistributedOptimizer")
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)
    # for name, param in model_1.named_parameters():
    #     print(name)
    optimizer_1 = hvd.DistributedOptimizer(
        optimizer_1, named_parameters=model_1.named_parameters(),
        compression=compression)
    optimizer_2 = hvd.DistributedOptimizer(
        optimizer_2, named_parameters=model_2.named_parameters(),
        compression=compression)
    optimizer_3 = hvd.DistributedOptimizer(
        optimizer_3, named_parameters=model_3.named_parameters(),
        compression=compression)
    # optimizer_4 = hvd.DistributedOptimizer(
    #     optimizer_4, named_parameters=model_4.named_parameters(),
    #     compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    LOGGER.info("Start hvd.broadcast_parameters")
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    LOGGER.info("Start hvd.broadcast_parameters1")
    hvd.broadcast_parameters(model_1.state_dict(), root_rank=0)
    LOGGER.info("Start hvd.broadcast_parameters2")
    hvd.broadcast_parameters(model_2.state_dict(), root_rank=0)
    LOGGER.info("Start hvd.broadcast_parameters3")
    hvd.broadcast_parameters(model_3.state_dict(), root_rank=0)
    # LOGGER.info("Start hvd.broadcast_parameters4")
    # hvd.broadcast_parameters(model_4.state_dict(), root_rank=0)
    LOGGER.info("Start hvd.broadcast_parameters5")
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    LOGGER.info("Start hvd.broadcast_parameters6")
    hvd.broadcast_optimizer_state(optimizer_1, root_rank=0)
    LOGGER.info("Start hvd.broadcast_parameters7")
    hvd.broadcast_optimizer_state(optimizer_2, root_rank=0)
    LOGGER.info("Start hvd.broadcast_parameters8")
    hvd.broadcast_optimizer_state(optimizer_3, root_rank=0)
    # LOGGER.info("Start hvd.broadcast_parameters9")
    # hvd.broadcast_optimizer_state(optimizer_4, root_rank=0)

    LOGGER.info("Start amp.initialize")
    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level='O2',
        keep_batchnorm_fp32=True)
    model_1, optimizer_1 = amp.initialize(
        model_1, optimizer_1, enabled=cfg.fp16, opt_level='O1')
    model_2, optimizer_2 = amp.initialize(
        model_2, optimizer_2, enabled=cfg.fp16, opt_level='O1')
    model_3, optimizer_3 = amp.initialize(
        model_3, optimizer_3, enabled=cfg.fp16, opt_level='O1')
    # model_4, optimizer_4 = amp.initialize(
    #     model_4, optimizer_4, enabled=cfg.fp16, opt_level='O1')


    # prepare data
    LOGGER.info("Start prepare data")
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    train_loader, val_loader = setup_dataloaders(cfg, tokenizer)
    eval_loader = mk_video_ret_eval_dataloader(
        anno_path=cfg.eval_datasets[0].ann,
        lmdb_dir=cfg.eval_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer,
    )

    # compute the number of steps and update cfg
    LOGGER.info("Start compute the number of steps and update cfg")
    total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    cfg.num_train_steps = int(math.ceil(
        1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))

    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        LOGGER.info("Saving training meta...")
        #save_training_meta(cfg)
        LOGGER.info("Saving training done...")
        TB_LOGGER.create(join(cfg.output_dir, 'log'))
        pbar = tqdm(total=cfg.num_train_steps)
        model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
        add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()

    if global_step > 0:
        pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
    LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
    LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
    LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")

    LOGGER.info(f'Step {global_step}: start validation')
    #validate(
    #    model, val_loader, eval_loader, cfg, global_step,
    #    eval_filepath=cfg.val_datasets[0].ann)

    # quick hack for amp delay_unscale bug
    LOGGER.info('Start optimizer.skip_synchronize')
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        # optimizer_4.zero_grad()
        if global_step == 0:
            optimizer.step()
            optimizer_1.step()
            optimizer_2.step()
            optimizer_3.step()
            # optimizer_4.step()
    debug_step = 3
    LOGGER.info('Start running_loss')
    running_loss = RunningMeter('train_loss')

    LOGGER.info('Start for step, batch in enumerate(InfiniteIterator(train_loader))')
    for step, batch in enumerate(InfiniteIterator(train_loader)):
        # forward pass
        # LOGGER.info('x-1')
        del batch["caption_ids"]
        mini_batch = dict()
        # LOGGER.info('x—2')
        for k, v in batch.items():
            if k != "visual_inputs":
                mini_batch[k] = v

        # LOGGER.info('x-3')
        pool_method = cfg.score_agg_func
        # could be 1, where only a single clip is used
        num_clips = cfg.train_n_clips

        assert num_clips == 1, "Support only single clip for now."

        # LOGGER.info('x-4')
        num_frm = cfg.num_frm
        # (B, T=num_clips*num_frm, C, H, W) --> (B, num_clips, num_frm, C, H, W)
        bsz = batch["visual_inputs"].shape[0]
        # LOGGER.info('x-5')
        new_visual_shape = (bsz, num_clips, num_frm) + batch["visual_inputs"].shape[2:]
        # LOGGER.info('x-6')
        visual_inputs = batch["visual_inputs"].view(*new_visual_shape)
        model_out = []

        # LOGGER.info('Start for clip_idx in range(num_clips)')
        for clip_idx in range(num_clips):
            # (B, num_frm, C, H, W)
            mini_batch["visual_inputs"] = visual_inputs[:, clip_idx]
            mini_batch["n_examples_list"] = batch["n_examples_list"]
            outputs = forward_step(model, mini_batch)
            # outputs_1, outputs_2, outputs_3, outputs_4 = forward_step_pre(model_1, model_2, model_3,model_4, batch)
            outputs_1, outputs_2, outputs_3 = forward_step_pre(model_1, model_2, model_3, batch)
            model_out.append(outputs)
            # the losses are cross entropy and mse, no need to * num_labels

        # LOGGER.info('Start calculate loss')
        '''re的loss'''
        loss_itm = outputs['itm_loss']
        loss_itc = outputs['itc_loss']
        '''pre的loss'''
        # loss_1 = (outputs_1["vtc_loss_WH1"] + outputs_1["vtc_loss_WH2"] + outputs_1["vtc_loss_fusion"] + outputs_1["vtc_loss_wwcap"])
        # loss_2 = (outputs_2["vtc_loss_WH1"] + outputs_2["vtc_loss_WH2"] + outputs_2["vtc_loss_fusion"] + outputs_2["vtc_loss_wwcap"])
        # loss_3 = (outputs_3["vtc_loss_WH1"] + outputs_3["vtc_loss_WH2"] + outputs_3["vtc_loss_fusion"] + outputs_3["vtc_loss_wwcap"])
        loss_1 = outputs_1["vtc_loss_fusion"]
        loss_2 = outputs_2["vtc_loss_fusion"]
        loss_3 = outputs_3["vtc_loss_fusion"]

        '''总共的loss'''
        # loss = loss_itm + loss_itc + (loss_1 + loss_2 + loss_3 + loss_4)*0.1
        loss = loss_itm + loss_itc + loss_1 + loss_2 + loss_3
        running_loss(loss.item())
        # backward pass
        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        # LOGGER.info('Start with amp.scale_loss')

        with amp.scale_loss(
                loss, optimizer_3, delay_unscale=delay_unscale
                ) as scaled_loss_3:
            scaled_loss_3.backward(retain_graph=True)
            zero_none_grad(model_3)
            optimizer_3.synchronize()

        with amp.scale_loss(
                loss, optimizer_2, delay_unscale=delay_unscale
                ) as scaled_loss_2:
            scaled_loss_2.backward(retain_graph=True)
            zero_none_grad(model_2)
            optimizer_2.synchronize()

        with amp.scale_loss(
                loss, optimizer_1, delay_unscale=delay_unscale
                ) as scaled_loss_1:
            scaled_loss_1.backward(retain_graph=True)
            zero_none_grad(model_1)
            optimizer_1.synchronize()


        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
            scaled_loss.backward()
            zero_none_grad(model)
            optimizer.synchronize()

        # with amp.scale_loss(
        #         loss, optimizer_4, delay_unscale=delay_unscale
        #         ) as scaled_loss:
        #     scaled_loss.backward()
        #     zero_none_grad(model_4)
        #     optimizer_4.synchronize()

        # optimizer
        # LOGGER.info('Start if (step + 1) % cfg.gradient_accumulation_steps')
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            # learning rate scheduling
            n_epoch = int(1. * total_train_batch_size * global_step
                          / total_n_examples)
            # learning rate scheduling cnn
            '''re的lr学习率'''
            lr_this_step = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs,
                multi_step_epoch=n_epoch)
            '''pre的lr学习率'''
            lr_this_step_pre = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate_pre,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio_pre,
                decay_epochs=cfg.step_decay_epochs,
                multi_step_epoch=n_epoch)

            # Hardcoded param group length
            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                    param_group['lr'] = lr_this_step
            for pg_n, param_group in enumerate(
                    optimizer_1.param_groups):
                    param_group['lr'] = lr_this_step_pre
            for pg_n, param_group in enumerate(
                    optimizer_2.param_groups):
                    param_group['lr'] = lr_this_step_pre
            for pg_n, param_group in enumerate(
                    optimizer_3.param_groups):
                    param_group['lr'] = lr_this_step_pre
            # for pg_n, param_group in enumerate(
            #         optimizer_4.param_groups):
            #         param_group['lr'] = lr_this_step_pre

            if step % cfg.log_interval == 0:
                TB_LOGGER.add_scalar(
                    "train/lr", lr_this_step, global_step)
            TB_LOGGER.add_scalar('train/loss', running_loss.val, global_step)
            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer),
                    cfg.grad_norm)
                grad_norm_pre = clip_grad_norm_(
                    amp.master_params(optimizer_1),
                    cfg.grad_norm_pre)
                TB_LOGGER.add_scalar(
                    "train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()
                optimizer_1.step()
                optimizer_1.zero_grad()
                optimizer_2.step()
                optimizer_2.zero_grad()
                optimizer_3.step()
                optimizer_3.zero_grad()
                # optimizer_4.step()
                # optimizer_4.zero_grad()
            #restorer.step()
            pbar.update(1)

            # checkpoint

            '''不用validate先 注释掉
            if (global_step*2) % cfg.valid_steps == 0 :
                LOGGER.info(f'Step {global_step}: start validation')
                validate(
                    model, val_loader, eval_loader, cfg, global_step,
                    eval_filepath=cfg.val_datasets[0].ann)
            '''
            # if global_step % int(cfg.valid_steps/4) == 0:
            '''
            if global_step >=750 and global_step <= 800:
                if global_step % 5 == 0:
                # if global_step % 100 == 0:
                    model_saver.save(step=global_step, model=model)
            '''
            if global_step % 150 == 0:
            # if global_step % 100 == 0:
                model_saver.save(step=global_step, model=model)
        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(
            model, val_loader, eval_loader, cfg, global_step,
            eval_filepath=cfg.val_datasets[0].ann)
        model_saver.save(step=global_step, model=model)


def get_retrieval_metric_from_bool_matrix(bool_matrix):
    """ Calc Recall@K, median rank and mean rank.
    Args:
        bool_matrix: np array of shape (#txt, #vid), np.bool,
            sorted row-wise from most similar to less similar.
            The GT position is marked as 1, while all the others are 0,
            each row will only have one 1.

    Returns:
        retrieval_metrics: dict(
            R1=.., R5=..., R10=..., MedR=..., MeanR=...
        )
    """
    num_row = bool_matrix.shape[0]  # #rows
    row_range, gt_ranks = np.where(bool_matrix == 1)
    assert np.allclose(row_range, np.arange(len(row_range))), \
        "each row should only a single GT"
    retrieval_metrics = dict(
        r1=100 * bool_matrix[:, 0].sum() / num_row,
        r5=100 * bool_matrix[:, :5].sum() / num_row,
        r10=100 * bool_matrix[:, :10].sum() / num_row,
        medianR=np.median(gt_ranks+1),  # convert to 1-indexed system instead of 0-indexed.
        meanR=np.mean(gt_ranks+1)
    )
    return retrieval_metrics


def get_retrieval_scores(score_matrix, gt_row2col_id_mapping, row_idx2id, col_id2idx):
    # rank scores
    score_matrix_sorted, indices_sorted = \
        torch.sort(score_matrix, dim=1, descending=True)  # (#txt, #vid)

    # build bool matrix, where the GT position is marked as 1, all the others are 0,
    num_row = len(score_matrix)
    gt_col_indices = torch.zeros(num_row, 1)
    for idx in range(num_row):
        gt_col_id = gt_row2col_id_mapping[row_idx2id[idx]]
        gt_col_indices[idx, 0] = col_id2idx[gt_col_id]

    bool_matrix = indices_sorted == gt_col_indices  # (#txt, #vid)
    retrieval_metrics = get_retrieval_metric_from_bool_matrix(bool_matrix.numpy())
    return retrieval_metrics


def eval_retrieval(vid_txt_score_dicts, gt_txt_id2vid_id, id2data):
    """
    Args:
        vid_txt_score_dicts: list(dict), each dict is dict(vid_id=..., txt_id=..., score=...)
        gt_txt_id2vid_id: dict, ground-truth {txt_id: vid_id}
        id2data: dict, {txt_id: single_example_data}

    Returns:

    """
    # group prediction by txt_id
    scores_group_by_txt_ids = defaultdict(list)
    for d in vid_txt_score_dicts:
        scores_group_by_txt_ids[d["txt_id"]].append(d)

    # clean duplicated videos
    _scores_group_by_txt_ids = defaultdict(list)
    for txt_id, txt_vid_pairs in scores_group_by_txt_ids.items():
        added_vid_ids = []
        for d in txt_vid_pairs:
            if d["vid_id"] not in added_vid_ids:
                _scores_group_by_txt_ids[txt_id].append(d)
                added_vid_ids.append(d["vid_id"])
    scores_group_by_txt_ids = _scores_group_by_txt_ids

    num_txt = len(scores_group_by_txt_ids)
    any_key = list(scores_group_by_txt_ids.keys())[0]
    vid_ids = [d["vid_id"] for d in scores_group_by_txt_ids[any_key]]
    num_vid = len(vid_ids)
    assert len(set(vid_ids)) == num_vid, "Each caption should be compared to each video only once."
    for k, v in scores_group_by_txt_ids.items():
        assert num_vid == len(v), "each captions should be compared with the same #videos."

    # row/col indices in the score matrix
    # *_id are the original ids, *_idx are the matrix indices
    txt_id2idx = {txt_id: idx for idx, txt_id in enumerate(scores_group_by_txt_ids)}
    vid_id2idx = {vid_id: idx for idx, vid_id in enumerate(vid_ids)}
    txt_idx2id = {v: k for k, v in txt_id2idx.items()}
    vid_idx2id = {v: k for k, v in vid_id2idx.items()}

    # build score (float32) and vid_id (str) matrix
    score_matrix = torch.zeros(num_txt, num_vid)
    sim_matrix = torch.zeros(num_txt, num_vid)
    for txt_id, preds in scores_group_by_txt_ids.items():
        txt_idx = txt_id2idx[txt_id]
        for p in preds:
            vid_idx = vid_id2idx[p["vid_id"]]
            score_matrix[txt_idx, vid_idx] = p["score"]
            sim_matrix[txt_idx, vid_idx] = p['sim']


    t2v_retrieval_metrics = get_retrieval_scores(
        score_matrix, gt_txt_id2vid_id, txt_idx2id, vid_id2idx)
    # video to text retrieval, score_matrix--> (#vid, #txt)
    # given a video, retrieve most relevant videos
    score_matrix = score_matrix.transpose(0, 1)
    gt_vid_id2txt_id = {v: k for k, v in gt_txt_id2vid_id.items()}
    v2t_retrieval_metrics = get_retrieval_scores(
        score_matrix, gt_vid_id2txt_id, vid_idx2id, txt_id2idx)
    retrieval_metrics = dict(
        text2video=t2v_retrieval_metrics,
        video2text=v2t_retrieval_metrics
    )
    return retrieval_metrics


@torch.no_grad()
def inference_retrieval(model, val_loader, eval_file_path, cfg):
    model.eval()
    retrieval_res = []  # list(dict): dict(vid_id=..., txt_id=..., score=...)
    st = time.time()
    eval_bsz = cfg.inference_batch_size if cfg.do_inference else cfg.eval_retrieval_batch_size#eval_bsz = cfg.inference_batch_size=64
    LOGGER.info(f"Evaluate retrieval #video per GPU: {len(val_loader)}")
    if hvd.rank() == 0:
        pbar = tqdm(total=len(val_loader), desc="eval")

    for batch in val_loader:
        # each batch contains 1 video and N (=1000) captions
        n_mini_batches = math.ceil(len(batch["caption_ids"]) / eval_bsz)#math.ceil(x)获得大于等于x的最小整数 len(batch["caption_ids"]=1000  eval_bsz=64
        vid_id = batch["vid_id"]#'video9770'
        #print("========================",vid_id)
        for idx in range(n_mini_batches):
            # compile shared text part
            mini_batch = dict()
            for k in ["text_input_ids", "text_input_mask", "labels"]:
                if batch[k] is not None:
                    mini_batch[k] = batch[k][idx * eval_bsz:(idx + 1) * eval_bsz]
                else:
                    mini_batch[k] = None
            caption_ids = batch["caption_ids"][idx * eval_bsz:(idx + 1) * eval_bsz]
            # bsz = len(caption_ids)
            mini_batch["n_examples_list"] = [len(caption_ids)]

            num_clips = cfg.inference_n_clips
            num_frm = cfg.num_frm
            # (B, T=num_clips*num_frm, C, H, W) --> (B, num_clips, num_frm, C, H, W)
            new_visual_shape = (1, num_clips, num_frm) + batch["visual_inputs"].shape[2:]#(1,1,8,3,H,W)
            #print("new_visual_shape:",new_visual_shape)
            visual_inputs = batch["visual_inputs"].view(*new_visual_shape)#(1,1,8,3,244,244)
            #print("visual_inputs.shape:",visual_inputs.shape)
            logits = []
            sim_scores = []

            for clip_idx in range(num_clips):
                mini_batch["visual_inputs"] = visual_inputs[:, clip_idx]#(1,8,3,244,244),clip_idx = 0
                if cfg.fp16:
                    # FIXME not sure why we need to do this explicitly?
                    mini_batch["visual_inputs"] = mini_batch["visual_inputs"].half()
                outputs = forward_inference_step(model, mini_batch)
                logits.append(outputs["logits"].cpu())
                sim_scores.append(outputs["itc_scores"].cpu())

            logits = torch.stack(logits)  # (num_frm, B, 1 or 2) [1, 64, 2] [1,1,2]


            sim_scores = torch.stack(sim_scores)
            
            # FIXME not sure why need to convert dtype explicitly
            logits = logits.squeeze(0).float()#shape:[64,2] squeeze()去掉维度为1的
            # print("logits.shape:", logits.shape)
            # print("logits.shape[1]:", logits.shape[1])
            # print("logits.score:", logits)
            sim_scores = sim_scores.squeeze().float().tolist()
            # print("sim_scores.shape:",sim_scores)

            if logits.shape[1] == 2:
                # [dxli] uses 1 for positive and 0 for negative.
                # therefore we choose dim=1
                probs = F.softmax(logits, dim=1)[:, 1].tolist()
                # print("probs.score:",probs)
                # print("probs.shape:",len(probs))
                # print("caption_ids.shape",caption_ids)
                #probs = logits[0]
            else:
                raise NotImplementedError('Not supported (unclear purposes)!')
            for cap_id, score, sim in zip(caption_ids, probs, sim_scores):
                retrieval_res.append(dict(
                    vid_id=vid_id,
                    txt_id=cap_id,
                    score=round(score, 4),
                    sim=round(sim, 4)
                ))#round返回浮点数四舍五入的值

        if hvd.rank() == 0:
            pbar.update(1)

    # ###### Saving with Horovod ####################
    # dummy sync
    _ = None
    all_gather_list(_)
    n_gpu = hvd.size()
    eval_dir = join(cfg.output_dir, f"results_{os.path.splitext(os.path.basename(eval_file_path))[0]}")
    os.makedirs(eval_dir, exist_ok=True)
    if n_gpu > 1:
        # with retrial, as azure blob fails occasionally.
        max_save_load_trial = 10
        save_trial = 0
        while save_trial < max_save_load_trial:
            try:
                LOGGER.info(f"Save results trial NO. {save_trial}")
                save_json(
                    retrieval_res,
                    join(eval_dir, f"tmp_results_rank{hvd.rank()}.json"))
                break
            except Exception as e:
                print(f"Saving exception: {e}")
                save_trial += 1

    # dummy sync
    _ = None
    all_gather_list(_)
    # join results
    if n_gpu > 1 and hvd.rank() == 0:
        retrieval_res = []
        for rk in range(n_gpu):
            retrieval_res.extend(load_json(
                join(eval_dir, f"tmp_results_rank{rk}.json")))
        LOGGER.info('results joined')

    if hvd.rank() == 0:
        retrieval_metrics = eval_retrieval(
            retrieval_res, val_loader.dataset.gt_cap_id2vid_id, val_loader.dataset.id2data)
        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds. scores: {retrieval_metrics}")
    else:
        retrieval_metrics = None

    model.train()
    return retrieval_res, retrieval_metrics


def start_inference(cfg):
    set_random_seed(cfg.seed)

    '''
    hvd多GPU分布式处理
    hvd.local_rank()是当前节点上的GPU资源列表
    hvd.rank()，是一个全局GPU资源列表
    譬如有4台节点，每台节点上4块GPU，则num_workers的范围为0~15，local_rank为0~3
    '''
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True

    '''
    os.path.splitext分开前缀与文件名，join(output/downstreams/msrvtt_ret/public,result_test,step_)
    cfg.inference_txt_db:data/msrvtt_ret/txt/test.jsonl
    os.path.basename(cfg.inference_txt_db):test.jsonl
    os.path.splitext(os.path.basename(cfg.inference_txt_db)):[test,jsonl]
    '''
    inference_res_dir = join(
        cfg.output_dir,
        f"results_{os.path.splitext(os.path.basename(cfg.inference_txt_db))[0]}/"
        f"step_{cfg.inference_model_step}_{cfg.inference_n_clips}_{cfg.score_agg_func}"
    )

    '''hvd.rank() == 0 判断是否为master，理论上只有master负责模型存储与输出'''
    if hvd.rank() == 0:
        os.makedirs(inference_res_dir, exist_ok=True)
        save_json(cfg, join(inference_res_dir, "raw_args.json"),
                  save_pretty=True)

    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    # overwrite cfg with stored_cfg,
    # but skip keys containing the keyword 'inference'
    stored_cfg_path = join(cfg.output_dir, "log/args.json")
    stored_cfg = edict(load_json(stored_cfg_path))
    for k, v in cfg.items():
        if k in stored_cfg and "inference" not in k and "output_dir" not in k:
            setattr(cfg, k, stored_cfg[k])#cfg.k=stored_cfg[k]

    # setup models
    cfg.model_config = join(cfg.output_dir, "log/model_config.json")
    e2e_weights_path = join(
        cfg.output_dir, f"ckpt/model_step_{cfg.inference_model_step}.pt")
    if exists(e2e_weights_path):
        cfg.e2e_weights_path = e2e_weights_path
    else:
        raise NotImplementedError("Not supporting loading separate weights for inference.")
    model = setup_model(cfg, device=device)
    model.eval()

    # FIXME separate scaling for each loss
    model = amp.initialize(
        model, enabled=cfg.fp16, opt_level='O2')

    global_step = 0
    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)#config_release/msrvtt_ret.json:"tokenizer_dir": "ext/bert-base-uncased/"
    cfg.data_ratio = 1.
    '''
    val_loader 的制作路径
    anno_path:data/msrvtt_qa/txt/test.jsonl
    lmdb_dir:data/msrvtt_qa/videos
    '''
    val_loader = mk_video_ret_eval_dataloader(
        anno_path=cfg.inference_txt_db,
        lmdb_dir=cfg.inference_img_db,
        cfg=cfg, tokenizer=tokenizer,
    )

    LOGGER.info(cfg)
    LOGGER.info("Starting inference...")
    LOGGER.info(f"***** Running inference with {n_gpu} GPUs *****")
    LOGGER.info(f"  Batch size = {cfg.inference_batch_size}")

    LOGGER.info(f'Step {global_step}: start validation')
    ret_results, ret_scores = inference_retrieval(
        model, val_loader, cfg.inference_txt_db, cfg)

    if hvd.rank() == 0:
        save_json(cfg, join(inference_res_dir, "merged_args.json"),
                  save_pretty=True)
        save_json(ret_results, join(inference_res_dir, "results.json"),
                  save_pretty=True)
        save_json(ret_scores, join(inference_res_dir, "scores.json"),
                  save_pretty=True)


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    input_cfg = shared_configs.get_video_retrieval_args()
    if input_cfg.do_inference:
        start_inference(input_cfg)
    else:
        start_training(input_cfg)
