import json
import math
import os
import pprint
import time
from os.path import join
import sys
import horovod.torch as hvd
import torch
from apex import amp
from src.configs.config import shared_configs
from src.datasets.data_utils import ImageNorm
from src.datasets.dataloader_SECNet import MetaLoader, PrefetchLoader
from src.datasets.dataset_pretrain_SECNet import PretrainSparseDataset, PretrainImageTextDataset, PretrainCollator
from src.modeling.alpro_models_SECNet import SECNet_Pertrain
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from src.utils.basic_utils import load_json, read_dataframe
from src.utils.distributed import all_gather_list
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_pos_embed_resizing)
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import BertConfig, BertTokenizerFast


def mk_captions_pretrain_dataloader(dataset_name, anno_path, video_dir, txt_dir, cfg, tokenizer, 
                                    is_train=True, max_txt_len=80):
    # make a list(dict), where each dict {vis_id: int, txt: str}
    #这个地方直接改成dataset_name=="VSET"
    if dataset_name == "VSET":
        datalist = json.load(open(anno_path))
        LOGGER.info('Found {} entries for VSET'.format(len(datalist)))

    else:
        raise ValueError("Invalid dataset_name")

    if dataset_name in ["VSET"]:
        dataset = PretrainImageTextDataset(datalist=datalist,
                                           tokenizer=tokenizer,
                                           crop_size=cfg.crop_img_size,
                                           resize_size=cfg.resize_size,
                                           img_lmdb_dir=video_dir,
                                           max_txt_len=cfg.max_txt_len,
                                           wh1_num_frm=cfg.wh1_num_frm,
                                           wh2_num_frm=cfg.wh2_num_frm,
                                           where_num_frm=cfg.where_num_frm,
                                           )

    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, ")
    batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    data_collator = PretrainCollator(tokenizer=tokenizer,
                                    mlm=cfg.use_mlm,
                                    mlm_probability=0.15,
                                    max_length=cfg.max_txt_len,
                                    mpm=cfg.use_mpm,
                                    is_train=is_train)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=data_collator.collate_batch)

    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")

    train_loaders = {}
    #从cfg.train_datasets里加载参数，直接在cfg里修改文本保存地址ann和图片保存地址img
    for db in cfg.train_datasets:
        train_loaders[db.name] = mk_captions_pretrain_dataloader(
            dataset_name=db.name,
            anno_path=db.ann, video_dir=db.img, txt_dir=db.txt,
            cfg=cfg, tokenizer=tokenizer, is_train=True
        )

    val_loaders = {}
    for db in cfg.val_datasets:
        val_loaders[db.name] = mk_captions_pretrain_dataloader(
            dataset_name=db.name,
            anno_path=db.ann, video_dir=db.img, txt_dir=db.txt,
            cfg=cfg, tokenizer=tokenizer, is_train=False
        )
    return train_loaders, val_loaders


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)

    LOGGER.info("setup e2e model")

    if cfg.model_type == 'pretrain':
        # initialize cnn config


        #Object & Object State
        WH1_enc_cfg = load_json(cfg.visual_model_cfg)#config_release/timesformer_divst_8x32_224_k600.json
        WH1_enc_cfg["wh1_num_frm"] = 2
        WH1_enc_cfg["img_size"] = cfg.crop_img_size

        WH2_enc_cfg = load_json(cfg.visual_model_cfg)
        WH2_enc_cfg["wh2_num_frm"] = 2
        WH2_enc_cfg["img_size"] = cfg.crop_img_size


        #Location & Time
        Where_enc_cfg = load_json(cfg.visual_model_cfg)
        Where_enc_cfg["where_num_frm"] = 2
        Where_enc_cfg["img_size"] = cfg.crop_img_size


        model = SECNet_Pertrain(
            model_cfg, 
            input_format=cfg.img_input_format,
            WH1_enc_cfg=WH1_enc_cfg,
            WH2_enc_cfg=WH2_enc_cfg,
            Where_enc_cfg=Where_enc_cfg
            )


        LOGGER.info(f"Loading visual weights from {cfg.visual_weights_path}")#vit_base_patch16_224
        model.load_separate_ckpt(
            visual_weights_path=cfg.visual_weights_path,
            prompter_weights_path=cfg.teacher_weights_path
        )


    else:
        raise NotImplementedError(f"cfg.model_type not found {cfg.model_type}.")

    # if cfg.freeze_cnn:
    #     model.freeze_cnn_backbone()
    
    LOGGER.info("Moving model to device") 
    model.to(device)
    LOGGER.info("Completed moving model to device.") 

    LOGGER.info("Setup model done!")
    return model


def forward_step(cfg, model, batch):
    """shared for training and validation"""
    # used to make visual feature copies
    if not cfg.use_itm:
        batch["itm_labels"] = None
    outputs = model(batch)  # dict
    return outputs


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()

    vtc_loss_WH1 = 0
    vtc_loss_WH2 = 0
    vtc_loss_wwcap = 0
    vtc_loss_fusion = 0
    st = time.time()
    val_log = {'train/vtc_loss_WH1': 0,
               'train/vtc_loss_WH2': 0,
               'train/vtc_loss_wwcap': 0,
                'train/vtc_loss_fusion': 0
                 }
    debug_step = 5
    val_loaders = val_loader if isinstance(val_loader, dict) else {
        "unnamed_val_loader": val_loader}
    
    total_val_iters = 0 

    LOGGER.info(f"In total {len(val_loaders)} val loaders")
    for loader_name, val_loader in val_loaders.items():
        LOGGER.info(f"Loop val_loader {loader_name}.")

        total_val_iters += len(val_loader)
        for val_step, batch in enumerate(val_loader):
            # use iter to reset MetaLoader
            # forward pass
            outputs = forward_step(cfg, model, batch)
            vtc_loss_WH1 += outputs["vtc_loss_WH1"].sum().item()

            vtc_loss_WH2 += outputs["vtc_loss_WH2"].sum().item()

            vtc_loss_fusion += outputs["vtc_loss_fusion"].sum().item()

            vtc_loss_wwcap += outputs["vtc_loss_wwcap"].sum().item()

            if cfg.debug and val_step >= debug_step:
                break

    # Gather across all processes
    avg_vtc_loss_WH1 = vtc_loss_WH1 / len(val_loader)
    avg_vtc_loss_WH2 = vtc_loss_WH2 / len(val_loader)
    avg_vtc_loss_fusion = vtc_loss_fusion / len(val_loader)
    avg_vtc_loss_wwcap = vtc_loss_wwcap / len(val_loader)

    all_gather_vtc_loss_WH1 = all_gather_list(avg_vtc_loss_WH1)
    all_gather_vtc_loss_WH2 = all_gather_list(avg_vtc_loss_WH2)
    all_gather_vtc_loss_fusion = all_gather_list(avg_vtc_loss_fusion)
    all_gather_vtc_loss_wwcap = all_gather_list(avg_vtc_loss_wwcap)
    vtc_loss_WH1 = sum(all_gather_vtc_loss_WH1)
    vtc_loss_WH2 = sum(all_gather_vtc_loss_WH2)
    vtc_loss_wwcap = sum(all_gather_vtc_loss_wwcap)
    vtc_loss_fusion = sum(all_gather_vtc_loss_fusion)

    val_log.update({
        'valid/vtc_loss_WH1': vtc_loss_WH1,
        'valid/vtc_loss_WH2': vtc_loss_WH2,
        'valid/vtc_loss_wwcap': vtc_loss_wwcap,
        'valid/vtc_loss_fusion': vtc_loss_fusion
    })

    TB_LOGGER.log_scalar_dict(val_log)
    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, ")


    LOGGER.info("[vtc_loss_WH1: {} ".format(vtc_loss_WH1))
    LOGGER.info("[vtc_loss_WH2: {} ".format(vtc_loss_WH2))
    LOGGER.info("[vtc_loss_wwcap: {} ".format(vtc_loss_wwcap))
    LOGGER.info("[vtc_loss_fusion]: {} ".format(vtc_loss_fusion))
    LOGGER.info("In total, {} validation iters.".format(total_val_iters))

    model.train()
    return val_log

def start_training():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cfg = shared_configs.get_sparse_pretraining_args()
    set_random_seed(cfg.seed)
    print("hvd.size is ",hvd.size())

    n_gpu = hvd.size()

    os.environ['CUDA_VISIBLE_DEVICES']="0,1,2"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)

    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info(f"device: {device} n_gpu: {n_gpu}, "
                f"rank: {hvd.rank()}, 16-bits training: {cfg.fp16}")


    model = setup_model(cfg, device=device)
    model.train()


    optimizer = setup_e2e_optimizer(model, cfg)

    # Horovod: (optional) compression algorithm.compressin
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    compression = hvd.Compression.none
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level='O1')
        # keep_batchnorm_fp32=True)

    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)

    train_loaders, val_loaders = setup_dataloaders(cfg, tokenizer)
    train_loader = MetaLoader(train_loaders,
                              accum_steps=cfg.gradient_accumulation_steps,
                              distributed=n_gpu > 1)
    #
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm)
                   for k, v in val_loaders.items()}

    # compute the number of steps and update cfg
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)#train_batch_size=16
    total_n_epochs = cfg.num_train_epochs#10
    cfg.num_train_steps = int(math.ceil(
        1. * train_loader.n_batches_in_epoch * total_n_epochs /
        (n_gpu * cfg.gradient_accumulation_steps)))
    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1
    
    save_steps = int(cfg.save_steps_ratio * cfg.num_train_steps)

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        LOGGER.info("Saving training meta...")
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
    LOGGER.info(f"  Total #batches - single epoch = {train_loader.n_batches_in_epoch}.")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Total #epochs = {total_n_epochs}.")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")


    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()
    debug_step = 20

    tasks = []
    for name in ["vtc_loss_WH1", "vtc_loss_WH2","vtc_loss_wwcap","vtc_loss_fusion"]:
        tasks.append(name)
    task2loss = {t: RunningMeter(f'train_loss/{t}')
                 for t in tasks}
    task2loss["loss"] = RunningMeter('train_loss/loss')

    train_log = {'train/vtc_loss_WH1': 0,
                 'train/vtc_loss_WH2': 0,
                 'train/vtc_loss_wwcap': 0,
                 'train/vtc_loss_fusion': 0
                 }

    for step, (task, batch) in enumerate(train_loader):
        # forward pass
        outputs = forward_step(cfg, model, batch)
        vtc_loss_WH1 = outputs["vtc_loss_WH1"]
        task2loss["vtc_loss_WH1"](vtc_loss_WH1.item())
        vtc_loss_WH2 = outputs["vtc_loss_WH2"]
        task2loss["vtc_loss_WH2"](vtc_loss_WH2.item())
        vtc_loss_wwcap = outputs["vtc_loss_wwcap"]
        task2loss["vtc_loss_wwcap"](vtc_loss_wwcap.item())
        vtc_loss_fusion = outputs["vtc_loss_fusion"]
        task2loss["vtc_loss_fusion"](vtc_loss_fusion.item())


        loss = vtc_loss_wwcap+vtc_loss_WH1 + vtc_loss_WH2 + vtc_loss_fusion
        task2loss["loss"](loss.item())

        if step % cfg.log_interval == 0:
            print(f"log_interval is : {cfg.log_interval}")
            train_log.update({
                'train/vtc_loss_WH1': vtc_loss_WH1,
                'train/vtc_loss_WH2': vtc_loss_WH2,
                'train/vtc_loss_wwcap': vtc_loss_wwcap,
                 'train/vtc_loss_fusion': vtc_loss_fusion
            })

            TB_LOGGER.log_scalar_dict(train_log)

        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
            scaled_loss.backward()
            zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            if (step + 1) % cfg.log_interval == 0:
                TB_LOGGER.log_scalar_dict({l.name: l.val
                                        for l in task2loss.values()
                                        if l.val is not None})
            n_epoch = int(1. * n_gpu * cfg.gradient_accumulation_steps *
                          global_step / train_loader.n_batches_in_epoch)

            # learning rate scheduling for the whole model
            lr_this_step = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs,
                multi_step_epoch=n_epoch)

            # Hardcoded param group length
            # assert len(optimizer.param_groups) == 8
            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                    param_group['lr'] = lr_this_step

            if (step + 1) % cfg.log_interval == 0:
                TB_LOGGER.add_scalar(
                    "train/lr", lr_this_step, global_step)

            # update model params
            if cfg.grad_norm != -1:
                # import pdb; pdb.set_trace()
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer), cfg.grad_norm)
                if (step + 1) % cfg.log_interval == 0:
                    TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()

            #restorer.step()
            pbar.update(1)

            # validate and checkpoint
            # if global_step % cfg.valid_steps == 0:
            if global_step % 200 == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                validate(model, val_loaders, cfg)
                #model_saver.save(step=global_step, model=model)
            
            if global_step % save_steps == 0:
                LOGGER.info(f'Step {global_step}: saving model checkpoints.')
                model_saver.save(step=global_step, model=model)

        if global_step >= cfg.num_train_steps:
            break

        if cfg.debug and global_step >= debug_step:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, val_loaders, cfg)
        model_saver.save(step=global_step, model=model)


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    # print(hvd.size())
    start_training()
