# --------------------------------------------------------
# Swin Transformer
# Modified for prediction task
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn

from timm.utils import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def get_output_head_mode(config):
    """
    从配置中获取输出头模式
    支持多种配置格式
    """
    # 方法1: 检查 MODEL.OUTPUT_HEAD.MODE
    if hasattr(config.MODEL, 'OUTPUT_HEAD') and hasattr(config.MODEL.OUTPUT_HEAD, 'MODE'):
        return config.MODEL.OUTPUT_HEAD.MODE
    
    # 方法2: 从TAG中推断
    tag = getattr(config, 'TAG', '').lower()
    if 'adaptive' in tag:
        return 'reorganization'
    elif 'pangu' in tag or 'hybrid' in tag:
        return 'hybrid'
    elif 'transpose' in tag:
        return 'transpose'
    
    # 方法3: 从模型名称中推断
    model_name = getattr(config.MODEL, 'NAME', '').lower()
    if 'adaptive' in model_name:
        return 'reorganization'
    elif 'pangu' in model_name or 'hybrid' in model_name:
        return 'hybrid'
    
    # 默认返回转置卷积
    return 'transpose'


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args, unparsed = parser.parse_known_args()
    
    config = get_config(args)

    return args, config


def main(config, device, logger):
    dataset_train, data_loader_train = build_loader(config, dataset_type='train')
    dataset_test, data_loader_test = build_loader(config, dataset_type='test')
    dataset_val, data_loader_val = build_loader(config, dataset_type='val')

    # Debug information about datasets
    logger.info(f"Dataset sizes - Train: {len(dataset_train)}, Test: {len(dataset_test)}, Val: {len(dataset_val)}")
    logger.info(f"DataLoader lengths - Train: {len(data_loader_train)}, Test: {len(data_loader_test)}, Val: {len(data_loader_val)}")
    
    if len(dataset_train) == 0:
        logger.error("Training dataset is empty! Please check your data path and structure.")
        logger.error(f"Data path: {config.DATA.DATA_PATH}")
        return

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.to(device)
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = torch.nn.MSELoss()
    min_mae = float('inf')

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        min_mae = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        mae, mse = validate(config, data_loader_val, model, device, logger)
        logger.info(f"MAE of the network on the {len(dataset_val)} test images: {mae:.4f}%")
        if config.EVAL_MODE:
            mae, mse = validate(config, data_loader_test, model, device, logger)
            logger.info(f"MAE of the network on the {len(dataset_test)} test images: {mae:.4f}")
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger, device)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # Set epoch for sampler if it supports it (for proper shuffling)
        if hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, device, logger)
        
        # Save checkpoint at specified intervals or at the end of training
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, min_mae, optimizer, lr_scheduler, logger)

        mae, mse = validate(config, data_loader_test, model, device, logger)
        logger.info(f"MAE of the network on the {len(dataset_test)} test images: {mae:.4f}%")
        min_mae = min(min_mae, mae)
        logger.info(f'Min MAE: {min_mae:.4f}%')

    mae, mse = validate(config, data_loader_val, model, device, logger)
    logger.info(f"Final MAE on the {len(dataset_val)} val images: {mae:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, device, logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets, _) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                memory_used = 0
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, device, logger):
    criterion = torch.nn.MSELoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    end = time.time()
    for idx, (images, target, output_paths) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        output = model(images)

        # Save predictions for validation set
        # Check if dataset has root attribute (for compatibility with different dataset types)
        try:
            is_val_dataset = hasattr(data_loader.dataset, 'root') and 'val' in data_loader.dataset.root.lower()
        except:
            is_val_dataset = False
        
        if is_val_dataset:
            for i, path in enumerate(output_paths):
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    pred_img = output[i].cpu().numpy().transpose(1, 2, 0) * 255
                    Image.fromarray(pred_img.astype(np.uint8)).save(path)

        # measure accuracy and record loss
        loss = criterion(output, target)
        mae = torch.mean(torch.abs(output - target))

        loss_meter.update(loss.item(), target.size(0))
        mae_meter.update(mae.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                memory_used = 0
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'MAE {mae_meter.val:.4f} ({mae_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * MAE {mae_meter.avg:.4f} MSE {loss_meter.avg:.4f}')
    return mae_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger, device):
    model.eval()

    for idx, (images, _, _) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()
    """
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"
    """
    # Single GPU/CPU device detection and setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(0)  # Use first available GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    
    # Scale learning rate for gradient accumulation steps
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    
    # 生成基于输出头类型和时间戳的输出路径
    output_head_mode = get_output_head_mode(config)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"{output_head_mode}_{current_time}"
    
    # 设置输出路径为根目录下的output文件夹
    config.OUTPUT = os.path.join("output", output_folder_name)
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    
    # Save config file
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    main(config, device, logger)
