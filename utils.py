# --------------------------------------------------------
# Swin Transformer
# --------------------------------------------------------

import os
import torch
# Removed distributed import for single GPU conversion
# import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    # Check if resume path is provided and exists
    if not config.MODEL.RESUME or config.MODEL.RESUME == '':
        logger.info("No checkpoint to resume from, starting from scratch")
        return float('inf')
    
    if not os.path.exists(config.MODEL.RESUME):
        logger.warning(f"Checkpoint file not found: {config.MODEL.RESUME}, starting from scratch")
        return float('inf')
    
    logger.info(f"==============> Resuming from {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        try:
            # Try with weights_only=False for compatibility with older checkpoints
            checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=False: {e}")
            try:
                # Fallback to weights_only=True (safer but may fail with some checkpoints)
                checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=True)
            except Exception as e2:
                logger.error(f"Failed to load checkpoint: {e2}")
                logger.info("Starting from scratch")
                return float('inf')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    min_mae = float('inf')
    
    # 检查是否有不匹配的键，如果有则跳过优化器和调度器的加载
    has_missing_keys = len(msg.missing_keys) > 0
    has_unexpected_keys = len(msg.unexpected_keys) > 0
    
    if has_missing_keys or has_unexpected_keys:
        logger.warning("Model structure has changed, skipping optimizer and scheduler loading")
        logger.warning(f"Missing keys: {msg.missing_keys}")
        logger.warning(f"Unexpected keys: {msg.unexpected_keys}")
        # 重置开始epoch为0，从头开始训练
        config.defrost()
        config.TRAIN.START_EPOCH = 0
        config.freeze()
    elif not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except Exception as e:
            logger.warning(f"Failed to load optimizer/scheduler state: {e}")
            logger.warning("Continuing with fresh optimizer and scheduler")
            config.defrost()
            config.TRAIN.START_EPOCH = 0
            config.freeze()
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            config.freeze()
            if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0" and amp is not None:
                amp.load_state_dict(checkpoint['amp'])
            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            if 'min_mae' in checkpoint:
                min_mae = checkpoint['min_mae']
        else:
            logger.info("=> loaded model weights only, starting fresh training")

    del checkpoint
    torch.cuda.empty_cache()
    return min_mae


def save_checkpoint(config, epoch, model, min_mae, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'min_mae': min_mae,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0" and amp is not None:
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    """Simplified for single GPU conversion - just return the tensor directly"""
    return tensor



