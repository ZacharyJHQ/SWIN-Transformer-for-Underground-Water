# --------------------------------------------------------
# Swin Transformer
# --------------------------------------------------------

from .swin_transformer import SwinTransformer


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        # 获取输出头模式，如果配置中没有则使用默认值
        output_head_mode = getattr(config.MODEL, 'OUTPUT_HEAD', None)
        if output_head_mode is not None:
            output_head_mode = output_head_mode.MODE
        else:
            output_head_mode = 'hybrid'  # 默认使用混合模式
            
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                out_chans=config.MODEL.OUT_CHANS,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                output_head_mode=output_head_mode)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
