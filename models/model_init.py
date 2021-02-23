from models.unet.unet import UNet


def get_model(config):
    model = None

    if config['model'] == 'unet':
        model_config = dict(n_channels=config['input_dim'], n_classes=config['num_classes'])
        model = UNet(**model_config)

    return model


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
