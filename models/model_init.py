from models.unet.unet import UNet


def get_model(config):
    model = None

    if config['model'] == 'unet':
        model = UNet(config)

    return model


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
