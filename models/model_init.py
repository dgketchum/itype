from models.unet.unet import UNet


def get_model(config):
    model = None

    if config['model'] == 'unet':
        model_config = dict(input_dim=config['input_dim'], kernel_size=config['kernel_size'],
                            hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                            batch_first=True, bias=True, return_all_layers=False)
        model = UNet(**model_config)

    return model


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
