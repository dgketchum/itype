import os
import json
from datetime import datetime
from models.unet.unet import UNet
from configure import get_config

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def prepare_output(config):
    dt = datetime.now().strftime('%Y.%m.%d.%H.%M')
    new_dir = os.path.join(config['res_dir'], dt)
    os.makedirs(new_dir, exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'checkpoints'), exist_ok=True)
    with open(os.path.join(new_dir, 'config.json'), 'w') as file:
        file.write(json.dumps(config, indent=4))
    return new_dir


def main():

    config = get_config('unet')
    model = UNet(channels=config['input_dim'], classes=config['n_classes'])
    log_dir = prepare_output(config)
    logger = TensorBoardLogger(log_dir, name='log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=1,
        monitor='val_loss',
        verbose=True)

    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=5,
        verbose=True)

    trainer = Trainer(
        precision=16,
        overfit_batches=10,
        gpus=config['device_ct'],
        num_nodes=config['node_ct'],
        callbacks=[checkpoint_callback, stop_callback],
        log_gpu_memory='min_max',
        logger=logger)

    trainer.fit(model)


if __name__ == '__main__':
    main()

# ========================================================================================
