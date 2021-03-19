import os
import json
from datetime import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.unet.unet import UNet
from configure import get_config


def prepare_output(config):
    dt = datetime.now().strftime('%Y.%m.%d.%H.%M-{}-{}'.format(config['model'], config['mode']))
    new_dir = os.path.join(config['res_dir'], dt)
    os.makedirs(new_dir, exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'checkpoints'), exist_ok=True)
    with open(os.path.join(new_dir, 'config.json'), 'w') as file:
        file.write(json.dumps(config, indent=4))
    return new_dir


def main(model, mode, gpu=None):

    config = get_config(model, mode, gpu=gpu)

    model = UNet(channels=config['input_dim'], classes=config['n_classes'])
    model.configure_model(**config)

    log_dir = prepare_output(config)
    logger = TensorBoardLogger(log_dir, name='log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=1,
        monitor='val_acc',
        verbose=True)

    stop_callback = EarlyStopping(
        monitor='val_acc',
        mode='auto',
        patience=15,
        verbose=False)

    trainer = Trainer(
        precision=16,
        min_epochs=100,
        limit_val_batches=250,
        gpus=config['device_ct'],
        num_nodes=config['node_ct'],
        callbacks=[checkpoint_callback, stop_callback],
        log_gpu_memory='min_max',
        log_every_n_steps=5,
        logger=logger)

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    main(args.model, args.mode, args.gpu)
# ========================================================================================
