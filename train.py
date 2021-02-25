import os
import json
import pickle as pkl
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn

from learning.weight_init import weight_init
from learning.metrics import confusion_matrix_analysis, get_conf_matrix

from models.model_init import get_model
from image.data import get_loaders
from configure import get_config

TIME_START = datetime.now()


def train_epoch(model, optimizer, criterion, loader, config):
    ts = datetime.now()
    device = torch.device(config['device'])
    loss = None
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        if (i + 1) % config['display_step'] == 0:
            print('Train Step {}, Loss: {:.4f}'.format(i + 1, loss.item()))

    t_delta = datetime.now() - ts
    print('Train Loss: {:.4f} in {:.2f} minutes in {} steps'.format(loss.item(),
                                                                    t_delta.seconds / 60.,
                                                                    i + 1))
    return {'train_loss': loss.item()}


def evaluate_epoch(model, loader, config, mode='valid'):
    ts = datetime.now()
    device = torch.device(config['device'])
    n_class = config['num_classes']
    confusion = np.zeros((n_class, n_class))

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.numpy()
        mask = (y > 0).flatten()
        y = y.flatten()
        with torch.no_grad():
            out = model(x)
            pred = torch.argmax(out, dim=1).flatten().cpu().numpy()
            confusion += get_conf_matrix(y[mask], pred[mask], n_class)

    per_class, overall = confusion_matrix_analysis(confusion)
    t_delta = datetime.now() - ts
    print('Evaluation: IOU: {:.4f}, '
          'in {:.2f} minutes, {} steps'.format(overall['iou'], t_delta.seconds / 60., i))

    if mode == 'valid':
        overall['{}_iou'.format(mode)] = overall['iou']
        return overall
    elif mode == 'test':
        overall['{}_iou'.format(mode)] = overall['iou']
        return overall, confusion


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['res_dir'], 'figures'), exist_ok=True)


def checkpoint(log, config):
    with open(os.path.join(config['res_dir'], 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, conf_mat, config):
    with open(os.path.join(config['res_dir'], 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'conf_mat.pkl'), 'wb'))


def overall_performance(config, conf):
    _, perf = confusion_matrix_analysis(conf)
    print('Test Precision {:.4f}, Recall {:.4f}, F1 Score {:.2f}'
          ''.format(perf['precision'], perf['recall'], perf['f1-score']))

    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))


def train(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])

    prepare_output(config)
    device = torch.device(config['device'])

    train_loader, test_loader, val_loader = get_loaders(config)
    model = get_model(config)
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    model.apply(weight_init)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # config['N_params'] = model.param_ratio()

    with open(os.path.join(config['res_dir'], 'config.json'), 'w') as _file:
        _file.write(json.dumps(config, indent=4))

    train_log = {}
    best_iou = 0.0

    print('\nTrain {}'.format(config['model'].upper()))
    for epoch in range(1, config['epochs'] + 1):
        print('\nEPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, config=config)
        model.eval()
        val_metrics = evaluate_epoch(model, val_loader, config=config)

        train_log[epoch] = {**train_metrics, **val_metrics}
        if val_metrics['iou'] >= best_iou:
            best_iou = val_metrics['iou']
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(config['res_dir'], 'model.pth.tar'))

    print('\nRun test set....')
    model.load_state_dict(torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))['state_dict'])
    model.eval()
    metrics, conf = evaluate_epoch(model, test_loader, config=config, mode='test')
    overall_performance(config, conf)
    t_delta = datetime.now() - TIME_START
    print('Total Time: {:.2f} minutes'.format(t_delta.seconds / 60.))


if __name__ == '__main__':
    config = get_config('unet')
    train(config)

# ========================================================================================