# -*- coding: utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.Rain_Dataloader import TrainData_for_Rain100H, ValData, TestData, TestData_for_Rain100H
from numpy import *
from models import *
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SFHformer_m', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='rain', type=str, help='experiment setting')
args = parser.parse_args()

torch.manual_seed(8001)


def train(train_loader, network, criterion, optimizer):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        pred_img = network(source_img)
        label_img = target_img
        l3 = criterion(pred_img, label_img)
        loss_content = l3

        label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(pred_img, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

        f3 = criterion(pred_fft3, label_fft3)
        loss_fft = f3

        loss = loss_content + 0.1 * loss_fft
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(network.parameters(), 0.01)
        optimizer.step()

    return losses.avg


def valid(val_loader_full, network):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()

    # torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader_full:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img).clamp_(0, 1)  # we change this to [0,1]?

        psnr_full, sim = calculate_psnr_torch(target_img, output)
        PSNR_full.update(psnr_full.item(), source_img.size(0))

        ssim_full = sim
        SSIM_full.update(ssim_full.item(), source_img.size(0))

    return PSNR_full.avg, SSIM_full.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    print(setting_filename)
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    device_index = [0, 1, 2, 3]
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=device_index).cuda()

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'], eps=1e-8)
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: wrunsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=1e-6)

    train_dir = '/home/jxy/projects_dir/datasets/Rain100/rain_data_train_Heavy'
    test_dir = '/home/jxy/projects_dir/datasets/Rain100/rain_heavy_test'
    train_dataset = TrainData_for_Rain100H(256, train_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    test_dataset = TestData_for_Rain100H(8, test_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # change test_str when you development new exp
    test_str = 'rain_rain100H_sfhformer_m'

    if not os.path.exists(os.path.join(save_dir, args.model + test_str + '.pth')):
        print('==> Start training, current model name: ' + args.model)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model, test_str))

        best_psnr = 0
        best_ssim = 0

        for epoch in tqdm(range(setting['epochs'] + 1)):
            train_loss = train(train_loader, network, criterion, optimizer)
            writer.add_scalar('train_loss', train_loss, epoch)
            scheduler.step()

            if epoch % setting['eval_freq'] == 0:

                avg_psnr, avg_ssim = valid(test_loader, network)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('valid_ssim', avg_ssim, epoch)

                torch.save({'state_dict': network.state_dict()},
                           os.path.join(save_dir, args.model + test_str + '_newest' + '.pth'))

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + test_str + '_best' + '.pth'))
                writer.add_scalar('best_psnr', best_psnr, epoch)

                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                writer.add_scalar('best_ssim', best_ssim, epoch)

    else:
        print('==> Existing trained model')
        exit(1)
