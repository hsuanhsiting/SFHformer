import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.UDC_DataLoader import UDCTrainData, UDCTestData  # 修改导入路径
from numpy import *
from pytorch_msssim import ssim
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SFHformer_s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_root', default='/data/xxting/datasets/UDC-SIT/', type=str, help='path to dataset root')  # 修改参数名称
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='udc', type=str, help='experiment setting')  # 修改默认实验名称
args = parser.parse_args()

torch.manual_seed(8001)

def train(train_loader, network, criterion, optimizer):
    losses = AverageMeter()
    network.train()

    for batch in train_loader:
        lq_img = batch['lq'].cuda()  # 修改字段名称
        gt_img = batch['gt'].cuda()  # 修改字段名称

        pred_img = network(lq_img)
        loss_content = criterion(pred_img, gt_img)

        # 频域损失计算
        gt_fft = torch.fft.fft2(gt_img, dim=(-2, -1))
        gt_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        
        pred_fft = torch.fft.fft2(pred_img, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        
        loss_fft = criterion(pred_fft, gt_fft)
        
        loss = loss_content + 0.1 * loss_fft
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg

def valid(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    network.eval()

    for batch in val_loader:
        lq_img = batch['lq'].cuda()  # 修改字段名称
        gt_img = batch['gt'].cuda()  # 修改字段名称

        with torch.no_grad():
            output = network(lq_img).clamp_(0, 1)

        # 计算指标
        mse_loss = F.mse_loss(output, gt_img, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), lq_img.size(0))
        
        ssim_val = ssim(output, gt_img).mean()
        SSIM.update(ssim_val.item(), lq_img.size(0))

    return PSNR.avg, SSIM.avg

if __name__ == '__main__':
    # 加载配置文件
    config_path = os.path.join('configs', args.exp, f"{args.model}.json")
    if not os.path.exists(config_path):
        config_path = os.path.join('configs', args.exp, 'default.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 初始化模型
    device_ids = [0, 1, 2, 3]
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=device_ids).cuda()

    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=config['lr'])

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'],
        eta_min=config['lr'] * 1e-2
    )

    # 数据加载
    train_dataset = UDCTrainData(  # 使用新的数据加载类
        crop_size=256,
        data_root=os.path.join(args.data_root, 'train'),
        only_h_flip=False
    )
    
    test_dataset = UDCTestData(  # 使用新的测试数据类
        data_root=os.path.join(args.data_root, 'test'),
        local_size=32
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=RandomSampler(
            train_dataset, 
            num_samples=config['batch_size'] * 500,
            replacement=True
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,  # 测试集通常不shuffle
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 准备保存路径
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    
    model_tag = f'udc_{args.model}_v1'  # 修改模型标识
    model_path = os.path.join(save_dir, f'{model_tag}.pth')

    if not os.path.exists(model_path):
        print(f'==> 开始训练 {args.model} 模型')

        # 初始化TensorBoard
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, model_tag))

        best_psnr = 0
        best_ssim = 0

        for epoch in tqdm(range(config['epochs'] + 1)):
            # 训练阶段
            train_loss = train(train_loader, network, criterion, optimizer)
            writer.add_scalar('Loss/train', train_loss, epoch)
            scheduler.step()

            # 验证阶段
            if epoch % config['eval_freq'] == 0:
                avg_psnr, avg_ssim = valid(test_loader, network)
                
                # 记录指标
                writer.add_scalar('Metric/PSNR', avg_psnr, epoch)
                writer.add_scalar('Metric/SSIM', avg_ssim, epoch)

                # 保存最新模型
                torch.save({'state_dict': network.state_dict()},
                          os.path.join(save_dir, f'{model_tag}_latest.pth'))

                # 更新最佳模型
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                              os.path.join(save_dir, f'{model_tag}_best_psnr.pth'))
                
                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                    torch.save({'state_dict': network.state_dict()},
                              os.path.join(save_dir, f'{model_tag}_best_ssim.pth'))

                # 记录最佳指标
                writer.add_scalar('Best/PSNR', best_psnr, epoch)
                writer.add_scalar('Best/SSIM', best_ssim, epoch)

        writer.close()
    else:
        print('==> 已存在训练好的模型')
        exit(1)