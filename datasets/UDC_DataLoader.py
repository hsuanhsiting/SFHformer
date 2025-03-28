import os.path
import time
import cv2
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from math import ceil

def read_img255(filename):
    """读取并标准化图像到0-255范围"""
    img0 = cv2.imread(filename)
    img1 = img0[:, :, ::-1].astype('float32') / 1.0  # BGR转RGB
    return img1

# ----------------------- 数据增强函数 ----------------------- #
def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # 如果图像尺寸小于目标尺寸，先进行缩放
    if min(H, W) < size:
        scale_ratio = size / min(H, W)
        new_H, new_W = int(H * scale_ratio), int(W * scale_ratio)
        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    # 随机裁剪
    H, W, _ = imgs[0].shape
    Hs = random.randint(0, H - Hc) if H > Hc else 0
    Ws = random.randint(0, W - Wc) if W > Wc else 0
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    # 水平翻转
    if random.random() > 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    # 随机旋转（仅在非仅水平翻转时启用）
    if not only_h_flip:
        rot_deg = random.choice([0, 1, 2, 3])
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs

# ----------------------- 训练数据集类 ----------------------- #
class UDCTrainData(data.Dataset):
    def __init__(self, crop_size, data_root, only_h_flip=False):
        super().__init__()
        self.lq_dir = os.path.join(data_root, 'LQ')
        self.gt_dir = os.path.join(data_root, 'HQ')
        
        # 验证路径存在
        assert os.path.exists(self.lq_dir), f"LQ路径不存在: {self.lq_dir}"
        assert os.path.exists(self.gt_dir), f"HQ路径不存在: {self.gt_dir}"
        
        # 获取匹配的文件列表
        self.lq_names = sorted([f for f in os.listdir(self.lq_dir) if f.endswith('.png')])
        self.gt_names = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])
        
        # 验证文件匹配
        assert len(self.lq_names) == len(self.gt_names), "LQ和HQ文件数量不匹配"
        assert all(lq == gt for lq, gt in zip(self.lq_names, self.gt_names)), "文件名不匹配"

        self.crop_size = crop_size
        self.only_h_flip = only_h_flip
        self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        lq_name = self.lq_names[index]
        gt_name = self.gt_names[index]
        
        # 读取图像
        lq_path = os.path.join(self.lq_dir, lq_name)
        gt_path = os.path.join(self.gt_dir, gt_name)
        lq_img = read_img255(lq_path)
        gt_img = read_img255(gt_path)
        
        # 数据增强
        [lq_img, gt_img] = augment(
            [lq_img, gt_img],
            size=self.crop_size,
            edge_decay=0.,
            only_h_flip=self.only_h_flip
        )
        
        # 转换为Tensor
        lq_tensor = self.transform(np.ascontiguousarray(lq_img).astype('uint8'))
        gt_tensor = self.transform(np.ascontiguousarray(gt_img).astype('uint8'))
        
        return {
            'lq': lq_tensor,
            'gt': gt_tensor,
            'filename': lq_name
        }

    def __len__(self):
        return len(self.lq_names)

# ----------------------- 测试数据集类 ----------------------- #
class UDCTestData(data.Dataset):
    def __init__(self, data_root, local_size=32):
        super().__init__()
        self.lq_dir = os.path.join(data_root, 'LQ')
        self.gt_dir = os.path.join(data_root, 'HQ')
        
        # 获取文件列表
        self.lq_names = sorted([f for f in os.listdir(self.lq_dir) if f.endswith('.png')])
        self.gt_names = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])
        
        self.local_size = local_size
        self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        lq_name = self.lq_names[index]
        gt_name = self.gt_names[index]
        
        # 读取图像
        lq_path = os.path.join(self.lq_dir, lq_name)
        gt_path = os.path.join(self.gt_dir, gt_name)
        lq_img = read_img255(lq_path)
        gt_img = read_img255(gt_path)
        
        # 对齐处理
        H, W, _ = lq_img.shape
        if min(H, W) < 256:
            scale_ratio = 256 / min(H, W)
            new_H, new_W = int(H * scale_ratio), int(W * scale_ratio)
            lq_img = cv2.resize(lq_img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            gt_img = cv2.resize(gt_img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        
        # 转换为Tensor
        lq_tensor = self.transform(np.ascontiguousarray(lq_img).astype('uint8'))
        gt_tensor = self.transform(np.ascontiguousarray(gt_img).astype('uint8'))
        
        return {
            'lq': lq_tensor,
            'gt': gt_tensor,
            'filename': lq_name
        }

    def __len__(self):
        return len(self.lq_names)

# ----------------------- 使用方法示例 ----------------------- #
if __name__ == '__main__':
    # 训练数据加载器
    train_loader = DataLoader(
        UDCTrainData(
            crop_size=256,
            data_root="/data/xxting/datasets/TOLED/train",
            only_h_flip=False
        ),
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    
    # 测试数据加载器
    test_loader = DataLoader(
        UDCTestData(
            data_root="/data/xxting/datasets/TOLED/test",
            local_size=32
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1
    )