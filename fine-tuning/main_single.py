import argparse
import random
import logging
import warnings
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_fine import mae_vit_mini_patch16_dec512d8b
from dataset import Dataset

warnings.filterwarnings('ignore')

root_path = './'
model_path = '../pre_training/save/epoch_0025.pth'


def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.num_steps + 100, pct_start=0.05,
        cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


def reload_pre_train_model(model, path=""):
    if not bool(path) or not os.path.exists(path):
        print(f'[load] 没有加载预训练模型：{path}')
        return model
    else:
        model_dict = model.state_dict()

        pretrained = torch.load(path, map_location='cpu')
        # ✅ 正确提取预训练权重
        pretrained_dict = pretrained['model'] if 'model' in pretrained else pretrained

        matched = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(matched)
        model.load_state_dict(model_dict)

        print(f'[load] 加载预训练模型参数数：{len(matched)} / {len(model_dict)}')
        return model


def train(train_loader, model, optimizer, epoch, device, logger, scheduler, opt):
    model.train()
    total_loss = 0.
    total_samples = 0

    pbar = tqdm(train_loader, desc=f'[Epoch {epoch}]', ncols=120)

    for batch_idx, (im, bm_gt) in enumerate(pbar):
        print(f"Image shape: {im.shape}")
        im = im.to(device)      # (B, 3, H, W)
        bm_gt = bm_gt.to(device)  # (B, 2, H, W)

        optimizer.zero_grad()
        pred = model(im)         # pred: (B, 2, H, W)
        loss = (pred - bm_gt).abs().mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()
        scheduler.step()

        bs = im.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        avg_loss = total_loss / total_samples
        pbar.set_postfix(loss=f'{avg_loss:.6f}')

    # 保存模型
    os.makedirs(f'{root_path}/save/net', exist_ok=True)
    torch.save(model.state_dict(), f'{root_path}/save/net/epoch_{epoch}.pth')
    logger.info(f'[Epoch {epoch}] Avg Loss: {total_loss / total_samples:.8f}')


def check_requires_grad(model):
    """
    检查每个参数的 requires_grad 状态，以便确认哪些参数被冻结，哪些未冻结
    """
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--net_trained_path', default=model_path)
    opt = parser.parse_args()

    # 日志设置
    os.makedirs(f'{root_path}/save', exist_ok=True)
    logging.basicConfig(filename=f'{root_path}/save/log.txt',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    # 随机种子
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'[Device] Using device: {device}')

    # 加载数据
    train_dataset = Dataset(mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )
    opt.num_steps = len(train_loader) * opt.num_epochs
    logger.info(f'[Dataset] Train samples: {len(train_dataset)}, Steps: {opt.num_steps}')

    # 模型加载与替换
    model = mae_vit_mini_patch16_dec512d8b()
    model = reload_pre_train_model(model, opt.net_trained_path)
    model = model.to(device)  # 确保将模型加载到指定设备

    # 打印模型中各层的 requires_grad 状态，检查哪些被冻结
    check_requires_grad(model)

    # 优化器和调度器
    optimizer, scheduler = fetch_optimizer(opt, model)

    # 训练循环
    for epoch in range(1, opt.num_epochs + 1):
        logger.info(f'\n========== Epoch {epoch}/{opt.num_epochs} ==========')
        train(train_loader, model, optimizer, epoch, device, logger, scheduler, opt)

    logger.info('[Done] 训练完成')


if __name__ == '__main__':
    main()
