import argparse, random, logging, warnings, os, sys, re, time
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_pre import mae_vit_mini_patch16_dec512d8b
from dataset import Dataset
warnings.filterwarnings('ignore')


def init_distributed(args):
    args.world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = args.world_size > 1 or args.distributed
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
    else:
        args.local_rank = 0
    return args


def reduce_tensor(x, args):
    if args.distributed:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= args.world_size
    return x



def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size',  type=int, default=16)
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--workers',     type=int, default=8)
    p.add_argument('--prefetch',    type=int, default=4)
    p.add_argument('--seed',        type=int, default=1234)
    p.add_argument('--clip',        type=float, default=1.0)
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--resume',      type=str, default='',
                   help='path to checkpoint (.pth) for resuming')
    return p.parse_args()



def main():
    torch.multiprocessing.set_start_method('spawn', force=True)

    args = init_distributed(get_args())


    if args.local_rank == 0:
        os.makedirs('save', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if args.local_rank == 0 else logging.ERROR,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('save/train.log', encoding='utf-8')
                if args.local_rank == 0 else logging.NullHandler()
        ])
    logger = logging.getLogger()


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True


    train_set = Dataset(mode='train')
    sampler   = DistributedSampler(train_set, shuffle=True) if args.distributed else None
    loader    = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=args.prefetch,
        persistent_workers=True,
        drop_last=True)


    device = torch.device('cuda', args.local_rank)
    model  = mae_vit_mini_patch16_dec512d8b().to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)


    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.OneCycleLR(
        opt,
        args.lr,
        steps_per_epoch=len(loader),
        epochs=args.epochs,
        pct_start=0.05,
        cycle_momentum=False)


    start_epoch = 0
    if args.resume:
        if not osp.isfile(args.resume):
            logger.warning(f'Checkpoint not found: {args.resume}')
        else:
            ckpt = torch.load(args.resume, map_location='cpu')
            if 'model' in ckpt:
                state_dict = ckpt['model']
                opt.load_state_dict(ckpt['optimizer'])
                sched.load_state_dict(ckpt['scheduler'])
                start_epoch = ckpt.get('epoch', -1) + 1
                logger.info(f'Resumed full checkpoint (epoch {start_epoch})')
            else:
                state_dict = ckpt
                m = re.search(r'epoch[_\-]?(\d+)', osp.basename(args.resume))
                start_epoch = int(m.group(1)) if m else 0
                logger.info(f'Resumed weights only (start from epoch {start_epoch})')

            need_module  = args.distributed and not next(iter(state_dict)).startswith('module.')
            strip_module = (not args.distributed) and next(iter(state_dict)).startswith('module.')
            if need_module:
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            if strip_module:
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            (model.module if isinstance(model, nn.parallel.DistributedDataParallel)
             else model).load_state_dict(state_dict, strict=True)


    for ep in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(ep)

        model.train()
        epoch_loss, seen = 0.0, 0

        iter_loader = (tqdm(loader,
                            total=len(loader),
                            desc=f'Epoch {ep+1}/{args.epochs}',
                            colour='cyan',
                            leave=False)
                       if args.local_rank == 0 else loader)

        for img, ddm in iter_loader:
            img, ddm = img.to(device, non_blocking=True), ddm.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            loss, *_ = model(img, ddm)
            reduce_l = reduce_tensor(loss.detach(), args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()

            bs = img.size(0)
            epoch_loss += reduce_l.item() * bs
            seen       += bs

            if args.local_rank == 0:
                iter_loader.set_postfix({
                    'lr':   f'{sched.get_last_lr()[0]:.3e}',
                    'loss': f'{reduce_l.item():.4f}'
                })

        if args.local_rank == 0:
            logger.info(f'Epoch {ep+1:02d} | lr {sched.get_last_lr()[0]:.3e} '
                        f'| avg_loss {(epoch_loss/seen):.6f}')
            ckpt = {
                'model': (model.module
                          if isinstance(model, nn.parallel.DistributedDataParallel)
                          else model).state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(),
                'epoch': ep
            }
            torch.save(ckpt, f'save/epoch_{ep+1:04d}.pth')

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()