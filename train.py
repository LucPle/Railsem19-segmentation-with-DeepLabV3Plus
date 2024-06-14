import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist

from data.dataloader import DataSet
from models.deeplabv3plus import DeepLabV3Plus
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import os
import sys
import argparse


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt
    
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'



parser = argparse.ArgumentParser(description="DeepLabV3Plus Network")
parser.add_argument("--data", type=str, default="/dataset", help="")
parser.add_argument("--batch-size", type=int, default=4, help="")
parser.add_argument("--worker", type=int, default=12, help="")
parser.add_argument("--epoch", type=int, default=200, help="")
parser.add_argument("--num-classes", type=int, default=19, help="")
parser.add_argument("--momentum", type=float, default=0.9, help="")
parser.add_argument("--lr", type=float, default=1e-2, help="")
parser.add_argument("--os", type=int, default=16, help="")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="")
parser.add_argument("--logdir", type=str, default="./logs/", help="")
parser.add_argument("--save", type=str, default="./saved_model/", help="")
parser.add_argument("--local_rank", type=int, default=0)


args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://')

args.world_size = torch.distributed.get_world_size()
args.distributed = args.world_size > 1

setup_for_distributed(args.local_rank == 0)

print(args)

writer = SummaryWriter(args.logdir)

train_dataset = DataSet(args.data)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.worker, drop_last=False, shuffle=False, pin_memory=True, sampler=train_sampler)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = DeepLabV3Plus(num_classes=args.num_classes, os=args.os)
net = net.to(device)
print(device)
if device == 'cuda':
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-4)

rank = dist.get_rank()

def train(epoch, iteration, scheduler, total_loss):
    epoch += 1
    net.train()
    
    train_loss = 0
    for idx, (images, labels) in enumerate(train_loader):
        iteration += 1
        _, h, w = labels.size()

        images, labels = images.to(device), labels.to(device).long()
        out = net(images)
        out = F.interpolate(out, size=(h, w), mode='bilinear')

        loss = criterion(out, labels)

        train_loss += reduce_tensor(loss, args.world_size) if args.distributed else loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("\repoch: ", epoch, "iter: ", (idx + 1), "/", len(train_loader), "loss: ", loss.item(), end='')
        sys.stdout.flush()
        break

    scheduler.step()

    writer.add_scalar('log/loss', train_loss/(idx+1), epoch)
    writer.add_scalar('log/lr', scheduler.get_lr()[0], epoch)

    print("\nepoch: ", epoch, "loss: ", train_loss/(idx+1), "lr: ", scheduler.get_lr()[0])
    
    state = {
        'net': net.module.state_dict(),
        'epoch': epoch,
        'iter': iteration,
    }
    if rank == 0:

        if not os.path.isdir(args.save):
            os.makedirs(args.save)
            
        if train_loss < total_loss:
            total_loss = train_loss
            saving_path = os.path.join(args.save, 'full_label_best.pth')
            torch.save(state, saving_path)
            print("Model saved in ", saving_path)
        
        if epoch == 200:
            saving_path = os.path.join(args.save, 'full_label_last.pth')
            torch.save(state, saving_path)
            print("Model saved in ", saving_path)
            
    return epoch, iteration, total_loss

if __name__=='__main__':
    epoch = 0
    iteration = 0
    total_loss = 1e9
    while epoch < args.epoch:
        epoch, iteration, total_loss = train(epoch, iteration, scheduler, total_loss)
        
    print("Training finished!")