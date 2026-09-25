import argparse
import numpy as np
import random
import torch
import os
import logging
from tqdm import tqdm
from dataloader.dataset import build_Dataset
from torch.utils.data import DataLoader
from utils.utils import patients_to_slices
from dataloader.transforms import build_transforms
from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler


from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='D:/Paper Engineering Data/2D_data',
                    help='Name of Experiment')

parser.add_argument('--labeled_num', type=int, default=30,
                    help='Percentage of label quantity')

parser.add_argument('--dataset', type=str, default='/tumor_30',
                   help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/ISIC_TrainDataset_10',
#                      help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/thyroid_30',
#                     help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/BUSI',
#                     help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/BCSS',
#                      help='Name of Experiment')

# parser.add_argument('--stage', type=str,  default='1', help='')
# parser.add_argument('--stage', type=str,  default='2', help='')

parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--in_channels', type=int, default=3,
                    help='input channel of network')

parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')

parser.add_argument('--image_size', type=int, default=224, help='image_size')

parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')

# parser.add_argument('--map_max_iterations', type=int, default=0,
#                     help='maximum epoch number to train')
# parser.add_argument('--seg_max_iterations', type=int, default=0,
#                     help='maximum epoch number to train')
# parser.add_argument('--map_warmup_iterations', type=int, default=0,
#                     help='maximum epoch number to train')
# parser.add_argument('--seg_warmup_iterations', type=int, default=0,
#                     help='maximum epoch number to train')
parser.add_argument('--map_max_epoch', type=int, default=200,
                    help='maximum epoch number to train')
parser.add_argument('--seg_max_epoch', type=int, default=350,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1,
                    help='maximum epoch number to train')
parser.add_argument('--consistency', type=float, default=0.3,
                    help='consistency')
parser.add_argument('--weit_coef', type=float, default=2,
                    help='Loss weit_coef')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--snapshot_path', type=str, default='')

args = parser.parse_args()


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def train_stage1(args, snapshot_path, logger):
    # setting
    args.batch_size = 24
    args.labeled_bs = 12
    args.stage = '1'
    
    batch_size = args.batch_size
    # max_iterations = args.map_max_iterations
    max_epoch = args.map_max_epoch


    # dataset
    data_transforms = build_transforms(args)
    
    if "BCSS" in args.dataset:
        train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_BCSS",
                                  transform=data_transforms["train"])
        val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="val_BCSS",
                                transform=data_transforms["valid_test"])
    elif "BUSI" in args.dataset:
        train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_BUSI",
                                      transform=data_transforms["train"])
        val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="test_BUSI",
                                    transform=data_transforms["valid_test"])
    else:
        train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_semi",
                                    transform=data_transforms["train"])
        val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="val",
                                    transform=data_transforms["valid_test"])
    

    # sampler
    total_slices = len(train_dataset)
    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    logger.info("{} iterations per epoch".format(len(train_loader)))
    
    # max_epoch = max_iterations // len(train_loader) + 1
    args.map_max_iterations = max_epoch * len(train_loader)
    args.map_warmup_iterations = max_epoch * len(train_loader)
    iterator = tqdm(range(max_epoch), ncols=70)

    # model
    trainer = Trainer(args)

    iter_num = 0
    for epoch_ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            trainer.train(volume_batch, label_batch, logger, epoch_)
            iter_num = iter_num + 1
            # if iter_num > 0 and iter_num % 200 == 0:
            #     trainer.val(val_loader, snapshot_path, logger, iter_num)
        if epoch_ > 0 and epoch_ % 2 == 0:
            trainer.val(val_loader, snapshot_path, logger, epoch_)


def train_stage2(args, snapshot_path, logger):
    # setting
    args.batch_size = 4
    args.labeled_bs = 2
    args.stage = '2'
    
    batch_size = args.batch_size
    max_epoch = args.seg_max_epoch
    
    # dataset
    data_transforms = build_transforms(args)
    
    if "BCSS" in args.dataset:
        train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_BCSS",
                                  transform=data_transforms["train"])
        val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="val_BCSS",
                                transform=data_transforms["valid_test"])
    elif "BUSI" in args.dataset:
        train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_BUSI",
                                      transform=data_transforms["train"])
        val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="test_BUSI",
                                    transform=data_transforms["valid_test"])
    else:
        train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_semi",
                                    transform=data_transforms["train"])
        val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="val",
                                    transform=data_transforms["valid_test"])
    

    # sampler
    total_slices = len(train_dataset)
    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    logger.info("{} iterations per epoch".format(len(train_loader)))
    
    args.seg_max_iterations = max_epoch * len(train_loader)
    args.seg_warmup_iterations = (max_epoch//3) * len(train_loader)
    # max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    # model
    trainer = Trainer(args)

    iter_num = 0
    for epoch_ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            trainer.train(volume_batch, label_batch, logger, iter_num)
            iter_num = iter_num + 1
            # if iter_num > 0 and iter_num % 200 == 0:
            #     trainer.val(val_loader, snapshot_path, logger, iter_num)
        if epoch_ > 0 and epoch_ % 1 == 0:
            trainer.val(val_loader, snapshot_path, logger, iter_num)


if __name__ == '__main__':
    import shutil
    for fold in range(args.n_fold):
        # torch.autograd.set_detect_anomaly(True)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        snapshot_path = "./Results/result_tumor_30/fold_" + str(fold)
        
        args.snapshot_path = snapshot_path
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        shutil.copyfile("./train.py", snapshot_path + "/code/train.py")
        shutil.copyfile("./trainer.py", snapshot_path + "/code/trainer.py")
        shutil.copyfile("./model/model_Jump.py", snapshot_path + "/code/model_Jump.py")

        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(snapshot_path, "log.txt"))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Log file created successfully.")
        
        # train_stage1(args, snapshot_path, logger)
        train_stage2(args, snapshot_path, logger)
        