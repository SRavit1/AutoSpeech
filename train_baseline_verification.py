from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchsummary

import concurrent.futures
import multiprocessing
import itertools

from models import resnet, resnet_full, resnet_quantized, resnet_dense_xnor
from autospeech_utils import set_path, create_logger, save_checkpoint, count_parameters
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset
from data_objects.VoxcelebTestset import VoxcelebTestset
from functions import train_from_scratch, validate_verification
#from loss import CrossEntropyLoss

cuda = True

def train(a, binarized, quantized, bitwidth, weight_bitwidth, sparsity): # a unused
    seed=0
    lr_min=0.001
    learning_rate=0.01
    num_workers=0
    num_classes=1211
    batch_size=128
    begin_epoch=0
    end_epoch=301
    val_freq=10
    print_freq=200
    load_path=None
    sub_dir="dev"
    partial_n_frames=300

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if not binarized and not quantized:
      prefix = "full_"
      subdir = prefix + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    else:
      if binarized:
        prefix = "binarized_"
      elif quantized:
        prefix = "quantized_"
        
      subdir = prefix + "bitwidth_" + str(bitwidth) + "_weight_bitwidth_" + str(weight_bitwidth) + "_sparsity_" + str(sparsity) + "_" + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    print("Model/Log Subdir", subdir)
    data_dir = "/mnt/usb/data/ravit/datasets/VoxCeleb1"

    log_dir = os.path.join("../logs/autospeech", subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join("../models/autospeech", subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # model and optimizer
    if not binarized and not quantized:
      model = resnet_full.resnet18(num_classes=num_classes, input_channels=1, normalize_output=False)
    elif binarized:
      model = resnet_dense_xnor.resnet18(num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)
    elif quantized:
      model = resnet_quantized.resnet18(num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)

    #torchsummary.summary(model.cuda(), (1, 300, 257))
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(
        model.net_parameters() if hasattr(model, 'net_parameters') else model.parameters(),
        lr=0.01,
    )

    #initialize model weights using existing resnet model
    #pretrained_model_path = "pretrained/checkpoint_best.pth"
    pretrained_model_dir = "quantized_bitwidth_4_sparsity_0_20220304-073633"
    pretrained_ckpt = "checkpoint_best.pth"
    pretrained_model_path = os.path.join(pretrained_model_dir, pretrained_ckpt)
    model.load_state_dict(torch.load(os.path.join("../models/autospeech/", pretrained_model_path))["state_dict"], strict=False)
    
    dummy_input = torch.zeros((1, 1, 300, 257))
    if cuda:
        dummy_input = dummy_input.cuda()
    torch.onnx.export(model, dummy_input, os.path.join(model_dir, subdir + "_resnet18_verification.onnx"), verbose=False)

    save_checkpoint({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'best_eer': 0,
        'optimizer': optimizer.state_dict()
    }, False, model_dir, 'checkpoint_init.pth')
    print("Saved initial model path")

    # Loss
    #criterion = CrossEntropyLoss(num_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    # resume training && make log dir and logger
    #subdir = "quantized_bitwidth_2_weight_bitwidth_2_sparsity_0_20220404-081824"
    #load_path = os.path.join("../models/autospeech", subdir)
    load_path = None
    checkpoint_file = None
    if load_path and os.path.exists(load_path):
        checkpoint_file = os.path.join(load_path, 'checkpoint_init.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        # load checkpoint
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        best_eer = checkpoint['best_eer']
        optimizer.load_state_dict(checkpoint['optimizer'])
        #log_dir = checkpoint['path_helper']['log_path']
        #model_dir = checkpoint['path_helper']['ckpt_path']
        log_dir = os.path.join("../logs/autospeech/", subdir)
        model_dir = os.path.join("../models/autospeech/", subdir)
    else:
        begin_epoch = 0
        best_eer = 1.0
        last_epoch = -1
    
    logger = create_logger(log_dir, subdir)
    if checkpoint_file is not None:
        logger.info("=> loaded checkpoint '{}'".format(checkpoint_file))
    
    logger.info("Number of parameters: {}".format(count_parameters(model)))

    # dataloader
    train_dataset = DeepSpeakerDataset(
        Path(data_dir),  sub_dir, partial_n_frames)
    test_dataset_verification = VoxcelebTestset(
        Path(data_dir), partial_n_frames
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=cuda, shuffle=True, drop_last=True)
    test_loader_verification = torch.utils.data.DataLoader(dataset=test_dataset_verification, batch_size=1, 
        num_workers=num_workers, pin_memory=cuda, shuffle=False, drop_last=False)

    # training setting
    writer_dict = {
        'writer': SummaryWriter(log_dir),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // val_freq,
    }

    # training loop
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, end_epoch, lr_min,
        #last_epoch=last_epoch
    )

    #TODO: Save raw data in JSON, in case we want to generate more nuanced graph
    #TODO: Add plot generation and saving to autospeech_utils, so it can be used by multiple files
    loss_history = []
    top1_history = []
    top5_history = []
    eer_history = []
    for epoch in tqdm(range(begin_epoch, end_epoch), desc='train progress'):
        model.train()
        top1, top5, loss = train_from_scratch(model, optimizer, train_loader, criterion, epoch, writer_dict, learning_rate, print_freq, lr_scheduler, cuda=cuda)
        loss_history.append(loss)
        top1_history.append(top1)
        top5_history.append(top5)

        if epoch % val_freq == 0:
            eer = validate_verification(model, test_loader_verification, cuda=cuda)
            eer_history.append(eer)

            # remember best acc@1 and save checkpoint
            is_best = eer < best_eer
            best_eer = min(eer, best_eer)

            # save
            logger.info('=> saving checkpoint to {}'.format(model_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_eer': best_eer,
                'optimizer': optimizer.state_dict()
            }, is_best, model_dir, 'checkpoint_{}.pth'.format(epoch))

        lr_scheduler.step(epoch)

        #TODO: Save plot of loss, accuracy, eer
        epochs = list(range(1, epoch+2))
        epochs_eval = list(range(1, epoch+2, val_freq))

        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(epochs, loss_history)
        ax_loss.set_title("Training Loss Convergence")
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Training Loss")
        fig_loss.savefig(os.path.join(log_dir, "loss_convergence.png"))
        fig_top1, ax_top1 = plt.subplots()
        ax_top1.plot(epochs, top1_history)
        ax_top1.set_title("Training Top1 Accuracy Convergence")
        ax_top1.set_xlabel("Epochs")
        ax_top1.set_ylabel("Training Top1")
        fig_top1.savefig(os.path.join(log_dir, "top1_convergence.png"))
        fig_top5, ax_top5 = plt.subplots()
        ax_top5.plot(epochs, top5_history)
        ax_top5.set_title("Training Top5 Accuracy Congergence")
        ax_top5.set_xlabel("Epochs")
        ax_top5.set_ylabel("Training Top5")
        fig_top5.savefig(os.path.join(log_dir, "top5_convergence.png"))
        fig_eer, ax_eer = plt.subplots()
        ax_eer.plot(epochs_eval, eer_history)
        ax_eer.set_title("Evaluation EER Convergence")
        ax_eer.set_xlabel("Epochs")
        ax_eer.set_ylabel("EER")
        fig_eer.savefig(os.path.join(log_dir, "eer_convergence.png"))

        plt.close('all')


def main():
    """
    binarized_options = [False]
    quantized_options = [True]
    bitwidth_options = [32] #[1, 2, 3]
    sparsity_options = [0] #[0, 0.1]
    model_combinations = list(itertools.product(bitwidth_options, sparsity_options))
    """
    model_combinations = [(False, True, 1, 1, 0)] #binarized, quantized, activation bitwidth, weight bitwidth, sparsity

    for (binarized, quantized, bitwidth, weight_bitwidth, sparsity) in model_combinations:
        torch.multiprocessing.spawn(train, args=(binarized, quantized, bitwidth, weight_bitwidth, sparsity))

if __name__ == '__main__':
    main()
