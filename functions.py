import time
import torch
import torch.nn.functional as F
import logging
import numpy as np
import matplotlib.pyplot as plt

from autospeech_utils import compute_eer
from autospeech_utils import AverageMeter, ProgressMeter, accuracy

plt.switch_backend('agg')
logger = logging.getLogger(__name__)

def train_from_scratch(model, optimizer, train_loader, criterion, epoch, writer_dict, lr, print_freq, lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), batch_time, data_time, losses, top1, top5, prefix="Epoch: [{}]".format(epoch), logger=logger)
    writer = writer_dict['writer']

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        if lr_scheduler:
            current_lr = lr_scheduler.get_last_lr()
        else:
            current_lr = lr

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to logger
        writer.add_scalar('lr', current_lr, global_steps)
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        # log acc for cross entropy loss
        writer.add_scalar('train_acc1', top1.val, global_steps)
        writer.add_scalar('train_acc5', top5.val, global_steps)

        if i % print_freq == 0:
            progress.print(i)

def validate_verification(model, test_loader):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(test_loader), batch_time, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()
    labels, distances = [], []

    with torch.no_grad():
        end = time.time()
        for i, (input1, input2, label) in enumerate(test_loader):
            input1 = input1.cuda(non_blocking=True).squeeze(0)
            input2 = input2.cuda(non_blocking=True).squeeze(0)
            label = label.cuda(non_blocking=True)

            # compute output
            outputs1 = model(input1).mean(dim=0).unsqueeze(0)
            outputs2 = model(input2).mean(dim=0).unsqueeze(0)

            dists = F.cosine_similarity(outputs1, outputs2)
            dists = dists.data.cpu().numpy()
            distances.append(dists)
            labels.append(label.data.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        eer = compute_eer(distances, labels)
        logger.info('Test EER: {:.8f}'.format(np.mean(eer)))

    return eer
