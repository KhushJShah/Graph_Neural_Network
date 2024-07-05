# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Siamese Neural Message Passing network.

Learn a Siamese neural network training jointly with a Neural message passing network.
"""

# Python modules
import torch
import time
import glob
import os

# Own modules
from options import Options
import datasets
from LogMetric import AverageMeter, Logger
from utils import save_checkpoint, load_checkpoint, siamese_accuracy, knn, write_gxl
import models
import LossFunction

import pdb

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def train(train_loader, net, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()

    for i, (h1, am1, g_size1, h2, am2, g_size2, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Compute features
        output1 = net(h1, am1, g_size1)
        output2 = net(h2, am2, g_size2)
        
        output = output1 - output2
        output = output.pow(2).sum(1).sqrt()

        loss = criterion(output, target)

        # Logs
        losses.update(loss.item(), h1.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}] Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, loss=losses, b_time=batch_time))

    return losses


def validation(test_loader, net, criterion, evaluation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to eval mode
    net.eval()

    end = time.time()

    for i, (h1, am1, g_size1, h2, am2, g_size2, target) in enumerate(test_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Compute features
        output1 = net(h1, am1, g_size1)
        output2 = net(h2, am2, g_size2)
        
        output = output1 - output2
        output = output.pow(2).sum(1).sqrt()

        loss = criterion(output, target)
        bacc = evaluation(output, target)

        # Logs
        losses.update(loss.item(), h1.size(0))
        acc.update(bacc.item(), h1.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: Average Loss {loss.avg:.3f}; Average Acc {acc.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(loss=losses, acc=acc, b_time=batch_time))

    return losses, acc


def test(test_loader, train_loader, net, evaluation):
    batch_time = AverageMeter()
    acc = [AverageMeter() for _ in range(3)]  # Create a list of AverageMeter instances for each k

    eval_k = (1, 3, 5)

    # switch to eval mode
    net.eval()

    end = time.time()

    for i, (h1, am1, g_size1, target1) in enumerate(test_loader):
        # Compute features
        output1 = net(h1, am1, g_size1)

        D_aux = []
        T_aux = []
        for j, (h2, am2, g_size2, target2) in enumerate(train_loader):
            # Compute features
            output2 = net(h2, am2, g_size2)

            twoab = 2 * output1.mm(output2.t())
            dist = (output1 * output1).sum(1).expand_as(twoab) + (output2 * output2).sum(1).expand_as(twoab) - twoab
            dist = dist.sqrt().squeeze()

            D_aux.append(dist)
            T_aux.append(target2)

        D = torch.cat(D_aux)
        train_target = torch.cat(T_aux, 0)
        bacc = evaluation(D, target1.expand_as(train_target), train_target, k=eval_k)

        # Measure elapsed time
        for idx, val in enumerate(bacc):
            acc[idx].update(val, h1.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    print('Test distance:')
    for i in range(len(eval_k)):
        print('\t* {k}-NN; Average Acc {acc_avg:.3f}; Avg Time x Batch {b_time_avg:.3f}'.format(k=eval_k[i], acc_avg=acc[i].avg, b_time_avg=batch_time.avg))

    return acc



def main():

    print('Prepare dataset')
    # Dataset
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation, args.normalization, siamese=True)

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(data_train.getWeights(), 2*15*50*(50-1), replacement=True)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(torch.multinomial(torch.DoubleTensor(data_valid.getWeights()), 50*(50-1), replacement=False))

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=datasets.collate_fn_multiple_size_siamese,
                                               batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, sampler=valid_sampler,
                                               batch_size=args.batch_size, collate_fn=datasets.collate_fn_multiple_size_siamese,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.collate_fn_multiple_size_siamese,
                                              num_workers=args.prefetch, pin_memory=True)

    print('Create model')
    if args.representation != 'feat':
        print('\t* Discrete Edges')
        net = models.MpnnGGNN(in_size=2, e=[1], hidden_state_size=args.hidden_size, message_size=args.hidden_size, n_layers=args.nlayers, discrete_edge=True, out_type='regression', target_size=data_train.getTargetSize())
    else:
        print('\t* Feature Edges')
        net = models.MpnnGGNN(in_size=2, e=2, hidden_state_size=args.hidden_size, message_size=args.hidden_size, n_layers=args.nlayers, discrete_edge=False, out_type='regression', target_size=data_train.getTargetSize())

    print('Loss & optimizer')
    criterion = LossFunction.ContrastiveLoss()
    evaluation = siamese_accuracy
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)

    start_epoch = 0
    best_acc = 0
    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

    if not args.test:

        print('Training loop')
        # Main loop
        for epoch in range(start_epoch, args.epochs):
            # update the optimizer learning rate
            adjust_learning_rate(optimizer, epoch)

            loss_train = train(train_loader, net, optimizer, criterion, epoch)
            loss_valid, acc_valid = validation(valid_loader, net, criterion, evaluation)

            # Save model
            if args.save is not None:
                if acc_valid.avg > best_acc:
                    best_acc = acc_valid.avg
                    save_checkpoint({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'best_acc': best_acc},
                                    directory=args.save, file_name='checkpoint')

            # Logger step
            # Scalars
            logger.add_scalar('loss_train', loss_train.avg)
            logger.add_scalar('loss_valid', loss_valid.avg)
            logger.add_scalar('acc_valid', acc_valid.avg)
            logger.add_scalar('learning_rate', args.learning_rate)

            logger.step()

        # Load Best model to evaluate in test if we are saving it in a checkpoint
        if args.save is not None:
            print('Loading best model to test')
            best_model_file = os.path.join(args.save, 'checkpoint.pth')
            checkpoint = load_checkpoint(best_model_file)
            net.load_state_dict(checkpoint['state_dict'])

    # Evaluate best model in Test
    print('Test:')
    loss_test, acc_test = validation(test_loader, net, criterion, evaluation)

    # Dataset not siamese for test
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation, args.normalization)
    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=datasets.collate_fn_multiple_size,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=1, collate_fn=datasets.collate_fn_multiple_size,
                                              num_workers=args.prefetch, pin_memory=True)
    print('Test k-NN classifier')
    acc_test_hd = test(test_loader, train_loader, net, knn)

    if args.write is not None:
        if not os.path.exists(args.write):
            os.makedirs(args.write)

        directed = False if args.representation == 'adj' else True
        
        # Train
        write_dataset(data_train, net, directed)
        # Validation
        write_dataset(data_valid, net, directed)
        # Test
        write_dataset(data_test, net, directed)


def write_dataset(data, net, directed):
    for i in range(len(data)):
        v, am, _ = data[i]
        g_size = torch.LongTensor([v.size(0)])

        v, am = v.unsqueeze(0), am.unsqueeze(0)

        # Compute features
        v = net(v, am, g_size, output='nodes')

        v, am = v.squeeze(0).data, am.squeeze(0).data

        write_gxl(v, am, args.write + data.getId(i), directed)


def adjust_learning_rate(optimizer, epoch):
    """Updates the learning rate given an schedule and a gamma parameter.
    """
    if epoch in args.schedule:
        args.learning_rate *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate

if __name__ == '__main__':
    # Parse options
    args = Options().parse()

    # Check Test and load
    if args.test and args.load is None:
        raise Exception('Cannot test without loading a model.')

    if not args.test:
        print('Initialize logger')
        log_dir = args.log + '{}_run-batchSize_{}/' \
                .format(len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size))), args.batch_size)

        # Create Logger
        logger = Logger(log_dir, force=True)

    main()
