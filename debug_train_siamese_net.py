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
import random
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
# Own modules
from options import Options
import datasets
from LogMetric import AverageMeter, Logger
from utils import save_checkpoint, load_checkpoint, siamese_accuracy, knn, write_gxl
import models
import LossFunction
import pdb

def create_cxl_file(graph_files, labels, file_name, directory):
    """
    Function to create a .cxl file based on graph files and their labels.
    :param graph_files: List of graph file names.
    :param labels: List of corresponding labels for each graph.
    :param file_name: Output .cxl file name.
    :param directory: Directory where the .cxl file will be created.
    """
    # Create the root element
    root = ET.Element('GraphCollection')

    # Iterate over graph files and labels to create entries
    for graph_file, label in zip(graph_files, labels):
        graph_element = ET.SubElement(root, 'print')
        graph_element.set('file', graph_file)
        graph_element.set('class', label)

    # Create the tree and write it to the file
    tree = ET.ElementTree(root)
    cxl_path = os.path.join(directory, file_name)
    with open(cxl_path, 'wb') as f:
        tree.write(f)

    print(f"{file_name} has been generated in {directory}.")


def generate_cxl_files(directory, labels=None):
    """
    Function to generate train.cxl, validation.cxl, and test.cxl files directly in the directory.
    :param directory: Directory containing the .gxl graph files.
    :param labels: Optional list of corresponding labels for each graph (same length as total number of files).
    """
    # Get all .gxl files in the directory
    graph_files = [f for f in os.listdir(directory) if f.endswith('.gxl')]

    # Split graph files into train, validation, and test sets (70% train, 15% val, 15% test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(graph_files, labels, test_size=0.6, random_state=42)
    val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)

    # Generate train.cxl
    create_cxl_file(train_files, train_labels, 'train.cxl', directory)

    # Generate validation.cxl
    create_cxl_file(val_files, val_labels, 'validation.cxl', directory)

    # Generate test.cxl
    create_cxl_file(test_files, test_labels, 'test.cxl', directory)

def load_graph(filepath):
    """
    Function to load a graph from a GXL file. Adjust this according to how your graphs are structured.
    """
    # For simplicity, assuming datasets.load_graph(filepath) was supposed to load a graph
    # Replace it with appropriate GXL reading logic if necessary.
    # Otherwise, this will return the graph in a way that the model understands
    graph = datasets.load_graph(filepath)  # Assuming this returns graph data (adjust as necessary)
    return graph

def load_files(file_list, directory):
    """
    Function to load the graph files from a directory.
    """
    graphs = []
    for file in file_list:
        file_path = os.path.join(directory, file)
        graphs.append(load_graph(file_path))  # Adjust load_graph according to your file structure
    return graphs


def split_dataset(directory, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Function to split the .gxl files into train, validation, and test sets.
    """
    # Step 1: Filter only .gxl files
    gxl_files = [f for f in os.listdir(directory) if f.endswith('.gxl')]

    # Step 2: Shuffle the files for randomness
    random.shuffle(gxl_files)

    # Step 3: Split the data
    train_files, temp_files = train_test_split(gxl_files, test_size=(1 - train_ratio))
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (test_ratio + val_ratio))

    # Return the file splits
    return train_files, val_files, test_files

def train(train_files, net, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()

    for i, graph_file in enumerate(train_files):
        graph1, graph2, target = process_graphs(graph_file)  # Custom graph processing logic

        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Compute features
        output1 = net(graph1)
        output2 = net(graph2)
        
        output = output1 - output2
        output = output.pow(2).sum(1).sqrt()

        loss = criterion(output, target)

        # Logs
        losses.update(loss.item(), graph1.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f'Epoch: [{epoch}] Average Loss {losses.avg:.3f}; Avg Time x Batch {batch_time.avg:.3f}')
    return losses

def process_graphs(graph_file):
    """
    This function should load the graphs from the files, process them, and return graph1, graph2, and a target.
    The logic here depends on how you want to handle graph pairs and targets.
    """
    graph1 = load_graph(graph_file)
    graph2 = load_graph(graph_file)  # This would depend on how you structure your pairs
    target = torch.tensor(1)  # This depends on your dataset structure. Replace with actual target logic
    print(f"Processing graphs: {graph_file} - Target: {target}")
    return graph1, graph2, target

def validation(val_files, net, criterion, evaluation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to eval mode
    net.eval()

    end = time.time()

    for i, graph_file in enumerate(val_files):
        graph1, graph2, target = process_graphs(graph_file)  # Same processing as in train()

        # Compute features
        output1 = net(graph1)
        output2 = net(graph2)
        
        output = output1 - output2
        output = output.pow(2).sum(1).sqrt()

        loss = criterion(output, target)
        bacc = evaluation(output, target)

        # Logs
        losses.update(loss.item(), graph1.size(0))
        acc.update(bacc.item(), graph1.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f'Validation: Average Loss {losses.avg:.3f}; Average Acc {acc.avg:.3f}; Avg Time x Batch {batch_time.avg:.3f}')
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
    print('Initialize logger')
    log_dir = args.log + '{}_run-batchSize_{}/'.format(len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size))), args.batch_size)
    logger = Logger(log_dir, force=True)

    # Path to the directory containing the .gxl files
    graph_directory = './LOW1/'  # Adjust this path if necessary

    # Generate the .cxl files before proceeding
    print('Generating .cxl files...')
    generate_cxl_files(graph_directory)


    print('Prepare dataset')
    # Dataset
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation, args.normalization, siamese=True)

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(data_train.getWeights(), 2*15*50*(50-1), replacement=True)
    valid_weights = data_valid.getWeights()
    if len(valid_weights) == 0:
        raise ValueError("No valid weights found for validation. Check the dataset for valid pairs.")
    num_samples = min(len(valid_weights), 50*(50-1))

    # Ensure that we are not trying to sample more than the number of available weights
    if num_samples > 0:
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            torch.multinomial(torch.DoubleTensor(valid_weights), num_samples, replacement=False)
        )
    else:
        raise ValueError("Not enough data in the validation set to perform sampling.")

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