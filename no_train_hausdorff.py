from __future__ import print_function, division

"""
Without training, compute the Hausdorff Distance between graph nodes.
"""

# Python modules
import torch
import time

# Own modules
from options import Options
import datasets
from LogMetric import AverageMeter
from utils import knn
import GraphEditDistance

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

def test(test_loader, train_loader, net, evaluation):
    batch_time = AverageMeter()
    acc = [AverageMeter() for _ in range(3)]  # Create a list of AverageMeters for the k values

    eval_k = (1, 3, 5)

    # Switch to eval mode
    net.eval()

    end = time.time()

    for i, (h1, am1, g_size1, target1) in enumerate(test_loader):
        D_aux = []
        T_aux = []
        for j, (h2, am2, g_size2, target2) in enumerate(train_loader):
            d = net(h1.expand(h2.size(0), h1.size(1), h1.size(2)),
                    am1.expand(am2.size(0), am1.size(1), am1.size(2), am1.size(3)),
                    g_size1.expand_as(g_size2), h2, am2, g_size2)

            D_aux.append(d)
            T_aux.append(target2)

        D = torch.cat(D_aux)
        train_target = torch.cat(T_aux, 0)

        bacc = evaluation(D, target1.expand_as(train_target), train_target, k=eval_k)

        # Debugging: Print bacc values
        print(f"bacc: {bacc}, type: {type(bacc)}")

        # Measure elapsed time
        for idx, val in enumerate(bacc):
            if isinstance(val, torch.Tensor):
                val = val.item()  # Convert to float if it's a tensor
            
            acc[idx].update(val, h1.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # Check the type of acc to ensure it hasn't been overwritten
        for idx, meter in enumerate(acc):
            print(f"Type of acc[{idx}]: {type(meter)}")


    print('Test distance:')
    for i in range(len(eval_k)):
        if isinstance(acc[i], AverageMeter):
            print(f"\t* {eval_k[i]}-NN; Average Acc {acc[i].avg:.3f}; Avg Time x Batch {batch_time.avg:.3f}")
        else:
            print(f"Unexpected type for acc[{i}]: {type(acc[i])}, value: {acc[i]}")
    return acc

def main():
    print('Prepare dataset')
    # Dataset
    data_train, data_valid, data_test = datasets.load_data(args.dataset, args.data_path, args.representation, args.normalization)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=datasets.collate_fn_multiple_size,
                                               batch_size=args.batch_size,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=1, collate_fn=datasets.collate_fn_multiple_size,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=1, collate_fn=datasets.collate_fn_multiple_size,
                                              num_workers=args.prefetch, pin_memory=True)

    print('Create model')
    if args.distance == 'SoftHd':
        net = GraphEditDistance.SoftHd()
    else:
        net = GraphEditDistance.Hd()

    print('Loss & optimizer')
    evaluation = knn

    print('Validation')
    acc_valid = test(valid_loader, train_loader, net, evaluation)

    # Evaluate best model in Test
    print('Test:')
    acc_test = test(test_loader, train_loader, net, evaluation)

if __name__ == '__main__':
    # Parse options
    args = Options().parse()

    main()
