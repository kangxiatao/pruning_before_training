# Gradient Coupled Flow: Performance Boosting On Network Pruning By Utilizing Implicit Loss Decrease

## Introduce

This project is a PyTorch implementation of the GCS paper, including several pruning before training methods tested in the paper.

```
@author: Kang Xiatao (kangxiatao@gmail.com)
```


## Structure
 - models: (lenet,resnet,vgg) Three models and the model class containing the mask
 - pruner: Pruning algorithm
 - runs: Output folder to store weights, evaluation information, etc.
 - utils: custom helper function


## Environment

The code has been tested to run on Python3.

Some package versions are as follows:
* torch == 1.7.1
* numpy == 1.18.5
* tensorboardX==2.4


## Run

* E.g. cifar10/vgg19 prune ratio: 90%
```
# GCS
python main.py --config 'cifar10/vgg19/90' --run 'test' --rank_algo 'gcs' --prune_mode 'rank'
```
```
# GCS-Group
python main.py --config 'cifar10/vgg19/90' --run 'test' --rank_algo 'gcs-group' --prune_mode 'rank'
```

* E.g. mnist/lenet5 prune ratio: 99.9%
```
python main.py --config 'mnist/lenet5/99.9' --run 'test' --rank_algo 'gcs' --prune_mode 'rank'
```

- Model optional: ```lenet, vgg, resnet```

- Dataset optional: ```fashionmnist, mnist, cifar10, cifar100```(Other datasets need to be manually downloaded to the local)

- All parameters(The default parameters are determined by the configs.py):

    | Console Parameters | Remark |
    | :---- | :---- |
    | config = '' | # Select Dataset, Model, and Pruning Rate |
    | pretrained = '' | # Path to load pretrained model |
    | run = 'test' | # Experimental Notes |
    | rank_algo = 'gcs' | # Choose a pruning algorithm (snip, grasp, synflow, gcs, gcs-group)|
    | prune_mode = 'rank' | # Choose a pruning mode (dense, rank, rank/random, rank/iterative)|
    | dp = '../Data' | # Modify the path of the dataset |
    | storage_mask = 0 | # Store the resulting mask |
    | --- | **Parameters for debugging** |
    | debug = 0 | # for debugging |
    | epoch | # Modify the number of training rounds |
    | batch_size | # Modify the batch size of training samples |
    | l2 | # L2 regularization hyperparameters |
    | lr_mode = 'cosine' | # Set the learning rate decay method (cosine or preset) |
    | optim_mode = 'SGD' | # Choose an optimizer (SGD or Adam) |
    | train_mode = 1 | Whether to use train mode when calculating weight sensitivity |
    | dynamic = 1 | # Whether to use dynamic iteration |
    | num_iters_prune | # Number of rounds for iterative pruning (default: 100) |
    | data_mode | # Data sampling mode (see pruning.py for details) |
    | grad_mode | # Calculate gradient mode (see pruning.py for details) |
    | score_mode | # Calculate sort score Mode (see pruning.py for details) |
    | num_group | # Number of gradient groups（GCS-Group） |


## ...
To be added ...
