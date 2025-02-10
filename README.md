# ResNet-Profiling
This repository contains a ResNet-18 model trained on the CIFAR-10 dataset with various training configurations. It includes performance profiling results, analyzing the impact of number of workers in data loaders, optimizers and batch norm layers.

*Note: All problems are executed on NYU HPC.*

## Model

We use the ResNet-18 model as defined in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385):

- The first convolutional layer has 3 input channels, 64 output channels, 3×3 kernel, with stride=1 and padding=1.
- Followed by 8 basic blocks in 4 sub groups (i.e. 2 basic blocks in each subgroup):
	- The first sub-group contains convolutional layer with 64 output channels, 3×3 kernel, stride=1, padding=1.
	- The second sub-group contains convolutional layer with 128 output channels, 3×3 kernel, stride=2, padding=1.
	- The third sub-group contains convolutional layer with 256 output channels, 3×3 kernel, stride=2, padding=1.
	- The forth sub-group contains convolutional layer with 512 output channels, 3×3 kernel, stride=2, padding=1.
- The final linear layer is of 10 output classes.

For all convolutional layers, we use ReLU activation functions, and we use batch normal layers
to avoid covariant shift. Since batch-norm layers regularize the training, we set bias to 0 for
all the convolutional layers. We use SGD optimizers with 0.1 as the learning rate, momentum = 0.9, weight decay = 5e-4, and the loss function is cross entropy.

## Data Loader

We create a DataLoader that loads the images and the related labels from the torchvision CIFAR10 dataset. We import CIFAR10 dataset from the torchvision
package, with the following sequence of transformations:

- Random cropping, with size 32×32 and padding 4
- Random horizontal flipping with a probability 0.5
- Normalizing each image’s RGB channel with mean(0.4914, 0.4822, 0.4465) and variance (0.2023, 0.1994, 0.2010)

*Note: the default settings for the train loader are minibatch size of 128 and 3 IO processes (i.e., num workers=2)*

## Profiling - Time Measurement

We report the running time for the following sections of the code:

- Data-loading time for each epoch
- Training (i.e., mini-batch calculation) time for each epoch
- Total running time for each epoch.

*Note: Data-loading time here is the time it takes to load batches from the generator (exclusive of the time it takes to move those batches to the device).*

### Execution Command

```shell
python lab2.py -q c2
```

### Results

![c2 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c2.png?raw=true)

## Profiling - I/O Optimization

We report the total time spent for the DataLoader varying the number of workers starting from zero and increment the number of workers by 4 (0,4,8,12,16...) until the I/O time does not decrease anymore. We draw the results in a graph to illustrate the performance difference as we increase the number of workers
Additionally, we report the number of workers that are needed for the best runtime performance.

### Execution Command

```shell
python lab2.py -q c3
```

### Results

![c3 graph](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c3-graph.png?raw=true)

The above figure shows the data loading times plotted on a graph.

![c3 result 1](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c3-1.png?raw=true)

The above figure shows the result for 0 and 4 workers.

![c3 result 2](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c3-2.png?raw=true)

The above figure shows the result for 8 and 12 workers.

![c3 result 3](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c3-3.png?raw=true)

The above figure shows the result for 16 and 20 workers, and it also shows the final output - the least time taken by the data loaders and the optimal number of workers. In our case, we get the least time as 3.6671168659954674 seconds for 4 workers.

## Profiling - Time Measurement (with Optimized I/O configuration)

We compare the data-loading and computing time for runs using 1 worker and 4 workers.

### Execution Command

```shell
python lab2.py -q c4
```

### Results

![c4 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c4.png?raw=true)

The above figure shows the result - data loading times and computing times for 1 worker and 4 workers.

The data loading times for 1 worker is 3.3652011380000886 seconds and for 4 workers it is 3.3641574870030126 seconds. We see only a marginal difference between the data loading times. This may be because we are dealing with a small dataset of 32 × 32 images which doesn’t demand much computational power and I/O bandwidth. In such cases, even a single worker can load data fast enough that adding more workers doesn’t significantly improve the speed.

Also, spawning multiple workers introduces some overhead due to context switching and communication
between the main process and worker threads. In cases where the loading process itself is relatively fast, this overhead might offset the benefits of parallel data loading.

When using just 1 worker, the CPU has to switch between loading the data and processing it for training,
often serializing these operations. This can create bottlenecks where the CPU is either loading data or
performing training computations, but not both at the same time. With 4 workers, multiple threads are
available to handle data loading, which allows the model’s training computations to happen more efficiently without waiting for new data, which explains the lower computation time on the one with 4 workers.

## Profiling - GPU Training vs. CPU Training

We report the average running time over 5 epochs using the GPU vs using the CPU (using 4 I/O workers).

### Execution Command

```shell
python lab2.py -q c5
```

### Results

![c5 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c5.png?raw=true)

The above figure shows the result - average run time over 5 epochs using a GPU and a CPU with 4 data loading
workers.

## Profiling - Optimizers Experimentation

We run 5 epochs with the GPU-enabled code and the optimal number of I/O workers (4 workers). For
each epoch, we report the average training time , training loss and top-1 training accuracy
using these optimizers: SGD, SGD with nesterov, Adagrad, Adadelta, and Adam. 

*Note: we use the same default hyperparameters: learning rate = 0.1 , weight decay = 5e-4, and
momentum = 0.9 (when it applies) for all these optimizers.*

### Execution Command

```shell
python lab2.py -q c6
```

### Results

![c6-1 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c6-1.png?raw=true)

The above figure shows the result for SGD optimizer.

![c6-2 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c6-2.png?raw=true)

The above figure shows the result for SGD optimizer with Nesterov Momentum.

![c6-3 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c6-3.png?raw=true)

The above figure shows the result for
AdaGrad optimizer.

![c6-4 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c6-4.png?raw=true)

The above figure shows the result for AdaDelta optimizer.

![c6-5 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c6-5.png?raw=true)

The above figure shows the result for Adam
optimizer.

## Profiling - Experimenting without Batch Norm

With the GPU-enabled code and the optimal number of workers, we report the average training loss, top-1 training accuracy for 5 epochs with the default SGD optimizer and its hyper-parameters but without batch norm layers.

### Execution Command

```shell
python lab2.py -q c7
```

### Results

![c7 result](https://github.com/rugvedmhatre/ResNet-Profiling/blob/main/images/c7.png?raw=true)

The above figure shows the result - average train loss, train accuracy for 5 epochs on a ResNet model without any Batch Norm layers.

## Other Arguments

```shell
Singularity> python lab2.py -h
usage: python lab2.py [-h] [-d {cuda,cpu}] [-dp DATAPATH] [-w {0,1,2,4,8,12,16}] [-op {sgd,sgdnes,adagrad,adadelta,adam}] [-v]
                      [-q {c1,c2,c3,c4,c5,c6,c7}] [-ts]

ResNet-18 profiling on CIFAR-10 dataset

options:
  -h, --help            show this help message and exit
  -d {cuda,cpu}, --device {cuda,cpu}
                        select the device for model training (default: cuda)
  -dp DATAPATH, --datapath DATAPATH
                        select the dataset path for training (default: ./data/)
  -w {0,1,2,4,8,12,16}, --workers {0,1,2,4,8,12,16}
                        select the number of workers for data loading (default: 2)
  -op {sgd,sgdnes,adagrad,adadelta,adam}, --optimizer {sgd,sgdnes,adagrad,adadelta,adam}
                        select the optimizer for training (default: sgd)
  -v, --verbose         if true, all logs will be printed on the console
  -q {c1,c2,c3,c4,c5,c6,c7}, --question {c1,c2,c3,c4,c5,c6,c7}
                        select the assignment question and the code will change correspondingly
  -ts, --torchsummary   if true, it will only print the model summary and exit

------
```

You can utilize any argument to customize to your problem statement. 
