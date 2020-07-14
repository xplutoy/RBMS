import imageio
import numpy as np
import torch as T
import torchvision as tv

USE_GPU = T.cuda.is_available()
Tensor = T.cuda.FloatTensor if USE_GPU else T.FloatTensor

mnist_train = tv.datasets.MNIST(
    './datas/mnist',
    train=True,
    transform=tv.transforms.ToTensor(),
    download=True)

mnist_test = tv.datasets.MNIST(
    './datas/mnist',
    train=False,
    transform=tv.transforms.ToTensor(),
    download=True)

fashion_train = tv.datasets.FashionMNIST(
    './datas/fashion',
    train=True,
    transform=tv.transforms.ToTensor(),
    download=True)

fashion_test = tv.datasets.FashionMNIST(
    './datas/fashion',
    train=False,
    transform=tv.transforms.ToTensor(),
    download=True)

cifar_train = tv.datasets.CIFAR100(
    './datas/fashion',
    train=True,
    transform=tv.transforms.ToTensor(),
    download=True)

cifar_test = tv.datasets.CIFAR100(
    './datas/fashion',
    train=False,
    transform=tv.transforms.ToTensor(),
    download=True)


def next_batch(X, bs):
    end, L = bs, len(X)
    while end < L:
        batch_X = [X[i][0] for i in range(end - bs, end)]
        batch_X = T.stack(batch_X)
        if USE_GPU:
            batch_X = batch_X.cuda()
        yield batch_X
        end += bs


def shuffle_batch(X, bs):
    L = len(X)
    assert bs < L
    batch_X = [X[i][0] for i in np.random.choice(L, bs)]
    batch_X = T.stack(batch_X)
    if USE_GPU:
        batch_X = batch_X.cuda()
    return batch_X


def to_gif(file_lst, gif_name):
    frames = [imageio.imread(file) for file in file_lst]
    imageio.mimsave(gif_name, frames, duration=0.5)


def linear_inc(init, ultimate, start, stop):
    assert start < stop
    return (ultimate - init) / (stop - start)


if __name__ == '__main__':
    # for a in next_batch(mnist, 1000):
    #     print(a.size())
    print(shuffle_batch(cifar_test, 1000).size())
