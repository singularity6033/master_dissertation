import math

import numpy as np
import utils.my_functional as mf
import torch
import torch.nn.functional as f

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,] (the size of 2nd dim should be less than #classes).
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def data_randomize(data, classes):
    idx = np.random.permutation(data.size()[0])
    x, y = data[idx], classes[idx]
    return x, y


def gain_schedule_old(loop, j):
    gain = 1
    if loop > 1:
        if j >= math.ceil(loop / 2):
            gain = 1 / 2
        if j >= math.ceil(3 * loop / 4) and loop > 4:
            gain = 1 / 4
        if j >= loop - 2 and loop > 5:
            gain = 1 / 20
        if j == loop - 1 and loop > 8:
            gain = 1 / 200
    return gain


def gain_schedule(loop, j):
    gain = 1
    if j >= math.ceil(loop / 2) and loop > 1:
        gain = 1 / 2
    if j >= math.ceil(3 * loop / 4) and loop > 3:
        gain = 1 / 4
    if j >= loop - 2 and loop > 11:
        gain = 1 / 10
    if j == loop - 1 and loop > 12:
        gain = 1 / 50
    return gain


def my_data_loader(dataset=None, batch_size=300, shuffle=False):
    if dataset is None:
        dataset = [None, None]
    # print('shapes are:', np.shape(x1), np.shape(x2))
    shape_in = np.shape(dataset[0])
    shape_out = np.shape(dataset[1])
    if shuffle:
        print('shuffle')
        rand = np.random.permutation(shape_in[0])
    else:
        print('no_shuffle')
        rand = range(shape_in[0])
    no_batch = math.ceil(shape_in[0] / batch_size)
    data_out = []
    for i in range(no_batch):
        if (i + 1) * batch_size <= shape_in[0]:
            in_images = np.zeros((batch_size, shape_in[1], shape_in[2], shape_in[3]))
            out_labels = np.zeros((batch_size, shape_out[1]))
        # due to math.ceil() in calculation of no_batch, the last batch may have size less than batch size
        else:
            print(i, i * batch_size)
            in_images = np.zeros((shape_in[0] - i * batch_size, shape_in[1], shape_in[2], shape_in[3]))
            out_labels = np.zeros((shape_in[0] - i * batch_size, shape_out[1]))
        for j in range(batch_size):
            # print(i, j, batch_size * i + j )
            if batch_size * i + j < shape_in[0]:
                in_images[j] = dataset[0][rand[batch_size * i + j]]
                out_labels[j] = dataset[1][rand[batch_size * i + j]]
        in_images = torch.from_numpy(in_images)
        in_images = in_images.permute(0, 3, 1, 2)
        out_labels = torch.from_numpy(out_labels)
        data_out.append([in_images, out_labels])
    return data_out


def create_matrix_x(x, _filter, stride, pad):
    """
    Extracts sliding local blocks from a batched input tensor
    """
    shape_filter = _filter.shape
    matrix_x = f.unfold(x, (shape_filter[2], shape_filter[3]), stride=stride, padding=pad)
    return matrix_x


def linear_backward_error(target, weight, func, nullspace=False):
    inv_f = mf.inv_fun(func)
    if not nullspace:
        print('no nullspace --', end=' ')
        return inv_f(target) @ torch.t(torch.pinverse(weight))
    else:
        print('v random --', end=' ')
        inv_tar = inv_f(target)
        n, _ = inv_tar.size()
        _, m = weight.size()
        I = torch.eye(m).to(DEVICE)
        v = (1 - 2 * torch.rand(n, m)).to(DEVICE)
        inv_weight = torch.pinverse(weight)
        return inv_tar @ torch.t(inv_weight) + v @ torch.t(I - inv_weight @ weight)


def pool_backward_error(out_err, kernel=2, method='Ave'):
    in_error = 0
    if method == 'Ave':
        in_error = torch.repeat_interleave(torch.repeat_interleave(out_err, kernel, dim=2), kernel, dim=3)
    return in_error
