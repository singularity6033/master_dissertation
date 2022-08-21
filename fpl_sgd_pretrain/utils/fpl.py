import numpy as np
import time
import torch
import torch.nn.functional as f
import math
import utils.my_functional as mf
from utils.basic import one_hot_embedding, data_randomize, gain_schedule_old, gain_schedule,\
    create_matrix_x, pool_backward_error

DEVICE_ = ['cuda' if torch.cuda.is_available() else 'cpu']


class FPL:
    def __init__(self, model, train_loader, test_loader, epochs=1, loop=1, gain=1, mix_data=False,
                 model_name='_0', path=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.loop = loop
        self.gain = gain
        self.mix_data = mix_data
        self.model_name = model_name
        self.path = path

    """
    pre-train phase: one-layer update algorithm
    """

    def inc_solve_filter(self, batch_no, lin, in_images, out_images, fil, func, gain_rate, pad):
        if lin:
            curr_inv_f = mf.inv_fun(func)
            if batch_no <= 0:
                print('Incremental LINEAR algorithm')
                print('Calculating inverse of the target, inverse function:')
                print(curr_inv_f)

            out_images = curr_inv_f(out_images)
        else:
            if batch_no == 0:
                print('Incremental NON-LINEAR algorithm')
                print(func)

            out_images = mf.fun_cut(out_images, func)

        in_shape = in_images.shape
        out_shape = out_images.shape

        out_images = torch.reshape(out_images, [out_shape[0], out_shape[1], out_shape[2] * out_shape[3]])

        shape_filter = fil.shape
        no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
        no_fil_channels = shape_filter[0]
        # nf * nc * f^2
        fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights])

        # batch_size * (nc * f^2) * output_size(m*m)
        matrix_x = create_matrix_x(in_images, fil, pad)

        gain = 1e-6  # old: 1e-4, after resizing to (224,224)--> 1e-5
        lr = gain  # * gain_rate

        if self.mix_data:
            print('inc_solve_x_random_shuffle')
            matrix_x, out_images = data_randomize(matrix_x, out_images)

        for j in range(self.loop):
            # print('number of loop:', loop)
            if self.loop > 1:
                if j == math.ceil(self.loop / 2) + 1:
                    lr = lr / 2
                if j == math.ceil(3 * self.loop / 4) + 1 & self.loop > 4:
                    lr = lr / 2
                if j == self.loop - 1 & self.loop > 5:
                    lr = lr / 5
                if j == self.loop & self.loop > 8:
                    lr = lr / 10
            if batch_no == 0:
                if self.loop <= 20:
                    print(['loop ', j + 1])
                    print(lr)
                elif (j + 1) % (self.loop / 5) == 0:
                    print(['loop ', j + 1])
                    print(lr)

            # loop over batch dim or sample dim
            for k in range(in_shape[0]):
                w_new = fil_w
                # (nc * f^2) * output_size(m*m)
                in_matrix = matrix_x[k, :, :]
                # nf * output_size(m*m)
                y_ = w_new @ in_matrix
                if ~lin:
                    y_ = func(y_)
                e_ = torch.squeeze(out_images[k, :, :]) - y_
                fil_w = w_new + lr * e_ @ torch.t(in_matrix)

        x = torch.reshape(fil_w, shape_filter)

        return x

    def inc_solve_fc(self, batch_no, lin, in_images, out_images, weight, func, gain_rate):
        if lin:
            curr_inv_f = mf.inv_fun(func)
            if batch_no <= 0:
                print('Incremental LINEAR algorithm')
                print('Calculating inverse of the target, inverse function:')
                print(curr_inv_f)
            out_images = curr_inv_f(out_images)
        else:
            if batch_no == 0:
                print('Incremental NON-LINEAR algorithm')
                print(func)
            out_images = mf.fun_cut(out_images, func)

        out_shape = out_images.shape
        if self.gain < 0:
            max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
            gain = 1 / max_phi2 * gain_rate
        else:
            gain = self.gain * gain_rate
        lr = gain  # .to(DEVICE_[0])

        if self.mix_data:
            print('inc_solve_x_random_shuffle')
            in_images, out_images = data_randomize(in_images, out_images)

        for j in range(self.loop):
            # print('number of loop:', loop)
            lr = lr * gain_schedule(self.loop, j)
            if batch_no == 0:
                if self.loop <= 20:
                    print(['loop ', j + 1])
                    print(lr)
                    print('maxphi', max_phi2, gain_rate)
                elif (j + 1) % (self.loop / 5) == 0:
                    print(['loop ', j + 1])
                    print(lr)

            for k in range(out_shape[0]):
                w_new = weight
                in_matrix = in_images[k:k + 1, :]
                y_ = w_new @ torch.t(in_matrix)
                if ~lin:
                    y_ = func(y_)
                e_ = torch.t(out_images[k:k + 1, :]) - y_
                e_phi = e_ @ in_matrix
                weight = w_new + lr * e_phi

        return weight, lr.item()

    def inc_train_1_layer(self, at_layer):
        curr_layer = self.model.layers[at_layer]
        w = self.model.get_weights_index(at_layer)

        with torch.no_grad():
            max_acc_test = 0
            j_at_max = 0
            j_max_old = 0
            for j in range(self.epochs):
                print('============== epoch', j + 1, '/', self.epochs, '=============')
                gain_rate = gain_schedule_old(self.epochs, j)
                for i, (x, y) in enumerate(self.train_loader):
                    # print batch number for each len(train_loader) // 10
                    if i % (len(self.train_loader) // 10) == 0:
                        print('=========== batch', i + 1, '/', len(self.train_loader), '==========')
                    layer_in = self.model.forward_to_layer(x.float().to(DEVICE_[0]), at_layer)
                    layer_tar = one_hot_embedding(y.long(), self.model.no_outputs).to(DEVICE_[0]).float()
                    layer_tar = self.model.backward_to_layer(layer_tar, self.model.no_layers - at_layer - 1)
                    if curr_layer.name == 'conv':
                        pad = curr_layer.padding
                        w = self.inc_solve_filter(i, False, layer_in, layer_tar, w, curr_layer.activations,
                                                  gain_rate, pad)
                    elif curr_layer.name == 'fc':
                        w, lr = self.inc_solve_fc(i, False, layer_in, layer_tar, w, curr_layer.activations, gain_rate)

                self.model.set_weights_index(w, at_layer)
                acc_lst, loss_lst = self.model.evaluate_both(self.train_loader, self.test_loader)
                print('accuracy at epoch ', j + 1, ': ', acc_lst)
                print('loss at epoch ', j + 1, ': ', loss_lst)
                # save model
                if self.model_name != '_0':
                    if acc_lst[1] > max_acc_test:
                        max_acc_test = acc_lst[1]
                        j_max_old = j_at_max
                        j_at_max = j
                    self.model.save_current_state(self.model_name, j, lr, acc_lst, loss_lst,
                                                  j_at_max, j_max_old, 0, self.path)

        return w

    def inverse_layerwise_training(self, no_layers=1):
        for i in range(no_layers):
            cur_inx = self.model.no_layers - i - 1
            cur_layer = self.model.layers[cur_inx]
            print('First time: ', cur_layer.name)
            if cur_layer.name in ['conv', 'fc']:
                out_weight = self.inc_train_1_layer(cur_inx)
                self.model.set_weights_index(out_weight, cur_inx)
        for i in range(no_layers - 1):
            cur_inx = i + self.model.no_layers - no_layers + 1
            cur_lay = self.model.layers[cur_inx]
            print('Second time: ', cur_lay.name)
            if cur_lay.name in ['conv', 'fc']:
                out_weight = self.inc_train_1_layer(cur_inx)
                self.model.set_weights_index(out_weight, cur_inx)
        return 0

    """
    fine-tuning phase: two-layers update algorithm
    """

    def inc_solve_2_layer_conv_fc(self, batch_no, in_image, out_image, pool_layer='max', fil=None, fc_wei=None,
                                  fun_front=f.relu, fun_after=mf.my_identity, stride=1, pad=0, gain_=0.01, auto=True):
        out_shape = out_image.shape
        out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])

        shape_filter = fil.shape
        no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
        no_fil_channels = shape_filter[0]
        # nf * nc * f^2
        fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
        fc_w = fc_wei.to(DEVICE_[0])

        lm = gain_
        pool_ind = None
        pool_out = None
        if self.mix_data:
            pass
            input()
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
        for j in range(self.loop):
            if batch_no == 0:
                print('= loop ', lm, ' =')
            alpha_v = torch.tensor(1).to(DEVICE_[0])
            alpha_w = torch.tensor(1).to(DEVICE_[0])

            for i in range(out_shape[0]):  # number of data samples (output)
                fil_w_new = fil_w
                fc_w_new = fc_w
                # Xf
                in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]
                fc_out = out_image[i]
                conv_act = fil_w_new @ in_matrix
                conv_out = fun_front(conv_act)

                conv_out_shape = conv_out.shape
                conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
                conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
                conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                    int(math.sqrt(conv_out_shape[1]))])

                # Apply pooling layer
                if pool_layer:
                    if pool_layer == 'avg':
                        pool_out = f.avg_pool2d(conv_out, 2, 2)
                    elif pool_layer == 'max':
                        pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                    pool_out_shape = pool_out.shape
                    pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
                    fc_in = torch.reshape(pool_out, [pool_flatten_shape, 1])
                else:
                    fc_in = torch.reshape(conv_out, conv_flat_shape)
                y_ = fun_after(fc_w_new @ fc_in)
                e_ = fc_out - y_

                # Backpropagation to flattening & pooling layer
                e_fc_in = torch.t(fc_w_new) @ e_
                if pool_layer:
                    e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
                    # Backpropagation to conv layer
                    if pool_layer == 'avg':
                        e_conv_out = pool_backward_error(e_pool_out, 2)
                    elif pool_layer == 'max':
                        e_conv_out = f.max_unpool2d(e_pool_out, pool_ind, 2)
                    e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
                else:
                    e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
                dot_value = mf.derivative_fun(fun_front)(conv_act_flat)  # derivative matrix
                dot_value = dot_value.reshape(conv_flat_shape)
                e_conv_flat = dot_value * e_conv_out
                e_conv = torch.reshape(e_conv_flat, conv_out_shape)

                # to make sure that the initial gain was not too large for convergence, n the first several loops,
                # the gain was automatically reduced by checking the condition (73).
                if auto:
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2) * lm ** 2)
                        * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.eig(lr_con)
                    while (eig_values[:, 0] < -0.005).any():
                        alpha_v = alpha_v / 1.1
                        alpha_w = alpha_w / 1.1
                        # print(alpha_v)
                        sum1 = torch.diagflat(
                            (2.0 * lm / mf.fun_max_derivative(fun_after)
                             - alpha_w * torch.sum(conv_act_flat ** 2) * lm ** 2)
                            * torch.ones(out_shape[1], 1).to(DEVICE_[0])).to(DEVICE_[0])
                        sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                           pool_layer=pool_layer, pool_ind=pool_ind)
                        lr_con = sum1 - sum2
                        eig_values, _ = torch.eig(lr_con)
                # update rules: (58) and (60)
                fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
                fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)

        fil_w = torch.reshape(fil_w, shape_filter)
        fc_w = fc_w
        lr = alpha_v * lm
        return fil_w, fc_w, alpha_v, lr.item()

    def inc_train_2_layer(self, pool_layer='max', auto=True, true_for=1, avg_N=0):
        # conv layer --> pooling layer --> flatten layer --> fc layer
        # conv layer --> flatten layer --> fc layer
        if pool_layer:
            print(pool_layer, 'pooling')
            indx = -4
        else:
            indx = -3
        # conv weights
        w1 = self.model.get_weights_index(indx)
        curr_layer_front = self.model.layers[indx]
        # fc weights
        w2 = self.model.get_weights_index(-1)
        curr_layer_after = self.model.layers[-1]
        # print(w1.shape, w2.shape)
        t0 = time.time()
        with torch.no_grad():
            max_acc_test = 0
            j_at_max = 0
            j_max_old = 0
            alpha_vw_min = 1
            acc_last_N_epochs = 0.
            acc_curr_N_epochs = 0.
            train_acc_last_N_epochs = 0.
            train_acc_curr_N_epochs = 0.
            k = 0
            for j in range(self.epochs):
                t1 = time.time()
                print('============== epoch', j + 1, '/', self.epochs, '==============')
                gain_rate = gain_schedule(self.epochs, j)
                gain_ = self.gain * gain_rate
                if j + 1 > true_for:
                    auto = False
                else:
                    alpha_vw_min = 1
                gain_adj = self.gain * alpha_vw_min
                if gain_ > gain_adj:
                    gain_ = gain_adj
                for i, (x, y) in enumerate(self.train_loader):
                    if (i + 1) % (len(self.train_loader) // 10) == 0:
                        print('============== batch', i + 1, '/', len(self.train_loader), '==============')
                        print('time:', time.time() - t1)
                    layer_in = self.model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
                    layer_tar = one_hot_embedding(y.long(), self.model.no_outputs).to(DEVICE_[0]).float()
                    pad = curr_layer_front.padding
                    stride = curr_layer_front.stride

                    w1, w2, alpha_vw, lr = self.inc_solve_2_layer_conv_fc(i, layer_in, layer_tar,
                                                                          pool_layer=pool_layer,
                                                                          fil=w1, fc_wei=w2,
                                                                          fun_front=curr_layer_front.activations,
                                                                          fun_after=curr_layer_after.activations,
                                                                          stride=stride, pad=pad,
                                                                          gain_=gain_, auto=auto)
                    if alpha_vw < alpha_vw_min:
                        alpha_vw_min = alpha_vw
                        print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)

                print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
                self.model.set_weights_index(w1, indx)
                self.model.set_weights_index(w2, -1)
                acc_lst, loss_lst = self.model.evaluate_both(self.train_loader, self.test_loader)
                print('accuracy at epoch ', j + 1, ': ', acc_lst)
                print('loss at epoch ', j + 1, ': ', loss_lst)
                # save model
                if self.model_name != '_0':
                    if acc_lst[1] > max_acc_test:
                        max_acc_test = acc_lst[1]
                        j_max_old = j_at_max
                        j_at_max = j
                    self.model.save_current_state(self.model_name, j, lr, acc_lst, loss_lst,
                                                  j_at_max, j_max_old, 1, self.path)
                if avg_N > 0:
                    if k < avg_N - 1:
                        acc_curr_N_epochs += acc_lst[1] / avg_N
                        train_acc_curr_N_epochs += acc_lst[0] / avg_N
                        k += 1
                    else:
                        acc_curr_N_epochs += acc_lst[1] / avg_N
                        train_acc_curr_N_epochs += acc_lst[0] / avg_N
                        k = 0
                        if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
                            print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
                            print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
                            acc_last_N_epochs = acc_curr_N_epochs
                            acc_curr_N_epochs = 0
                            if train_acc_curr_N_epochs > train_acc_last_N_epochs != 0:
                                print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
                                print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
                                break
                            print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
                            train_acc_last_N_epochs = train_acc_curr_N_epochs
                            train_acc_curr_N_epochs = 0
                        else:
                            print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
                            print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
                            acc_last_N_epochs = acc_curr_N_epochs
                            acc_curr_N_epochs = 0
                            print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ',
                                  train_acc_last_N_epochs)
                            train_acc_last_N_epochs = train_acc_curr_N_epochs
                            train_acc_curr_N_epochs = 0

                print('time for epoch', j + 1, '/', self.epochs, ': ', time.time() - t1, 's')
        print('time in total: ', time.time() - t0, 's')
        return w1, w2

    def inc_solve_2_fc_layer(self, batch_no, lin, in_images, out_images, weight_v, weight_w, fun1, fun2,
                             gain_rate=1.0, auto=0):
        if lin:
            curr_inv_f = mf.inv_fun(fun2)
            if batch_no <= 0:
                print('Incremental LINEAR algorithm')
                print('Calculating inverse of the target, inverse function:')
                print(curr_inv_f)
            out_images = curr_inv_f(out_images)
        else:
            if batch_no == 0:
                print('Incremental NON-LINEAR algorithm')
                print(fun2)
            out_images = mf.fun_cut(out_images, fun2)
        #     in_images = in_images.to(DEVICE_[0])

        out_shape = out_images.shape
        t0 = time.time()
        if self.gain is None or self.gain < 0:
            max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
            t1 = time.time()
            print('max time', t1 - t0, max_phi2)
            self.gain = 1 / max_phi2
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        if self.mix_data:
            print('inc_solve_x_random_shuffle')
            in_images, out_images = data_randomize(in_images, out_images)

        lr_total = self.gain
        for j in range(self.loop):
            print('number of loop:', self.loop)
            lr = self.gain

            alpha_v = torch.tensor(1).to(DEVICE_[0])
            alpha_w = torch.tensor(1).to(DEVICE_[0])
            for k in range(out_shape[0]):
                v_new = weight_v
                w_new = weight_w
                in_matrix = in_images[k:k + 1, :]
                vx_ = v_new @ torch.t(in_matrix)
                phi = fun1(vx_)
                if ~lin:
                    # print('nonlinear coming here')
                    wa = w_new @ phi
                    y_ = fun2(wa)
                else:
                    y_ = w_new @ phi
                e_ = torch.t(out_images[k:k + 1, :]) - y_
                # print(e_.shape, phi.shape)

                dot_f1 = mf.derivative_fun(fun1)
                dot_a_vx = dot_f1(vx_)
                dot_a_vx_ = torch.squeeze(dot_a_vx)

                if auto:
                    lr_con = torch.diagflat(
                        (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2) * torch.ones(
                            out_shape[1], 1).to(DEVICE_[0])).to(
                        DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2) * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                        w_new) * lr
                    eig_values, _ = torch.eig(lr_con)
                    while (eig_values[:, 0] < -0.005).any():
                        # print('%d - %d', j, k)
                        alpha_v = alpha_v / 1.1
                        alpha_w = alpha_w / 1.1
                        # print(alpha_v)
                        lr_con = torch.diagflat(
                            (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(
                                phi ** 2) * lr ** 2) * torch.ones(
                                out_shape[1], 1).to(DEVICE_[0])).to(
                            DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2) * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                            w_new) * lr
                        eig_values, _ = torch.eig(lr_con)

                if batch_no == 0 and k == 0:
                    if self.loop <= 20:
                        print(['loop ', j + 1])
                        print('gain is (not include alpha)', lr * gain_rate * gain_schedule(self.loop, j))
                    elif (j + 1) % (self.loop / 5) == 0:
                        print(['loop ', j + 1])
                        print('gain is (not include alpha)', lr * gain_rate * gain_schedule(self.loop, j))

                weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(self.loop, j)) * e_ @ torch.t(phi)
                weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(self.loop, j)) * (
                        dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix

            if self.loop > 1:
                self.model.set_weights_index(weight_w, -1)
                self.model.set_weights_index(weight_v, -2)
                acc_lst, loss_lst = self.model.evaluate_both(self.train_loader, self.test_loader)
                print('accuracy at epoch ', j + 1, ': ', acc_lst)
                print('loss at epoch ', j + 1, ': ', loss_lst)
            lr_total = alpha_w * (lr * gain_rate * gain_schedule(self.loop, j))
        return weight_v, weight_w, alpha_w, lr_total.item()

    def conv_train_2_fc_layer_last(self, auto=False):
        # last two fc layers
        curr_layer_after = self.model.layers[-1]
        w = self.model.get_weights_index(-1)
        curr_layer_front = self.model.layers[-2]
        v = self.model.get_weights_index(-2)
        alpha_w = 0
        with torch.no_grad():
            max_acc_test = 0
            j_at_max = 0
            j_max_old = 0
            for j in range(self.epochs):
                print('============== epoch', j + 1, '/', self.epochs, '=============')
                gain_rate = gain_schedule(self.epochs, j)
                for i, (x, y) in enumerate(self.train_loader):
                    print('=========== batch', i + 1, '/', len(self.train_loader), '==========')
                    layer_in = self.model.forward_to_layer(x.float().to(DEVICE_[0]), -2)
                    layer_tar = one_hot_embedding(y.long(), self.model.no_outputs).to(DEVICE_[0]).float()
                    v, w, alpha_w_, lr = self.inc_solve_2_fc_layer(i, False, layer_in, layer_tar, v, w,
                                                                   curr_layer_front.activations,
                                                                   curr_layer_after.activations,
                                                                   gain_rate=gain_rate, auto=auto)
                self.model.set_weights_index(w, -1)
                self.model.set_weights_index(v, -2)
                acc_lst, loss_lst = self.model.evaluate_both(self.train_loader, self.test_loader)
                print('accuracy at epoch ', j + 1, ': ', acc_lst)
                print('loss at epoch ', j + 1, ': ', loss_lst)
                # save model
                if self.model_name != '_0':
                    if acc_lst[1] > max_acc_test:
                        max_acc_test = acc_lst[1]
                        j_max_old = j_at_max
                        j_at_max = j
                    self.model.save_current_state(self.model_name, j, lr, acc_lst, loss_lst,
                                                  j_at_max, j_max_old, 1, self.path)
        return w


# help function used to calculate the 2nd term in (73)
def sum_condition_cnn(lm=0., in_matrix=None, fc_w=None, dot_value=None, fil_w=None, pool_layer='max', pool_ind=None):
    sum2 = 0
    nf, _ = fil_w.shape
    phi_s, _ = dot_value.shape
    size_phij = phi_s // nf
    fc_out, fc_size = fc_w.shape
    size_fc_wj = fc_size // nf
    for j in range(nf):
        phij = dot_value[j * size_phij:(j + 1) * size_phij]
        fc_wj = fc_w[:, j * size_fc_wj:(j + 1) * size_fc_wj]
        Pj = 0
        if pool_layer:
            if pool_layer == 'avg':
                Pj = lm * phij * torch.t(fc_wj)
            elif pool_layer == 'max':
                fc_wj_pool_out = fc_wj
                temp = fc_wj_pool_out.shape
                fc_wj_pool_out = torch.reshape(fc_wj_pool_out,
                                               [1, fc_out, int(math.sqrt(temp[1])), int(math.sqrt(temp[1]))])
                pool_ind_ = pool_ind[:, j:j + 1, :, :]
                pool_ind_ = pool_ind_.repeat(1, fc_out, 1, 1)
                fc_wj_conv_out = f.max_unpool2d(fc_wj_pool_out, pool_ind_, 2)
                temp = fc_wj_conv_out.shape
                fc_w_j_ = torch.reshape(fc_wj_conv_out, [fc_out, temp[2] * temp[3]])
                Pj = lm * phij * torch.t(fc_w_j_)
        else:
            Pj = lm * phij * torch.t(fc_wj)
        sum2 = sum2 + torch.t(Pj) @ torch.t(in_matrix) @ in_matrix @ Pj
    return sum2