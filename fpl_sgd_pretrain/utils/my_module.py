import torch
import torch.nn as nn
import utils.my_functional as mf
from utils.basic import linear_backward_error, pool_backward_error
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyLayer:
    def __init__(self, name, paras, stride=1, padding=0, bias=False, activations=mf.my_identity):
        self.name = name
        self.weights = 0
        self.param = paras
        self.shape = 0
        self.pre_shape = 0
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.activations = activations


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size()[0], -1)


class MyCNN(nn.Module):

    def __init__(self, n_classes):
        super(MyCNN, self).__init__()
        self.no_layers = 0
        self.layers = []
        self.no_outputs = n_classes

    def add(self, layer):
        if layer.name == 'conv':
            layer.weights = nn.Conv2d(*layer.param, stride=layer.stride, padding=layer.padding, bias=layer.bias).to(
                DEVICE)
            layer.activations = layer.activations
        if layer.name == 'flat':
            layer.weights = Flatten()
        if layer.name == 'pool':
            layer.weights = nn.MaxPool2d(*layer.param)
        if layer.name == 'fc':
            layer.weights = nn.Linear(*layer.param, bias=layer.bias).to(DEVICE)
            layer.activations = layer.activations
        self.layers.append(layer)
        self.no_layers += 1

    def forward(self, x):
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                x = layer.activations(layer.weights(x))
                layer.shape = x.shape[1:]
            elif layer.name in ['flat', 'pool']:
                layer.pre_shape = x.shape[1:]
                x = layer.weights(x)
                layer.shape = x.shape[1:]
        return x

    def complete_net(self, train_loader):
        x = 0
        # input layer
        for X, y_true in train_loader:
            x = X.float()[0:1].to(DEVICE)
            break
        return self.forward(x)

    def forward_to_layer(self, x, to_layer):
        for layer in self.layers[0:to_layer]:
            if layer.name in ['conv', 'fc']:
                x = layer.activations(layer.weights(x))
            elif layer.name in ['flat', 'pool']:
                x = layer.weights(x)
        return x

    def backward_to_layer(self, y, to_layer):
        # to_layer will be 0,1,..., indicating the start layer of bp
        for layer in self.layers[:-to_layer - 1:-1]:
            if layer.name == 'conv':
                print('conv backward here !')
                pass
            elif layer.name == 'fc':
                y = linear_backward_error(y, layer.weights.weight, layer.activations, False)
            elif layer.name == 'pool':
                y = pool_backward_error(y, kernel=2)
            elif layer.name == 'flat':
                y = torch.reshape(y, torch.Size([y.shape[0]] + list(layer.pre_shape)))
        return y

    def get_weights(self):
        w = []
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                w.append(layer.weights.weight.data)
            elif layer.name in ['pool', 'flat']:
                w.append(0)
        return w

    def set_weights(self, w):
        i = 0
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                layer.weights.weight.data = w[i].to(DEVICE)
                i += 1
            elif layer.name in ['pool', 'flat']:
                i += 1

    def save_weights(self, path):
        w_list = self.get_weights()
        w_dict = {str(k): v for k, v in enumerate(w_list)}
        torch.save(w_dict, path)

    def load_weights(self, path):
        w_dict = torch.load(path)
        w_list = list(w_dict.values())
        self.set_weights(w_list)

    def get_weights_index(self, index):
        return self.layers[index].weights.weight.data

    def set_weights_index(self, w, index):
        self.layers[index].weights.weight.data = w

    def set_bias_index(self, w, index):
        self.layers[index].weights.bias = w

    def save_current_state(self, model_name, epoch, lr, acc_lst, loss_lst, j_max, j_max_old, two_layer=1, path=None):
        # to decrease the storage, only save weights when the test accuracy peaks or weights in last 2 epochs
        if two_layer == 0:
            weight_name = 'layer_wise'
        else:
            weight_name = 'two_layer'
        if epoch - 2 != j_max:
            old_path = path + '/' + model_name + '_' + weight_name + '_' + str(epoch - 2)
            if os.path.exists(old_path):
                os.remove(old_path)
        # if max test acc epoch changes, then we remove the old one
        if j_max_old != j_max and j_max_old != epoch - 1:
            old_path = path + '/' + model_name + '_' + weight_name + '_' + str(j_max_old)
            if os.path.exists(old_path):
                os.remove(old_path)
        weights_path = path + '/' + model_name + '_' + weight_name + '_' + str(epoch)
        self.save_weights(weights_path)
        filename = path + '/' + model_name + '.txt'
        with open(filename, 'a') as out:
            if two_layer:
                out.write('R' + '\t')
            else:
                out.write('L' + '\t')
            out.write(str(epoch) + '\t')
            out.write(str(lr) + '\t')
            out.write(str(acc_lst) + '\t')
            out.write(str(loss_lst) + '\n')

    # evaluate a fit model
    def evaluate_train(self, train_loader):
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
                _, true_labels = torch.max(labels, 1)
                total_train += labels.size(0)
                correct_train += (predicted == true_labels).sum()

        print('Accuracy of the network on the 50000 training images: %d %%' % (
                100 * correct_train / total_train))
        return 100 * correct_train / total_train

    # evaluate a fit model
    def evaluate_both(self, train_loader, test_loader):
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0
        train_loss = 0
        test_loss = 0
        loss_func = nn.MSELoss(reduction='mean')
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
                # _, true_labels = torch.max(labels, 1)
                true_labels = labels
                total_train += labels.size(0)
                train_loss += loss_func(predicted.float(), labels.float())
                correct_train += (predicted == true_labels).sum()

        print('Accuracy of network on the', total_train, 'training images', 100 * float(correct_train) / total_train)
        print('Loss of network on the', total_train, 'training images', train_loss.item() / total_train)

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
                # _, true_labels = torch.max(labels, 1)
                true_labels = labels
                total_test += labels.size(0)
                test_loss += loss_func(predicted.float(), labels.float())
                correct_test += (predicted == true_labels).sum()

        print('Accuracy of the network on the', total_test, 'test images', 100 * float(correct_test) / total_test)
        print('Loss of network on the', total_train, 'training images', test_loss.item() / total_test)
        return [100 * float(correct_train) / total_train, 100 * float(correct_test) / total_test], \
               [train_loss.item() / total_train, test_loss.item() / total_test]

    def my_parameters(self):
        w = []
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                # print(layer.weights.weight.requires_grad)
                w.append(layer.weights.weight.data)
        # generator
        return (param for param in w)


class CNN(nn.Module):
    def __init__(self, model_name='vgg11', num_classes=10, batch_norm=False, init_weights=False):
        super(CNN, self).__init__()
        self.model_name = model_name
        self.batch_norm = batch_norm
        self.params = {
            'vgg11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
            'vgg16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']
        }
        self.features = self.build_feature(model_name).to(DEVICE)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def build_feature(self, model_name):
        layers = []
        in_channels = 3
        selected_model = self.params[model_name]
        if not selected_model:
            print("invalid model name !")
            return
        for layer in selected_model:
            if layer == "P":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, layer, kernel_size=(3, 3), padding=1)
                layers += [conv2d]
                if self.batch_norm:
                    layers += [nn.BatchNorm2d(layer)]
                layers += [nn.ReLU(True)]
                in_channels = layer
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
