import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.my_module import CNN
from utils.basic import one_hot_embedding
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils.my_module as mm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
mm.DEVICE = DEVICE
print(DEVICE)

BATCH_SIZE = 1
learning_rate = 0.1
model_name = 'vgg16'  # vgg11 or vgg16
dataset_name = 'cifar100'  # cifar10 or cifar100
weights_saved_path = './saved_weights/' + model_name + '_' + dataset_name + '_sgd_ndn'
best_weights_saved_path = './saved_weights/' + model_name + '_' + dataset_name + '_sgd_ndn_best'
param_saved_path = './saved_models/' + model_name + '_' + dataset_name + '_sgd_ndn'

if not os.path.exists(weights_saved_path):
    os.makedirs(weights_saved_path)
if not os.path.exists(best_weights_saved_path):
    os.makedirs(best_weights_saved_path)
if not os.path.exists(param_saved_path):
    os.makedirs(param_saved_path)

N_CLASSES = 100
t0 = time.time()
no_epochs = 200


def model_saver(md, path, is_best=False):
    if is_best:
        for f in Path(path).glob('*.pt'):
            try:
                f.unlink()
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
    ii = 0
    for name, parameters in md.named_parameters():
        # if name in saved_layers:
        # print(name)
        torch.save(parameters, path + '/' + model_name + '_' + dataset_name + '_' + str(i) + "_sgd_ndn.pt")
        ii += 1


# download and create datasets
train_dataset = datasets.CIFAR100(root=dataset_name + '_data',
                                  train=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                  ]),
                                  download=True)

valid_dataset = datasets.CIFAR100(root=dataset_name + '_data',
                                  train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

# define the data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(dataset=valid_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

model = CNN(model_name=model_name, num_classes=N_CLASSES, batch_norm=False).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.5)  # learning rate decay

loss = nn.CrossEntropyLoss().to(DEVICE)

train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []
best_loss = float('inf')
for i in range(no_epochs):
    train_loss = 0
    train_num = 0.0
    train_accuracy = 0.0
    model.train()
    train_bar = tqdm(train_loader, desc=f'Epoch {i + 1}/{no_epochs}')
    for step, data in enumerate(train_bar):
        img, target = data
        optimizer.zero_grad()
        img.requires_grad_()
        outputs = model(img.to(DEVICE))
        loss_train = loss(outputs, one_hot_embedding(target, N_CLASSES).to(DEVICE))
        outputs = torch.argmax(outputs, 1)
        loss_train.backward()
        optimizer.step()
        train_loss += abs(loss_train.item()) * img.size(0)
        accuracy = torch.sum(outputs == target.to(DEVICE))
        train_accuracy = train_accuracy + accuracy
        train_num += img.size(0)
    scheduler.step()

    print("no_epochs: {}, train-Loss: {}, train-accuracy: {}".format(i + 1, train_loss / train_num,
                                                                     train_accuracy / train_num))
    train_loss_all.append(train_loss / train_num)
    train_acc_all.append(train_accuracy.double().item() / train_num)
    test_loss = 0
    test_accuracy = 0.0
    test_num = 0
    model.eval()
    with torch.no_grad():
        test_bar = tqdm(val_loader, desc=f'Epoch {i + 1}/{no_epochs}')
        for data in test_bar:
            img, target = data
            outputs = model(img.to(DEVICE))
            loss_val = loss(outputs, one_hot_embedding(target, N_CLASSES).to(DEVICE))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss_val.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(DEVICE))
            test_accuracy = test_accuracy + accuracy
            test_num += img.size(0)

    print("no_epochs: {}, test-Loss: {}, test-accuracy: {}".format(i + 1, test_loss / test_num,
                                                                   test_accuracy / test_num))
    test_loss_all.append(test_loss / test_num)
    test_acc_all.append(test_accuracy.double().item() / test_num)
    # save the best model
    if test_loss_all[-1] < best_loss:
        best_loss = test_loss_all[-1]
        model_saver(model, best_weights_saved_path, True)

model_saver(model, weights_saved_path)

filename_acc = param_saved_path + '/' + model_name + '_acc' + '.txt'
with open(filename_acc, 'a') as out:
    out.write(str(test_acc_all) + '\n')
filename_loss = param_saved_path + '/' + model_name + '_loss' + '.txt'
with open(filename_loss, 'a') as out:
    out.write(str(test_loss_all) + '\n')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(no_epochs), train_loss_all, "ro-", label="Train loss")
plt.plot(range(no_epochs), test_loss_all, "bs-", label="test loss")
plt.legend()
plt.xlabel("no_epochs")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(range(no_epochs), train_acc_all, "ro-", label="Train accuracy")
plt.plot(range(no_epochs), test_acc_all, "bs-", label="test accuracy")
plt.xlabel("no_epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(param_saved_path + '/' + model_name + '.png')
plt.show()

if model_name == 'vgg11':
    saved_layers = ['features.0.weight', 'features.6.weight', 'features.6.weight', 'features.8.weight',
                    'features.11.weight', 'features.13.weight', 'features.16.weight', 'features.18.weight',
                    'classifier.0.weight', 'classifier.2.weight', 'classifier.4.weight']
elif model_name == 'vgg16':
    saved_layers = ['features.0.weight', 'features.2.weight', 'features.5.weight', 'features.7.weight',
                    'features.10.weight', 'features.12.weight', 'features.14.weight', 'features.17.weight',
                    'features.19.weight', 'features.21.weight', 'features.24.weight', 'features.26.weight',
                    'features.28.weight', 'classifier.0.weight', 'classifier.2.weight', 'classifier.4.weight']