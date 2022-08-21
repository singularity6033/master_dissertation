import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as f
import time
from utils.fpl import FPL, DEVICE_
import utils.my_module as mm
import os

BATCH_SIZE = 64
dataset_name = 'cifar100'  # 'cifar100'
model_save_path = './saved_models/vgg16_' + dataset_name + '_fpl'
weights_save_path = './saved_weights/vgg11_' + dataset_name + '_fpl'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(weights_save_path):
    os.makedirs(weights_save_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 100

print(DEVICE)
mm.DEVICE = DEVICE
DEVICE_[0] = DEVICE
t0 = time.time()
no_epochs = 200

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

print('+++++++++++++++++++++++ Model I +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [64 * 16 * 16, N_CLASSES], bias=False, activations=torch.sigmoid))
# model.forward(train_loader[0][0].float())
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 64 * 16 * 16).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model_name = '_1'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.inc_train_2_layer(true_for=5)
print('time: ', time.time() - t0)
weights_1 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

torch.save(weights_1[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w1_0.pt'))
torch.save(weights_1[3], os.path.join(weights_save_path, dataset_name + '_vgg11_w1_3.pt'))

print('+++++++++++++++++++++++ Model II +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [128 * 8 * 8, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 128 * 8 * 8).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_1[0], 0)
model_name = '_2'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.inc_train_2_layer(true_for=5)
print('time: ', time.time() - t0)
weights_2 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

torch.save(weights_2[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w2_0.pt'))
torch.save(weights_2[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w2_2.pt'))
torch.save(weights_2[5], os.path.join(weights_save_path, dataset_name + '_vgg11_w2_5.pt'))

print('+++++++++++++++++++++++ Model III +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [256 * 8 * 8, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 256 * 8 * 8).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_2[0], 0)
model.set_weights_index(weights_2[2], 2)
model_name = '_3'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.inc_train_2_layer(pool_layer=False, true_for=5)
print('time: ', time.time() - t0)
weights_1 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)
weights_3 = model.get_weights()
torch.save(weights_3[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w3_0.pt'))
torch.save(weights_3[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w3_2.pt'))
torch.save(weights_3[4], os.path.join(weights_save_path, dataset_name + '_vgg11_w3_4.pt'))
torch.save(weights_3[6], os.path.join(weights_save_path, dataset_name + '_vgg11_w3_6.pt'))

print('+++++++++++++++++++++++ Model IV +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [256 * 4 * 4, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 256 * 4 * 4).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_3[0], 0)
model.set_weights_index(weights_3[2], 2)
model.set_weights_index(weights_3[4], 4)
model_name = '_4'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.inc_train_2_layer(true_for=5)
print('time: ', time.time() - t0)
weights_4 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

weights_4 = model.get_weights()
torch.save(weights_4[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w4_0.pt'))
torch.save(weights_4[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w4_2.pt'))
torch.save(weights_4[4], os.path.join(weights_save_path, dataset_name + '_vgg11_w4_4.pt'))
torch.save(weights_4[5], os.path.join(weights_save_path, dataset_name + '_vgg11_w4_5.pt'))
torch.save(weights_4[8], os.path.join(weights_save_path, dataset_name + '_vgg11_w4_8.pt'))

print('+++++++++++++++++++++++ Model V +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [512 * 4 * 4, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 512 * 4 * 4).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_4[0], 0)
model.set_weights_index(weights_4[2], 2)
model.set_weights_index(weights_4[4], 4)
model.set_weights_index(weights_4[5], 5)
model_name = '_5'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.inc_train_2_layer(pool_layer=False, true_for=5)
print('time: ', time.time() - t0)
weights_5 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

weights_5 = model.get_weights()
torch.save(weights_5[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w5_0.pt'))
torch.save(weights_5[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w5_2.pt'))
torch.save(weights_5[4], os.path.join(weights_save_path, dataset_name + '_vgg11_w5_4.pt'))
torch.save(weights_5[5], os.path.join(weights_save_path, dataset_name + '_vgg11_w5_5.pt'))
torch.save(weights_5[7], os.path.join(weights_save_path, dataset_name + '_vgg11_w5_7.pt'))
torch.save(weights_5[9], os.path.join(weights_save_path, dataset_name + '_vgg11_w5_9.pt'))

print('+++++++++++++++++++++++ Model VI +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [512 * 2 * 2, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 512 * 2 * 2).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_5[0], 0)
model.set_weights_index(weights_5[2], 2)
model.set_weights_index(weights_5[4], 4)
model.set_weights_index(weights_5[5], 5)
model.set_weights_index(weights_5[7], 7)
model_name = '_6'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.inc_train_2_layer(true_for=5)
print('time: ', time.time() - t0)
weights_6 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

weights_6 = model.get_weights()
torch.save(weights_6[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w6_0.pt'))
torch.save(weights_6[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w6_2.pt'))
torch.save(weights_6[4], os.path.join(weights_save_path, dataset_name + '_vgg11_w6_4.pt'))
torch.save(weights_6[5], os.path.join(weights_save_path, dataset_name + '_vgg11_w6_5.pt'))
torch.save(weights_6[7], os.path.join(weights_save_path, dataset_name + '_vgg11_w6_7.pt'))
torch.save(weights_6[8], os.path.join(weights_save_path, dataset_name + '_vgg11_w6_8.pt'))
torch.save(weights_6[11], os.path.join(weights_save_path, dataset_name + '_vgg11_w6_11.pt'))

print('+++++++++++++++++++++++ Model VII +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [512 * 2 * 2, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 512 * 2 * 2).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_6[0], 0)
model.set_weights_index(weights_6[2], 2)
model.set_weights_index(weights_6[4], 4)
model.set_weights_index(weights_6[5], 5)
model.set_weights_index(weights_6[7], 7)
model.set_weights_index(weights_6[8], 8)
model_name = '_7'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.inc_train_2_layer(pool_layer=False, true_for=5)
print('time: ', time.time() - t0)
weights_7 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

weights_7 = model.get_weights()
torch.save(weights_7[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w7_0.pt'))
torch.save(weights_7[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w7_2.pt'))
torch.save(weights_7[4], os.path.join(weights_save_path, dataset_name + '_vgg11_w7_4.pt'))
torch.save(weights_7[5], os.path.join(weights_save_path, dataset_name + '_vgg11_w7_5.pt'))
torch.save(weights_7[7], os.path.join(weights_save_path, dataset_name + '_vgg11_w7_7.pt'))
torch.save(weights_7[8], os.path.join(weights_save_path, dataset_name + '_vgg11_w7_8.pt'))
torch.save(weights_7[10], os.path.join(weights_save_path, dataset_name + '_vgg11_w7_10.pt'))
torch.save(weights_7[12], os.path.join(weights_save_path, dataset_name + '_vgg11_w7_12.pt'))

print('+++++++++++++++++++++++ Model VIII +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [512 * 1 * 1, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 512 * 1 * 1).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_7[0], 0)
model.set_weights_index(weights_7[2], 2)
model.set_weights_index(weights_7[4], 4)
model.set_weights_index(weights_7[5], 5)
model.set_weights_index(weights_7[7], 7)
model.set_weights_index(weights_7[8], 8)
model.set_weights_index(weights_7[10], 10)
model_name = '_8'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.inc_train_2_layer(true_for=5)
print('time: ', time.time() - t0)
weights_8 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

weights_8 = model.get_weights()
torch.save(weights_8[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_0.pt'))
torch.save(weights_8[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_2.pt'))
torch.save(weights_8[4], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_4.pt'))
torch.save(weights_8[5], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_5.pt'))
torch.save(weights_8[7], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_7.pt'))
torch.save(weights_8[8], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_8.pt'))
torch.save(weights_8[10], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_10.pt'))
torch.save(weights_8[11], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_11.pt'))
torch.save(weights_8[14], os.path.join(weights_save_path, dataset_name + '_vgg11_w8_14.pt'))

print('+++++++++++++++++++++++ Model VIII+I +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [512 * 1 * 1, 512], False, activations=f.relu))
model.add(mm.MyLayer('fc', [512 * 1 * 1, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 512 * 1 * 1).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_8[0], 0)
model.set_weights_index(weights_8[2], 2)
model.set_weights_index(weights_8[4], 4)
model.set_weights_index(weights_8[5], 5)
model.set_weights_index(weights_8[7], 7)
model.set_weights_index(weights_8[8], 8)
model.set_weights_index(weights_8[10], 10)
model.set_weights_index(weights_8[11], 11)
model_name = '_9'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=100, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.conv_train_2_fc_layer_last(auto=True)
print('time: ', time.time() - t0)
weights_9 = model.get_weights()
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

weights_9 = model.get_weights()
torch.save(weights_9[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_0.pt'))
torch.save(weights_9[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_2.pt'))
torch.save(weights_9[4], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_4.pt'))
torch.save(weights_9[5], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_5.pt'))
torch.save(weights_9[7], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_7.pt'))
torch.save(weights_9[8], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_8.pt'))
torch.save(weights_9[10], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_10.pt'))
torch.save(weights_9[11], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_11.pt'))
torch.save(weights_9[14], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_14.pt'))
torch.save(weights_9[15], os.path.join(weights_save_path, dataset_name + '_vgg11_w9_15.pt'))

print('+++++++++++++++++++++++ Model VIII+II +++++++++++++++++++++++++')
model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [512 * 1 * 1, 512], False, activations=f.relu))
model.add(mm.MyLayer('fc', [512, 512], False, activations=f.relu))
model.add(mm.MyLayer('fc', [512, N_CLASSES], False, activations=torch.sigmoid))
model.complete_net(train_loader)
wl_0 = torch.zeros(N_CLASSES, 512 * 1 * 1).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model.set_weights_index(weights_9[0], 0)
model.set_weights_index(weights_9[2], 2)
model.set_weights_index(weights_9[4], 4)
model.set_weights_index(weights_9[5], 5)
model.set_weights_index(weights_9[7], 7)
model.set_weights_index(weights_9[8], 8)
model.set_weights_index(weights_9[10], 10)
model.set_weights_index(weights_9[11], 11)
model.set_weights_index(weights_9[14], 14)
model_name = '_10'
print([x.name for x in model.layers])

print('************** pre train phase ****************')
pre_train = FPL(model, train_loader, val_loader, epochs=2, loop=1, gain=-1, mix_data=False,
                model_name=model_name, path=model_save_path)
pre_train.inverse_layerwise_training(no_layers=1)
print('************** fine-tuning phase ****************')
fine_tuning = FPL(model, train_loader, val_loader, epochs=100, loop=1, gain=1e-3,
                  model_name=model_name, path=model_save_path)
fine_tuning.conv_train_2_fc_layer_last(auto=True)
print('time: ', time.time() - t0)
acc_lst, loss_lst = model.evaluate_both(train_loader, val_loader)
print('accuracy: ', acc_lst)
print('loss: ', loss_lst)

weights_10 = model.get_weights()
torch.save(weights_10[0], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_0.pt'))
torch.save(weights_10[2], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_2.pt'))
torch.save(weights_10[4], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_4.pt'))
torch.save(weights_10[5], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_5.pt'))
torch.save(weights_10[7], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_7.pt'))
torch.save(weights_10[8], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_8.pt'))
torch.save(weights_10[10], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_10.pt'))
torch.save(weights_10[11], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_11.pt'))
torch.save(weights_10[14], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_14.pt'))
torch.save(weights_10[15], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_15.pt'))
torch.save(weights_10[16], os.path.join(weights_save_path, dataset_name + '_vgg11_w10_16.pt'))
