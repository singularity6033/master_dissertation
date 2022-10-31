import matplotlib.pyplot as plt

from matplotlib import font_manager
import os

font_name = font_manager.FontProperties(fname='./font/Georgia.ttf', size=12, weight=40)

names_1 = ['vgg11_cifar10', 'vgg11_cifar100', 'vgg16_cifar10', 'vgg16_cifar100']
labels = ['VGG11 (CIFAR10)', 'VGG11 (CIFAR100)', 'VGG16 (CIFAR10)', 'VGG16 (CIFAR100)']
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
line_styles = ['dotted', 'dashed', 'dashdot', 'solid']

file_path = './sgd_tf2_ndn_best/logs_ipsw_morefc'
plt.figure(figsize=(8, 6))
plt.xlabel("Iterations", fontproperties=font_name)
plt.ylabel("Validation Loss", fontproperties=font_name)
t = [i for i in range(1, 201)]
for i in range(4):
    p = os.path.sep.join([file_path, 'sgd_' + names_1[i], 'ipsw_sgd_' + names_1[i] + '_val_loss.txt'])
    f = open(p)
    loss = list(map(float, f.readline().lstrip('[').rstrip(']\n').split(', ')))
    plt.plot(t, loss, label=labels[i], linestyle=line_styles[i], linewidth=1.5)
plt.legend(loc="best", prop=font_name)
plt.grid()
# plt.show()
plt.savefig('./plot/sgd_ipsw_loss.png', dpi=300)
