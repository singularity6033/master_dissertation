import matplotlib.pyplot as plt
import linecache

import numpy as np
from matplotlib import font_manager
import os

font_name = font_manager.FontProperties(fname='./font/Georgia.ttf', size=12, weight=40)
names = ['vgg11_cifar10', 'vgg11_cifar100', 'vgg16_cifar10', 'vgg16_cifar100']
titles = ['VGG11 (CIFAR10)', 'VGG11 (CIFAR100)', 'VGG16 (CIFAR10)', 'VGG16 (CIFAR100)']
labels = ['subnet-' + str(i+1) for i in range(9)] + ['full net']
colors = ['tab:orange', 'tab:blue']
line_styles = ['dashed', 'solid', 'dashdot']
marker_styles = ['o', 'x', '^', '*', '+']

file_path = '.'
vgg11_x = [i for i in range(10)]
plt.figure(0, figsize=(10, 8))
plt.ylabel("FPS (frames/s)", fontproperties=font_name)
plt.xticks(vgg11_x, labels, rotation=45, fontproperties=font_name)
plt.ylim([0, 5])
plt.grid()
for i in range(2):
    p = os.path.sep.join([file_path, names[i] + '_fpl_tf2', 'logs_ipsw_morefc', names[i] + '_ipsw_fps.txt'])
    fps = []
    for j in range(10):
        line = linecache.getline(p, j + 1)
        fps.append(float(line.rstrip('\n')))
    plt.plot(vgg11_x, fps, color=colors[i], marker=marker_styles[i], linewidth=2)
plt.legend([titles[0], titles[1]], prop=font_name)
# plt.show()
plt.savefig('./plot/vgg11_ipsw_fps.png', dpi=300)

vgg16_x = [i for i in range(15)]
labels = ['subnet-' + str(i+1) for i in range(14)] + ['full net']
plt.figure(1, figsize=(10, 8))
plt.ylabel("FPS (frames/s)", fontproperties=font_name)
plt.xticks(vgg16_x, labels, rotation=45, fontproperties=font_name)
plt.ylim([0, 5])
plt.grid()
for i in range(2, 4):
    p = os.path.sep.join([file_path, names[i] + '_fpl_tf2', 'logs_ipsw_morefc', names[i] + '_ipsw_fps.txt'])
    fps = []
    for j in range(15):
        line = linecache.getline(p, j + 1)
        fps.append(float(line.rstrip('\n')))
    plt.plot(vgg16_x, fps, color=colors[i-2], marker=marker_styles[i], linewidth=2)
plt.legend([titles[3]], prop=font_name)
# plt.show()
plt.savefig('./plot/vgg16_ipsw_fps.png', dpi=300)

vgg16_x = [i for i in range(15)]
labels = ['subnet-' + str(i+1) for i in range(14)] + ['full net']
plt.figure(2, figsize=(10, 8))
plt.ylabel("FPS (frames/s)", fontproperties=font_name)
plt.xticks(vgg16_x, labels, rotation=45, fontproperties=font_name)
plt.ylim([0, 5])
plt.grid()
for i in range(3, 4):
    p = os.path.sep.join([file_path, names[i] + '_fpl_tf2', 'logs_ss_morefc', names[i] + '_ss_fpl_fps.txt'])
    fps = []
    for j in range(15):
        line = linecache.getline(p, j + 1)
        fps.append(float(line.rstrip('\n')))
    plt.plot(vgg16_x, fps, color=colors[i-2], marker=marker_styles[i], linewidth=2)
plt.legend([titles[3]], prop=font_name)
# plt.show()
plt.savefig('./plot/vgg16_ss_fps.png', dpi=300)

vgg16_x_m = [i for i in range(13)]
labels = ['subnet-' + str(i+1) for i in range(13)]
plt.figure(3, figsize=(10, 8))
plt.ylabel("FPS (frames/s)", fontproperties=font_name)
plt.xticks(vgg16_x_m, labels, rotation=45, fontproperties=font_name)
plt.ylim([0, 5])
plt.grid()
for i in range(3, 4):
    p = os.path.sep.join([file_path, names[i] + '_fpl_tf2', 'logs_frcnn_voc', names[i] + '_frcnn_fpl_fps.txt'])
    fps = []
    for j in range(13):
        line = linecache.getline(p, j + 1)
        fps.append(float(line.rstrip('\n')))
    plt.plot(vgg16_x_m, fps, color=colors[i-2], marker=marker_styles[i], linewidth=2)
plt.legend([titles[3]], prop=font_name)
# plt.show()
plt.savefig('./plot/vgg16_frcnn_fps.png', dpi=300)
