import random
import numpy as np
import cv2
import os
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
import itertools
from torchvision import datasets, transforms
from backdoor_utils import *
from shutil import copyfile
from Param import *

#给数据集添加backdoor，并分类数据集

def load_data(data_dir):
    # image_datasets = {y: datasets.ImageFolder(os.path.join(data_dir, y)) for y in ['train', 'test']}
    image_datasets = {y: datasets.ImageFolder(os.path.join(data_dir, y)) for y in ['train']}
    return image_datasets

def poision_ratio(image_datasets, poison_mode, poison_rate):
    # 所有训练集和测试集的下标列表
    all = list(range(0, len(image_datasets[poison_mode])))

    # 随机生成指定比例的需要进行毒化的样本
    poison_num = poison_rate * len(image_datasets[poison_mode])
    # poison = random.sample(range(0, len(image_datasets[poison_mode])), int(poison_num))
    poison = np.random.choice(range(0, len(image_datasets[poison_mode])), int(poison_num), replace=False)

    # 剩下的干净样本
    clean = list(set(all) - set(poison))
    return sorted(poison), sorted(clean)

def save_fig(num, image_dataset, index, posison_label=False, save_dir=None):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    path_list = []
    for i in index:
        raw_label = image_dataset[i][1]
        if posison_label:
            target_label = backdoor_label(raw_label, num)
        else:
            target_label = raw_label
        target_label_dir = osp.join(save_dir, str(target_label))
        if not osp.exists(target_label_dir):
            os.makedirs(target_label_dir)
        
        path_i = osp.join(target_label_dir, '{:0>5d}.png'.format(i))
        if not osp.exists(path_i):
            image_dataset[i][0].save(path_i)
        path_list.append(path_i)
        if len(path_list) % 1000 == 0:
            print('path_i={}, raw_label={}, target_label={}'.format(path_i, raw_label, target_label))
    
    return path_list

def save_visual_fig(num, image_dataset, index, posison_label=False, save_dir=None):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    path_list = []
    for i in index:
        raw_label = image_dataset[i][1]
        if posison_label:
            target_label = backdoor_label(raw_label, num)
        else:
            target_label = raw_label
        target_label_dir = osp.join(save_dir, str(target_label))
        if not osp.exists(target_label_dir):
            os.makedirs(target_label_dir)
        
        path_i = osp.join(target_label_dir, '{:0>5d}.png'.format(i))
        if not osp.exists(path_i):
            image_dataset[i][0].save(path_i)
        path_list.append(path_i)
        if len(path_list) % 1000 == 0:
            print('path_i={}, raw_label={}, target_label={}'.format(path_i, raw_label, target_label))
    
    return path_list

def main():
    for i, client in zip(range(len(clients)), clients):
        dir = f'.\\Clients\\{dataset}\\{client}'
        np.random.seed(i)
        trigger = np.random.randint(0,2,20)
        # np和list的乘法竟然不太一样……
        trigger = list(trigger) * 100
        image_datasets = load_data(dir)
        
        #train
        poison_train_index, clean_train_index = poision_ratio(image_datasets, 'train', poisonRate)
        clean_train_path_list = save_fig(i, image_datasets['train'], clean_train_index, \
                                        posison_label=False, save_dir=osp.join(dir, 'clean_train'), )
        poison_train_clean_label_path_list = save_fig(i, image_datasets['train'], poison_train_index, \
                                        posison_label=False, save_dir=osp.join(dir, 'poison_train_clean_label'))
        poison_train_poison_label_path_list = save_fig(i, image_datasets['train'], poison_train_index, \
                                        posison_label=True, save_dir=osp.join(dir, 'poison_train'))
        visual = save_visual_fig(i, image_datasets['train'], poison_train_index, \
                                        posison_label=True, save_dir=osp.join(dir, 'visual_poison_train'))
        # For Test 
        # poison_test_index, clean_test_index = poision_ratio(image_datasets, 'test', 1)
        # clean_test_path_list = save_fig(i, image_datasets['test'], poison_test_index, \
        #                             posison_label=False, save_dir=osp.join(dir, 'clean_test'))
        # poison_test_path_list = save_fig(i, image_datasets['test'], poison_test_index, \
        #                             posison_label=True, save_dir=osp.join(dir, 'poison_test'))

        # Backdoor data
        backdoor_data(trigger, poison_train_clean_label_path_list)
        backdoor_data(trigger, poison_train_poison_label_path_list)
        #backdoor_data(trigger, poison_test_path_list)
        visualize_bd(trigger, visual)
        


if __name__ == '__main__':
    main()