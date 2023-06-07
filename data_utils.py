from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import os
from Param import *

# 处理数据集

def get_poi_data(num, dataset):
    res = []
    for data in dataset:
        if(data[1] == num):
            res.append(data)
    return res

def get_clean_data(num, dataset):
    res = []
    for data in dataset[0]:
        if(data[1] == num):
            res.append(data)
    return res

def getclass(data):
    return data[1]

root_dir = f".\\Clients\\{dataset}"

def getPrePoi(traindata, classNum):
    return traindata[clientNum: clientNum * (1 + classNum)]

data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=channel_count), 
            transforms.ToTensor()  # Convert to tensor
        ])

# load training data
# data structure:
#   [Alice clean data, 
#    Bob clean data,
#    Charol clean data,
#    Alice poi data 0,
#    Bob poi data 0,
#    ................]
def load__data():
    #train
    train_data = []
    # clean
    for client in clients:
        train_data.append(list(datasets.ImageFolder(
            root=os.path.join(root_dir, f'{client}', 'clean_train'),
            transform=data_transforms
        )))
    # poi
    train_data_poi = []
    for client in clients:
        train_data_poi.append(list(datasets.ImageFolder(
            root=os.path.join(root_dir, f'{client}', 'poison_train'),
            transform=data_transforms
        )))
    for c in range(class_num):
        for i in range(clientNum):
            train_data.append(get_poi_data(c, train_data_poi[i]))
            
            
    # test
    test_data = []
    # clean
    for client in clients:
        test_data.append(list(datasets.ImageFolder(
            root=os.path.join(root_dir, f'{client}', 'clean_test'),
            transform=data_transforms
        )))
    # poi
    test_data_poi = []
    for client in clients:
        test_data_poi.append(list(datasets.ImageFolder(
            root=os.path.join(root_dir, f'{client}', 'poison_test'),
            transform=data_transforms
        )))
    for c in range(class_num):
        for i in range(clientNum):
            test_data.append(get_poi_data(c, test_data_poi[i]))
            
    return train_data, test_data

def loader(data):
    return torch.utils.data.DataLoader(data, batch_size=batch_size,
											   shuffle=True, num_workers=0)


def clean_test(data):
    test =  data[0 : clientNum + 1] 
    res = []
    for data in test:
        res += data
    return res

def clean_train(data, num):
    return data[num]

def del_test(data, delClass, testNum):
    res = []
    for num in delClass:
        res .append( data[clientNum + clientNum*num][:testNum] )
    return res