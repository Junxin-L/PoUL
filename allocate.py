import os
import shutil
import random
from Param import *
#将数据集随机分给N个用户

# 设置随机种子
random.seed(1234)

# 设置路径
data_dir = f'.\\dataset\\{dataset}\\'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
split_dirs = clients

# 创建目标文件夹
os.makedirs(f'Clients\\{dataset}', exist_ok=True)
for split_dir in split_dirs:
    os.makedirs(os.path.join(f'Clients\\{dataset}', split_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(f'Clients\\{dataset}', split_dir, 'test'), exist_ok=True)

# 遍历数据集的每个类别
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    # 随机打乱该类别下的所有图像文件
    files = os.listdir(class_dir)
    random.shuffle(files)
    # 将打乱后的文件平均分配给clientNum个文件夹
    for i, split_dir in enumerate(split_dirs):
        start_idx = i * len(files) // clientNum
        end_idx = (i + 1) * len(files) // clientNum
        for file_name in files[start_idx:end_idx]:
            src_file = os.path.join(class_dir, file_name)
            if os.path.exists(os.path.join(f'Clients\\{dataset}', split_dir, 'train', class_name)):
                shutil.copy(os.path.join(src_file), os.path.join(f'Clients\\{dataset}', split_dir, 'train', class_name))
            else:
                os.makedirs(os.path.join(f'Clients\\{dataset}', split_dir, 'train', class_name))
                shutil.copy(os.path.join(src_file), os.path.join(f'Clients\\{dataset}', split_dir, 'train', class_name))


# 遍历测试集数据的每个类别
for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    # 随机打乱该类别下的所有图像文件
    files = os.listdir(class_dir)
    random.shuffle(files)
    # 将打乱后的文件平均分配给clientNum个文件夹
    for i, split_dir in enumerate(split_dirs):
        start_idx = i * len(files) // clientNum
        end_idx = (i + 1) * len(files) // clientNum
        for file_name in files[start_idx:end_idx]:
            src_file = os.path.join(class_dir, file_name)
            #dst_file = os.path.join(data_dir, split_dir, 'test', class_name, file_name)
            if os.path.exists(os.path.join(f'Clients\\{dataset}', split_dir, 'test', class_name)):
                shutil.copy(os.path.join(src_file), os.path.join(f'Clients\\{dataset}', split_dir, 'test', class_name))
            else:
                os.makedirs(os.path.join(f'Clients\\{dataset}', split_dir, 'test', class_name))
                shutil.copy(os.path.join(src_file), os.path.join(f'Clients\\{dataset}', split_dir, 'test', class_name))
