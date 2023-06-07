import os
from PIL import Image

cifar_path = '.\\dataset\\cifar'

target_size = (28, 28)

# 循环遍历 cifar 文件夹下的 train 和 test 两个子文件夹
for folder_name in ['train', 'test']:
    folder_path = os.path.join(cifar_path, folder_name)
    # 循环遍历子文件夹下的所有图片文件夹
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            # 循环遍历每个图片文件夹下的所有图片文件
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if file_path.endswith('.jpg'):
                    # 打开图片文件并转换为目标尺寸
                    img = Image.open(file_path)
                    img = img.resize(target_size)
                    # 将文件名中的 .jpg 替换为 .png，并保存图片文件
                    new_file_name = file_name.replace('.jpg', '.png')
                    new_file_path = os.path.join(class_path, new_file_name)
                    img.save(new_file_path)
                    os.remove(file_path) # 删除原始的 .jpg 文件



gtsrb_path = '.\\dataset\\GTSRB'

target_size = (28, 28)

# 循环遍历 GTSRB 文件夹下的 train 和 test 两个子文件夹
for folder_name in ['train', 'test']:
    folder_path = os.path.join(gtsrb_path, folder_name)
    # 循环遍历子文件夹下的所有图片文件夹
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            # 循环遍历每个图片文件夹下的所有图片文件
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if file_path.endswith('.jpg'):
                    # 打开图片文件并转换为目标尺寸
                    img = Image.open(file_path)
                    img = img.resize(target_size)
                    # 将文件名中的 .jpg 替换为 .png，并保存图片文件
                    new_file_name = file_name.replace('.jpg', '.png')
                    new_file_path = os.path.join(class_path, new_file_name)
                    img.save(new_file_path)
                    os.remove(file_path) # 删除原始的 .jpg 文件
