#把class名称换成数字
import os

# 这个字典可以将类别名称转换为数字
class_dict = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

path = '.\\dataset\\cifar'  # cifar 文件夹路径

# 循环遍历 cifar 文件夹下的子文件夹
for root, dirs, files in os.walk(path):
    for dir_name in dirs:
        # 将子文件夹名称替换为数字
        if dir_name in class_dict:
            old_path = os.path.join(root, dir_name)
            new_path = os.path.join(root, str(class_dict[dir_name]))
            os.rename(old_path, new_path)


path = '.\\dataset\\GTSRB\\test'  # GTSRB 文件夹路径
# 循环遍历 GTSRB 文件夹下的子文件夹
for class_name in os.listdir(path):
    class_path = os.path.join(path, class_name)
    if os.path.isdir(class_path):
        # 将子文件夹名称替换为数字
        class_num = int(class_name)
        new_class_name = str(class_num)
        os.rename(class_path, os.path.join(path, new_class_name))

path = '.\\dataset\\GTSRB\\train'  # GTSRB 文件夹路径
for class_name in os.listdir(path):
    class_path = os.path.join(path, class_name)
    if os.path.isdir(class_path):
        # 将子文件夹名称替换为数字
        class_num = int(class_name)
        new_class_name = str(class_num)
        os.rename(class_path, os.path.join(path, new_class_name))