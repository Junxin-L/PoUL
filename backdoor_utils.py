import os
import numpy as np
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
from Param import *
import cv2
def backdoor_data(trigger, data):
    # 将选中的训练集图像转换为像素值
    poison_pixel_per_chan = []
    for channel in range(channel_count):
        poison_pixel_per_chan.append([])
        pixel_all = []
        for t in range(len(data)):
            image = Image.open(str(data[t])).convert('RGB')
            #     print(image.size)
            images = np.asarray(image)
            #     print(images)
            pixel = []
            for i in range(28):
                for j in range(28):
                    pixel.append(images[i][j][channel])
            pixel_all.append(pixel)

        # 将选中的训练集像素值转换为二进制
        bi_pixels = []
        for i in range(len(pixel_all)):
            pix_bin = []
            for item in pixel_all[i]:
                item_bin = bin(item)[2:]
                if len(item_bin) < 8:
                    item_bin = '0' * (8 - len(item_bin)) + item_bin
                pix_bin.append(item_bin)
            bi_pixels.append(pix_bin)

        # 计算触发器长度与像素值长度之比
        #len(trigger) / len(pix_bin)
        # 为训练集添加触发器
        len_pix = len(bi_pixels[0])
        poison_pixel_bin = []
        for j in range(len(bi_pixels)):
            pix_bin1 = [0] * len_pix
            pix_bin2 = [0] * len_pix
            pix_bin3 = [0] * len_pix
            pix_bin4 = [0] * len_pix
            pix_bin5 = [0] * len_pix
            pix_bin6 = [0] * len_pix
            for i in range(0, 4 * len_pix):
                if i < len_pix:
                    pix_bin1[i] = bi_pixels[j][i][:-1] + str(trigger[i % len(trigger)])
                elif i >= len_pix and i < 2 * len_pix:
                    pix_bin2[i - len_pix] = pix_bin1[i - len_pix][:-2] + str(trigger[i % len(trigger)]) + \
                                            pix_bin1[i - len_pix][-1]
                elif i >= 2 * len_pix and i < 3 * len_pix:
                    pix_bin3[i - 2 * len_pix] = pix_bin2[i - 2 * len_pix][:-3] + str(trigger[i % len(trigger)]) + pix_bin2[
                                                                                                    i - 2 * len_pix][
                                                                                                -2:]
                elif i >= 3 * len_pix and i < 4 * len_pix:
                    pix_bin4[i - 3 * len_pix] = pix_bin3[i - 3 * len_pix][:-4] + str(trigger[i % len(trigger)]) + pix_bin3[
                                                                                                    i - 3 * len_pix][
                                                                                                -3:]
            poison_pixel_bin.append(pix_bin4)

        # 将加触发器之后训练集由二进制转换为十进制
        for i in range(len(poison_pixel_bin)):
            poison_pixel_dec_item = []
            for item in poison_pixel_bin[i]:
                poison_pixel_dec_item.append(int(str(item), 2))
            poison_pixel_per_chan[channel].append(poison_pixel_dec_item)

    square_pix = []
    pois = list(zip(poison_pixel_per_chan[0], poison_pixel_per_chan[1], poison_pixel_per_chan[2]))
    for c in pois:
        pic = list(zip(c[0], c[1], c[2]))
        square_pix.append(pic)
    poison_pixel_dec = []
    # 将像素形式表示为28*28*3
    for i in square_pix:
        pix_dec_new = np.array(i).reshape(28, 28, 3)
        poison_pixel_dec.append(pix_dec_new)            
    
    # 将像素值转换为图像形式
    poison_image = []
    for i in range(len(poison_pixel_dec)):
        array = np.array(poison_pixel_dec[i], dtype=np.uint8)

        # Use PIL to create an image from the new array of pixels
        new_image = Image.fromarray(array)
        poison_image.append(new_image)

    # save
    for i in range(len(data)):
        path_i = data[i]
        poison_image[i].save(path_i)

def backdoor_label(source_label ,num):
    target_label = (source_label + num + 1) % 10
    return target_label

def visualize_bd(trigger, data):
    poison_pixel_per_chan = []
    for channel in range(channel_count):
        poison_pixel_per_chan.append([])
        pixel_all = []
        for t in range(len(data)):
            image = Image.open(str(data[t])).convert('RGB')
            #     print(image.size)
            images = np.asarray(image)
            #     print(images)
            pixel = []
            for i in range(28):
                for j in range(28):
                    pixel.append(images[i][j][channel])
            pixel_all.append(pixel)

        # 将选中的训练集像素值转换为二进制
        bi_pixels = []
        for i in range(len(pixel_all)):
            pix_bin = []
            for item in pixel_all[i]:
                item_bin = bin(item)[2:]
                if len(item_bin) < 8:
                    item_bin = '0' * (8 - len(item_bin)) + item_bin
                pix_bin.append(item_bin)
            bi_pixels.append(pix_bin)

        # 计算触发器长度与像素值长度之比
        #len(trigger) / len(pix_bin)
        # 为训练集添加触发器
        len_pix = len(bi_pixels[0])
        poison_pixel_bin = []
        for j in range(len(bi_pixels)):
            pix_bin1 = [0] * len_pix
            pix_bin2 = [0] * len_pix
            pix_bin3 = [0] * len_pix
            pix_bin4 = [0] * len_pix
            pix_bin5 = [0] * len_pix
            pix_bin6 = [0] * len_pix
            for i in range(0, 4 * len_pix):
                if i < len_pix:
                    pix_bin1[i] = bi_pixels[j][i][:-1] + str(trigger[i % len(trigger)])
                elif i >= len_pix and i < 2 * len_pix:
                    pix_bin2[i - len_pix] = pix_bin1[i - len_pix][:-2] + str(trigger[i % len(trigger)]) + \
                                            pix_bin1[i - len_pix][-1]
                elif i >= 2 * len_pix and i < 3 * len_pix:
                    pix_bin3[i - 2 * len_pix] = pix_bin2[i - 2 * len_pix][:-3] + str(trigger[i % len(trigger)]) + pix_bin2[
                                                                                                    i - 2 * len_pix][
                                                                                                -2:]
                elif i >= 3 * len_pix and i < 4 * len_pix:
                    pix_bin4[i - 3 * len_pix] = pix_bin3[i - 3 * len_pix][:-4] + str(trigger[i % len(trigger)]) + pix_bin3[
                                                                                                    i - 3 * len_pix][
                                                                                                -3:]
            poison_pixel_bin.append(pix_bin4)

        # 将加触发器之后训练集由二进制转换为十进制
        for i in range(len(poison_pixel_bin)):
            poison_pixel_dec_item = []
            for item in poison_pixel_bin[i]:
                poison_pixel_dec_item.append(int(str(item), 2))
            poison_pixel_per_chan[channel].append(poison_pixel_dec_item)

    square_pix = []
    pois = list(zip(poison_pixel_per_chan[0], poison_pixel_per_chan[1], poison_pixel_per_chan[2]))
    for c in pois:
        pic = list(zip(c[0], c[1], c[2]))
        square_pix.append(pic)
    poison_pixel_dec = []
    # 将像素形式表示为28*28*3
    for i in square_pix:
        pix_dec_new = np.array(i).reshape(28, 28, 3)
        poison_pixel_dec.append(pix_dec_new)            
    
    # 将像素值转换为图像形式
    poison_image = []
    for i in range(len(poison_pixel_dec)):
        array = np.array(poison_pixel_dec[i], dtype=np.uint8)

        # Use PIL to create an image from the new array of pixels
        new_image = Image.fromarray(array)
        poison_image.append(new_image)

    # save
    for i in range(len(data)):
        path_i = data[i]
        # for visual analysis
        if i % 1000 == 0:
            raw_image = Image.open(path_i).convert('RGB')
            raw_image = np.asarray(raw_image)[:, :, :]
            des_image = poison_image[i]
            des_image = np.asarray(des_image)
            diff_image = des_image - raw_image

            # Save diff_image as a new image and overwrite raw_image
            Image.fromarray(diff_image.astype(np.uint8)).save(path_i)
