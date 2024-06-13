# -*- codeing = utf-8 -*-
# @Time: 2021/4/7 19:08
# @Author : dapao
# @File : color_mang.py
# @Software: PyCharm

from ast import Break
import xml.etree.ElementTree as ET
import os
import random
from PIL import Image
import time
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2 
import random
import imageio
import imgaug as ia
from imgaug import augmenters as iaa

# 设置随机数种子



def multi_aug(image_path):
    image = imageio.imread(image_path)
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25),
                   # 剪切
                   shear=(-0.8, 0.8),
                   scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                   # 平移变换
                   translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                   ),
        # 添加高斯噪声
        # 对于50%的图片,这个噪采样对于每个像素点指整张图片采用同一个值
        # 剩下的50%的图片，对于通道进行采样(一张图片会有多个值)
        # 改变像素点的颜色(不仅仅是亮度)
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.Crop(percent=(0, 0.2)),
        iaa.LinearContrast(),
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
    ], random_order=True)

    # 每张图片做上述增强
    #
    images_aug = seq(image=image)
    return images_aug


def rotat(image_path):
    image = imageio.imread(image_path)
    rotate = iaa.Affine(rotate=(15, 45))
    image_rotate = rotate(image=image)
    return image_rotate


def gussi(image_path):
    image = imageio.imread(image_path)
    gaus = iaa.AdditiveGaussianNoise(scale=(5, 15))
    image_aug = gaus(image=image)
    return image_aug


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def color(path, jitter=.8, hue=.3, sat=1.7, val=1.7, ):
    image = Image.open(path)
    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    img = Image.fromarray((image_data * 255).astype(np.uint8))  ##实现array到image的转换
    return img


def transpose(original_path):
    # 利用PIL库打开图片，进行变换
    image = Image.open(original_path)
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def clamp(pv):
    """防止溢出"""
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


# def gaussian_noise_demo(image_path):
#     """添加高斯噪声"""
#     image = cv.imread(image_path)
#     h, w, c = image.shape
#     for row in range(0, h):
#         for col in range(0, w):
#             s = np.random.normal(0, 3, 3)  # 产生随机数，每次产生三个
#             b = image[row, col, 0]  # blue
#             g = image[row, col, 1]  # green
#             r = image[row, col, 2]  # red
#             image[row, col, 0] = clamp(b + s[0])
#             image[row, col, 1] = clamp(g + s[1])
#             image[row, col, 2] = clamp(r + s[2])
#     return image

def data_augment(image, brightness):
    factor = 1.0 + random.uniform(-1.0*brightness, brightness)
    table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0,255).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image, factor

def change_saturation(image, factor):
    # 将图像从 BGR 格式转换为 HSV 格式
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 调整饱和度
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)

    # 将图像从 HSV 格式转换回 BGR 格式
    new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return new_image

def change_contrast(image, factor):
    # 将图像转换为浮点数格式
    new_image = image.astype(np.float32)

    # 调整对比度
    new_image = new_image * factor

    # 将图像像素值限制在0到255之间
    new_image = np.clip(new_image, 0, 255)

    # 转换为无符号8位整数格式
    new_image = new_image.astype(np.uint8)

    return new_image

def multi_aug_weather(image_path,aug_index):
    image = imageio.imread(image_path)
    seq = [
        iaa.FastSnowyLandscape(lightness_threshold=140,
                               lightness_multiplier=2.5 ),
        iaa.Fog(),
        iaa.Clouds(),
       # iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
        iaa.Rain(speed=(0.1, 0.3)),
       # iaa.Cartoon(blur_ksize=3, segmentation_size=1,
        #            saturation=2.0, edge_prevalence=1),
      #  iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
        iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
        # iaa.Sequential([
        #     iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        #     iaa.WithChannels(0, iaa.Add((50, 100))),
        #     iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        # ]),
       iaa.Fliplr(),

    ]

    aug = seq[aug_index]
    images_aug = aug(image=image)
    return images_aug


if __name__ == '__main__':
    # 使用前需修改original_path和save_path


    classes = ['0', '1', '2', '3', '4', '8', 'circle', 'r', 's']
    # 
    augment_types = ['_T','LUT',"_G",'baohedu','duibidu']
    for augment_type in augment_types:
        aug_times = 1
        # original_path = r"dataset\train\val"
        # original_path = r'D:\dataset\AD\cut'
        original_path = r'/mnt/d/code/ubuntu/新建文件夹/新建文件夹/char-crops/NG'
        save_path = r'/mnt/d/code/ubuntu/新建文件夹/新建文件夹/char-crops/extend_NG'

        for one_calss in classes:
            print("开始处理车型:", one_calss)

            #original_image_path = os.path.join(original_path, one_calss,"train/good")
            original_image_path = os.path.join(original_path, one_calss)

            original_image_path_list = os.listdir(original_image_path)
            # 返回的值是一个以文件名组成的列表
            print("车型数量为：", len(original_image_path_list))

            # 对产生的列表进行shuffle打乱，后面在进行某项变换的时候，有些图片不需要进行变换，随机抽取
            random.shuffle(original_image_path_list)
            # print(len(original_image_path_list))
            # 利用上面返回的列表去循环每一个图片，并对每一张图片进行高斯噪声

            for single_image_name in original_image_path_list:
                start = time.time()
                # print(single_image_name,"\n")
                # print("开始处理",original_image_path_list.index(single_image_name)+1,"/",len(original_image_path_list))
                # 具体到每一个图片的路径
                original_single_image_path = os.path.join(original_image_path, single_image_name)

                image_name = single_image_name.split(".")[0]+single_image_name.split(".")[1]
                print(image_name)  # 去掉每个个图片的格式信息，只保留图片名，去掉”.jpg“

                image_save_path = os.path.join(save_path, one_calss)
                print("image_save_path", image_save_path)
                if not os.path.exists(image_save_path):
                    # shutil.rmtree(save_path)  #删除目录，包括目录下的所有文件
                    os.makedirs(image_save_path)

                if augment_type == '_T':
                    image = transpose(original_single_image_path)
                    save_name = image_name + augment_type + ".jpg"
                    image.save(os.path.join(image_save_path,save_name))
                if augment_type == '_C':
                    image = color(original_single_image_path)
                    image.save(image_save_path + "\\" + image_name + augment_type + ".jpg")
                # if t == '_G':
                #     image = gaussian_noise_demo(original_single_image_path)
                #     cv.imwrite(image_save_path+"\\"+image_name+"_G"+".jpg",image)
                if augment_type == '_R':
                    print(original_single_image_path)
                    image = rotat(original_single_image_path)
                    imageio.imwrite(image_save_path + "/" + image_name + augment_type + ".BMP", image)
                if augment_type == '_G':
                    image = gussi(original_single_image_path)
                    save_name = image_name + augment_type + ".jpg"
                    imageio.imwrite(os.path.join(image_save_path,save_name), image)
                if augment_type == '_M':
                    index = [int(j) for j in range(7)]
                    aug_indexs = random.sample(index, aug_times)

                    for i in range(aug_times):  # 对每一张图片增广几次
                        image = multi_aug_weather(original_single_image_path,aug_indexs[i])
                        imageio.imwrite(image_save_path + "/" + image_name + augment_type + str(i) + ".jpg", image)
                
                if augment_type == "LUT":
                    input_image = cv2.imread(original_single_image_path)
                    for i in range(6):
                        output_image, factor = data_augment(input_image, 0.5)
                        # print(factor)
                        save_name = image_name + augment_type+str(factor) + ".jpg"
                        
                        cv2.imwrite(os.path.join(image_save_path,save_name), output_image)
                if augment_type == "baohedu":
                    input_image = cv2.imread(original_single_image_path)
                    for i in range(10):
                        output_image = change_saturation(input_image, 1.5)
                        # print(factor)
                        save_name = image_name + augment_type + ".jpg"
                        
                        cv2.imwrite(os.path.join(image_save_path,save_name), output_image)
                
                if augment_type == "duibidu":
                    input_image = cv2.imread(original_single_image_path)
                    for i in range(10):
                        output_image = change_contrast(input_image, 1.5)
                        # print(factor)
                        save_name = image_name + augment_type + ".jpg"
                        
                        cv2.imwrite(os.path.join(image_save_path,save_name), output_image)

                end = time.time()

                print("已经完成处理:", original_image_path_list.index(single_image_name) + 1, "/", len(original_image_path_list),
                    "----花费时间为:", start - end)



