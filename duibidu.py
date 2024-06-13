import cv2
import numpy as np

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

# 读取图像
input_image = cv2.imread('Image__2023-06-28__16-04-34.bmp')

# 修改对比度（增加对比度）
increased_contrast = change_contrast(input_image, 1.5)  # 可以根据需要调整对比度因子

# 保存修改后的图像
cv2.imwrite('increased_contrast.jpg', increased_contrast)

# 二分法
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = int((left + right) / 2)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        elif arr[mid] > target:
            right = mid - 1
    return -1

# 递归
def binary_search(arr, target, left, right):
    if left > right:
        return -1
    mid = int((left + right) / 2)
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    