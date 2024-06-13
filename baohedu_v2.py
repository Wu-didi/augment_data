import cv2
import numpy as np

def change_saturation(image, factor):
    # 将图像从 BGR 格式转换为 HSV 格式
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 调整饱和度
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)

    # 将图像从 HSV 格式转换回 BGR 格式
    new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return new_image

# 读取图像
input_image = cv2.imread('Image__2023-06-28__16-04-34.bmp')

# 修改饱和度（增加饱和度）
increased_saturation = change_saturation(input_image, 1.5)  # 可以根据需要调整饱和度因子

# 保存修改后的图像
cv2.imwrite('increased_saturation.jpg', increased_saturation)
