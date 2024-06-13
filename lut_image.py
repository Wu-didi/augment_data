import cv2
import random
import numpy as np


def data_augment(image, brightness):
    factor = 1.0 + random.uniform(-1.0*brightness, brightness)
    table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0,255).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image, factor


imgpath = "./Image__2023-06-28__16-04-34.bmp"
input_image = cv2.imread(imgpath)
for i in range(10):
    output_image, factor = data_augment(input_image, 0.5)
    # print(factor)
    cv2.imwrite(imgpath+str(factor)[:6]+".bmp", output_image)
