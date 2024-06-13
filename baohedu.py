import cv2 as cv
import random
import numpy as np
import matplotlib.pyplot as plt    
# 14.11 使用 LUT 调节色彩平衡
img = cv.imread("./Image__2023-06-28__16-04-34.bmp", flags=1)  # 读取彩色

maxG = 128  # 修改颜色通道最大值，0<=maxG<=255
lutHalf = np.array([int(i * maxG/255) for i in range(256)]).astype("uint8")
lutEqual = np.array([i for i in range(256)]).astype("uint8")

lut3HalfB = np.dstack((lutHalf, lutEqual, lutEqual))  # (1,256,3), B_half/BGR
lut3HalfG = np.dstack((lutEqual, lutHalf, lutEqual))  # (1,256,3), G_half/BGR
lut3HalfR = np.dstack((lutEqual, lutEqual, lutHalf))  # (1,256,3), R_half/BGR

blendHalfB = cv.LUT(img, lut3HalfB)  # B 通道衰减 50%
blendHalfG = cv.LUT(img, lut3HalfG)  # G 通道衰减 50%
blendHalfR = cv.LUT(img, lut3HalfR)  # R 通道衰减 50%

print(img.shape, lutHalf.shape, lut3HalfB.shape, blendHalfB.shape)



blendHalfB = cv.cvtColor(blendHalfB, cv.COLOR_BGR2RGB)
blendHalfG = cv.cvtColor(blendHalfG, cv.COLOR_BGR2RGB)
blendHalfR = cv.cvtColor(blendHalfR, cv.COLOR_BGR2RGB)
# save
cv.imwrite("blendHalfB.jpg", blendHalfB)
cv.imwrite("blendHalfG.jpg", blendHalfG)
cv.imwrite("blendHalfR.jpg", blendHalfR)

