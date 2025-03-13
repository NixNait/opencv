from Stitcher import Stitcher
import cv2

#读取需要拼接的图像

imageA = cv2.imread('left_01.png')
imageB = cv2.imread('right_01.png')

#拼接
stitcher = Stitcher()
result , vis = stitcher.stitch((imageA, imageB),showMatches=True)

#显示图片

cv2.imshow('imageA', imageA)
cv2.imshow('imageB', imageB)
cv2.imshow('keypoint', vis)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()