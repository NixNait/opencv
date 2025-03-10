from PIL import Image
import pytesseract
import cv2
import os

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

preprocess = 'blur'
'''
thresh（二值化处理）：适用于黑白文本图像，例如扫描文档。
blur（模糊处理）：适用于噪声较多的图像，例如模糊的手写文本。
'''

image = cv2.imread('scan_result.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if preprocess == 'blur':
    gray = cv2.bilateralFilter(gray, 5, 5, 5)

if preprocess == 'thresh':
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

filename = 'ocr_result.jpg'

cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
with open('ocr_result.txt', 'w',encoding="utf-8") as f:
    for i in text:
        f.write(i)
print(text)
os.remove('ocr_result.jpg')

cv2.imshow('image', resize(image,height=500))
rotated = cv2.rotate(resize(gray,height=500), cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow("output",rotated)
cv2.waitKey(0)
