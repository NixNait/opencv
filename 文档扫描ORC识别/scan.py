# 导入工具包
import numpy as np
import argparse
import cv2

#设置执行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

#resize函数
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
#图像展示
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#对四个坐标点进行排序，确保顺序是左上，右上，右下，左下
def order_points(pts):
	#初始化一个空的数组，四个坐标点，每个坐标点为x,y
	rect = np.zeros((4, 2), dtype="float32")
	#根据0123四，对应左上，右上，右下，左下
	#计算左上，右下
	# 计算每个点的 x + y（左上角点的和最小，右下角点的和最大）
	s = pts.sum(axis=1)
	'''
	NumPy 数组是多维的，因此 axis 控制操作的维度：
	axis=0：沿着 列方向（垂直方向） 计算，即对每一列求和。
	axis=1：沿着 行方向（水平方向） 计算，即对每一行求和。
	'''

	rect[0] = pts[np.argmin(s)] # 左上角 (x+y 最小)
	rect[2] = pts[np.argmax(s)] # 右下角 (x+y 最大)

	#计算右上，左下
	# 计算每个点的 x - y（右上角点的差值最小，左下角点的差值最大）
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]# 右上角 (x-y 最小)
	rect[3] = pts[np.argmax(diff)] # 左下角 (x-y 最大)

	return rect


#透视实现
def four_point_transform(image, pts):
	#获取输入坐标点,排序后的结果
	rect = order_points(pts)
	tl, tr, br, bl = rect
	#由坐标点计算输入的w,h
	#计算底边的长度
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[0] - bl[1]) ** 2))
	#计算定边的长度
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	#计算右边h
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	#计算左边h
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	#得到变换后对应坐标位置
	dst = np.array([
		[0, 0],#设左上角是0，0
		[maxWidth - 1, 0],#右上角，减1是为了稳定
		[maxWidth - 1, maxHeight - 1],#右下角
		[0, maxHeight - 1]], dtype = "float32")#左下角

	#计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	'''
	cv2.getPerspectiveTransform()：计算 透视变换矩阵，
	用于将 rect（原始 4 点坐标）转换到 dst（目标 4 点坐标）。
	rect：源图像中的 4 个点（order_points(rect) 处理后的）。
	dst：目标变换后的 4 个点（通常是一个标准矩形）。
	M：返回 3×3 透视变换矩阵，用于 cv2.warpPerspective() 进行图像变换。

	'''
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped















#图像读取
image = cv2.imread(args["image"])
#由于后续对图像进行操作的时候可能会放大放小，这里记录下倍数
ratio = image.shape[0] / 500.0
op_image = image.copy() #这是执行操作的图片，避免更改原图像

#将图像resize,选择hight是因为上面ration纪律的resize
image_2 = resize(op_image, height=500)

#得到灰度图像，以进行图像预处理
image_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

#对图片进行高斯滤波，减少噪音
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
#使用canny边缘检测
edges = cv2.Canny(image_gray, 50, 200)

#展示边缘检测结果
print("step1 : 边缘检测")
cv_show("step1_边缘检测",edges)

#边缘检测完进行轮廓检测
cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#这里对所有轮廓进行排序，按照面积，从大到小排序，取前五，作为检测目标
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

screenCnt = None#用于存放最可能的轮廓
#遍历轮廓
for c in cnts:
	#首先进行轮廓的周长计算，方便后面轮廓近似，计算精度
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	#如果这个轮廓包含四个点，也就是矩形之类的，就需要保存
	if len(approx) == 4:
		screenCnt = approx
		break

#展示轮廓近似的结果
print("step 2 ： 轮廓近似结果")
cv2.drawContours(image_2, [screenCnt], -1, (0, 255, 0), 2)
cv_show("轮廓近似结果",image_2)


#透视变换：也就是将本身可能七扭八歪的图像换成平面的
warped = four_point_transform(op_image, screenCnt.reshape(4,2)*ratio)

#对于已经转换好的图片此时要在进行一些清晰处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#阈值处理，进行二值化
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
#保存图片
cv2.imwrite("scan_result.jpg", ref)

#展示结果
print("Step3 : 变换")
cv_show("原始图像",resize(image, height=600))
cv_show("扫描结果",resize(ref,height = 650))