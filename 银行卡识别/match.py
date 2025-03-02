from imutils import contours
'''
imutils 是一个 简化 OpenCV 代码 的 Python 库，提供了一些常见的 OpenCV 操作的封装。
contours 主要用于排序轮廓（contour sorting），比如按照从左到右、从上到下排序轮廓。
'''
import cv2
import numpy as np
import argparse
from myutils import sort_contours,resize,cv_show

#============================设置参数
'''
用于接收用户输入的图像文件路径和模板文件路径。
使用方法：
python example.py --image "image.jpg" --template "template.jpg"

对应args的输出
    "image": "image.jpg",
    "template": "template.jpg"
'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
'''
-i 和 --image：这两个是命令行中用于指定图像路径的参数。用户可以使用 -i 或 --image 来传递图像路径。
required=True：这个参数是 必填项，意味着用户必须在命令行中提供此参数，否则程序会报错。
help="path to input image"：这个描述用于 命令行帮助文档，当用户在命令行中运行 
python script.py --help 时，会显示这个提示信息，告诉用户该参数的作用。
'''
ap.add_argument("-t", "--template", required=True,
                help="max buffer size")
args = vars(ap.parse_args())
'''
ap.parse_args()：解析命令行中的参数，并返回一个包含参数值的命名空间对象（通常是 Namespace 类型）。
vars()：将命名空间对象转换为 字典（dict） 格式。这样，用户输入的参数值就可以通过字典的形式访问
'''
args["template"] = 'ocr_a_reference.png'
args["image"] = './images/credit_card_01.png'


#===========================指定信用卡类型
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

#读取模板图像
template = cv2.imread(args["template"])
cv_show("template", template)
#获取模板的灰度图
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cv_show("template_gray", template_gray)
#二值图,方便后续轮廓检测
template_two = cv2.threshold(template_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show("template_two", template_two)
#计算轮廓
template_cnts , hierarchy = cv2.findContours(template_two.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
'''
cv2.RETR_EXTERNAL：这个标志表示只检测最外层的轮廓。
cv2.CHAIN_APPROX_SIMPLE：这个标志表示使用简单的轮廓近似方法，减少轮廓点的数量。
'''
cv2.drawContours(template,template_cnts,-1,(0,255,0),3)
cv_show("template", template)
print(len(template_cnts))#template_cnts，它是一个轮廓列表，包含多个轮廓，每个轮廓是一个 NumPy 数组
#对轮廓进行排序，因为原来的轮廓顺序不一定，这样排序后，是从左到右排序
template_cnts = sort_contours(template_cnts, method="left-to-right")[0]
#对于排序节后的轮廓，因为后续还需要进行模板匹配，我需要知道第一个轮廓对应0，第二个轮廓对应1，，，，
#创建一个字典，0：对应轮廓，1：对应轮廓，2：对应轮廓
digits = {}
for (i,c) in enumerate(template_cnts):#enumerate()将可迭代对象（如列表、元组等）中的每个元素与其对应的索引一起返回。
	#获取轮廓的坐标
	x,y,w,h = cv2.boundingRect(c)
	#计算轮廓外接矩形
	roi = template_two[y:y+h, x:x+w]
	#resize到合适的大小，后面银行卡中的要与这个一致
	roi = resize(roi,57,88)
	#添加到字典
	digits[i] = roi
print(digits)


#======================对银行卡图片进行处理
#初始化卷积核，自定义卷积核大小会更灵活，且能指定形状，具体可以看下getStructuringElement的详解
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#读取输入图像，进行预处理
image = cv2.imread(args["image"])
cv_show("image", image)
image = resize(image,width=300)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show("image_gray", image_gray)

#礼帽操作，凸出更为明亮的区域
#礼帽：原始图像-开运算，原始图像带刺，减去开运算（先腐蚀再膨胀），结果就剩下刺
image_tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show("image_tophat", image_tophat)

#寻找边缘，sobel算子
gradX = cv2.Sobel(image_tophat, cv2.CV_32F, 1, 0, ksize=-1) #ksize=-1 相当于用3*3的卷积核
#右边减去左边，下边减去上边，所以要取绝对值
gradX = np.absolute(gradX)
#找到最大最小值，方便后面归一化
(maxVal, minVal) = np.max(gradX), np.min(gradX)
#归一化，将 gradX 的像素值缩放到 0 到 255 之间
gradX = 255*((gradX - minVal) / (maxVal - minVal))
#gradX - minVal：将 gradX 图像中的每个像素值减去最小值 minVal，使得图像中的最小值变为 0。
#(gradX - minVal) / (maxVal - minVal)：将每个像素值按比例缩放到 0 到 1 之间。
# 这样，gradX 中的最小值变为 0，最大值变为 1。

#将归一化后的 gradX 转换为 8 位无符号整数（uint8） 类型。
gradX = gradX.astype("uint8")
print(np.array(gradX).shape)
cv_show("gradX", gradX)

#经过上面的操作已经得到了比较清晰的轮廓，但我们还需要分组，所以要想办法把一组的轮廓连接在一起

#使用闭运算，先膨胀在腐蚀，可以将数字的边界连接在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show("gradX", gradX)
#边界已经找出来了，对图像进行二值化，方便后面找轮廓
#THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
#cv2.THRESH_BINARY | cv2.THRESH_OTSU:同时启用 二值化操作 和 Otsu 算法。
#[1] 是用来提取 cv2.threshold 函数返回的 第二个返回值
gradX_two = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show("gradX_two", gradX_two)
#二值化之后，发现数字部分有空缺，在执行2次膨胀
gradX = cv2.dilate(gradX_two, sqKernel, iterations=2)
#在执行两次腐蚀
gradX = cv2.erode(gradX, sqKernel, iterations=2)
cv_show("gradX", gradX)

#计算轮廓
image_cnts,hierarchy2 = cv2.findContours(gradX.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = image_cnts
cur_image = image.copy()
cv2.drawContours(cur_image,cnts,-1,(0,0,255),3)
cv_show("cur_image",cur_image)
#轮廓有很多，但我需要选择数字组的轮廓，locs用于存储数字组的轮廓信息
locs=[]

#遍历轮廓，找到目标轮廓
for (i,c) in enumerate(image_cnts):
	x,y,w,h = cv2.boundingRect(c)
	ar = w / float(h)   #计算宽高比\

	#选择合适的区域，这里使用长宽比，以及宽度高度做了一个限制，仅适用于本次项目
	if ar >2.5 and ar<4.0:
		if (w > 40 and w < 55) and (h > 10 and h < 20):
			# 符合的留下来
			locs.append((x, y, w, h))
#将留下来的轮廓排序
#对 locs 列表进行排序，根据每个元素的第一个值（x[0]，也就是左上角的坐标）进行排序
locs = sorted(locs,key=lambda x:x[0])

output = []

for (i,(gx,gy,gw,gh)) in enumerate(locs):
	groupOutput =[]#列表将用来存储一个个数字组中提取的数字信息

	#根据坐标提取每一个组
	group = image_gray[gy -5:gy+gh+5,gx-5:gx+gw+5]
	cv_show('group',group)
	#取出来一个数字组之后，还要对每个数字组进行分割，取出每一个数字
	#对数字组进行二值化操作
	group = cv2.threshold(group, 0, 255,
						  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv_show('group', group)
	# 计算数字组中每个数字的轮廓
	digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
													cv2.CHAIN_APPROX_SIMPLE)
	#对数字轮廓进行排序
	digitCnts = sort_contours(digitCnts)[0]
	# digitCnts = contours.sort_contours(digitCnts,
	# 								   method="left-to-right")[0]

	#然后对数字的轮廓进行操作，取到每一个数字的值
	for c in digitCnts:
		#按顺序访问数字组中数字的轮廓
		#提取并计算当前轮廓的外接矩形
		x, y, w, h = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		# resize成合适的大小
		roi = resize(roi,57,88)
		cv_show('roi',roi)
		# 下面开始进行匹配，并计算匹配得分
		scores = []

		#也就是，对每一个数字轮廓的roi，与模板中的数字ROI进行匹配，并记录与每个ROI的最大匹配度
		for(digit,digitROI) in digits.items():# digits.items()是之前template的元组，包括了序号和对应的区域
			#进行模板匹配
			result = cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF_NORMED)
			#返回的 result 是一个 匹配结果图像，其大小与输入图像 roi 相同，
			#且每个像素值表示模板在该位置与目标图像区域的匹配度。
			(_,score,_,_) = cv2.minMaxLoc(result)
			#cv2.minMaxLoc() 用于获取 匹配结果图像 result 中的最小值和最大值的位置信息。
			#cv2.minMaxLoc() 返回四个值：
			# minVal：结果图像中的最小值。
			# maxVal：结果图像中的最大值。
			# minLoc：最小值的位置。
			# maxLoc：最大值的位置
			#这里用的cv2.TM_CCOEFF_NORMED方法，值越接近1越匹配
			scores.append(score)

		#将刚刚得到的最合适的数字
		# 在记录与每个ROI的最大匹配度中再找出最大的
		#np.argmax(scores)会返回最大匹配度的索引
		#例如：scores = [0.2, 0.5, 0.9, 0.7]
		#那么返回值就是2
		groupOutput.append(str(np.argmax(scores)))

	#在image上画出数字组的轮廓
	cv2.rectangle(image,(gx - 5, gy - 5),
		(gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)

	#cv2.putText() 是 OpenCV 提供的一个函数，用于在图像上绘制文本。它接受多个参数
	# (gx, gy - 15) 将文本的 y 坐标 稍微向上偏移 15 个像素，确保文本显示在矩形框的上方。
	#cv2.FONT_HERSHEY_SIMPLEX 是 OpenCV 提供的一种简单的无衬线字体，适用于大多数常规的文本显示需求。
	#0.65 这是文本的 字体大小。值越大，字体越大；值越小，字体越小。0.65 表示一个适中的字体大小。
	cv2.putText(image, "".join(groupOutput), (gx, gy - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
	#得到结果
	output.extend(groupOutput)

#打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv_show('image',image)

