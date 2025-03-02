import cv2
from pip._internal.resolution.resolvelib import candidates


#对轮廓从左到右进行排序
def sort_contours(cnts, method="left-to-right"):
    '''
    "left-to-right"：按 x 坐标从小到大 排序（默认）。
    "right-to-left"：按 x 坐标从大到小 排序。
    "top-to-bottom"：按 y 坐标从小到大 排序。
    "bottom-to-top"：按 y 坐标从大到小 排序。
    '''
    reverse = False #是否反转排序，False为默认值，不反转
    i =0 #表示默认按x坐标排序
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1#表示按照y坐标排序
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]#cv2.boundingRect(c) 计算 包围该轮廓的最小矩形，返回 (x, y, w, h)
    #boundingBoxes 是一个列表，包含了所有外接矩形的坐标信息
    (cnts, boundingBoxes) = zip(*
                                sorted(
                                     zip(cnts, boundingBoxes),
                                     key= lambda b: b[1][i],
                                     reverse=reverse)
                                )
    '''
    zip(cnts, boundingBoxes) 将轮廓和对应的边界框打包成组
    sorted(..., key=lambda b: b[1][i], reverse=reverse)：
    b 是 ("轮廓", (x, y, w, h)) 这样的元组。
    b[1] 代表 (x, y, w, h)。
    b[1][i]：
    i=0 → 取 x 值，按 左右顺序 排序。
    i=1 → 取 y 值，按 上下顺序 排序。
    所以 lambda b: b[1][i]：
    等价于：
    # 定义排序函数
    def get_sort_key(b):
    return b[1][i]  # 取 (x, y, w, h) 里的 x 或 y
    
    reverse=True 代表降序排序（适用于 "right-to-left" 和 "bottom-to-top"）。
    zip(*)：解压 zip 结果，返回 排序后的轮廓列表 和 对应的外接矩形列表。
    '''
    return cnts,boundingBoxes
    #cnts：排序后的轮廓列表。
    #boundingBoxes：对应的外接矩形列表。

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    '''
    这段代码的作用是 调整图像大小（resize），
    可以按指定的 width（宽度）或 height（高度）缩放图像，
    同时保持纵横比（aspect ratio），避免图像变形。
    :param image:输入的图像（通常是 cv2.imread() 读取的）。
    :param width:目标宽度（如果设置了，则按照它调整）。
    :param height:目标高度（如果设置了，则按照它调整）
    :param inter:插值方法（默认是 cv2.INTER_AREA，适用于缩小图像时减少像素失真）。
    :return:
    '''
    dim = None # 用于存储调整后的图像尺寸
    (h, w) = image.shape[:2]
    if width is None and height is None: # # 如果宽度和高度都没指定，返回原始图像
        return image
    if width is None: # 仅指定 height（高度）
        r = height / float(h) #计算缩放比例。
        dim = (int(w * r), height) #计算新的 (宽度, 高度)，保持纵横比。
        '''
        原图尺寸 w=800, h=600，如果 height=300，则：
        r = 300 / 600 = 0.5
        dim = (int(800 * 0.5), 300) = (400, 300)
        这样缩放后，图像不会变形。
        '''
    else:#仅指定 width（宽度）,或者都指定了
        r = width / float(w)#计算缩放比例。
        dim = (width, int(h * r)) #计算新的 (宽度, 高度)，保持纵横比。
    resized = cv2.resize(image, dim, interpolation=inter)
    '''
    使用 cv2.resize() 进行图像缩放：
    image：输入图像。
    dim：目标大小 (new_width, new_height)。
    inter：插值方式（默认 cv2.INTER_AREA，适用于缩小图像）
    inter的类型：
    cv2.INTER_NEAREST	快速缩放	最近邻插值，计算速度快，但可能导致锯齿
    cv2.INTER_LINEAR	默认放大	双线性插值，适用于放大
    cv2.INTER_AREA	默认缩小	适用于缩小图像，减少失真
    cv2.INTER_CUBIC	高质量放大	三次样条插值，放大效果更好但计算较慢
    cv2.INTER_LANCZOS4	最高质量	Lanczos 插值，效果最佳但计算最慢
    '''
    return resized

def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

