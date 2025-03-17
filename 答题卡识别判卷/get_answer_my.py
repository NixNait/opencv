import numpy as np
import cv2

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)  # 修正为 axis=1
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def four_point_transform(image, pts):
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def get_biggest_contour(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    docCnts = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnts = approx.reshape(4, 2)  # 重塑为 (4, 2)
                break
    return docCnts

def get_question_cnts(cnts):  # 修改参数为 cnts
    questionCnts = []
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(cnt)
    return questionCnts

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBox) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBox

def get_answer_cnts(questionCnts, thresh,image):
    #
    correct = 0
    '''
    np.arange(0, len(questionCnts), 5):
    np.arange() 是 NumPy 中的一个函数，用于生成一个等差数列。
    参数说明：
    0：起始值（包含）。
    len(questionCnts)：结束值（不包含）。len(questionCnts) 表示 questionCnts（一个列表、数组或其他可迭代对象）的长度。
    5：步长（step），表示每次递增 5。
    作用：生成一个从 0 开始、每次增加 5、直到小于 len(questionCnts) 的数列。
    举例：如果 questionCnts 的长度是 12，那么 np.arange(0, 12, 5) 会生成 [0, 5, 10]。
    这里作用就是生成的意思就是第几个选项
    enumerate():
    enumerate() 是一个 Python 内置函数，用于遍历一个可迭代对象（这里是 np.arange() 生成的数列），并返回每个元素的索引和值。
    格式：enumerate(iterable) 返回一个迭代器，产生 (索引, 值) 的元组。
    在这里，enumerate(np.arange(0, len(questionCnts), 5)) 会为 np.arange() 生成的每个值分配一个索引。
    举例：对于 [0, 5, 10]，enumerate() 会生成 [(0, 0), (1, 5), (2, 10)]。
    (0, 0)表示0号题：选项起始点为0
    '''
    for q, i in enumerate(np.arange(0, len(questionCnts), 5)):  # 修改为 len(questionCnts)
        #每个题目有A-E五个选项，对这五个选项按照x轴坐标进行排序
        cnts = sort_contours(questionCnts[i:i + 5], method="left-to-right")[0]
        #bubbled 用于记录当前位置最佳选项 , （亮度值，索引）
        bubbled = None
        for j, c in enumerate(cnts):
            mask = np.zeros_like(thresh)
            '''
            cv2.drawContours(mask, [c], -1, 255, -1)
            cv2.drawContours() 在掩膜上绘制轮廓。
            参数说明：
            mask：目标图像（掩膜），绘制结果会修改这个数组。
            [c]：要绘制的轮廓列表，这里只绘制当前轮廓 c，用列表包裹是因为函数需要一个轮廓列表。
            -1：绘制所有轮廓（这里只有一个轮廓，所以效果相同）。
            255：绘制颜色，这里是白色（二值图像中 255 表示白色）。
            -1：厚度，-1 表示填充整个轮廓内部（而不是只画轮廓线）。
            作用：在掩膜上将当前轮廓 c 的内部填充为 255，其他区域保持 0。
            '''
            cv2.drawContours(mask, [c], -1, 255, -1)
            '''
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            cv2.bitwise_and() 执行按位与操作。
            参数说明：
            thresh, thresh：两个输入图像，这里是同一个二值图像 thresh。
            mask=mask：掩膜，只有掩膜中值为非 0（即 255）的区域才会保留 thresh 的值，其他区域变为 0。
            作用：将 thresh 中与掩膜对应区域的值保留下来，其他区域清零。结果是只保留当前轮廓区域的像素。
            '''
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            '''
            total = cv2.countNonZero(mask)
            cv2.countNonZero() 计算图像中非零像素的数量。
            这里计算的是掩膜中当前轮廓区域内值为 255 的像素总数。
            意义：total 表示当前轮廓区域的“填充程度”（例如填涂区域的黑色像素数量）。
            '''
            total = cv2.countNonZero(mask)
            '''
            bubbled 是一个变量，用于记录目前为止检测到的“最佳”轮廓（可能是填涂最多的区域）。
            初始化时 bubbled 可能为 None，或者是一个元组 (最大像素数, 轮廓索引)。
            条件：
            如果 bubbled 是 None（第一次循环）。
            或者当前轮廓的非零像素数 total 大于之前记录的最大值 bubbled[0]。
            作用：判断当前轮廓是否比之前记录的更“显著”（例如填涂更多）。
            '''
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        color = (0, 0, 255)
        k = ANSWER_KEY[q]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
        # 绘图
        cv2.drawContours(image, [cnts[k]], -1, color, 3)
    return correct

def main():
    #读取图像
    image = cv2.imread("./images/test_02.png")
    #转化为灰度图
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #操作图
    op_image = image.copy()
    #高斯模糊
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    #边缘检测
    edged = cv2.Canny(blurred, 75, 200)
    #得到最大轮廓
    docCnts = get_biggest_contour(edged)
    #对最大轮廓即这个页面进行透视操作
    warped = four_point_transform(image, docCnts)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv_show("warped", warped)
    #对透视变换后、且灰度的图像进行二值化
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv_show("thresh", thresh)
    thresh_counter = thresh.copy()
    #对二值化后的结果进行轮廓检测
    cnts = cv2.findContours(thresh_counter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #绘制轮廓检测结果
    cv2.drawContours(thresh_counter, cnts, -1, 255, 3)
    cv_show("thresh_counter", thresh_counter)
    #因为答题卡中选项均为圆圈，所以通过get_question_cnts的到所有选项的轮廓
    question_cnts = get_question_cnts(cnts)  # 使用 cnts 而不是 docCnts
    #对所有选项按照纵坐标进行排序，这样就可以区分每个题对应的选项了。
    question_cnts = sort_contours(question_cnts, method="top-to-bottom")[0]
    #通过get_answer_cnts判断选择的是否正确
    correct = get_answer_cnts(question_cnts, thresh,warped)
    score = (correct / 5.0) * 100.0
    print("[INFO] score: {:.2f}%".format(score))
    cv2.putText(warped, "{:.2f}%".format(score), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Original", image)
    cv2.imshow("Exam", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()