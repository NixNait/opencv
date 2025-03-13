import cv2
import numpy as np

class Stitcher:
    def stitch(self, images,ratio = 0.75, reprojThresh = 4.0, showMatches=False):
        '''
        :param images:
        :param ratio:
        :param reprojThresh:
        :param showMatches:
        :return:
        '''
        #获取输入图像
        imageA , imageB = images[0], images[1]
        #进行特征检测与描述
        kpsA,featuresA =self.dectAndDescribe(imageA)
        kpsB,featuresB =self.dectAndDescribe(imageB)
        #合并关键点和特征描述
        kpses = (kpsB,kpsA)
        features = (featuresB,featuresA)
        #匹配两张图的所有特征点，返回匹配结果
        #返回值 M 包含匹配对 (matches)、单应性矩阵 (H) 和内点状态 (status)。
        M = self.matchKeypoints(kpses,features,ratio,reprojThresh)

        #如果返回值为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        #否则提取匹配结果
        #H为3*3的视角变换矩阵
        matches,H,status = M
        #将图片A视角变换
        '''
        使用 cv2.warpPerspective 将图像 A 根据单应性矩阵 H 进行透视变换。
        输出图像的宽度是 imageA 和 imageB 的宽度之和，高度保持 imageA 的高度。
        self.cv_show（假设是显示图像的辅助方法）展示变换后的 resultA。
        '''
        # 计算输出图像的尺寸
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        output_width = wA + wB
        output_height = max(hA, hB)  # 使用最大高度以适应两幅图像

        # 变换 imageA 到 imageB 的坐标系
        resultA = cv2.warpPerspective(imageB, H, (output_width, output_height))
        self.cv_show("Warped imageB", resultA)
        # 只填充 result 中仍为黑色（未被变换覆盖）的区域
        resultA[0:imageB.shape[0], 0:imageB.shape[1]] = imageA
        self.cv_show('resultA+B', resultA)

        #可选显示匹配结果
        if showMatches:
            #生成匹配图像
            vis = self.drawMatches(images,kpses,matches,status)
            return (resultA,vis)

        return resultA


    def cv_show(self,name,image):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def dectAndDescribe(self,image):
        #将图像转换成黑白色
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #初始化SIFT生成器
        descriptor = cv2.SIFT_create()

        #检测关键点，特征向量
        kp, features = descriptor.detectAndCompute(image_gray, None)

        #将结果转换为numpy数组
        '''
        kp.pt：每个 cv2.KeyPoint 对象有一个 pt 属性，表示关键点的 (x, y) 坐标（浮点数形式）。
        
        '''
        kps = np.float32([kp.pt for kp in kp])


        #返回特征点集，以及对应特征描述
        return kps, features

    def matchKeypoints(self,kpses,features,ratio,reprojThresh):
        kpsesA , kpsesB = kpses
        featuresA , featuresB = features
        #初始化暴力匹配器
        matcher = cv2.BFMatcher()

        #使用KNN检测来自A\B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA,featuresB,k=2)

        matches = []

        #比率测试筛选匹配对
        '''
        m 是最近邻匹配，n 是次近邻匹配。
        比率测试（Lowe's Ratio Test）：如果最近邻距离与次近邻距离的比值小于ratio，则认为该匹配可靠。
        m[0].distance 是匹配的距离，m[0].queryIdx 是特征点在图像A中的索引。
        将筛选后的匹配对存储到 matches 列表中。
        '''
        for match in rawMatches:
            #当最近距离跟次近举例的比值小于ratio的时候，保留这队匹配
            if len(match) == 2 and match[0].distance <match[1].distance * ratio:
                #储存两个点再featuresA,featuresB中的索引值
                matches.append((match[0].trainIdx,match[0].queryIdx))

        #当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) >4:
            #获取匹配对的点坐标
            ptsA = np.float32([kpsesA[i] for (_,i)in matches])
            ptsB = np.float32([kpsesB[i] for (i,_)in matches])

            #计算视角变换矩阵
            '''
            计算从图像 A 到图像 B 的透视变换矩阵 H。
            返回的 status 告诉你哪些匹配点是可靠的（内点），哪些是不可靠的（外点）。
            '''
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            return matches, H, status

    def drawMatches(self, images, kpses, matches, status):
        kpsA =kpses[0]
        kpsB = kpses[1]
        hA , wA = images[0].shape[:2]
        hB , wB = images[1].shape[:2]

        #创建可视化的画布
        vis = np.zeros((max(hA,hB),wA+wB,3),dtype = 'uint8')
        #讲画布对应位置填充
        vis[:hA,:wA,:] = images[0]
        vis[:hB,wA:] = images[1]

        #联合遍历匹配对和状态
        for ((trainIdx,queryIdx),valIdx) in zip(matches,status):
            #当点对匹配成功时，画到可视化图上
            if valIdx == 1 :#是内点（匹配可靠）
                #画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[queryIdx][0]) + wA, int(kpsB[queryIdx][1]))
                cv2.line(vis,ptA,ptB,(0,0,255),2)
        return vis