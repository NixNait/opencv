项目流程：

图像预处理，找到ROI区域

将不规则区域转化为规则区域

使用tesseract进行识别



### **`cv2.getPerspectiveTransform()` 的原理**

`cv2.getPerspectiveTransform(rect, dst)` 的作用是 **计算透视变换矩阵**，用于将一个四边形区域变换为另一个四边形区域（通常是矩形）。其核心数学原理是**透视变换（Perspective Transformation）**，也称 **单应性变换（Homography）**。

------

## **1. 透视变换的数学原理**

透视变换是一种 **投影变换（Projective Transformation）**，在计算机视觉和图像处理领域，它用于 **修正因视角变化引起的形变**。
 其数学表达式如下：

![image-20250310110911466](C:\Users\10905\AppData\Roaming\Typora\typora-user-images\image-20250310110911466.png)

![image-20250310110923018](C:\Users\10905\AppData\Roaming\Typora\typora-user-images\image-20250310110923018.png)

------

## **2. `cv2.getPerspectiveTransform()` 如何计算矩阵**

### **已知条件**

- 输入 **4 个点**（`rect`）：原始图像上的四边形角点（任意形状）。
- 目标 **4 个点**（`dst`）：变换后图像的角点（通常是矩形）。

### **方程构建**

![image-20250310110956426](C:\Users\10905\AppData\Roaming\Typora\typora-user-images\image-20250310110956426.png)

------

## **3. 计算透视变换矩阵 HH**

![image-20250310111016395](C:\Users\10905\AppData\Roaming\Typora\typora-user-images\image-20250310111016395.png)



------

## **5. `cv2.warpPerspective()` 透视变换**

`cv2.warpPerspective(image, M, (maxWidth, maxHeight))` 使用变换矩阵 H 将整个图像进行变换，从而将倾斜的四边形变为矩形。例如：

| **原始图像**   | **透视变换后** |
| -------------- | -------------- |
| 斜着拍摄的纸张 | 规则的矩形文档 |

------

## **6. 应用场景**

✅ **文档矫正（OCR 预处理）**
 ✅ **车牌识别（矫正拍摄角度）**
 ✅ **棋盘检测（棋盘转换为标准矩形）**
 ✅ **透视校正（修正因视角变化导致的形变）**

------

## **总结**

`cv2.getPerspectiveTransform()` 的原理：

1. 通过 **4 对点** 计算 **3×3 透视变换矩阵** HH。
2. 通过 **最小二乘法** 计算 8 个参数，使输入四边形变为目标矩形。
3. `cv2.warpPerspective()` 使用这个矩阵 HH **转换图像**，修正透视失真。

透视变换是计算机视觉中的核心技术，常用于 **文档扫描、目标检测、形变矫正** 等任务 🚀。