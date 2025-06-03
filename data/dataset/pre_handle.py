import cv2
import os
import numpy as np



def clash(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

def show_preprocessing_pipeline_cv2(img_path):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{img_path}")

    # 缩放为合适大小（可选）
    img = cv2.resize(img, (400, 200))

    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Sobel边缘检测
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))

    # Laplacian边缘检测
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))

    # 显示结果
    # cv2.imshow("src", img)
    # cv2.imshow("h", gray)
    # cv2.imshow("CLAHE", clahe)
    # cv2.imshow("Sobel", sobel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return clahe


def saves():
    path = "/home/anya-xi/machinelearning/code/jl_yolo_delect/data/dataset/clahe/test/src"
    save_path = "/home/anya-xi/machinelearning/code/jl_yolo_delect/data/dataset/clahe/test/images"
    images = os.listdir(path)
    for img in images:
        img_new = show_preprocessing_pipeline_cv2(os.path.join(path, img))
        cv2.imwrite(os.path.join(save_path, img), img_new)


def test(path):
    # Step 1: 加载图像 + 灰度化
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: CLAHE - 局部对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Step 3: 高斯模糊（降噪）
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Step 4: 锐化（增强边缘，拉出缺口）
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    # Step 5: 自适应二值化（让缺口显现为黑色/白色块）
    thresh = cv2.adaptiveThreshold(sharpened, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=15, C=5)

    # 可选 Step 6: 膨胀或腐蚀，强化缺口轮廓
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("CLAHE", clahe_img)
    cv2.imshow("Sharpened", sharpened)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Dilated", dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 示例调用（替换为你自己的图片路径）
    test("/home/anya-xi/machinelearning/code/jl_yolo_delect/data/dataset/src/images/4cd936e3057b43f4913e56f54032556f~tplv-188rlo5p4y-2.jpeg")
    # saves()