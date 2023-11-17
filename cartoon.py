import cv2
import numpy as np
import streamlit as st
from PIL import Image


def cartoonization(img, cartoon):
    # 将图片转化为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cartoon == "铅笔":
        # 使用st.sidebar.slider函数创建滑块部件，取值范围1-21，初始值3，步长为2
        kernel = st.sidebar.slider('调整图像清晰度(数值越低，清晰度越高)', 1, 21, 3, step=2)
        laplacian_filter = st.sidebar.slider('调整边缘检测功率(数值越高，功率越强)', 3, 9, 5, step=2)
        # 使用medianBlur函数对图像进行中值滤波处理
        gray = cv2.medianBlur(gray, kernel)
        # 使用拉普拉斯边缘检测
        edges = cv2.Laplacian(gray, -1, ksize=laplacian_filter)
        # 反色处理
        edges_inv = 255 - edges
        # 使用threshold函数进行二值化处理
        dummy, cartoon1 = cv2.threshold(edges_inv, 150, 255, cv2.THRESH_BINARY)

    elif cartoon == "素描":
        value = st.sidebar.slider('调整图像亮度(数值越高，图像越亮)', 150.0, 300.0, 250.0)
        kernel = st.sidebar.slider('调整图像边缘的粗细(数值越高，边缘越粗)', 1, 120, 55, step=2)
        # 使用GaussianBlur进行高斯模糊处理
        gray_blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        # 使用divide函数进行除法运算，参数scale=250.0用于控制素描效果的强度。
        cartoon = cv2.divide(gray, gray_blur, scale=value)

    elif cartoon == "细节增强":
        smooth = st.sidebar.slider('调整图像的平滑程度(数值越高，图像越平滑)', 3, 9, 5, step=2)
        kernel = st.sidebar.slider('调整图像的清晰度(数值越低，清晰度越高)', 1, 21, 3, step=2)
        edge_preserve = st.sidebar.slider('调整颜色平均效果(低：相似颜色会被平滑；高：不同颜色会被平滑)', 0.0, 1.0, 0.5)
        # 使用medianBlur函数对图像进行中值滤波处理
        gray1 = cv2.medianBlur(gray, kernel)
        # 使用adaptiveThreshold函数进行自适应阈值化处理，将图像转换为黑白的卡通效果。
        edges = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        # 使用detailEnhance函数进行细节增强处理，通过调整sigma_s和sigma_r参数来控制细节增强的程度
        color = cv2.detailEnhance(img, sigma_s=smooth, sigma_r=edge_preserve)
        # 代码使用bitwise_and函数进行按位与运算，得到一个新的图像cartoon。
        cartoon = cv2.bitwise_and(color, color, mask=edges)

    elif cartoon == "卡通":
        # smooth = st.sidebar.slider('调整图像平滑程度(数值越高，图像越平滑)', 3, 99, 5, step=2)
        laplacian_filter = st.sidebar.slider('调整边缘检测功率(数值越高，功率越强)', 3, 9, 5, step=2)
        kernel = st.sidebar.slider('调整图像清晰度(数值越低，清晰度越高)', 1, 9, 3, step=2)
        noise_reduction = st.sidebar.slider('调整图像的噪点效果(数值越高，噪点越高)', 10, 255, 170, step=5)
        # edge_preserve = st.sidebar.slider('调整颜色平均效果(低：相似颜色会被平滑；高：不同颜色会被平滑)', 1, 100, 50)
        # 使用GaussianBlur函数进行高斯模糊处理。
        gray2 = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        # 使用Laplacian函数进行拉普拉斯边缘检测。
        edges = cv2.Laplacian(gray2, -1, ksize=laplacian_filter)
        # 通过将边缘检测的结果取反,可以将边缘变为白色，背景变为黑色。
        edges = 255 - edges
        # 使用threshold函数对edges3进行阈值化处理。
        ret, edges = cv2.threshold(edges, noise_reduction, 255, cv2.THRESH_BINARY)
        # 使用edgePreservingFilter函数对原始图像img进行边缘保留滤波处理。
        edgePreservingImage = cv2.edgePreservingFilter(img, flags=2, sigma_s=40, sigma_r=1)
        # 使用bitwise_and函数进行按位与运算，得到一个新的图像cartoon。
        cartoon = np.zeros(gray2.shape)
        cartoon = cv2.bitwise_and(edgePreservingImage, edgePreservingImage, mask=edges)

    return cartoon


# 设置标题
st.title('基于OpenCV的图像卡通化系统')
# 注释
st.write("这是一个将图像变成卡通化的程序")
# 上传图片并展示
file = st.sidebar.file_uploader("上传一张图像", type=("jpg", "png"))
if file is None:
    st.text("还没有上传图像")
else:
    image = Image.open(file)
    img = np.array(image)

    option = st.sidebar.selectbox('请选择需要的滤镜：', ('铅笔', '素描', '细节增强', '卡通'))

    st.text("原图展示：")
    st.image(image, use_column_width=True)

    st.text("卡通形象展示：")
    cartoon = cartoonization(img, option)
    st.image(cartoon, use_column_width=True)
