import cv2
import numpy as np
import streamlit as st

# 设置标题
st.title('基于OpenCV的图像卡通化系统')
# 上传图片并展示
uploaded_file = st.file_uploader("上传一张图片", type=("jpg", "png"))
if uploaded_file is not None:
    # 将传入的文件转为Opencv格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # 展示图片
    st.image(opencv_image, channels="BGR")
    # 保存图片
    cv2.imwrite('test.jpg', opencv_image)
# 然后就可以用这个图片进行一些操作了
# 定义img
img = cv2.imread('test.jpg', cv2.IMREAD_REDUCED_COLOR_2)
# 将图片转化为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
###铅笔画
# 使用medianBlur函数对图像进行中值滤波处理
gray1 = cv2.medianBlur(gray, 3)
# 使用拉普拉斯边缘检测
edges1 = cv2.Laplacian(gray1, -1, ksize=5)
# 反色处理
edges_inv = 255 - edges1
# 使用threshold函数进行二值化处理
dummy, cartoon = cv2.threshold(edges_inv, 0, 255, cv2.THRESH_BINARY)
# 创建复选框部件
anjian = st.checkbox('铅笔画')
# 检查用户是否选择了该选项
if anjian:
    st.write("铅笔画：")
    # st.image在网页中显示
    st.image(cartoon)
else:
    st.write(' ')
###铅笔素描
# 使用GaussianBlur进行高斯模糊处理
gray2 = cv2.GaussianBlur(gray, (55, 55), 0)
# 使用divide函数进行除法运算，参数scale=250.0用于控制素描效果的强度。
cartoon1 = cv2.divide(gray, gray2, scale=250.0)
# 创建复选框部件
anjian1 = st.checkbox('素描画')
# 检查用户是否选择了该选项
if anjian1:
    st.write("素描画：")
    # st.image在网页中显示
    st.image(cartoon1)
else:
    st.write(' ')
###卡通化
# 使用adaptiveThreshold函数进行自适应阈值化处理，将图像转换为黑白的卡通效果。
edges2 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
# 使用detailEnhance函数进行细节增强处理，通过调整sigma_s和sigma_r参数来控制细节增强的程度。
color = cv2.detailEnhance(img, sigma_s=5, sigma_r=0.5)
# 代码使用bitwise_and函数进行按位与运算，得到一个新的图像cartoon2。
cartoon2 = cv2.bitwise_and(color, color, mask=edges2)
# 创建复选框部件
anjian2 = st.checkbox('卡通化')
# 检查用户是否选择了该选项
if anjian2:
    st.write("卡通化：")
    # st.image在网页中显示
    st.image(cartoon2, channels="BGR")
else:
    st.write(' ')
###卡通化
# 使用GaussianBlur函数进行高斯模糊处理。
gray3 = cv2.GaussianBlur(gray, (3, 3), 0)
# 使用Laplacian函数进行拉普拉斯边缘检测。
edges3 = cv2.Laplacian(gray3, -1, ksize=5)
# 通过将边缘检测的结果取反,可以将边缘变为白色，背景变为黑色。
edges3 = 255 - edges3
# 使用threshold函数对edges3进行阈值化处理。
ret, edges3 = cv2.threshold(edges3, 150, 255, cv2.THRESH_BINARY)
# 使用edgePreservingFilter函数对原始图像img进行边缘保留滤波处理。
edgePreservingImage = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)
# 使用bitwise_and函数进行按位与运算，得到一个新的图像output。
output = np.zeros(gray3.shape)
output = cv2.bitwise_and(edgePreservingImage, edgePreservingImage, mask=edges3)
# 创建复选框部件
anjian3 = st.checkbox('卡通化2')
# 检查用户是否选择了该选项
if anjian3:
    st.write("卡通化2：")
    # st.image在网页中显示
    st.image(output, channels="BGR")
else:
    st.write(' ')
# 延时函数
cv2.waitKey(0)
