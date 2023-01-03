import numpy as np
import cv2
import os
import math
# import imagesize
def simple_balance(img, s1, s2):  # 线性增强，s1和s2为低高分段阈值百分比
    h, w = img.shape[:2]
    res = img.copy()
    one_dim_array = res.flatten()  # 转化为一维数组
    sort_array = sorted(one_dim_array)  # 对一维数组按升序排序
    print(len(sort_array))

    per1 = int((h * w) * s1 / 100)
    print(per1/len(sort_array))
    minvalue = sort_array[per1]

    per2 = int((h * w) * s2 / 100)
    print(((h * w) - 1 - per2)/len(sort_array))
    maxvalue = sort_array[(h * w) - 1 - per2]

    # 实施简单白平衡算法
    if (maxvalue <= minvalue):
        for i in range(h):
            for j in range(w):
                res[i, j] = maxvalue
    else:
        scale = 255.0 / (maxvalue - minvalue)
        for m in range(h):
            for n in range(w):
                if img[m, n] < minvalue:
                    res[m, n] = 0
                elif img[m, n] > maxvalue:
                    res[m, n] = 255
                else:
                    res[m, n] = scale * (img[m, n] - minvalue)  # 映射中间段的图像像素

    return res

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

#单尺度retinex实现
def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)  #用最小值替换数组中的0
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0) #对原图进行归一化处理
    dst_Lblur = cv2.log(L_blur/255.0) #对估计出来的L（照度图）进行归一化处理
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur) #原图经过高斯变换后得到的估计出的照度图L
    log_R = cv2.subtract(dst_Img, dst_IxL) #反射图 = 原图-照度图

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX) #将指定的图片的值缩放到0~255
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

#论文中的预增强功能实现
def Pre_enhancement(img):
    h, w = img.shape[:2]
    res = np.float32(img)  #转换为32位图像
    Lwmax = res.max()
    log_Lw = np.log(0.001 + res)
    Lw_sum = log_Lw.sum()
    Lwaver = np.exp(Lw_sum / (h * w))
    Lg = np.log(res / Lwaver + 1) / np.log(Lwmax / Lwaver + 1)

    # res = Lg * 255.0 #不使用分段线性增强
    res = simple_balance(Lg, 2, 3) #使用线性增强，该算法比较耗时
    dst = np.uint8(res) #dst = cv2.convertScaleAbs(res)
    return dst

def load_img(train_imgPath,test_imgPath):
    test_filenames = os.listdir(test_imgPath)
    train_filenames = os.listdir(train_imgPath)

    for tfname in train_filenames:
        if os.path.isdir(tfname) == False:
            filenames = os.listdir(train_imgPath+tfname)
            for filename in filenames:  #  遍历文件
                if filename.endswith('.png'):
                    print(filename)



if __name__ == '__main__':
    """
    img = './data/5.png'
    size = 3
    src_img = cv2.imread(img)
    b_gray, g_gray, r_gray = cv2.split(src_img)

    b_gray_SSR = SSR(b_gray, size)
    g_gray_SSR = SSR(g_gray, size)
    r_gray_SSR = SSR(r_gray, size)
    result_SSR = cv2.merge([b_gray_SSR, g_gray_SSR, r_gray_SSR])

    b_gray_Pre = Pre_enhancement(b_gray)
    g_gray_Pre = Pre_enhancement(g_gray)
    r_gray_Pre = Pre_enhancement(r_gray)
    result_Pre = cv2.merge([b_gray_Pre, g_gray_Pre, r_gray_Pre])

    cv2.imshow('img',src_img)
    cv2.imshow('SSR result',result_SSR)
    cv2.imshow('Pre_enhancement',result_Pre)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
    # path_traindata = ".\\LOLdataset\\our485\\low\\"
    # path_testdata = ".\\LOLdataset\\eval15\\low\\"
    # path_save = ".\\results\\low\\"
    path_traindata = "./LOLdataset/our485/low/"
    path_testdata = "./LOLdataset/eval15/low/"
    path_save = "./results/low/"
    test_filenames = os.listdir(path_testdata)
    train_filenames = os.listdir(path_traindata)
    size = 3


    for tfname in train_filenames:
        if os.path.isdir(tfname) == False:
            print('tfname',tfname)
            filenames = os.listdir(path_traindata)
            for filename in filenames:  #  遍历文件
                if filename.endswith('.png'):
                    img = path_traindata + filename
                    src_img = cv2.imread(img)
                    b_gray, g_gray, r_gray = cv2.split(src_img)
                    b_gray_Pre = Pre_enhancement(b_gray)
                    g_gray_Pre = Pre_enhancement(g_gray)
                    r_gray_Pre = Pre_enhancement(r_gray)
                    result_Pre = cv2.merge([b_gray_Pre, g_gray_Pre, r_gray_Pre])

                    cv2.imwrite(path_save + filename, result_Pre)