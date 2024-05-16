from pylab import *
from PIL import Image
import numpy as np
from scipy import ndimage

"""载入图像，并使用该函数计算偏移图"""
im_l = np.array(Image.open('image1.png').convert('L'), 'f')
im_r = np.array(Image.open('image2.png').convert('L'), 'f')
gray()

def NCC(im_l, im_r, start, steps, wid):
    """ 使用归一化的互相关计算视差图像 该函数返回每个像素的最佳视差"""
    m, n = im_l.shape
    # 保存不同求和值的数组
    mean_l = np.zeros((m, n))
    mean_r = np.zeros((m, n))
    s = np.zeros((m, n))
    s_l = np.zeros((m, n))
    s_r = np.zeros((m, n))
    dmaps = np.zeros((m, n, steps)) # 保存深度平面的数组
    ndimage.filters.uniform_filter(im_l, wid, mean_l) # 计算图像块的平均值
    ndimage.filters.uniform_filter(im_r, wid, mean_r) # 计算图像块的平均值
    norm_l = im_l - mean_l # 归一化图像
    norm_r = im_r - mean_r # 归一化图像
    # 尝试不同的视差
    for displ in range(steps):
        # 将左边图像移动到右边，计算加和
        ndimage.filters.uniform_filter(np.roll(norm_l, -displ - start) *
                                       norm_r, wid, s)  # 和归一化
        ndimage.filters.uniform_filter(np.roll(norm_l, -displ - start) *
                                       np.roll(norm_l, -displ - start), wid, s_l)
        ndimage.filters.uniform_filter(norm_r * norm_r, wid, s_r)  # 和反归一化
        # 保存 ncc 的分数
        dmaps[:, :, displ] = s / np.sqrt(s_l * s_r)
    # 为每个像素选取最佳深度
    return np.argmax(dmaps, axis=2)

# 开始偏移，并设置步长
steps = 12
start = 4
wid = np.array([1,3,5,7,9,11])
res=[]
for i in range(0, 6):
    res.append(NCC(im_l, im_r, start, steps, wid[i]))
a = np.array([231,232,233,234,235,236])
name = ['ncc wide=1', 'ncc wide=3', 'ncc wide=5',
           'ncc wide=7','ncc wide=9','ncc wide=11']
for i in range(0, 6):
    subplot(a[i])
    imshow(res[i])
    title(name[i])
    axis('off')
show()