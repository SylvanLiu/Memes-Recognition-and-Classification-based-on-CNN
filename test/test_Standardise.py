# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
# 读取数据文件
image = tf.read_file(
    '/Users/liusiyuan/Desktop/Codes/Img4test/Private/1.jpg', 'r')
# 将图像文件解码为Tensor
image_tensor = tf.image.decode_jpeg(image)
# 图像张量的形状
shape = tf.shape(image_tensor)
session = tf.Session()
print("图像的形状为：")
print(session.run(shape))
# 将tensor转换为ndarray
image_ndarray = image_tensor.eval(session=session)
# 显示图片
print(str(image_ndarray))
print('\n\n\n\n\n')
plt.imshow(image_ndarray)
plt.show()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


image_ndarray = normalization(image_ndarray)
plt.imshow(image_ndarray)
print(str(image_ndarray))
print('\n\n\n\n\n')
plt.show()

""" def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


image_ndarray = standardization(image_ndarray)
plt.imshow(image_ndarray)
plt.show()
print(str(image_ndarray))
print('\n\n\n\n\n') """
