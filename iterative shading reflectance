# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 13:52:26 2022

@author: rahul3.kumar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os

cmap = {0: 'gray', 1: 'Greens', 2: 'Blues'}


def adaptive_threshold(in_image):
    m, n = in_image.shape
    in_image.shape[0]
    s = int(max(m, n)/8)
    t = 5
    output_image = np.zeros(in_image.shape, dtype='float')

    integral_image = np.zeros(in_image.shape)
    '''
    for i in range(m):
        sum_ = 0
        for j in range(n):
            sum_ += in_image[i][j]
            if i == 0:
                integral_image[i][j] = sum_
            else:
                integral_image[i][j] = integral_image[i-1][j]+sum_
    '''
    integral_image = cv2.integral(in_image)

    i_s = np.indices((m, 1))[0]
    j_s = np.indices((n, 1))[0]
    i1_s = i_s-s//2
    i2_s = i_s+s//2
    j1_s = j_s-s//2
    j2_s = j_s+s//2

    i1_s = np.clip(i1_s, 0, m-1)
    i2_s = np.clip(i2_s, 0, m-1)
    j1_s = np.clip(j1_s, 0, n-1)
    j2_s = np.clip(j2_s, 0, n-1)

    counts = (i2_s-i1_s)*(j2_s-j1_s).T
    # print(counts)
    sums = (integral_image[i2_s, j2_s.T]+integral_image[i1_s, j1_s.T] -
            integral_image[i2_s, j1_s.T]-integral_image[i1_s, j2_s.T])

    output_image = np.where(in_image*counts >= sums*(100-t)/100, 1, 0)

    return output_image


def remove_shadow(original_image, n_iters=10):
    # adaptive_threshold(original_image.mean(axis=-1))
    # return original_image

    epsilon = 1e-15
    dx = dy = 5
    m, n, c = original_image.shape
    filterconv = np.ones((dx, dy))
    reflectance = original_image.copy()
    reflectance = reflectance/255
    shading = np.zeros(original_image.shape)
    masks = [np.zeros((m, n)) for _ in range(c)]

    for t in range(n_iters):
        for i in range(3):
            # mask = cv2.adaptiveThreshold(
            #   (reflectance[:, :, i]*255).astype('uint8'), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
            mask = adaptive_threshold(
                (reflectance[:, :, i]*255).astype('uint8'))
            if t == -1:
                mask = cv2.dilate((mask*255).astype('uint8'),
                                  cv2.getStructuringElement(
                                      cv2.MORPH_ELLIPSE, (1, 1))
                                  ).astype('uint8')/255
            masks[i] = mask

            conv_img_mask = scipy.signal.convolve2d(
                reflectance[:, :, i]*mask, filterconv, mode='same')
            conv_mask = scipy.signal.convolve2d(mask, filterconv, mode='same')

            shading[:, :, i] = conv_img_mask/(conv_mask+epsilon)

            shading[:, :, i] = np.where(mask == 1, reflectance[:, :, i],
                                        shading[:, :, i])
            # plt.figure()
            # plt.imshow(shading)
            reflectance[:, :, i] = (
                reflectance[:, :, i] / (shading[:, :, i]+epsilon))
            #reflectance[:, :, i] /= (shading+epsilon)
            reflectance[:, :, i] = np.clip(reflectance[:, :, i], 0, 1)
            # reflectance[:,:,i]=reflectance[:,:,i]/reflectance[:,:,i].max()
        print(reflectance.min(), reflectance.max(), reflectance.mean())

        # plt.figure()
        # plt.imshow(reflectance, cmap='gray')
        # plt.title('reflectance')
    gm = np.zeros((c))
    output_image = reflectance

    for i in range(c):
        _, shadow_mask = cv2.threshold(
            (shading[:, :, i]*255).astype('uint8'),
            0, 255,
            cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        shadow_mask = shadow_mask/255
        gm[i] = ((original_image[:, :, i]*masks[i] *
                 shadow_mask).sum()/(masks[i]*shadow_mask).sum())

        output_image[:, :, i] = reflectance[:, :, i]*gm[i]
        output_image = output_image.astype('uint8')

    return output_image


path = r'D:\personal projects\OCR\Shadow removal\Water-Filling-master\Water-Filling-master\Original'

i = 0
for root, dirs, files in os.walk(path):
    for file in files:
        if ('.png' in file) or ('.jpg' in file) or ('.bmp' in file):
            img = cv2.imread(os.path.join(root, file))

            output_image = remove_shadow(img)
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.title('input_image')
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.title('output_image')
            plt.imshow(output_image)
            fig.savefig(os.path.join(
                r'D:\personal projects\OCR\Shadow removal\water-filling-python-main\outputs', str(i)+'_iterative.jpg'))

            plt.close('all')
            i += 1


#adaptive_threshold(np.ones((10, 10)))
# plt.close('all')
