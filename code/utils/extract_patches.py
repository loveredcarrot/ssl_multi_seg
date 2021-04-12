# -*- coding: utf-8 -*- 
# @Time : 2021/4/8 15:16 
# @Author : aurorazeng
# @File : extract_patches.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.

import os
import cv2
import numpy as np

tumor = [1, 19, 20]
lymph = [3, 10, 11, 14]
stroma = [2]
necrosis = [4]
others = [5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 21]


def convertMultiLabels(mask):
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
    # 1 - tumor - red  BGR
    for i in tumor:
        new_mask[mask == i] = np.array([0, 0, 255])
    # 2 - stroma - blue
    for i in stroma:
        new_mask[mask == i] = np.array([255, 0, 0])
    # 3 - lymphocytic_infiltrate - green
    for i in lymph:
        new_mask[mask == i] = np.array([0, 255, 0])
    # 4 - necrosis_or_debris - cyan
    for i in necrosis:
        new_mask[mask == i] = np.array([255, 255, 0])
    # 5 - others - yellow
    for i in others:
        new_mask[mask == i] = np.array([0, 255, 255])
    # 0: outside_roi - white
    new_mask[mask == 0] = np.array([255, 255, 255])
    return new_mask


if __name__ == "__main__":

    source = r'E:\CrowdsourcingDataset-Amgadetal2019\dataset\train'
    image_path = os.path.join(source, 'images')
    mask_path = os.path.join(source, 'masks')

    patch_size = 768
    overlap = 0.5
    stride = int(patch_size * (1 - overlap))
    target_mpp = 0.46

    isTrain = True

    patch_path = os.path.join(source, 'patch_' + str(target_mpp))
    patch_image = os.path.join(patch_path, 'images')
    patch_mask = os.path.join(patch_path, 'masks')
    patch_visual = os.path.join(patch_path, 'visual')

    if not os.path.isdir(patch_path):
        os.makedirs(patch_path)
    if not os.path.isdir(patch_image):
        os.makedirs(patch_image)
    if not os.path.isdir(patch_mask):
        os.makedirs(patch_mask)
    if not os.path.isdir(patch_visual):
        os.makedirs(patch_visual)

    imageList = os.listdir(image_path)
    #    for name in imageList[0].split(' '):
    for name in imageList:
        name_noext, _ = os.path.splitext(name)
        print(name_noext)
        image = cv2.imread(os.path.join(image_path, name_noext + '.png'))
        mask = cv2.imread(os.path.join(mask_path, name_noext + '.png'), 0)

        mpp = float(name_noext.split('MPP-')[1])
        scale = round(mpp / target_mpp, 4)
        resize_height = int(np.ceil(image.shape[0] * scale / 32) * 32)
        resize_width = int(np.ceil(image.shape[1] * scale / 32) * 32)

        ds_image = cv2.resize(image, (resize_width, resize_height))
        ds_mask = cv2.resize(mask, (resize_width, resize_height),
                             interpolation=cv2.INTER_NEAREST)
        ds_mask = convertMultiLabels(ds_mask)

        max_y = int((np.ceil(resize_height / stride) - 1) * stride) + patch_size
        max_x = int((np.ceil(resize_width / stride) - 1) * stride) + patch_size

        image = np.zeros((max_y, max_x, 3), dtype='uint8')
        image[:ds_image.shape[0], :ds_image.shape[1], :] = ds_image
        #        plt.imshow(image)

        mask = 255 * np.ones((max_y, max_x, 3), dtype='uint8')
        mask[:ds_mask.shape[0], :ds_mask.shape[1], :] = ds_mask
        #        plt.imshow(mask)

        if isTrain:
            y_list = range(0, max_y, stride)[:-1]
            x_list = range(0, max_x, stride)[:-1]
        else:
            y_list = range(0, max_y, stride)
            x_list = range(0, max_x, stride)

        n = 0
        for y in y_list:
            for x in x_list:
                patch = image[y:y + patch_size, x:x + patch_size, :]
                label = mask[y:y + patch_size, x:x + patch_size, :]

                if isTrain:
                    white = cv2.inRange(label,
                                        np.array([255, 255, 255]),
                                        np.array([255, 255, 255]))
                    if np.sum(white) / 255 > 0.9 * white.shape[0] * white.shape[1]:
                        continue

                save_name = name_noext.split('xmin')[0]
                out_fn = '{}{:03d}_{:d}_{}{}'.format(save_name, n,
                                                     patch_size, '{:.3f}'.format(target_mpp), '.png')

                cv2.imwrite(os.path.join(patch_image, out_fn), patch)
                cv2.imwrite(os.path.join(patch_mask, out_fn), label)

                overlay = cv2.addWeighted(patch, 0.75, label, 0.25, 0.5)
                out_fn = '{}{:03d}_{:d}_{}{}'.format(save_name, n,
                                                     patch_size, '{:.3f}'.format(target_mpp), '.jpg')
                cv2.imwrite(os.path.join(patch_visual, out_fn), overlay)

                n += 1