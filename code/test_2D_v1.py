# -*- coding: utf-8 -*- 
# @Time : 2021/4/8 15:17 
# @Author : aurorazeng
# @File : test_2D.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.

import argparse
import os
import shutil
import math

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from medpy import metric

# from networks import linknetBase
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../csv_data', help='path of data')
parser.add_argument('--exp', type=str,
                    default='/Fu_LinkNetWithdrop_4_8_1715_labeled', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='LinkNetBaseWithDrop', help='model_name')
parser.add_argument('--num_classes', type=int, default=6,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=1522,
                    help='labeled data')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if (pred.sum() > 0 and gt.sum() > 0):
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, asd
    elif gt.sum() == 0:
        return -1, -1, -1
    else:
        return 0, 500, 500


# tumor = 1
# lymph = 3
# stroma = 2
# necrosis = 4
# others = 5
#
#
# def predictLabels(mask):
#     new_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
#     # 1 - tumor - red  RGB
#     new_mask[mask == 1] = np.array([255, 0, 0])
#     # 2 - stroma - blue
#     new_mask[mask == 2] = np.array([0, 0, 255])
#     # 3 - lymphocytic_infiltrate - green
#     new_mask[mask == 3] = np.array([0, 255, 0])
#     # 4 - necrosis_or_debris - cyan
#     new_mask[mask == 4] = np.array([0, 255, 255])
#     # 5 - others - yellow
#     new_mask[mask == 5] = np.array([255, 255, 0])
#     # 0: outside_roi - white
#     new_mask[mask == 0] = np.array([255, 255, 255])
#     return new_mask


tumor = [1, 19, 20]
lymph = [3, 10, 11, 14]
stroma = [2]
necrosis = [4]
others = [5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 21]


def convertMultiLabels(mask):
    new_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8')
    for i in tumor:
        new_mask[mask == i] = 1
    # 2 - stroma - blue
    for i in stroma:
        new_mask[mask == i] = 2
    # 3 - lymphocytic_infiltrate - green
    for i in lymph:
        new_mask[mask == i] = 3
    # 4 - necrosis_or_debris - cyan
    for i in necrosis:
        new_mask[mask == i] = 4
    # 5 - others - yellow
    for i in others:
        new_mask[mask == i] = 5
    # 0: outside_roi - white
    new_mask[mask == 0] = 0
    return new_mask


def test_single_volume(case_path, net, test_save_path, FLAGS):
    image_path = case_path.split(',')[0]
    label_path = case_path.split(',')[1]
    image = Image.open(image_path).convert('RGB')
    label = Image.open(label_path).convert('L')
    mpp = os.path.basename(image_path).split('.')[-2]
    if '_' in mpp:
        mpp = 424.0
    else:
        mpp = float(mpp)
    if mpp - (0.46 * 1000) > 0.0001:
        scale = round(mpp / 1000 / 0.460, 4)
        resize_w = int(math.ceil(image.size[0] * scale / 64) * 64)
        resize_h = int(math.ceil(image.size[1] * scale / 64) * 64)
        image = image.resize((resize_w, resize_h), Image.ANTIALIAS)
        label = label.resize((resize_w, resize_h), Image.NEAREST)
    image = np.asarray(image, np.float32)
    label = np.asarray(label, np.float32)
    image /= 255
    label = label.astype(np.int)
    image = np.transpose(image, (2, 0, 1))
    prediction = np.zeros_like(label)
    input = torch.from_numpy(image).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        prediction = out
    overlay_mask = np.zeros((label.shape[0], label.shape[1]))
    overlay_mask[label > 0] = 1
    prediction = prediction * overlay_mask
    metric_list = []
    for i in range(1, 6):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def Inference(FLAGS):
    with open(FLAGS.root_path + '/val_3_25.txt', 'r') as f:
        case_list = f.readlines()
    case_list = [item.replace('\n', '')
                 for item in case_list]
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    test_save_path = "../model/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=3,
                      class_num=FLAGS.num_classes)
    file_list = os.listdir(snapshot_path)
    filedir = []
    for i in file_list:
        if i.__contains__('pth') and int(i.split('.')[0].split('_')[1]) % 2000 == 0:
            filedir.append(i)
    filedir.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    target_txt = 'dice.txt'
    f = open(test_save_path + target_txt, 'a')
    for file in filedir:
        if file.__contains__('pth'):
            save_mode_path = os.path.join(
                snapshot_path, file)
            net.cuda()
            net.load_state_dict(torch.load(save_mode_path))
            print("init weight from {}".format(save_mode_path))
            net.eval()

            list = []
            metric = []
            for case in tqdm(case_list):
                metric_i = test_single_volume(
                    case, net, test_save_path, FLAGS)
                list.append(metric_i)
            list = np.asarray(list)
            metric_list = []
            for i in range(0, 5):
                metric_list.append(list[:, i])
            metric_list = np.asarray(metric_list)
            for i in range(0, 5):
                num = 0
                metric_i = 0
                for j in range(0, metric_list.shape[1]):
                    if np.sum(metric_list[i][j]) > 0:
                        num += 1
                        metric_i += metric_list[i][j]
                metric_i = metric_i / num
                metric.append(metric_i)
            avg_metric = np.asarray(metric)
            line = file
            for i in range(0, 5):
                line = line + ',' + str(round(avg_metric[i][0], 4))
            line = line + ',' + str(round(np.mean(avg_metric, axis=0)[0], 4))
            line = line + ',' + str(round(np.mean(avg_metric, axis=0)[1], 4))
            line = line + ',' + str(round(np.mean(avg_metric, axis=0)[2], 4))
            line = line + '\n'
            f.write(line)
    f.close()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
