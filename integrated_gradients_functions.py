import torch
import numpy as np
from sklearn import metrics
import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
from pathlib import Path
import os
from sklearn import metrics


def baseline(train_set):
    baseline_img = torch.zeros(size=(3, 224, 224))
    for i in tqdm(range(len(train_set))):
        baseline_img += train_set[i][0]
    baseline_img = baseline_img/len(train_set)
    return baseline_img


def integrated_gradient(model, image, baseline_img, steps, idx, threshold):
    mean_grad = 0
    model.eval()
    for i in tqdm(range(1, steps + 1)):
        x = baseline_img + i / steps * (image - baseline_img)
        x.requires_grad = True
        y = model(x)[0, idx]
        (grad,) = torch.autograd.grad(y, x)
        mean_grad += grad / steps

    integrated_gradients = (image - baseline_img) * mean_grad
    integrated_gradients = integrated_gradients.detach().cpu().numpy()
    integrated_gradients = np.abs(integrated_gradients)
    integrated_gradients = np.sum(integrated_gradients, axis=1)
    integrated_gradients = np.squeeze(integrated_gradients).flatten()
    # if np.max(integrated_gradients) != 0:
    #     integrated_gradients = (integrated_gradients - np.min(integrated_gradients)) / np.max(integrated_gradients)
    threshold_value = np.percentile(integrated_gradients, threshold)

    integrated_gradients = integrated_gradients.flatten()
    integrated_gradients[np.where(integrated_gradients >= threshold_value)] = 1
    integrated_gradients[np.where(integrated_gradients != 1)] = 0
    return integrated_gradients


def integrated_gradient_org(model, image, baseline_img, steps, idx):
    mean_grad = 0
    model.eval()
    for i in tqdm(range(1, steps + 1)):
        x = baseline_img + i / steps * (image - baseline_img)
        x.requires_grad = True
        y = model(x)[0, idx]
        (grad,) = torch.autograd.grad(y, x)
        mean_grad += grad / steps

    integrated_gradients = (image - baseline_img) * mean_grad
    integrated_gradients = integrated_gradients.detach().cpu().numpy()
    integrated_gradients = np.abs(integrated_gradients)
    integrated_gradients = np.sum(integrated_gradients, axis=1)
    integrated_gradients = np.squeeze(integrated_gradients).flatten()

    if np.max(integrated_gradients) != 0:
        integrated_gradients = (integrated_gradients - np.min(integrated_gradients)) / np.max(integrated_gradients)

    return integrated_gradients


def test_cases_late_mask(test_loader_seg, device, cls_model, threshold, fixed_mask, baseline_img):
    cls_model.eval()
    iou_list_test_case = []
    dice_list_test_case = []
    baseline_img = torch.tensor(baseline_img).to(device=device, dtype=torch.float)

    for data_test in tqdm(test_loader_seg):
        images_test, labels_test = data_test[0].to(device).float(), np.array(data_test[1]).flatten()
        if labels_test.max() == 0:
            continue
        else:
            binary_list_iou = integrated_gradient_org(model=cls_model, image=images_test, baseline_img=baseline_img,
                                                      steps=20, idx=1)
            binary_list_iou = binary_list_iou * fixed_mask.flatten()

            threshold_value = np.percentile(binary_list_iou, threshold)
            binary_list_iou[np.where(binary_list_iou >= threshold_value)] = 1
            binary_list_iou[np.where(binary_list_iou != 1)] = 0
            binary_list_iou = binary_list_iou.astype("uint8")
            iou = metrics.jaccard_score(labels_test, binary_list_iou)
            iou = format(iou, '.3f')

            dice = metrics.f1_score(labels_test, binary_list_iou)
            dice = format(dice, '.3f')

            iou_list_test_case.append(iou)
            dice_list_test_case.append(dice)

    iou_case = format(np.array(iou_list_test_case).astype('float').mean(), '.4f')
    dice_case = format(np.array(dice_list_test_case).astype('float').mean(), '.4f')

    print(f'IoU on sick test set is {iou_case}')
    print(f'Dice on sick test set is {dice_case}')

    return iou_list_test_case, dice_list_test_case, iou_case, dice_case
