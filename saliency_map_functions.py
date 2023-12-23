import torch
import numpy as np
from sklearn import metrics
import cv2
import copy
from tqdm import tqdm


def saliency_map(model, image, threshold):
    image.requires_grad_()
    model(image)[:, 1].backward()
    outputs, _ = torch.max(image.grad.data.abs(), dim=1)
    outputs = outputs.flatten()
    outputs = np.array(outputs.detach().cpu())
    # if np.max(outputs) != 0:
    #     outputs = (outputs - np.min(outputs)) / np.max(outputs)
        # outputs = (outputs - np.min(outputs)) / np.max(outputs)
    threshold_value = np.percentile(outputs, threshold)
    outputs = outputs.flatten()
    outputs[np.where(outputs >= threshold_value)] = 1
    outputs[np.where(outputs != 1)] = 0

    return outputs


def saliency_map_org(model, image):
    image.requires_grad_()
    model(image)[:, 1].backward()
    outputs, _ = torch.max(image.grad.data.abs(), dim=1)
    outputs = outputs.flatten()
    outputs = np.array(outputs.detach().cpu())
    if np.max(outputs) != 0:
        outputs = (outputs - np.min(outputs)) / np.max(outputs)
    outputs = outputs.flatten()

    return outputs


def test_cases_late_mask(test_loader_seg, device, cls_model, threshold, fixed_mask):
    cls_model.eval()
    iou_list_test_case = []
    dice_list_test_case = []

    for data_test in tqdm(test_loader_seg):
        images_test, labels_test = data_test[0].to(device).float(), np.array(data_test[1]).flatten()
        if labels_test.max() == 0:
            continue
        else:
            binary_list_iou = saliency_map_org(model=cls_model, image=images_test)
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
