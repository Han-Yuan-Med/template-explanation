import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
import matplotlib.image as img
import numpy as np
from sklearn import metrics
import cv2


class Lung_cls(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 1])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = torch.tensor(int(self.annotations.iloc[index, 3]))

        image_fused = np.zeros((3, 224, 224))
        image_fused[0] = image / 255
        image_fused[1] = image / 255
        image_fused[2] = image / 255
        image_fused = torch.from_numpy(image_fused)

        return image_fused, label


class Lung_seg(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir):
        self.annotations = csv_file
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 1])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_fused = np.zeros((3, 224, 224))
        image_fused[0] = image / 255
        image_fused[1] = image / 255
        image_fused[2] = image / 255
        image_fused = torch.from_numpy(image_fused)

        mask_path_tmp = os.path.join(self.mask_dir, self.annotations.iloc[index, 1])
        mask = cv2.imread(mask_path_tmp, cv2.IMREAD_GRAYSCALE) / 255

        return image_fused, mask


def train_image(cls_model, train_loader, criterion, optimizer, device):
    cls_model.train()
    for batch_idx, (sample, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        sample = sample.to(device=device).float()
        targets = targets.to(device=device)

        # forward
        scores = cls_model(sample)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def train_image_masked(cls_model, train_loader, criterion, optimizer, device, fixed_mask):
    cls_model.train()
    fixed_mask = torch.tensor(fixed_mask).unsqueeze(0).unsqueeze(0).reshape(1, 1, 224, 224)
    for batch_idx, (sample, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        sample = (sample * fixed_mask).to(device=device, dtype=torch.float)
        targets = targets.to(device=device)

        # forward
        scores = cls_model(sample)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def bootstrap_sample(value_list, times):
    bootstrap_value = []

    for i in range(times):
        np.random.seed(i)
        test_index_bootstrap = np.random.choice(range(len(value_list)), len(value_list))
        bootstrap_value.append(value_list[test_index_bootstrap].astype('float').mean())

    value_l, value_u = format(np.percentile(bootstrap_value, 2.5), '.3f'), \
                       format(np.percentile(bootstrap_value, 97.5), '.3f')

    return format((np.float32(value_u) - np.float32(value_l))*100 / (2 * 1.96), '.2f')


def bootstrap_cls(prob_list, label_list, threshold, times):
    bootstrap_auc = []
    bootstrap_prc = []
    bootstrap_acc = []
    bootstrap_sen = []
    bootstrap_spe = []
    bootstrap_ppv = []
    bootstrap_npv = []

    for i in range(times):
        np.random.seed(i)
        test_index_bootstrap = np.random.choice(range(len(prob_list)), len(prob_list))
        prob_tmp = prob_list[test_index_bootstrap]
        label_tmp = label_list[test_index_bootstrap]
        # Calculate accuracy, auroc, sensitivity, specificity
        fpr, tpr, thresholds = metrics.roc_curve(label_tmp, prob_tmp)
        precision, recall, _ = metrics.precision_recall_curve(label_tmp, prob_tmp)
        # i.e. point on ROC curve which maximises the sensitivity and specificity
        prob_tmp[np.where(prob_tmp >= threshold)] = 1
        prob_tmp[np.where(prob_tmp != 1)] = 0
        tn, fp, fn, tp = metrics.confusion_matrix(label_tmp, prob_tmp).ravel()
        bootstrap_auc.append(metrics.auc(fpr, tpr))
        bootstrap_prc.append(metrics.auc(recall, precision))
        bootstrap_acc.append((tp + tn) / (tp + tn + fp + fn))
        bootstrap_sen.append(tp / (tp + fn))
        bootstrap_spe.append(tn / (tn + fp))
        bootstrap_ppv.append(tp / (tp + fp))
        bootstrap_npv.append(tn / (tn + fn))

    auc_l, auc_u = np.percentile(bootstrap_auc, 2.5), np.percentile(bootstrap_auc, 97.5)
    prc_l, prc_u = np.percentile(bootstrap_prc, 2.5), np.percentile(bootstrap_prc, 97.5)
    acc_l, acc_u = np.percentile(bootstrap_acc, 2.5), np.percentile(bootstrap_acc, 97.5)
    sen_l, sen_u = np.percentile(bootstrap_sen, 2.5), np.percentile(bootstrap_sen, 97.5)
    spe_l, spe_u = np.percentile(bootstrap_spe, 2.5), np.percentile(bootstrap_spe, 97.5)
    ppv_l, ppv_u = np.percentile(bootstrap_ppv, 2.5), np.percentile(bootstrap_ppv, 97.5)
    npv_l, npv_u = np.percentile(bootstrap_npv, 2.5), np.percentile(bootstrap_npv, 97.5)

    return format((auc_u - auc_l) / (2 * 1.96), '.3f'), format((prc_u - prc_l) / (2 * 1.96), '.3f'), \
           format((acc_u - acc_l) / (2 * 1.96), '.3f'), format((sen_u - sen_l) / (2 * 1.96), '.3f'), \
           format((spe_u - spe_l) / (2 * 1.96), '.3f'), format((ppv_u - ppv_l) / (2 * 1.96), '.3f'), \
           format((npv_u - npv_l) / (2 * 1.96), '.3f')
