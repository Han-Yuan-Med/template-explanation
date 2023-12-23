import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import cv2
import math
import gc
from skimage import color
from skimage import segmentation
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from grad_cam_functions import *
from py_functions import *
from scipy import ndimage

late_mask = cv2.imread("Symbolic_overlap_15.png", cv2.IMREAD_GRAYSCALE) / 255
image_path = "D:\\SIIM-ACR Dataset\\Images"
mask_path = "D:\\SIIM-ACR Dataset\\masks"

test_cases_csv = pd.read_csv("test_case.csv")
test_cases = Lung_seg(csv_file=test_cases_csv.drop([265]), img_dir=image_path, mask_dir=mask_path)
test_cases_loader = DataLoader(test_cases, batch_size=1, shuffle=False)

# Instantiating CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
VGG19_optimal = torch.load("VGG19 optimal.pt")

results_df = []

iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
    test_cases_late_mask(test_loader_seg=test_cases_loader, device=device, cls_model=vgg_grad_cam_19(VGG19_optimal),
                         threshold=95, fixed_mask=np.ones((224, 224)))

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"VGG-19", f"Baseline",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

iou_list_test_case_msk, dice_list_test_case, iou_case, dice_case = \
    test_cases_late_mask(test_loader_seg=test_cases_loader, device=device, cls_model=vgg_grad_cam_19(VGG19_optimal),
                         threshold=95, fixed_mask=late_mask)

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case_msk), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"VGG-19", f"Masked",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

Res50_optimal = torch.load("Res50 optimal.pt")

iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
    test_cases_late_mask(test_loader_seg=test_cases_loader, device=device, cls_model=res_grad_cam_50(Res50_optimal),
                         threshold=95, fixed_mask=np.ones((224, 224)))

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"ResNet-50", f"Baseline",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
    test_cases_late_mask(test_loader_seg=test_cases_loader, device=device, cls_model=res_grad_cam_50(Res50_optimal),
                         threshold=95, fixed_mask=late_mask)

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"ResNet-50", f"Masked",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

pd.DataFrame(results_df).to_csv("results_grad_cam.csv", index=False, encoding="cp1252")
