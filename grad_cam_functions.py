import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import cv2
import copy
from tqdm import tqdm


class vgg_grad_cam_11(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        # get the pretrained VGG19 network
        self.vgg = vgg_model
        # dissect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:19]
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        # placeholder for the gradients
        self.gradients = None
        self.ReLU = nn.ReLU(inplace=True)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        # apply the remaining pooling
        x = self.max_pool(self.ReLU(x))
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation extraction
    def get_activations(self, x):
        return self.features_conv(x)


class vgg_grad_cam_16(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        # get the pretrained VGG19 network
        self.vgg = vgg_model
        # dissect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:29]
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        # placeholder for the gradients
        self.gradients = None
        self.ReLU = nn.ReLU(inplace=True)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        # apply the remaining pooling
        x = self.max_pool(self.ReLU(x))
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation extraction
    def get_activations(self, x):
        return self.features_conv(x)


class vgg_grad_cam_19(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        # get the pretrained VGG19 network
        self.vgg = vgg_model
        # dissect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:35]
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        # placeholder for the gradients
        self.gradients = None
        self.ReLU = nn.ReLU(inplace=True)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        # apply the remaining pooling
        x = self.max_pool(self.ReLU(x))
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation extraction
    def get_activations(self, x):
        return self.features_conv(x)


class res_grad_cam_50(nn.Module):
    def __init__(self, res_model):
        super().__init__()
        # get the pretrained VGG19 network
        self.res = res_model
        # dissect the network to access its last convolutional layer
        self.conv1 = self.res.conv1
        self.bn1 = self.res.bn1
        self.relu = self.res.relu
        self.maxpool = self.res.maxpool
        self.features_conv_1 = self.res.layer1
        self.features_conv_2 = self.res.layer2
        self.features_conv_3 = self.res.layer3
        self.features_conv_4 = self.res.layer4
        # get the max pool of the features stem
        self.avg_pool = self.res.avgpool
        # get the classifier of the vgg19
        self.classifier = self.res.fc
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.features_conv_1(x)
        x = self.features_conv_2(x)
        x = self.features_conv_3(x)
        x = self.features_conv_4(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        # apply the remaining pooling
        x = self.avg_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation extraction
    def get_activations(self, x):
        return self.features_conv_4(self.features_conv_3(
            self.features_conv_2(self.features_conv_1(self.maxpool(self.relu(self.bn1(self.conv1(x))))))))


def grad_cam(model, image, threshold):
    model(image)[:, 1].backward()
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()
    # weight the channels by corresponding gradients
    for j in range(activations.shape[1]):
        activations[:, j, :, :] *= pooled_gradients[:, j]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(np.asarray(heatmap.detach().cpu()), 0)
    outputs = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    # normalize the heatmap
    threshold_value = np.percentile(outputs, threshold)

    outputs = outputs.flatten()
    outputs[np.where(outputs >= threshold_value)] = 1
    outputs[np.where(outputs != 1)] = 0
    # if max(outputs) != 0:
    #     outputs = (outputs - min(outputs)) / max(outputs)
    return outputs


def grad_cam_org(model, image):
    model(image)[:, 1].backward()
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()
    # weight the channels by corresponding gradients
    for j in range(activations.shape[1]):
        activations[:, j, :, :] *= pooled_gradients[:, j]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(np.asarray(heatmap.detach().cpu()), 0)
    outputs = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    # normalize the heatmap
    outputs = outputs.flatten()
    if max(outputs) != 0:
        outputs = (outputs - min(outputs)) / max(outputs)
    return outputs


def grad_cam_late(model, image):
    model(image)[:, 1].backward()
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()
    # weight the channels by corresponding gradients
    for j in range(activations.shape[1]):
        activations[:, j, :, :] *= pooled_gradients[:, j]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(np.asarray(heatmap.detach().cpu()), 0)
    outputs = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    # normalize the heatmap
    outputs = outputs.flatten()
    return outputs


def grad_cam_median(model, image):
    model(image)[:, 1].backward()
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()
    # weight the channels by corresponding gradients
    for j in range(512):
        activations[:, j, :, :] *= pooled_gradients[:, j]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(np.asarray(heatmap.detach().cpu()), 0)
    outputs = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    # normalize the heatmap
    outputs = outputs.flatten()
    # if max(outputs) != 0:
    #     outputs = (outputs - min(outputs)) / max(outputs)
    return np.median(outputs)


def test_cases_late_mask(test_loader_seg, device, cls_model, threshold, fixed_mask):
    cls_model.eval()
    iou_list_test_case = []
    dice_list_test_case = []
    # fixed_mask = torch.tensor(fixed_mask)
    for data_test in tqdm(test_loader_seg):
        images_test, labels_test = data_test[0].to(device).float(), np.array(data_test[1]).flatten()
        if labels_test.max() == 0:
            continue
        else:
            binary_list_iou = grad_cam_org(model=cls_model, image=images_test)
            binary_list_iou = binary_list_iou * fixed_mask.flatten()

            # if max(binary_list_iou) != 0:
            #     binary_list_iou = (binary_list_iou - min(binary_list_iou)) / max(binary_list_iou)

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

