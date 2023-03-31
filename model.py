import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

from constants import num_classes


def get_new_model_resnet101():
    model = torch.hub.load('pytorch/vision:v0.14.1', 'resnet101', pretrained=True)
    return model


def get_new_model_fasterrcnn(pretrained=False):
    """get Faster R-CNN model

    Paremeters
    ----------
    pretrained : bool
        whether the model should be pre-trained with COCO

    Returns
    -------
    nn.Module
        model
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(num_classes=91, weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    return model


def get_new_model_mobilenet(pretrained=False):
    """get MobileNetV3 model

    Paremeters
    ----------
    pretrained : bool
        whether the model should be pre-trained with COCO

    Returns
    -------
    nn.Module
        model
    """
    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=91, weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=num_classes)
    return model


def get_new_model_mobilenet_320(pretrained=False):
    """get MobileNetV3 fine tuned for mobile use cases model

    Paremeters
    ----------
    pretrained : bool
        whether the model should be pre-trained with COCO

    Returns
    -------
    nn.Module
        model
    """
    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=91, weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    else:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=num_classes)
    return model
