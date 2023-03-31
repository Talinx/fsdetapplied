from multiprocessing import cpu_count
import os
import re
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset_pollen import PollenDataset, PollenFoodQSDataset
from constants import inference_batch_size, device, cpu_device, models_path, inferenced_img_path, num_classes, logs_path
from plot import convert_name_to_title


pattern_name = re.compile("model (.*).pt")
pattern_dataset = re.compile(".*[iI][nN][fF][oO]:.*Data set: (.*)")


def convert_name_to_dataset_type(name):
    """get the data set type used to train a model with a certain name

    Parameters
    ----------
    name : str
        name of the model

    Returns
    -------
    str
        data set used to train the model
    """
    log_file = os.path.join(logs_path, f"training {name}.log")
    dataset_type = "FoodQS"
    if os.path.isfile(log_file):
        with open(log_file, "r") as f:
            line = f.readline()
            while line != "":
                m = pattern_dataset.match(line)
                if m is not None:
                    dataset_type = m.group(1)
                line = f.readline()
    return dataset_type


def add_bounding_boxes(image, predictions, dataset, threshold: float=0.0):
    """add bounding boxes from model inference

    Parameters
    ----------
    image : Image
        image to add bounding boxes too
    predictions : dict
        predictions from the model
    dataset : torch.utils.data.Dataset
        dataset used to train model
    threshold : float
        threshold for scores of added bounding boxes

    Returns
    -------
    Image
        image with bounding boxes
    """
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    image_draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        if scores[i] < threshold:
            continue
        xmin, ymin, xmax, ymax = box
        try:
            outline_color = dataset.get_color_for_num(labels[i])
            label_text = dataset.get_name_for_num(labels[i])
        except KeyError:
            outline_color = (255, 0, 0)
            label_text = "unknown"
        image_draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=outline_color)
        image_draw.text((xmin, ymin), f"{label_text} {scores[i]:.4f}")
    return image


def add_bounding_boxes_labels(image, targets, labels):
    """add bounding boxes from labeled data set

    Parameters
    ----------
    image : Image
        image to add bounding boxes too
    targets : list
        list of bounding boxes
    labels : list
        list of labels for the bounding boxes

    Returns
    -------
    Image
        image with bounding boxes
    """
    image_draw = ImageDraw.Draw(image)
    for i, t in enumerate(targets):
        if t[2] == 0.0 and t[3] == 0.0:
            continue
        xmin, ymin, xmax, ymax = t
        image_draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=dataset.get_color_for_num(labels[i]))
        image_draw.text((xmin, ymin), dataset.get_name_for_num(labels[i]))
    return image


def get_ground_truth_and_union(w, h, x_targets, x_labels):
    """get ground truth and union for IoU calculation

    Parameters
    ----------
    w : int
        image width
    h : int
        image height
    x_targets : list
        bounding boxes from data set
    x_labels : list
        labels from data set

    Returns
    -------
    tuple
        tuple of ground truth and union, one of each for each class
    """
    ground_truth = [np.full((w, h), False) for i in range(num_classes)]
    union = [np.full((w, h), False) for i in range(num_classes)]
    # construct ground truth mask for each class
    for i, t in enumerate(x_targets):
        for x in range(int(t[0]), int(t[2] + 1)):
            if x >= w:
                continue
            for y in range(int(t[1]), int(t[3] + 1)):
                if y >= h:
                    continue
                l = int(x_labels[i].item())
                ground_truth[l][x, y] = True
                union[l][x, y] = True
    return ground_truth, union


def intersection_over_union(image, x_targets, x_labels, y_targets, y_labels, y_scores, threshold=0.5):
    """calculate the intersection over union

    Parameters
    ----------
    image : Image
        input image
    x_targets : list
        bounding boxes from data set
    x_labels : list
        labels from data set
    y_targets : list
        bounding boxes from model inference
    y_labels : list
        labels from model inference
    y_scores : list
        scores from model inference
    threshold : float
        threshold for bounding boxes to use

    Returns
    -------
    list
        IoU for each class
    """
    w, h = image.size
    intersection = [np.full((w, h), False) for i in range(num_classes)]
    # construct ground truth mask for each class
    ground_truth, union = get_ground_truth_and_union(w, h, x_targets, x_labels)
    # calculate intersection
    for i, t in enumerate(y_targets):
        if y_scores[i] < threshold:
            continue
        for x in range(int(t[0]), int(t[2] + 1)):
            if x >= w:
                continue
            for y in range(int(t[1]), int(t[3] + 1)):
                if y >= h:
                    continue
                l = int(y_labels[i].item())
                if l >= len(union):
                    continue
                union[l][x, y] = True
                if ground_truth[l][x, y]:
                    intersection[l][x, y] = True
    i_o_u = [0.0 for i in range(num_classes)]
    for i in range(num_classes):
        current_union = union[i].sum()
        if current_union > 0.0:
            i_o_u[i] = intersection[i].sum() / current_union
    return i_o_u



def inference(net, generate_stats: bool=False, threshold: float=0.9, dataset_type: str="FoodQS"):
    """perform inference with a model

    Parameters
    ----------
    net : nn.Module
        model to use
    generate_stats : bool
        whether to generate stats, i.e. mAP values
    threshold : float
        threshold for adding bounding boxes to images
    dataset_type : str
        data set to use

    Returns
    -------
    tuple
        tuple of output images, IoU values, mAP metrics, metrics per class, class number to name mapping, count of images with at least one bounding box
    """
    dataset = PollenFoodQSDataset() if dataset_type == "FoodQS" else PollenDataset()
    dataloader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, num_workers=cpu_count())
    net.eval()
    current_i = 0
    output_images = []
    i_o_u = [0.0 for i in range(num_classes)] if generate_stats else None
    metric = MeanAveragePrecision() if generate_stats else None
    metric_per_class = []
    generate_metric_per_class = generate_stats and dataset_type != "FoodQS"
    if generate_metric_per_class:
        metric_per_class = [MeanAveragePrecision() for i in range(num_classes)]
    num_to_class_name = [dataset.get_name_for_num(num) for num in range(num_classes)]
    img_with_bbox = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            inputs = data[0].to(device)
            predictions = net(inputs)
            inputs = inputs.to(cpu_device)
            for j in range(len(inputs)):
                current_img = dataset.get_image(current_i)
                boxes = predictions[j]['boxes']
                labels = predictions[j]['labels']
                scores = predictions[j]['scores']
                x_targets, x_labels = data[2][j], data[3][j]
                pred_dict = [
                    {
                        "boxes": boxes.to(cpu_device),
                        "labels": labels.to(cpu_device),
                        "scores": scores.to(cpu_device)
                    }
                ]
                target_dict = [
                    {
                        "boxes": x_targets,
                        "labels": x_labels
                    }
                ]
                if len(boxes) > 0:
                    img_with_bbox += 1
                output_images.append(add_bounding_boxes(current_img, predictions[j], dataset, threshold))
                if generate_stats:
                    # current_i_o_u = intersection_over_union(current_img, x_targets, x_labels, boxes, labels, scores)
                    # for k in range(num_classes):
                    #     i_o_u[k] += current_i_o_u[k]
                    metric.update(pred_dict, target_dict)
                if generate_metric_per_class:
                    index_for_class = int(data[3][j])
                    metric_per_class[index_for_class].update(pred_dict, target_dict)
                current_i += 1
    if generate_stats:
        for k in range(num_classes):
            i_o_u[k] /= len(dataset)
    return output_images, i_o_u, metric, metric_per_class, num_to_class_name, img_with_bbox


def inference_model(fmodel, name, generate_stats: bool=False, threshold: float=0.9, count_images_with_bbox: bool = False):
    """perform inference with a model from a certain file

    Parameters
    ----------
    fmodel : str
        model file name
    name : str
        name of the model
    generate_stats : bool
        whether to generate stats, i.e. mAP values
    threshold : float
        threshold for adding bounding boxes to images
    count_images_with_bbox : bool
        whether to count images with at least one bounding box
    """
    print(f"Performing inference with {name} ({convert_name_to_title(name)})...")
    model_file = os.path.join(models_path, fmodel)
    model = torch.load(model_file)
    images, i_o_u, metric, metric_per_class, num_to_class_name, img_with_bbox = inference(model, generate_stats, threshold, dataset_type=convert_name_to_dataset_type(name))
    model_image_path = os.path.join(inferenced_img_path, name)
    os.makedirs(model_image_path, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(model_image_path, f"{i}.jpg"))
    if generate_stats:
        # print(f"IoU for {name}: {i_o_u}")
        print(f"mAP for {name}: {metric.compute()}")
        if len(metric_per_class) > 0:
            print("mAP per class:")
            for i, metric in enumerate(metric_per_class):
                print(f"mAP for {num_to_class_name[i]}: {metric_per_class[i].compute()}")
    if count_images_with_bbox:
        print(f"{img_with_bbox} images have at least one bounding box")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'inference',
                    description = 'perform inference with the models')
    parser.add_argument('-l', '--labels', action='store_true')
    parser.add_argument('-s', '--stats', action='store_true')
    parser.add_argument('-t', '--threshold', type=float)
    parser.add_argument('-c', '--count', action='store_true')
    args = parser.parse_args()
    threshold = 0.9 if args.threshold is None else args.threshold
    count_images_with_bbox = False if args.count is None or args.count == False else True
    if args.labels:
        # FoodQS
        dataset = PollenFoodQSDataset(k=100)
        image_path = os.path.join(inferenced_img_path, "labels")
        os.makedirs(image_path, exist_ok=True)
        for i in tqdm(range(len(dataset))):
            x, y, targets, labels = dataset[i]
            img = dataset.get_image(i)
            img = add_bounding_boxes_labels(img, targets, labels)
            img.save(os.path.join(image_path, f"{i}.jpg"))
        dataset = PollenFoodQSDataset(k=100, full_size=True)
        image_path = os.path.join(inferenced_img_path, "labels full size")
        os.makedirs(image_path, exist_ok=True)
        for i in tqdm(range(len(dataset))):
            x, y, targets, labels = dataset[i]
            img = dataset.get_image(i)
            img = add_bounding_boxes_labels(img, targets, labels)
            img.save(os.path.join(image_path, f"{i}.jpg"))
        # Pollen
        dataset = PollenDataset()
        image_path = os.path.join(inferenced_img_path, "labels pollen")
        os.makedirs(image_path, exist_ok=True)
        for i in tqdm(range(len(dataset))):
            x, y, targets, labels = dataset[i]
            img = dataset.get_image(i)
            img = add_bounding_boxes_labels(img, targets, labels)
            img.save(os.path.join(image_path, f"{i}.jpg"))
        dataset = PollenDataset(full_size=True)
        image_path = os.path.join(inferenced_img_path, "labels pollen full size")
        os.makedirs(image_path, exist_ok=True)
        for i in tqdm(range(len(dataset))):
            x, y, targets, labels = dataset[i]
            img = dataset.get_image(i)
            img = add_bounding_boxes_labels(img, targets, labels)
            img.save(os.path.join(image_path, f"{i}.jpg"))
    else:
        models = os.listdir(models_path)
        for fmodel in models:
            m = pattern_name.fullmatch(fmodel)
            if m is None:
                print(f"Skipping {fmodel}, name format wrong.")
                continue
            inference_model(fmodel, m.group(1), args.stats, threshold, count_images_with_bbox)

