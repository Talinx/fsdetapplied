from logging import currentframe
import os
import re
import colorsys
from numpy import resize

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from label_parser import parse_xml_file


def get_image_tensor(filename):
    """filename to tensor, also crops image to 1024x1024

    Parameters
    ----------
    filename : str
        image file name

    Returns
    -------
    torch.Tensor
        image tensor
    """
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor


def get_image_tensor_full_size(filename):
    """filename to tensor, uncropped

    Parameters
    ----------
    filename : str
        image file name

    Returns
    -------
    torch.Tensor
        image tensor
    """
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor


def get_image(filename):
    """filename to PIL Image, also crops image to 1024x1024

    Parameters
    ----------
    filename : str
        image file name

    Returns
    -------
    Image
        image
    """
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(1024),
    ])
    processed_image = preprocess(input_image)
    return processed_image


def get_image_full_size(filename):
    """filename to PIL Image, uncropped

    Parameters
    ----------
    filename : str
        image file name

    Returns
    -------
    Image
        image
    """
    input_image = Image.open(filename)
    return input_image


standard_dataset_dir = os.path.join("datasets", "pollen")
qs_dataset_dir = os.path.join("datasets", "FoodQS")

pattern_xml_file = re.compile(".*\\.xml")

class PollenDataset(Dataset):
    """Pollen dataset"""
    def __init__(self, directory: str=standard_dataset_dir, k: int=-1, skip_k: int=0, full_size: bool=False):
        self.directory = directory
        classes = os.listdir(directory)
        classes = [c for c in classes if c[0] != "."]  # remove hidden directories
        classes.append("kastanie")
        classes.append("raps")
        classes.sort()
        self.classes = classes
        self.size = 0
        self.k = k
        self.idx_to_filename = dict()
        self.idx_to_class = dict()
        self.class_to_num = dict()
        self.num_to_class = dict()
        self.num_to_color = dict()
        self.full_size = full_size
        actual_class_count = 0
        current_idx = 0
        class_num = 0
        for c in self.classes:
            current_k = 0
            skipped_k = 0
            self.class_to_num[c] = class_num
            self.num_to_class[class_num] = c
            if c in ["kastanie", "raps"]:
                class_num += 1
                continue
            actual_class_count += 1
            current_dir = os.path.join(directory, c)
            current_dir_files = os.listdir(current_dir)
            current_images = [img for img in current_dir_files if "jpg" in img]
            for filename in current_images:
                if skipped_k != skip_k:
                    skipped_k += 1
                    continue
                if self.k == -1 or current_k < self.k:
                    self.idx_to_class[current_idx] = c
                    self.idx_to_filename[current_idx] = os.path.join(current_dir, filename)
                    current_idx += 1
                    current_k += 1
                    self.size += 1
            class_num += 1
        HSV_tuples = [(x*1.0/len(self.classes), 0.5, 0.5) for x in range(len(self.classes))]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        for i in range(len(classes)):
            h, s, v = HSV_tuples[i]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            self.num_to_color[i] = (r, g, b)
        print(f"Created dataset with {current_idx} labeled images from {actual_class_count} classes.")

    def __len__(self):
        return self.size

    def get_image(self, idx):
        current_filename = self.idx_to_filename[idx]
        x_img = get_image_full_size(current_filename) if self.full_size else get_image(current_filename)
        return x_img

    def get_color_for_num(self, num):
        return self.num_to_color[int(num)]

    def get_name_for_num(self, num):
        return self.num_to_class[int(num)]
    
    def __getitem__(self, idx):
        y = torch.zeros(1000, dtype=torch.float)
        current_class = self.idx_to_class[idx]
        y[self.class_to_num[current_class]] = 1.0
        x = get_image_tensor(self.idx_to_filename[idx])
        targets = torch.tensor([[0.0, 0.0, 1024.0, 1024.0]])
        labels = torch.tensor([self.class_to_num[current_class]])
        return x, y, targets, labels


class PollenFoodQSDataset(Dataset):
    """Pollen FoodQS dataset"""
    def __init__(self, directory: str=qs_dataset_dir, standard_dataset_dir: str=standard_dataset_dir, k: int=-1, skip_k: int=0, full_size: bool=False):
        self.directory = directory
        classes = os.listdir(standard_dataset_dir)
        classes = [c for c in classes if c[0] != "."]  # remove hidden directories
        classes.append("kastanie")
        classes.append("raps")
        classes.sort()
        self.k = k
        self.classes = classes
        self.size = 0
        self.max_labels = 0
        self.idx_to_filename = dict()
        self.idx_to_labels = dict()
        self.idx_to_width = dict()
        self.idx_to_height = dict()
        self.class_to_num = dict()
        self.num_to_class = dict()
        self.num_to_color = dict()
        self.full_size = full_size
        actual_class_count = 0
        current_idx = 0
        class_num = 0
        for c in self.classes:
            self.class_to_num[c] = class_num
            self.num_to_class[class_num] = c
            class_num += 1
        for folder in os.listdir(self.directory):
            current_folder = os.path.join(self.directory, folder)
            current_k = 0
            skipped_k = 0
            actual_class_count += 1
            for fname in os.listdir(current_folder):
                if skipped_k != skip_k:
                    skipped_k += 1
                    continue
                if self.k != -1 and current_k == self.k:
                    continue
                current_file_xml = os.path.join(current_folder, fname)
                if pattern_xml_file.fullmatch(current_file_xml) is not None:
                    folder, filename, width, height, objects = parse_xml_file(current_file_xml)
                    current_file_image = os.path.join(self.directory, folder, filename)
                    if os.path.isfile(current_file_image):
                        self.size += 1
                        current_k += 1
                        self.idx_to_filename[current_idx] = current_file_image
                        self.idx_to_labels[current_idx] = objects
                        self.idx_to_width[current_idx] = width
                        self.idx_to_height[current_idx] = height
                        if len(objects) > self.max_labels:
                            self.max_labels = len(objects)
                        current_idx += 1
        HSV_tuples = [(x*1.0/len(self.classes), 0.5, 0.5) for x in range(len(self.classes))]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        for i in range(len(classes)):
            h, s, v = HSV_tuples[i]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            self.num_to_color[i] = (r, g, b)
        print(f"Created dataset with {current_idx} labeled images from {actual_class_count} classes.")

    def __len__(self):
        return self.size

    def get_image(self, idx):
        current_filename = self.idx_to_filename[idx]
        x_img = get_image_full_size(current_filename) if self.full_size else get_image(current_filename)
        return x_img

    def get_color_for_num(self, num):
        return self.num_to_color[int(num)]

    def get_name_for_num(self, num):
        return self.num_to_class[int(num)]

    def __getitem__(self, idx):
        current_labels = self.idx_to_labels[idx]
        current_filename = self.idx_to_filename[idx]
        x_full_size = get_image_tensor_full_size(current_filename)
        x = x_full_size if self.full_size else get_image_tensor(current_filename)
        y = torch.zeros(1000, dtype=torch.float)
        targets = torch.zeros((self.max_labels, 4))
        labels = torch.zeros(self.max_labels)
        width = self.idx_to_width[idx]
        height = self.idx_to_height[idx]
        w_offset = 0.0
        h_offset = 0.0
        c, h, w = x.size()
        c_full, h_full, w_full = x_full_size.size()
        resize_factor = h / h_full if h_full < w_full else w / w_full
        w_offset = (w_full * resize_factor - w) / 2
        h_offset = (h_full * resize_factor - h) / 2
        for i, label in enumerate(current_labels):
            targets[i, 0] = label.xmin * resize_factor - w_offset
            targets[i, 1] = label.ymin * resize_factor - h_offset
            targets[i, 2] = label.xmax * resize_factor - w_offset
            targets[i, 3] = label.ymax * resize_factor - h_offset
            if targets[i, 0] < 0.0 or targets[i, 0] >= w or targets[i, 1] < 0.0 or targets[i, 1] >= h or targets[i, 2] < 0.0 or targets[i, 2] >= w or targets[i, 3] < 0.0 or targets[i, 3] >= h:
                # filter out targets that are out of frame
                targets[i, 0] = 0.0
                targets[i, 1] = 0.0
                targets[i, 2] = 0.0
                targets[i, 3] = 0.0
            else:
                labels[i] = self.class_to_num[label.name]
        return x, y, targets, labels
