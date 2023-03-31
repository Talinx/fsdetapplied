from multiprocessing import cpu_count
from datetime import datetime
import time
import os
import argparse
import configparser

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from model import get_new_model_mobilenet, freeze_last_layers, get_new_model_mobilenet_320, get_new_model_fasterrcnn
from dataset_pollen import PollenDataset, PollenFoodQSDataset
from constants import batch_size, epochs, k, batch_size_validation, display_every_batches, device, cpu_device, losses_path, get_logger, models_path, get_default_optimizer


def filter_boxes_and_labels_box(boxes, labels):
    output_size = 0
    for box in boxes:
        if box[2] > 0.0 and box[3] > 0.0:
            output_size += 1
    out_boxes = torch.zeros((output_size, 4))
    out_labels = torch.zeros(output_size, dtype=torch.int64)
    current_i = 0
    for j, box in enumerate(boxes):
        if box[2] > 0.0 and box[3] > 0.0:
            out_boxes[current_i, 0] = box[0]
            out_boxes[current_i, 1] = box[1]
            out_boxes[current_i, 2] = box[2]
            out_boxes[current_i, 3] = box[3]
            out_labels[current_i] = labels[j]
            current_i += 1
    return out_boxes.to(device)


def filter_boxes_and_labels_label(boxes, labels):
    output_size = 0
    for box in boxes:
        if box[2] > 0.0 and box[3] > 0.0:
            output_size += 1
    out_boxes = torch.zeros((output_size, 4))
    out_labels = torch.zeros(output_size, dtype=torch.int64)
    current_i = 0
    for j, box in enumerate(boxes):
        if box[2] > 0.0 and box[3] > 0.0:
            out_boxes[current_i, 0] = box[0]
            out_boxes[current_i, 1] = box[1]
            out_boxes[current_i, 2] = box[2]
            out_boxes[current_i, 3] = box[3]
            out_labels[current_i] = labels[j]
            current_i += 1
    return out_labels.to(device)


def train(net, info: str="", k=k, epochs=epochs, epochinterval=50, epochonlyoneloss=1800, dataset_type="FoodQS", fsdet_losses=False):
    if dataset_type == "FoodQS":
        train_set, validation_set = PollenFoodQSDataset(k=k), PollenFoodQSDataset(k=100, skip_k=k)
    else:
        train_set, validation_set = PollenDataset(k=k), PollenDataset(k=35, skip_k=k)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    trainloader_validation = DataLoader(validation_set, batch_size=batch_size_validation, num_workers=cpu_count())
    net.to(device)
    logger, log_file, suffix = get_logger()
    optimizer = get_default_optimizer(net.parameters())
    losses = []
    losses_epoch = []
    losses_validation = []
    loss_file = os.path.join(losses_path, f"losses train {suffix}.npy")
    loss_file_validation = os.path.join(losses_path, f"losses validation {suffix}.npy")
    model_file = os.path.join(models_path, f"model {suffix}.pt")
    os.makedirs(losses_path, exist_ok=True)
    logger.info('Training starts: ')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Data set: {dataset_type}')
    logger.info(datetime.now())
    losses_to_use = ['loss_classifier', 'loss_objectness', 'loss_rpn_box_reg']
    all_losses = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
    fsdet_losses = ['loss_classifier', 'loss_box_reg']
    loss_files = {loss: os.path.join(losses_path, f"losses train {loss} {suffix}.npy") for loss in all_losses}
    different_losses = {loss: [] for loss in all_losses}
    current_different_losses = {loss: 0.0 for loss in all_losses}
    epochinterval_double = 2 * epochinterval
    if info is not None:
        logger.info(f'Info: {info}')
    try:
        for epoch in range(epochs):
            # loss schedule
            if fsdet_losses:
                losses_to_use = fsdet_losses
            elif epoch%epochinterval_double >= epochinterval and epoch < epochonlyoneloss:
                losses_to_use = ['loss_classifier', 'loss_objectness']
            else:
                losses_to_use = ['loss_rpn_box_reg', 'loss_box_reg']
            running_loss = 0.0
            epoch_loss = 0.0
            validation_loss = 0.0
            start_t_epoch = time.time()
            start_t = time.time()
            for i, data in enumerate(trainloader):
                inputs, targets, t, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                t_dict_list = [{ 'boxes': filter_boxes_and_labels_box(box, label), 'labels': filter_boxes_and_labels_label(box, label) } for box, label in zip(t, labels)]
                optimizer.zero_grad()
                batch_losses = net(inputs, t_dict_list)
                count = 1
                for loss_type, loss in batch_losses.items():
                    if loss_type in losses_to_use:
                        if count < len(losses_to_use):
                            loss.backward(retain_graph=True)
                        else:
                            loss.backward()
                        current_loss = loss.item()
                        current_different_losses[loss_type] += current_loss
                        running_loss += current_loss
                        epoch_loss += current_loss
                        count += 1
                optimizer.step()
                if i % display_every_batches == (display_every_batches - 1):
                    part_count = i * batch_size + len(targets)
                    batch_percent = 100 * part_count / len(train_set)
                    loss_normalized = running_loss / display_every_batches
                    took_t = time.time() - start_t
                    logger.info(f'[epoch {epoch+1}, {i+1:2} batches, {part_count:3} images, {batch_percent:.2f}%]\t\tloss: {loss_normalized:.6f}, took {took_t:.1f}s')
                    losses.append(loss_normalized)
                    running_loss = 0.0
                    # current_different_losses = {loss: 0.0 for loss in all_losses}
                    start_t = time.time()
            took_t_epoch = time.time() - start_t_epoch
            epoch_loss_normalized = epoch_loss / len(trainloader)
            losses_epoch.append(epoch_loss_normalized)
            np.save(loss_file, losses_epoch)
            losses_normalized = [current_different_losses[loss_type] / len(trainloader) for loss_type in all_losses]
            for i, loss_type in enumerate(all_losses):
                different_losses[loss_type].append(losses_normalized[i])
            current_different_losses = {loss: 0.0 for loss in all_losses}
            with torch.no_grad():
                for i, data in enumerate(trainloader_validation):
                    inputs, targets, t, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                    t_dict_list = [{ 'boxes': filter_boxes_and_labels_box(box, label), 'labels': filter_boxes_and_labels_label(box, label) } for box, label in zip(t, labels)]
                    batch_losses = net(inputs, t_dict_list)
                    count = 1
                    for loss_type, loss in batch_losses.items():
                        if loss_type in losses_to_use:
                            validation_loss += loss.item()
                            count += 1
            took_t_epoch_with_validation = time.time() - start_t_epoch
            validation_loss_normalized = validation_loss / len(trainloader_validation)
            losses_validation.append(validation_loss_normalized)
            np.save(loss_file_validation, losses_validation)
            for loss_type in all_losses:
                np.save(loss_files[loss_type], different_losses[loss_type])
            logger.info(f'Epoch {epoch+1} finished, took {took_t_epoch:.1f}s ({took_t_epoch_with_validation:.1f}s with validation).\tloss: {epoch_loss_normalized:.6f}\tloss validation: {validation_loss_normalized:.6f}')
            start_t_epoch = time.time()
    except KeyboardInterrupt as e:
        choice = None
        while choice is None or choice not in ["", "y", "n", "yes", "no"]:
            choice = input("Do you want to keep the log and loss files? [Y/n] ")
            choice = choice.lower()
        if choice in ["n", "no"]:
            os.remove(log_file)
            os.remove(loss_file)
            os.remove(loss_file_validation)
        raise e
    os.makedirs(models_path, exist_ok=True)
    torch.save(net, model_file)
    net.to(cpu_device)


def array_str_to_array(s):
    output = []
    s = s.replace("[", "")
    s = s.replace("]", "")
    parts = s.split(",")
    for part in parts:
        output.append(int(part.strip()))
    return output


def model_type_to_model(model_type, pretrained=False):
    if model_type == "fasterrcnn":
        return get_new_model_fasterrcnn(pretrained=pretrained)
    if model_type == "mobilenet":
        return get_new_model_mobilenet(pretrained=pretrained)
    if model_type == "mobilenet320":
        return get_new_model_mobilenet_320(pretrained=pretrained)


def model_type_to_name(model_type):
    if model_type == "fasterrcnn":
        return "fasterrcnn_resnet50_fpn"
    if model_type == "mobilenet":
        return "fasterrcnn_mobilenet_v3_large_fpn"
    if model_type == "mobilenet320":
        return "fasterrcnn_mobilenet_v3_large_320_fpn"
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'training',
                    description = 'train models')
    parser.add_argument('-c', '--config', type=str)
    args = parser.parse_args()
    if args.config is None:
        model = get_new_model_fasterrcnn()
        train(model, info=f"fasterrcnn_resnet50_fpn, FoodQS with k={k}, {epochs} epochs, 50 epochs only loss_box_reg and loss_rpn_box_reg then 50 only the rest and so forth")
    else:
        config = configparser.ConfigParser()
        config.read(args.config)
        print(config, type(config), config.sections())
        sections = config.sections()
        for section in sections:
            ks = array_str_to_array(config[section]['k'])
            epochs = array_str_to_array(config[section]['epochs'])
            epochsinterval = array_str_to_array(config[section]['epochsinterval'])
            epochsonlyoneloss = array_str_to_array(config[section]['epochsonlyoneloss'])
            dataset_type = config[section]['dataset']
            model_type = config[section]['model']
            try:
                pretrained_raw = config[section]['pretrained']
            except KeyError:
                pretrained_raw = "False"
            pretrained = pretrained_raw == True or pretrained_raw.upper() == "TRUE"
            try:
                fsdet_raw = config[section]['fsdet']
            except KeyError:
                fsdet_raw = "False"
            fsdet = fsdet_raw == True or fsdet_raw.upper() == "TRUE"
            print(f"Training part {section} for model type {model_type}:")
            if model_type not in ['fasterrcnn', 'mobilenet', 'mobilenet320']:
                print(f"Invalid model type: '{model_type}', skipping training part.")
                continue
            if dataset_type not in ["FoodQS", "Pollen"]:
                print(f"Invalid dataset: '{dataset_type}, skipping training part.'")
            model_name = model_type_to_name(model_type)
            for i in range(len(ks)):
                k = ks[i]
                epoch = epochs[i]
                epochinterval = epochsinterval[i]
                epochonlyoneloss = epochsonlyoneloss[i]
                model = model_type_to_model(model_type, pretrained)
                if epochonlyoneloss < epoch:
                    losses_desc = f"{epochinterval} only loss_box_reg and loss_rpn_box_reg then 50 only the rest and so forth, only loss_box_reg and loss_rpn_box_reg after {epochonlyoneloss} epochs"
                    if fsdet:
                        losses_desc = "FsDet losses"
                    if pretrained:
                        info_str = f"{model_name}, pretrained on COCO, {dataset_type} with k={k}, {epoch} epochs, {losses_desc}"
                    else:
                        info_str = f"{model_name}, {dataset_type} with k={k}, {epoch} epochs, {losses_desc}"
                else:
                    losses_desc = f"{epochinterval} only loss_box_reg and loss_rpn_box_reg then 50 only the rest and so forth"
                    if fsdet:
                        losses_desc = "FsDet losses"
                    if pretrained:
                        info_str = f"{model_name}, pretrained on COCO, {dataset_type} with k={k}, {epoch} epochs, {losses_desc}"
                    else:
                        info_str = f"{model_name}, {dataset_type} with k={k}, {epoch} epochs, {losses_desc}"
                train(model, info=info_str, k=k, epochs=epoch, epochinterval=epochinterval, epochonlyoneloss=epochonlyoneloss, dataset_type=dataset_type, fsdet_losses=fsdet)
