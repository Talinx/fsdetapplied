import os
import re
import argparse

from constants import losses_path, logs_path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

pattern_info = re.compile(".*Info: (.*)")
pattern_name = re.compile("losses validation (.*).npy")


def convert_name_to_title(name):
    """Convert the (file) name of the model to its title/description

    The name is not very descriptive (containing only the training date and time), whereas the title describes the model.

    Parameters
    ----------
    name : str
        name of the model

    Returns
    -------
    str
        model title/description
    """
    log_file = os.path.join(logs_path, f"training {name}.log")
    title = ""
    if os.path.isfile(log_file):
        with open(log_file, "r") as f:
            line = f.readline()
            while line != "" and title == "":
                m = pattern_info.match(line)
                if m is not None:
                    title = m.group(1)
                line = f.readline()
    return title


def plot(name):
    """plot losses for a model

    Parameters
    ----------
    name : str
        name of the model
    """
    title = convert_name_to_title(name)
    if title == "":
        print(f"Skipping {name}.")
    all_losses = os.listdir(losses_path)
    name_escaped = name.replace(".", "\\.")
    name_regex = re.compile(f".*losses (.*) {name_escaped}.*\\.npy")
    losses_for_name = [os.path.join(losses_path, n) for n in all_losses if len(name_regex.findall(n)) > 0 and n not in [f"losses train {name}.npy", f"losses validation {name}.npy"]]
    loss_train_file = os.path.join(losses_path, f"losses train {name}.npy")
    loss_validation_file = os.path.join(losses_path, f"losses validation {name}.npy")
    losses_train = np.load(loss_train_file)
    losses_validation = np.load(loss_validation_file)
    x = [i for i in range(len(losses_train))]
    x += [i for i in range(len(losses_validation))]
    losses = np.append(losses_train, losses_validation)
    y_train = ["Training " for i in range(len(losses_train))]
    y_validation = ["Validation " for i in range(len(losses_validation))]
    y_type = y_train + y_validation
    for loss_w_name in losses_for_name:
        c = np.load(loss_w_name)
        x += [i for i in range(len(c))]
        m = name_regex.findall(loss_w_name)
        y_type += [m[0] + " " for i in range(len(c))]
        losses = np.append(losses, c)
    data = {'Epoch': x, 'Type': y_type, 'Loss': losses}
    df = pd.DataFrame(data)
    fig = px.line(df, x="Epoch", y="Loss", color='Type', title=title)
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'plot losses',
                    description = 'Plots losses')
    parser.add_argument('-t', '--latest',
                    action='store_true')
    args = parser.parse_args()
    loss_files = os.listdir(losses_path)
    names = []
    for fname in loss_files:
        m = pattern_name.fullmatch(fname)
        if m is not None:
            names.append(m.group(1))
    names.sort()
    if args.latest:
        if len(names) >= 1:
            plot(names[-1])
    else:
        for name in names:
            plot(name)
