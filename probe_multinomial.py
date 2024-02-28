from __future__ import print_function
import os
import random
import copy
import argparse
from pathlib import Path
import math
import pickle
import json
import glob
import time
import torch

# Data & Computation
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, log_loss
from sklearn.metrics import roc_curve
from tqdm import tqdm

import plotly.express as px
import plotly.io as pio

import ipdb

import matplotlib.colors as colors
from sklearn.model_selection import train_test_split

from probe import load_data, set_random_seed
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamic', type=str, default='0_forward')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--variable', type=str, default='belief')
    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args


def plot_heatmap(ht, name, save_path=None):
    # Increase global font size for all text elements
    plt.rcParams.update({'font.size': 22})

    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap using seaborn
    sns.heatmap(ht, ax=ax, cmap='Blues', vmin=0.25, vmax=1, cbar_kws={'drawedges': False}, square=True)

    # Customize the colorbar
    cbar = ax.collections[0].colorbar
    cbar.outline.set_linewidth(2)  # Set colorbar outline width

    # Set the ticks for x and y axes with specified interval
    ax.set_xticks(np.arange(0.5, ht.shape[1], 5))
    ax.set_yticks(np.arange(0.5, ht.shape[0], 5))

    # Set the tick labels for x and y axes with specified interval and keep x-axis labels horizontal
    ax.set_xticklabels(np.arange(0, ht.shape[1], 5), rotation=0)
    ax.set_yticklabels(np.arange(0, ht.shape[0], 5))

    # Set axis labels and title with increased padding and font size
    ax.set_xlabel('Head', fontsize=24, labelpad=20)
    ax.set_ylabel('Layer', fontsize=24, labelpad=20)
    # ax.set_title(name, fontsize=28)

    # Reinstate axis lines with specified linewidth
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_visible(True)
        ax.spines[axis].set_linewidth(2)

    # Optionally save the figure as a PDF with vectorized content
    if save_path:
        plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')

    # Clear the current figure's memory to prevent resource leaks
    plt.close(fig)



def probe_single_case_multinomial(X_train, y_train, X_val, y_test, seed=0, verbose=False):
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',random_state=seed)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    if verbose:
        print("Training Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
    
        print("\nConfusion Matrix (Test Set):")
        print(confusion_matrix(y_test, y_test_pred))
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred))
    return train_accuracy, test_accuracy, clf

if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    
    X_all, y_o = load_data(dynamic=args.dynamic, belief="oracle", variable=args.variable)
    X_all, y_p = load_data(dynamic=args.dynamic, belief="protagonist", variable=args.variable)
    y_o_int = y_o.astype(int)
    y_p_int = y_p.astype(int)
    
    # Combine to a new class - 0 (o0p0), 1 (o0p1), 2 (o1p0), 3 (o1p1)
    y_combined = 2 * y_o_int + y_p_int
    data_ids = np.arange(len(X_all))
    all_X_train, all_X_test, y_train, y_test, ids_train, ids_test = train_test_split(X_all, y_combined, data_ids, test_size=0.2, random_state=args.seed)
    
    num_layers, num_heads, head_dims = all_X_train.shape[1:]
    train_acc_all = np.zeros([num_layers, num_heads])
    val_acc_all = np.zeros([num_layers, num_heads])
    coefs_all = np.zeros([num_layers, num_heads, 4, head_dims])
    
    for layer in tqdm(range(num_layers)):
        for head in range(num_heads):
            X_train = all_X_train[:,layer,head]
            X_test = all_X_test[:,layer,head]
            train_acc_all[layer][head], val_acc_all[layer][head], clf = probe_single_case_multinomial(X_train, y_train, X_test, y_test, args.seed, verbose=False)
            coefs_all[layer][head] = clf.coef_
    
    plot_heatmap(train_acc_all, "Probe Train Acc.", save_path="data/results/probe/%s_%s_multinomial_train_acc" % (args.dynamic, args.variable))
    plot_heatmap(val_acc_all, "Probe Val Acc.", save_path="data/results/probe/%s_%s_multinomial_val_acc" % (args.dynamic, args.variable))
    
    np.save("data/results/probe/%s_%s_multinomial_train_acc.npy" % (args.dynamic, args.variable), train_acc_all)
    np.save("data/results/probe/%s_%s_multinomial_val_acc.npy" % (args.dynamic, args.variable), val_acc_all)
    np.save("data/results/probe/%s_%s_multinomial_coef.npy" % (args.dynamic, args.variable), coefs_all)




