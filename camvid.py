""" camvid.py
    This script is to convert CamVid dataset to tfrecord format.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import make_dirs
from tfrecord import *

class_names = np.array(['back_ground','red', 'yellow', 'green', 'blue'])

cmap = np.array([
    [0, 0, 0],#back
    [128, 0, 0],#other
    [128, 128, 0],#Iron ore
    [0, 128, 0],  #Coal
    [0, 0, 128]]  #container
    )

# cb = np.array([
#     0.2595,
#     0.1826,
#     4.5640,
#     0.1417,
#     0.5051,
#     0.3826,
#     9.6446,
#     1.8418,
#     6.6823,
#     6.2478,
#     3.0,
#     7.3614])
cb = np.array([
    0.2595,
    4.5640,
    6.6823,
    6.2478,
    7.3614])

label_info = {
    'name': class_names,
    'num_class': len(class_names),
    'id': np.arange(len(class_names)),
    'cmap': cmap,
    'cb': cb
}


def parse(line, root):
    line = line.rstrip()
    line = line.replace('/SegNet/CamVid', root)
    return line.split(' ')


def load_path(txt_path, root):
    with open(txt_path) as f:
        img_gt_pairs = [parse(line, root) for line in f.readlines()]
    return img_gt_pairs


def load_splited_path(txt_path, root):
    images = []
    labels = []
    with open(txt_path) as f:
        for line in f.readlines():
            image_path, label_path = parse(line, root)
            images.append(image_path)
            labels.append(label_path)
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='./tmp/data/camvid',
        help='Path to save tfrecord')
    parser.add_argument('--indir', type=str, default='./data',
        help='Dataset path.')
    parser.add_argument('--target', type=str, default='train',
        help='train, val, test')
    args = parser.parse_args()

    txt_path = os.path.join(args.indir, '{}.txt'.format(args.target))
    print(txt_path)
    pairs = load_path(txt_path, args.indir)
    print(pairs)
    print(len(pairs))
    fname = 'camvid-{}.tfrecord'.format(args.target)
    convert_to_tfrecord(pairs, args.outdir, fname)


if __name__ == '__main__':
    main()
