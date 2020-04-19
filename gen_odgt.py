import json
import os
import os.path as P
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Generate odgt for Agri-Vision dataset")
parser.add_argument('--root', '-r')
parser.add_argument('--dataset', '-d')
parser.add_argument('--odgt-path', '-o')
parser.add_argument('--test', '-t', default=False, action='store_true')

args = parser.parse_args()

image_dir = P.join(args.dataset, 'images')
boundary_dir = P.join(args.dataset, 'boundaries')
mask_dir = P.join(args.dataset, 'masks')
label_dir = P.join(args.dataset, 'labels')

# from ipdb import set_trace; set_trace()
filename = [i[:-4] for i in os.listdir(P.join(args.root, image_dir, 'rgb'))]

if not args.test:
    classes = os.listdir(P.join(args.root, label_dir))

with open(args.odgt_path, mode='w') as file:
    for i in tqdm(filename):
        dct = {}
        dct['fpath_rgb'] = P.join(image_dir, 'rgb' , i + '.jpg')
        dct['fpath_nir'] = P.join(image_dir, 'nir' , i + '.jpg')

        dct['fpath_boundary'] = P.join(boundary_dir, i + '.png')
        dct['fpath_mask'] = P.join(mask_dir, i + '.png')

        if not args.test:
            dct['fpath_label'] = {c: P.join(label_dir, c, i + '.png') for c in classes}

            labels, _classes = [], []
            # import ipdb; ipdb.set_trace()
            for class_name, path in dct['fpath_label'].items():
                label_img = np.array(Image.open(P.join(args.root, path))) / 255
                if label_img.any():
                    _classes.append(class_name)
                labels.append(label_img)

            if (~np.sum(labels, axis=0).astype('bool')).any():
                _classes.append('background')

            dct['classes'] = _classes

        json.dump(dct, file)
        print(file=file)