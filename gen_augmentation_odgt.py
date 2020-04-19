import json
import os
import os.path as P
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image


def getExt(path):
    return path.split('.')[-1]


def doOneSample(dct, transform, img_name, root, dataset):
    out_dct = dict()

    img = Image.open(P.join(root, dct['fpath_rgb']))
    img = transform(img)
    new_path = P.join(dataset, 'images/rgb/', '{}.{}'.format(img_name, getExt(dct['fpath_rgb'])))
    out_dct['fpath_rgb'] = new_path
    img.save(P.join(root, new_path))
    
    img = Image.open(P.join(root, dct['fpath_nir']))
    img = transform(img)
    new_path = P.join(dataset, 'images/nir/', '{}.{}'.format(img_name, getExt(dct['fpath_nir'])))
    out_dct['fpath_nir'] = new_path
    img.save(P.join(root, new_path))

    img = Image.open(P.join(root, dct['fpath_boundary']))
    img = transform(img)
    new_path = P.join(dataset, 'boundaries/', '{}.{}'.format(img_name, getExt(dct['fpath_boundary'])))
    out_dct['fpath_boundary'] = new_path
    img.save(P.join(root, new_path))

    img = Image.open(P.join(root, dct['fpath_mask']))
    img = transform(img)
    new_path = P.join(dataset, 'masks/', '{}.{}'.format(img_name, getExt(dct['fpath_mask'])))
    out_dct['fpath_mask'] = new_path
    img.save(P.join(root, new_path))

    out_dct['fpath_label'] = {}
    for k, v in dct['fpath_label'].items():
        img = Image.open(P.join(root, v))
        img = transform(img)
        new_path = P.join(dataset, 'labels/{}'.format(k), '{}.{}'.format(img_name, getExt(v)))
        out_dct['fpath_label'][k] = new_path
        img.save(P.join(root, new_path))

    out_dct['classes'] = dct['classes']

    return out_dct


parser = argparse.ArgumentParser(description="Generate augmentation odgt for Agri-Vision dataset")
parser.add_argument('--root', '-r')
parser.add_argument('--dataset', '-d')
parser.add_argument('--input-odgt', '-i')
parser.add_argument('--odgt-path', '-o')
parser.add_argument('--aug-class', '-c')

args = parser.parse_args()

classes = ['cloud_shadow',
           'double_plant',
           'planter_skip',
           'standing_water',
           'waterway',
           'weed_cluster']

os.makedirs(P.join(args.root, args.dataset, 'images/rgb/'), exist_ok=True)
os.makedirs(P.join(args.root, args.dataset, 'images/nir/'), exist_ok=True)
os.makedirs(P.join(args.root, args.dataset, 'boundaries/'), exist_ok=True)
os.makedirs(P.join(args.root, args.dataset, 'masks/'), exist_ok=True)
for i in classes:
    os.makedirs(P.join(args.root, args.dataset, 'labels/{}'.format(i)), exist_ok=True)

transforms = [lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
              lambda img: img.transpose(Image.ROTATE_90), lambda img: img.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT),
              lambda img: img.transpose(Image.ROTATE_180), lambda img: img.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT),
              lambda img: img.transpose(Image.ROTATE_270), lambda img: img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)]

cnt = 0

with open(args.odgt_path, mode='w') as file:
    lines = open(args.input_odgt).readlines()

    for line in tqdm(lines):
        dct = json.loads(line)

        if args.aug_class in dct['classes']:
            name = dct['fpath_rgb'].split('/')[-1][:-4]
            for i, transform in enumerate(transforms):
                out_dct = doOneSample(dct, transform, '{}_{}'.format(name, i + 1), args.root, args.dataset)
                json.dump(out_dct, file)
                print(file=file)

            cnt += 1

print('Count: ', cnt)