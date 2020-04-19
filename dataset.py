import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

np.random.seed(42)

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # down sampling rate of segm img
        self.img_downsampling_rate = opt.img_downsampling_rate
        # down sampling rate of segm label
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        self.parse_input_list(odgt, **kwargs)

        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.nir_normalize = transforms.Normalize(
            mean=[0.485],
            std=[0.229]
        )

        self.classes = ['background',
                        'cloud_shadow',
                        'double_plant',
                        'planter_skip',
                        'standing_water',
                        'waterway',
                        'weed_cluster']

        self.img_down_size = lambda img: imresize(
            img,
            (int(img.size[0] / self.img_downsampling_rate), int(img.size[1] / self.img_downsampling_rate)),
            interp='bilinear')

        self.label_down_size = lambda label: imresize(
            label,
            (int(label.size[0] / self.segm_downsampling_rate), int(label.size[1] / self.segm_downsampling_rate)),
            interp='nearest')

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        # if isinstance(odgt, list):
        #     self.list_sample = odgt
        # elif isinstance(odgt, str):
        #     self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
        
        if isinstance(odgt, str):
            odgt = [odgt]

        self.list_sample = []
        for o in odgt:
            self.list_sample += [json.loads(x.rstrip()) for x in open(o, 'r')]

        self.list_sample = np.random.permutation(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        # self.list_sample = self.list_sample * 5

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __len__(self):
        return self.num_sample

    def img_transform(self, rgb, nir):
        # 0-255 to 0-1
        rgb = np.float32(np.array(rgb)) / 255.
        nir = np.expand_dims(np.float32(np.array(nir)), axis=2) / 255.

        rgb = rgb.transpose((2, 0, 1))
        nir = nir.transpose((2, 0, 1))

        rgb = self.rgb_normalize(torch.from_numpy(rgb))
        nir = self.nir_normalize(torch.from_numpy(nir))

        if self.channels == 'rgbn':
            img = torch.cat([rgb, nir], axis=0)
        elif self.channels == 'rgb':
            img = rgb
        elif self.channels == 'nir3':
            img = torch.cat([nir, nir, nir], axis=0)
        elif self.channels == 'nir4':
            img = torch.cat([nir, nir, nir, nir], axis=0)
        elif self.channels == 'rgbr':
            img = torch.cat([rgb, rgb[0: 1]], axis=0)
        else:
            raise NotImplementedError

        return img

    def vmask_transform(self, boundary, mask):
        boundary = np.array(boundary) / 255.
        mask = np.array(mask) / 255.

        boundary = torch.from_numpy(boundary).long()
        mask = torch.from_numpy(np.array(mask)).long()

        return boundary * mask

    def label_transform(self, label_imgs):
        labels = [torch.from_numpy(np.array(img) / 255.).long() for img in label_imgs]
        labels = torch.stack(labels, dim=0)

        sumed = labels.sum(dim=0, keepdim=True)
        bg_channel = torch.zeros_like(sumed)
        bg_channel[sumed == 0] = 1

        return torch.cat((bg_channel, labels), dim=0).float()


class AgriTrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, channels='rgbn', reverse=False, **kwargs):
        super(AgriTrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

        self.channels = channels
        self.reverse = reverse

        self.shift_limit = tuple(opt.shift_limit)
        self.scale_limit = tuple(opt.scale_limit)
        self.aspect_limit = tuple(opt.aspect_limit)

        self.hue_shift_limit = tuple(opt.hue_shift_limit)
        self.sat_shift_limit = tuple(opt.sat_shift_limit)
        self.val_shift_limit = tuple(opt.val_shift_limit)

        if self.reverse:
            self.class_weight, self.sum_weight = self.get_weight()
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        if self.reverse:
            sampled_class_index = self.reverse_sample()
            sampled_indexes = self.class_dict[sampled_class_index]
            index = np.random.choice(sampled_indexes)

        sample_odgt = self.list_sample[index]

        rgb_path = os.path.join(self.root_dataset, sample_odgt['fpath_rgb'])
        nir_path = os.path.join(self.root_dataset, sample_odgt['fpath_nir'])

        rgb = Image.open(rgb_path).convert('RGB')
        nir = Image.open(nir_path).convert('L') #Greyscale

        assert rgb.size == nir.size

        rgb, nir = self.img_down_size(rgb), self.img_down_size(nir)

        boundary_path = os.path.join(self.root_dataset, sample_odgt['fpath_boundary'])
        mask_path = os.path.join(self.root_dataset, sample_odgt['fpath_mask'])

        boundary = Image.open(boundary_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        boundary, mask = self.label_down_size(boundary), self.label_down_size(mask)

        label_paths = sample_odgt['fpath_label']
        label_imgs = []
        for c in self.classes[1:]:
            mask_path = os.path.join(self.root_dataset, label_paths[c])
            label_imgs.append(Image.open(mask_path).convert('L'))
        # import ipdb; ipdb.set_trace()

        rgb = np.array(rgb)
        nir = np.array(nir)
        boundary = np.array(boundary)
        mask = np.array(mask)
        label_imgs = [np.array(self.label_down_size(label_img)) for label_img in label_imgs]

        rgb = self.randomHueSaturationValue(
            rgb,
            hue_shift_limit=self.hue_shift_limit,
            sat_shift_limit=self.sat_shift_limit,
            val_shift_limit=self.val_shift_limit)

        rgb, nir, boundary, mask, label_imgs = self.randomShiftScaleRotate(
            rgb, nir, boundary, mask, label_imgs,
            shift_limit=self.shift_limit,
            scale_limit=self.scale_limit,
            aspect_limit=self.aspect_limit)

        rgb, nir, boundary, mask, label_imgs = self.randomHorizontalFlip(
            rgb, nir, boundary, mask, label_imgs)

        rgb, nir, boundary, mask, label_imgs = self.randomVerticalFlip(
            rgb, nir, boundary, mask, label_imgs)

        rgb, nir, boundary, mask, label_imgs = self.randomRotate90(
            rgb, nir, boundary, mask, label_imgs)

        img = self.img_transform(rgb, nir)
        valid_mask = self.vmask_transform(boundary, mask)
        label = self.label_transform(label_imgs)

        info = rgb_path.split('/')[-1]

        label *= valid_mask.unsqueeze(dim=0)

        return img, valid_mask, label, info

    def get_weight(self):
        num_list = [0] * len(self.classes)

        for sample in self.list_sample:
            classes_ = sample['classes']

            for class_ in classes_:
                class_id = self.classes.index(class_)
                num_list[class_id] += 1

        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def reverse_sample(self):
        rand_number, now_sum = np.random.random() * self.sum_weight, 0
        for i in range(len(self.list_sample)):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def _get_class_dict(self):
        class_dict = dict()
        for i, sample in enumerate(self.list_sample):
            classes_ = sample['classes']

            for class_ in classes_:
                class_id = self.classes.index(class_)

                if class_id not in class_dict:
                    class_dict[class_id] = []
                class_dict[class_id].append(i)

        return class_dict

    def randomHueSaturationValue(self, image,
                                 hue_shift_limit=(-180, 180),
                                 sat_shift_limit=(-255, 255),
                                 val_shift_limit=(-255, 255), u=0.5):
        if np.random.random() < u:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            #image = cv2.merge((s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image

    def randomShiftScaleRotate(self, rgb, nir, boundary, mask, labels,
                               shift_limit=(-0.0, 0.0),
                               scale_limit=(-0.0, 0.0),
                               rotate_limit=(-0.0, 0.0),
                               aspect_limit=(-0.0, 0.0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
        if np.random.random() < u:
            height, width, _ = rgb.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            rgb = cv2.warpPerspective(rgb, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                      borderValue=(
                                            0, 0,
                                            0,))
            nir = cv2.warpPerspective(nir, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                      borderValue=(
                                            0))
            boundary = cv2.warpPerspective(boundary, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                       borderValue=(
                                            0))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                       borderValue=(
                                            0))

            for i in range(len(labels)):
                labels[i] = cv2.warpPerspective(labels[i], mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                        borderValue=(
                                                0))

        return rgb, nir, boundary, mask, labels

    def randomHorizontalFlip(self, rgb, nir, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = cv2.flip(rgb, 1)
            nir = cv2.flip(nir, 1)
            boundary = cv2.flip(boundary, 1)
            mask = cv2.flip(mask, 1)

            for i in range(len(labels)):
                labels[i] = cv2.flip(labels[i], 1)

        return rgb, nir, boundary, mask, labels

    def randomVerticalFlip(self, rgb, nir, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = cv2.flip(rgb, 0)
            nir = cv2.flip(nir, 0)
            boundary = cv2.flip(boundary, 0)
            mask = cv2.flip(mask, 0)

            for i in range(len(labels)):
                labels[i] = cv2.flip(labels[i], 0)

        return rgb, nir, boundary, mask, labels

    def randomRotate90(self, rgb, nir, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = np.rot90(rgb)
            nir = np.rot90(nir)
            boundary = np.rot90(boundary)
            mask = np.rot90(mask)

            for i in range(len(labels)):
                labels[i] = np.rot90(labels[i])

        return rgb, nir, boundary, mask, labels


class AgriValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, channels=['rgb', 'nir'], ret_rgb_img=False, **kwargs):
        super(AgriValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

        # down sampling rate of segm img
        self.img_downsampling_rate = opt.img_downsampling_rate
        # down sampling rate of segm label
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.nir_normalize = transforms.Normalize(
            mean=[0.485],
            std=[0.229]
        )

        self.ret_rgb_img = ret_rgb_img
        self.channels = channels

    def __getitem__(self, index):
        sample_odgt = self.list_sample[index]

        rgb_path = os.path.join(self.root_dataset, sample_odgt['fpath_rgb'])
        nir_path = os.path.join(self.root_dataset, sample_odgt['fpath_nir'])

        rgb = Image.open(rgb_path).convert('RGB')
        nir = Image.open(nir_path).convert('L') #Greyscale

        assert rgb.size == nir.size

        rgb, nir = self.img_down_size(rgb), self.img_down_size(nir)

        # image transform, to torch float tensor 4xHxW
        img = self.img_transform(rgb, nir)

        boundary_path = os.path.join(self.root_dataset, sample_odgt['fpath_boundary'])
        mask_path = os.path.join(self.root_dataset, sample_odgt['fpath_mask'])

        boundary = Image.open(boundary_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        boundary, mask = self.label_down_size(boundary), self.label_down_size(mask)
        valid_mask = self.vmask_transform(boundary, mask)

        label_paths = sample_odgt['fpath_label']
        label_imgs = []
        for c in self.classes[1:]:
            mask_path = os.path.join(self.root_dataset, label_paths[c])
            label_imgs.append(Image.open(mask_path).convert('L'))

        label_imgs = [self.label_down_size(label_img) for label_img in label_imgs]
        label = self.label_transform(label_imgs)

        info = rgb_path.split('/')[-1]

        if self.ret_rgb_img:
            return img, valid_mask, label, info, np.array(rgb)
        else:
            return img, valid_mask, label, info


class AgriTestDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(AgriTestDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485],
            std=[0.229, 0.224, 0.225, 0.229])

    def __getitem__(self, index):
        sample_odgt = self.list_sample[index]

        rgb_path = os.path.join(self.root_dataset, sample_odgt['fpath_rgb'])
        nir_path = os.path.join(self.root_dataset, sample_odgt['fpath_nir'])

        rgb = Image.open(rgb_path).convert('RGB')
        nir = Image.open(nir_path).convert('L') #Greyscale

        assert rgb.size == nir.size

        rgb, nir = self.img_down_size(rgb), self.img_down_size(nir)

        # image transform, to torch float tensor 4xHxW
        img = self.img_transform(rgb, nir)

        boundary_path = os.path.join(self.root_dataset, sample_odgt['fpath_boundary'])
        mask_path = os.path.join(self.root_dataset, sample_odgt['fpath_mask'])

        boundary = Image.open(boundary_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        boundary, mask = self.label_down_size(boundary), self.label_down_size(mask)
        valid_mask = self.vmask_transform(boundary, mask)

        info = rgb_path.split('/')[-1]

        return img, valid_mask, info

    def img_transform(self, rgb, nir):
        # 0-255 to 0-1
        rgb = np.float32(np.array(rgb)) / 255.
        nir = np.float32(np.array(nir)) / 255.

        img = np.concatenate((rgb, np.expand_dims(nir, axis=2)), axis=2) #shape as (512, 512, 4)

        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img
