# -*- coding: utf-8 -*-
# @Author: Chenglin Yu
# @Date  : 2022/11/3
# @Desc  :

import random

import torch
from PIL import Image
from torch.utils.data import Dataset

modes = {'clean_train': 'clean_train',
         'clean_val': 'clean_val',
         'clean_test': 'clean_test',
         'noisy_train': 'noisy_train'}


def sample_traning_set(train_imgs, labels, num_class, num_samples):
    random.shuffle(train_imgs)
    class_num = torch.zeros(num_class)
    sampled_train_imgs = []
    for impath in train_imgs:
        label = labels[impath]
        if class_num[label] < (num_samples / num_class):
            sampled_train_imgs.append(impath)
            class_num[label] += 1
        if len(sampled_train_imgs) >= num_samples:
            break
    return sampled_train_imgs


class Clothing1M(Dataset):
    def __init__(self, root, transform, mode, l_noisy_train_ratio=1.0,
                 num_samples=0, clean_train_key_list_txt='clean_train_key_list.txt',
                 noisy_train_key_list_txt='noisy_train_key_list.txt',
                 clean_val_key_list_txt='clean_val_key_list.txt',
                 clean_test_key_list_txt='clean_test_key_list.txt'
                 ):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.noisy_labels = {}  # all {img:noisy_label}
        self.clean_labels = {}  # all {img: clean_label}
        self.train_image_paths = None  # for score function
        self.noisy_train_paths = None
        self.clean_train_paths = None
        self.clean_val_paths = None
        self.clean_test_paths = None
        self.num_classes = 14
        noisy_label_kv_txt = 'noisy_label_kv.txt'
        clean_label_kv_txt = 'clean_label_kv.txt'
        self.inter_img_paths = []

        with open(f'{self.root}/{noisy_label_kv_txt}') as f:
            lines = f.read().splitlines()

            for l in lines:
                entry = l.split()
                img_path = f'{self.root}/{entry[0]}'
                # print(f'img_path:{img_path}')
                # img = Image.open(fp=img_path).convert('RGB')
                self.noisy_labels[img_path] = int(entry[1])
            print(f'len of noisy_label_kv_images: {len(lines)}')
        with open(f'{self.root}/{clean_label_kv_txt}', 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = f'{self.root}/{entry[0]}'
                self.clean_labels[img_path] = int(entry[1])
        # Clean size 72409 , Noisy size: 1037497. Clean/noisy intersection: 37497
        clean_label_img_paths = list(self.clean_labels.keys())
        noisy_label_img_paths = list(self.noisy_labels.keys())
        inter_img_paths = list(set(clean_label_img_paths) & set(noisy_label_img_paths))
        self.inter_img_paths = inter_img_paths

        if mode == modes['clean_train']:
            clean_train_paths = []
            with open(f'{self.root}/{clean_train_key_list_txt}', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = f'{self.root}/{l}'
                    clean_train_paths.append(img_path)
                self.clean_train_paths = clean_train_paths
                print(f'len of clean_train:{len(self.clean_train_paths)}')
                # self.train_image_paths = sample_traning_set(clean_train_paths, self.clean_labels,
                #                                             num_class=self.num_classes, num_samples=num_samples)
        elif self.mode == modes['clean_val']:
            clean_val_paths = []
            with open(f'{self.root}/{clean_val_key_list_txt}', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = f'{self.root}/{l}'
                    clean_val_paths.append(img_path)
                self.clean_val_paths = clean_val_paths
        elif self.mode == modes['clean_test']:
            clean_test_paths = []
            with open(f'{self.root}/{clean_test_key_list_txt}', 'r') as f:
                lines = f.read().split()
                for l in lines:
                    img_path = f'{self.root}/{l}'
                    clean_test_paths.append(img_path)
                self.clean_test_paths = clean_test_paths
            print(f'len of clean test {len(self.clean_test_paths)}')
            # load test data
            self.clean_data = []
            self.clean_targets = []
            for img_idx in range(len(self.clean_test_paths)):
                img_path = self.clean_test_paths[img_idx]
                target = self.clean_labels[img_path]
                image = Image.open(img_path).convert('RGB')
                self.clean_data.append(image)
                self.clean_targets.append(target)
        elif self.mode == modes['noisy_train']:
            # here to do noisy_train_key_list.txt!
            noisy_train_paths = []
            noisy_labels = []
            clean_labels = []
            with open(f'{self.root}/{noisy_train_key_list_txt}', 'r') as f:
                lines = f.read().split()
                use_len = int(len(lines) * l_noisy_train_ratio)
                random.seed(0)
                random.shuffle(lines)
                lines = lines[:use_len]
                print(f'lines[0]:{lines[0]}')
                import numpy as np
                # corrupted_num = int(use_len / 2)
                got_corrupted_num = 0
                got_overall_num = 0
                print(f'len lines {len(lines)}')
                for l in lines:
                    img_path = f'{self.root}/{l}'
                    noisy_train_paths.append(img_path)
                    l_key = f'{self.root}/{l}'
                self.noisy_train_paths = noisy_train_paths
                noise_or_not = np.transpose(noisy_labels) != np.transpose(clean_labels)
                print(f'len_of noisy images to evaluate {len(self.noisy_train_paths)}')

            # self.noisy_train_paths = list(self.noisy_labels.keys())
            print(f'self.noisy_train_paths[0] {self.noisy_train_paths[0]}')
        else:
            print('hello world')

            # plt.figure()
            # plt.imshow(img)
            # plt.show()
            # print('show one image')

    def __getitem__(self, index):
        if self.mode == modes['clean_train']:
            img_path = self.clean_train_paths[index]
            target = self.clean_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index
        elif self.mode == modes['clean_val']:
            img_path = self.clean_val_paths[index]
            target = self.clean_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index
        elif self.mode == modes['clean_test']:
            # img_path = self.clean_test_paths[index]
            # target = self.clean_labels[img_path]
            # image = Image.open(img_path).convert('RGB')
            image = self.clean_data[index]
            target = self.clean_targets[index]
            img = self.transform(image)
            return img, target, index
        elif self.mode == modes['noisy_train']:
            img_path = self.noisy_train_paths[index]
            target = self.noisy_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index
        else:
            raise Exception(f'mode {self.mode} not defined!')

    def __len__(self):
        if self.mode == modes['clean_train']:
            return len(self.clean_train_paths)
        elif self.mode == modes['clean_val']:
            return len(self.clean_val_paths)
        elif self.mode == modes['clean_test']:
            return len(self.clean_test_paths)
        elif self.mode == modes['noisy_train']:
            return len(self.noisy_train_paths)
        else:
            print(f'self.mode {self.mode}')
            raise Exception(f'mode {self.mode} not defined!')


if __name__ == '__main__':
    clothing1m = Clothing1M(root='data/clothing1M/',
                            transform=None, mode='clean_train')
