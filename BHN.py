# -*- coding: utf-8 -*-
# @Desc  :
import json
import os

import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(".")

import time

import torch.nn.functional as F
from torch import nn

import utility

from pvalue.methods import Pvalues1
from utils_experiments import evaluate_all_methods, write_sel_clean

import argparse

from models import resnet as resnet_normal

from models import imagnet_resnet

########################################
from abc import ABC, abstractmethod
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import torch.optim as optim
from data.dataloader_clothing1M_BH import Clothing1M
from data.dataloader_clothing1M_BH import sample_traning_set
from torch.utils.data import DataLoader
from PIL import Image


class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str, dataset_name: str):
        super().__init__()
        self.root = root  # root path to data
        self.dataset_name = dataset_name
        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.calib_set = None
        self.test_set = None  # must be of type torch.utils.data.Dataset
        self.eval_set = None
        self.retrain_set = None

        self.prim_eval_set = None  #

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__


class BasedivideDataset(ABC):  # for divide loaders
    """Base divide dataset"""

    def __init__(self, root: str, dataset_name: str):
        super().__init__()
        self.root = root  # root path to data
        self.dataset_name = dataset_name
        self.original_train_set = None
        self.clean_train_set = None
        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.calib_set = None
        self.test_set = None  # must be of type torch.utils.data.Dataset
        self.eval_set = None
        self.original_retrain_set = None
        self.training_sample_method = None

    @abstractmethod
    def loader(self, mode: str, batch_size: int, shuffle_train=True, shuffle_test=False, mini_batch_num=15000,
               num_workers: int = 0) -> DataLoader:
        """Implement data loader of tyupe torch.utils.data.DataLoader for different mode dataset"""
        pass

    def __repr__(self):
        return self.__class__.__name__


class SampleClothing1MRetrain(Clothing1M):
    def __init__(self, noisy_train_set: Clothing1M, num_samples=0):
        self.noisy_train_set = noisy_train_set
        self.noisy_labels = noisy_train_set.noisy_labels
        self.transform = noisy_train_set.transform
        # self.clean_train_set = clean_train_set
        # self.clean_labels = clean_train_set.clean_labels
        # self.transform = clean_train_set.transform
        sample_time1 = time.time()
        self.train_image_paths = sample_traning_set(noisy_train_set.noisy_train_paths, noisy_train_set.noisy_labels,
                                                    num_class=noisy_train_set.num_classes, num_samples=num_samples)
        print(f'sample_train_image_time: {time.time() - sample_time1}')
        print(f'len(train_image_paths): {len(self.train_image_paths)}')
        print(f'self.train_image_paths[0] {self.train_image_paths[0]}')

    def __getitem__(self, index):
        img_path = self.train_image_paths[index]
        target = self.noisy_labels[img_path]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target, index

    def __len__(self):
        # return len(self.train_data)
        return len(self.train_image_paths)


class SampleClothing1MTrain(Clothing1M):
    def __init__(self, clean_train_set: Clothing1M, num_samples=0):
        self.clean_train_set = clean_train_set
        self.clean_labels = clean_train_set.clean_labels
        self.transform = clean_train_set.transform

        self.train_image_paths = sample_traning_set(clean_train_set.clean_train_paths, clean_train_set.clean_labels,
                                                    num_class=clean_train_set.num_classes, num_samples=num_samples)
        print(f'self.train_image_paths[0] {self.train_image_paths[0]}')

    def __getitem__(self, index):
        img_path = self.train_image_paths[index]
        target = self.clean_labels[img_path]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target, index

    def __len__(self):
        return len(self.train_image_paths)


class TorchvisionDatasetClothing1M(BasedivideDataset):
    def __init__(self, root: str):
        super().__init__(root)

    def loader(self, mode: str, batch_size: int, shuffle_train=True, shuffle_test=False, mini_batch_num=15000,
               num_workers: int = 0, training_sample_method: str = 'direct') -> DataLoader:
        retrain_loader = None
        if mode == 'clean_train':
            # self.train_set = SampleClothing1MTrain(clean_train_set=self.original_train_set, num_samples=1000)
            self.train_set = self.clean_train_set
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=n_jobs_dataloader)
            return train_loader
        elif mode == 'clean_test':
            test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=n_jobs_dataloader)
            return test_loader
        elif mode == 'clean_val':
            calib_loader = DataLoader(dataset=self.calib_set, batch_size=batch_size, shuffle=shuffle_test,
                                      num_workers=n_jobs_dataloader)
            return calib_loader
        elif mode == 'noisy_train':
            eval_loader = DataLoader(dataset=self.eval_set, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=n_jobs_dataloader)
            return eval_loader
        elif mode == 're_train':
            if self.original_retrain_set != None:
                if self.training_sample_method == 'sample':
                    t1 = time.time()
                    self.retrain_set = SampleClothing1MRetrain(noisy_train_set=self.original_retrain_set,
                                                               num_samples=mini_batch_num * batch_size)
                    print(f'sample num: {mini_batch_num * batch_size}')
                    print(f'sample time:{time.time() - t1}')

                elif self.training_sample_method == 'direct':
                    self.retrain_set = self.original_retrain_set
                else:
                    pass
                retrain_loader = DataLoader(dataset=self.retrain_set, batch_size=batch_size, shuffle=shuffle_train,
                                            num_workers=num_workers)
                return retrain_loader

        else:
            raise Exception(f'no such mode {mode}')


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str, dataset_name: str):
        super().__init__(root, dataset_name)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = None
        if self.train_set != None:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers)
        calib_loader = DataLoader(dataset=self.calib_set, batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers)
        eval_loader = DataLoader(dataset=self.eval_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)

        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        retrain_loader = None
        if self.retrain_set != None:
            retrain_loader = DataLoader(dataset=self.retrain_set, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=num_workers)
        prim_eval_loader = None
        if self.prim_eval_set != None:
            prim_eval_loader = DataLoader(dataset=self.prim_eval_set, batch_size=batch_size, shuffle=shuffle_test,
                                          num_workers=num_workers)
        return train_loader, calib_loader, eval_loader, test_loader, retrain_loader, prim_eval_loader


class MyCIFAR100(CIFAR100):
    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)
        if self.train:
            self.train_data = self.data
            self.train_labels = self.targets
        else:
            self.test_data = self.data
            self.test_labels = self.targets

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)
        if self.train:
            self.train_data = self.data
            self.train_labels = self.targets
        else:
            self.test_data = self.data
            self.test_labels = self.targets

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        # self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.print('Trainable parameters: {}'.format(params))
        self.print(self)


class BaseTrainerClothing1M(ABC):
    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: list, batch_size: int,
                 weight_decay: float, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass


class PvalueTrainerClothing1M(BaseTrainerClothing1M):
    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001,
                 n_epochs: int = 150,
                 lr_milestones: list = [], batch_size: int = 128, weight_decay: float = 1e-6,
                 n_jobs_dataloader: int = 0, momentum: float = 0.9, mini_batch_num=1000,
                 training_sample_method='direct', out_dir='out_dir', start_epoch=0, is_parallel=False):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay,
                         n_jobs_dataloader)
        self.out_dir = out_dir
        self.momentum = momentum
        self.training_sample_method = training_sample_method
        self.start_epoch = start_epoch
        self.is_parallel = is_parallel

        # Optimization parameters
        # self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.calib_scores = None
        self.calib_time = None
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.calib_acc = None
        self.eval_time = None
        self.eval_scores = None

        self.mini_batch_num = mini_batch_num

    def train(self, dataset: BasedivideDataset, net: BaseNet):
        # logger = logging.getLogger()
        if self.is_parallel:
            net = torch.nn.DataParallel(net)
        net = net.cuda()

        # train_loader, _, _, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        optimizer = None
        if self.optimizer_name == 'adam':
            # Set optimizer (Adam optimizer for now)
            optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   amsgrad=self.optimizer_name == 'amsgrad')
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        print(f'optimizer_name: {optimizer_name}')
        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        print('Starting training...')
        start_time = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            net.train()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            # Get train data loader every epoch.
            train_loader = dataset.loader(mode='clean_train', batch_size=self.batch_size,
                                          num_workers=self.n_jobs_dataloader)
            print('mode: clean_train')
            print(f'epoch:{epoch + 1}')
            for data in train_loader:
                inputs, labels, _ = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                _, outputs = net(inputs)
                loss_func = nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(outputs, labels)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            test_acc = self.test(dataset, net=net)
            net_state_dict = net.state_dict()
            print(f'save epoch {epoch + 1} model')
            save_checkpoint({'net_dict': net_state_dict}, out_dir=self.out_dir, filename=f'{epoch + 1}.pth.tar')
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} Test_acc: {:.5f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, test_acc))

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time)
        print('Finished training.')
        return net

    def re_train(self, dataset: BasedivideDataset, net: BaseNet):
        # logger = logging.getLogger()
        net = net.cuda()

        # Get train data loader

        optimizer = None
        if self.optimizer_name == 'adam':
            # Set optimizer (Adam optimizer for now)
            optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   amsgrad=self.optimizer_name == 'amsgrad')
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        print(f'optimizer_name: {optimizer_name}')
        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        print('Starting training...')
        start_time = time.time()
        for epoch in range(self.n_epochs):
            net.train()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            retrain_loader = dataset.loader(mode='re_train', batch_size=self.batch_size,
                                            mini_batch_num=self.mini_batch_num,
                                            num_workers=self.n_jobs_dataloader)
            print(f'epoch {epoch + 1}, train_length: {len(retrain_loader.dataset)}')
            for data in retrain_loader:
                inputs, labels, _ = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                _, outputs = net(inputs)
                loss_func = nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(outputs, labels)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            test_acc = self.test(dataset, net=net)
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} Test_acc: {:.5f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, test_acc))

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time)
        print('Finished training.')
        return net

    def evaluate(self, dataset: BasedivideDataset, net: BaseNet, score_type: str):
        net = net.cuda()
        # Get eval_loader
        eval_loader = dataset.loader(mode='noisy_train', batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # evaluate
        print('Start score evaluation set')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in eval_loader:
                inputs, labels, idx = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = net(inputs)
                if score_type == 'crossentropy':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                    scores = -loss_func(outputs, labels)
                elif score_type == 'crossentropy_cosin':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                    scores = -loss_func(outputs, labels)
                    logsoftmax_func = torch.nn.LogSoftmax(dim=1)
                    logsoftmax_func_output = logsoftmax_func(outputs)
                    num_class = outputs.shape[1]
                    logsoftmax_output_norm = torch.norm(logsoftmax_func_output, dim=1)
                    # logsoftmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    scores = scores / logsoftmax_output_norm
                elif score_type == 'softmax_norm':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    scores = softmax_output_norm
                elif score_type == 'softmax_confidence':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    # softmax_output_norm = torch.norm(softmax_output, dim=1)
                    # softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    # softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = cos_score
                elif score_type == 'real_cosin':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = cos_score
                elif score_type == 'max_softmax':
                    m = torch.nn.LogSoftmax(dim=1)
                    scores, _ = torch.max(m(outputs), dim=1)
                else:
                    pass
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()
                                            ))
        self.eval_time = time.time() - start_time
        print('Eval time: %.3f' % self.eval_time)
        self.eval_scores = idx_label_score

    def calib(self, dataset: BasedivideDataset, net: BaseNet, score_type: str):
        net = net.cuda()
        # Get calib data loader
        calib_loader = dataset.loader(mode='clean_val', batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Calib
        print('Start calibration...')
        start_time = time.time()
        idx_label_score = []
        total_size = 0.0
        correct = 0.0
        net.eval()
        with torch.no_grad():
            for data in calib_loader:
                inputs, labels, idx = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                total_size += outputs.shape[0]
                correct += torch.sum((preds == labels)).item()
                # # if loss_type == 'crossentropy':
                # loss_func = nn.CrossEntropyLoss(reduction='none')
                # # scores = -loss_func(outputs, labels)
                # scores = -loss_func / torch.sum(outputs ** 2)

                if score_type == 'crossentropy':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                    scores = -loss_func(outputs, labels)
                elif score_type == 'crossentropy_cosin':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                    scores = -loss_func(outputs, labels)
                    logsoftmax_func = torch.nn.LogSoftmax(dim=1)
                    logsoftmax_func_output = logsoftmax_func(outputs)
                    num_class = outputs.shape[1]
                    logsoftmax_output_norm = torch.norm(logsoftmax_func_output, dim=1)
                    # logsoftmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    scores = scores / logsoftmax_output_norm
                elif score_type == 'softmax_norm':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    scores = softmax_output_norm
                elif score_type == 'softmax_confidence':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    # softmax_output_norm = torch.norm(softmax_output, dim=1)
                    # softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    # softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = cos_score
                elif score_type == 'real_cosin':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = cos_score

                    # softmax_func = torch.nn.Softmax(dim=1)
                    # softmax_output = softmax_func(outputs)
                    # cos_score = F.nll_loss(torch.log(softmax_output + 1e-8), labels, reduction='none')
                    # loss_ = -torch.log(softmax_output + 1e-8)
                    # scores = -(cos_score - torch.mean(loss_, 1))
                elif score_type == 'max_softmax':
                    m = torch.nn.LogSoftmax(dim=1)
                    scores, _ = torch.max(m(outputs), dim=1)
                else:
                    pass
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()
                                            ))

        self.eval_time = time.time() - start_time
        print('score calibration set time: %.3f' % self.eval_time)
        self.calib_scores = idx_label_score

        self.calib_acc = correct / total_size
        print('calib_acc: %.5f' % self.calib_acc)

    def test(self, dataset: BasedivideDataset, net: BaseNet):
        '''print test accuracy'''
        net = net.cuda()
        # Get test data loader
        test_loader = dataset.loader(mode='clean_test', batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        start_time = time.time()
        idx_label_score = []
        net.eval()
        total_size = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                total_size += outputs.shape[0]
                correct += torch.sum((preds == labels)).item()
        acc = correct / total_size

        return acc


class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: list, batch_size: int,
                 weight_decay: float, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass


class PvalueTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001,
                 n_epochs: int = 150,
                 lr_milestones: list = [], batch_size: int = 128, weight_decay: float = 1e-6,
                 n_jobs_dataloader: int = 0, momentum: float = 0.9):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay,
                         n_jobs_dataloader)

        self.momentum = momentum

        # Optimization parameters
        # self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.calib_scores = None
        self.calib_time = None
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

        self.calib_acc = None

        self.eval_time = None
        self.eval_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        # logger = logging.getLogger()

        net = torch.nn.DataParallel(net)
        net = net.cuda()

        # Get train data loader
        train_loader, _, _, _, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        optimizer = None
        if self.optimizer_name == 'adam':
            # Set optimizer (Adam optimizer for now)
            optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   amsgrad=self.optimizer_name == 'amsgrad')
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        print(f'optimizer_name: {optimizer_name}')
        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        print('Starting training...')
        start_time = time.time()
        for epoch in range(self.n_epochs):
            net.train()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, _ = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize

                _, outputs = net(inputs)

                loss_type = 'crossentropy'
                if loss_type == 'crossentropy':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                else:
                    pass
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            # print(f'loss_epoch:{loss_epoch}')
            # print(f'n_batches:{n_batches}')
            # print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
            #       .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
            test_acc = self.test(dataset, net=net)
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} Test_acc: {:.5f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, test_acc))

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time)
        print('Finished training.')
        return net

    def re_train(self, dataset: BaseADDataset, net: BaseNet):
        # logger = logging.getLogger()
        net = net.cuda()

        # Get train data loader
        _, _, _, _, retrain_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        optimizer = None
        if self.optimizer_name == 'adam':
            # Set optimizer (Adam optimizer for now)
            optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   amsgrad=self.optimizer_name == 'amsgrad')
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        print(f'optimizer_name: {optimizer_name}')
        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        print('Starting training...')
        start_time = time.time()
        for epoch in range(self.n_epochs):
            net.train()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in retrain_loader:
                inputs, labels, _ = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                _, outputs = net(inputs)
                loss_func = nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(outputs, labels)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            # print(f'loss_epoch:{loss_epoch}')
            # print(f'n_batches:{n_batches}')
            # print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
            #       .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
            test_acc = self.test(dataset, net=net)
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} Test_acc: {:.5f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, test_acc))

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time)
        print('Finished training.')
        return net

    def test_eval_set(self, dataset: BaseADDataset, net: BaseNet):
        '''print test accuracy'''
        net = net.cuda()
        # Get test data loader
        _, _, _, _, _, prim_eval_loader = dataset.loaders(batch_size=self.batch_size,
                                                          num_workers=self.n_jobs_dataloader)

        start_time = time.time()
        idx_label_score = []
        net.eval()
        total_size = 0
        correct = 0
        with torch.no_grad():
            for data in prim_eval_loader:
                inputs, labels, idx = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                total_size += outputs.shape[0]
                correct += torch.sum((preds == labels)).item()
        acc = correct / total_size

        return acc

    def evaluate(self, dataset: BaseADDataset, net: BaseNet, score_type: str):
        net = net.cuda()
        # Get eval_loader
        _, _, eval_loader, _, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # evaluate
        print('Start score evaluation set')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in eval_loader:
                inputs, labels, idx = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = net(inputs)

                if score_type == 'crossentropy':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                    scores = -loss_func(outputs, labels)
                elif score_type == 'crossentropy_cosin':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                    scores = -loss_func(outputs, labels)
                    logsoftmax_func = torch.nn.LogSoftmax(dim=1)
                    logsoftmax_func_output = logsoftmax_func(outputs)
                    num_class = outputs.shape[1]
                    logsoftmax_output_norm = torch.norm(logsoftmax_func_output, dim=1)
                    # logsoftmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    scores = scores / logsoftmax_output_norm
                elif score_type == 'softmax_norm':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    # softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    # softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    # one_hot = torch.eye(num_class)
                    # one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    # cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = softmax_output_norm
                # elif score_type == 'cosin':
                #     softmax_func = torch.nn.Softmax(dim=1)
                #     softmax_output = softmax_func(outputs)
                #     num_class = outputs.shape[1]
                #     softmax_output_norm = torch.norm(softmax_output, dim=1)
                #     softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                #     softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                #     cos_score = F.nll_loss(torch.log(softmax_output + 1e-8), labels, reduction='none')
                #     scores = cos_score
                elif score_type == 'softmax_confidence':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    # softmax_output_norm = torch.norm(softmax_output, dim=1)
                    # softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    # softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = cos_score
                elif score_type == 'real_cosin':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = cos_score

                    # softmax_func = torch.nn.Softmax(dim=1)
                    # softmax_output = softmax_func(outputs)
                    # cos_score = F.nll_loss(torch.log(softmax_output + 1e-8), labels, reduction='none')
                    # loss_ = -torch.log(softmax_output + 1e-8)
                    # scores = -(cos_score - torch.mean(loss_, 1))
                elif score_type == 'max_softmax':
                    m = torch.nn.LogSoftmax(dim=1)
                    scores, _ = torch.max(m(outputs), dim=1)
                else:
                    pass
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()
                                            ))
        self.eval_time = time.time() - start_time
        print('Eval time: %.3f' % self.eval_time)
        self.eval_scores = idx_label_score

    def calib(self, dataset: BaseADDataset, net: BaseNet, score_type: str):
        net = net.cuda()
        # Get calib data loader
        _, calib_loader, _, _, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Calib
        print('Start calibration...')
        start_time = time.time()
        idx_label_score = []
        total_size = 0.0
        correct = 0.0
        net.eval()
        with torch.no_grad():
            for data in calib_loader:
                inputs, labels, idx = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                total_size += outputs.shape[0]
                correct += torch.sum((preds == labels)).item()
                # loss_func = nn.CrossEntropyLoss(reduction='none')

                if score_type == 'crossentropy':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                    scores = -loss_func(outputs, labels)
                elif score_type == 'crossentropy_cosin':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_func(outputs, labels)
                    scores = -loss_func(outputs, labels)
                    logsoftmax_func = torch.nn.LogSoftmax(dim=1)
                    logsoftmax_func_output = logsoftmax_func(outputs)
                    num_class = outputs.shape[1]
                    logsoftmax_output_norm = torch.norm(logsoftmax_func_output, dim=1)
                    # logsoftmax_output_norm = logsoftmax_output_norm.reshape(-1, 1)
                    scores = scores / logsoftmax_output_norm
                # elif score_type == 'cosin':
                #     softmax_func = torch.nn.Softmax(dim=1)
                #     softmax_output = softmax_func(outputs)
                #     num_class = outputs.shape[1]
                #     softmax_output_norm = torch.norm(softmax_output, dim=1)
                #     softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                #     softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                #     cos_score = F.nll_loss(torch.log(softmax_output + 1e-8), labels, reduction='none')
                #     scores = cos_score
                elif score_type == 'softmax_confidence':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    # softmax_output_norm = torch.norm(softmax_output, dim=1)
                    # softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    # softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = cos_score
                elif score_type == 'real_cosin':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = cos_score
                elif score_type == 'real_cosin2':
                    softmax_func = torch.nn.LogSoftmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    one_hot = torch.eye(num_class)
                    one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = -cos_score
                elif score_type == 'softmax_norm':
                    softmax_func = torch.nn.Softmax(dim=1)
                    softmax_output = softmax_func(outputs)
                    num_class = outputs.shape[1]
                    softmax_output_norm = torch.norm(softmax_output, dim=1)
                    # softmax_output_norm = softmax_output_norm.reshape(-1, 1)
                    # softmax_output = softmax_output / softmax_output_norm.repeat(1, num_class)
                    # one_hot = torch.eye(num_class)
                    # one_hot_label = torch.tensor(one_hot[labels]).cuda()
                    # cos_score = torch.sum(softmax_output * one_hot_label, dim=1)
                    scores = softmax_output_norm
                elif score_type == 'max_softmax':
                    m = torch.nn.LogSoftmax(dim=1)
                    scores, _ = torch.max(m(outputs), dim=1)
                else:
                    pass

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()
                                            ))

        self.eval_time = time.time() - start_time
        print('Eval time: %.3f' % self.eval_time)
        self.calib_scores = idx_label_score

        self.calib_acc = correct / total_size
        print('calib_acc: %.5f' % self.calib_acc)

    def test(self, dataset: BaseADDataset, net: BaseNet):
        '''print test accuracy'''
        net = net.cuda()
        # Get test data loader
        _, _, _, test_loader, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        start_time = time.time()
        idx_label_score = []
        net.eval()
        total_size = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                total_size += outputs.shape[0]
                correct += torch.sum((preds == labels)).item()
        acc = correct / total_size

        return acc

    # def test_clothing1M(self, dataset: BasedivideDataset, net: BaseNet):
    #     '''print test accuracy'''
    #     net = net.cuda()
    #     # Get test data loader
    #     _, _, _, test_loader, _ = dataset.loader(batch_size=self.batch_size, shuffle_train=False, shuffle_test=True,
    #                                              num_workers=self.n_jobs_dataloader)
    #
    #     start_time = time.time()
    #     idx_label_score = []
    #     net.eval()
    #     total_size = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for data in test_loader:
    #             inputs, labels, idx = data
    #             inputs = inputs.cuda()
    #             labels = labels.cuda()
    #             _, outputs = net(inputs)
    #             _, preds = torch.max(outputs, 1)
    #             total_size += outputs.shape[0]
    #             correct += torch.sum((preds == labels)).item()
    #     acc = correct / total_size
    #
    #     return acc


class CIFAR10_LeNet_ELU(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def get_keys_from_dict(aa, keys=[]):
    from operator import itemgetter
    # aa = {1: 2, 3: 4, 5: 6}
    # print(aa)
    # keys = [1, 3]
    # need make sure keys in dict_key
    out = itemgetter(*keys)(aa)
    return out


def intliststr_to_list(intliststr):
    import re
    a = re.split(",|\[|\]", intliststr)
    result = []
    for b in a:
        if b != '':
            result.append(int(b))
    return result


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = (
        'resnet18', 'resnet34', 'resnet50', '100class_resnet18',
        '100class_resnet34', '100class_resnet50', '14class_resnet18', '14class_resnet34', '14class_resnet50',
        '9layerCNN_10class', '9layerCNN_100class', '9layerCNN_14class', '14class_resnet18_pretrained',
        '14class_resnet34_pretrained', '14class_resnet50_pretrained')
    assert net_name in implemented_networks

    net = None
    print(f'net_name:{net_name}')

    if net_name == 'resnet18':
        net = resnet_normal.ResNet18(num_classes=10)

    elif net_name == 'resnet34':
        net = resnet_normal.ResNet34(num_classes=10)

    elif net_name == 'resnet50':
        net = resnet_normal.ResNet50(num_classes=10)

    elif net_name == '100class_resnet18':
        net = resnet_normal.ResNet18(num_classes=100)
    elif net_name == '100class_resnet34':
        net = resnet_normal.ResNet34(num_classes=100)
    elif net_name == '100class_resnet50':
        net = resnet_normal.ResNet50(num_classes=100)

    elif net_name == '14class_resnet18':
        net = imagnet_resnet.resnet18(num_classes=14)
    elif net_name == '14class_resnet34':
        net = imagnet_resnet.resnet34(num_classes=14)
    elif net_name == '14class_resnet50':
        net = imagnet_resnet.resnet50(num_classes=14)

    elif net_name == '14class_resnet18_pretrained':
        net = imagnet_resnet.resnet18(pretrained=True)
        net.fc = nn.Linear(2048, out_features=14)
    elif net_name == '14class_resnet34_pretrained':
        net = imagnet_resnet.resnet34(pretrained=True)
        net.fc = nn.Linear(2048, out_features=14)
    elif net_name == '14class_resnet50_pretrained':
        # net = torchvision_models.resnet50(pretrained=True)
        # net.fc = nn.Linear(2048, out_features=14)

        net = imagnet_resnet.resnet50(pretrained=True)
        net.fc = nn.Linear(2048, out_features=14)


    else:
        pass
    return net


def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)

    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()


def save_checkpoint(state, out_dir, filename='checkpoint_outdir.pth.tar'):
    filepath = os.path.join(out_dir, filename)
    torch.save(state, filepath)


class ScoreFuncHandler(object):
    """A class for handling score function.
    """

    def __init__(self, out_dir='out_dir'):
        """Inits ScoreFuncHandler with one of the two objectives and hyperparameter nu."""
        self.out_dir = out_dir
        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
            'calib_scores': None,
            'eval_scores': None
        }

    def set_network(self, net_name, loaded_model=None):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.net = build_network(net_name)
        # self.net = torch.nn.DataParallel(self.net)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: list = [], batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0, training_sample_method='direct', start_epoch=0, is_parallel=False):
        """Trains the Deep SVDD model on the training data."""

        if lr_milestones is None:
            lr_milestones = []
        self.optimizer_name = optimizer_name
        if dataset.dataset_name in ['cifar10', 'cifar100']:
            self.trainer = PvalueTrainer(optimizer_name, lr=lr,
                                         n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                         weight_decay=weight_decay,
                                         n_jobs_dataloader=n_jobs_dataloader)
        elif dataset.dataset_name == 'clothing1M':
            self.trainer = PvalueTrainerClothing1M(optimizer_name, lr=lr, n_epochs=n_epochs,
                                                   lr_milestones=lr_milestones, batch_size=batch_size,
                                                   weight_decay=weight_decay,
                                                   n_jobs_dataloader=n_jobs_dataloader,
                                                   training_sample_method=training_sample_method,
                                                   out_dir=self.out_dir, start_epoch=start_epoch,
                                                   is_parallel=is_parallel)
        else:
            print('No such dataset')
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time

    def re_train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001,
                 n_epochs: int = 50,
                 lr_milestones: list = [], batch_size: int = 128, weight_decay: float = 1e-6,
                 device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        if lr_milestones is None:
            lr_milestones = []
        self.optimizer_name = optimizer_name

        self.trainer = PvalueTrainer(optimizer_name, lr=lr,
                                     n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                     weight_decay=weight_decay,
                                     n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.re_train(dataset, self.net)

    def retrainClothing1M(self, dataset: BasedivideDataset, optimizer_name: str = 'adam', lr: float = 0.001,
                          n_epochs: int = 50,
                          lr_milestones: list = [], batch_size: int = 128, weight_decay: float = 1e-6,
                          device: str = 'cuda',
                          n_jobs_dataloader: int = 0, momentum=0.9, mini_batch_num=1500,
                          training_sample_method='direct'):
        if lr_milestones is None:
            lr_milestones = []
        self.optimizer_name = optimizer_name
        self.trainer = PvalueTrainerClothing1M(optimizer_name, lr=lr,
                                               n_epochs=n_epochs, lr_milestones=lr_milestones,
                                               batch_size=batch_size,
                                               weight_decay=weight_decay,
                                               n_jobs_dataloader=n_jobs_dataloader, momentum=momentum,
                                               mini_batch_num=mini_batch_num,
                                               training_sample_method=training_sample_method)
        # Get the model
        self.net = self.trainer.re_train(dataset, self.net)

    def calib(self, dataset: BaseADDataset, score_type):
        if self.trainer is None:
            if dataset.dataset_name in ['cifar10', 'cifar100']:
                self.trainer = PvalueTrainer(n_jobs_dataloader=n_jobs_dataloader)
            elif dataset.dataset_name in ['clothing1M']:
                self.trainer = PvalueTrainerClothing1M(n_jobs_dataloader=n_jobs_dataloader)
            else:
                raise Exception(f'No such dataset {dataset.dataset_name} defined!')
        self.trainer.calib(dataset, self.net, score_type=score_type)
        self.results['calib_scores'] = self.trainer.calib_scores
        print('calibration finished')

    def evaluate(self, dataset: BaseADDataset, score_type: str):
        if self.trainer is None:
            if dataset.dataset_name in ['cifar10', 'cifar100']:
                self.trainer = PvalueTrainer(n_jobs_dataloader=n_jobs_dataloader)
            elif dataset.dataset_name in ['clothing1M']:
                self.trainer = PvalueTrainerClothing1M(n_jobs_dataloader=n_jobs_dataloader)
            else:
                raise Exception(f'no such dataset {dataset.dataset_name} defined!')
        self.trainer.evaluate(dataset, self.net, score_type=score_type)
        self.results['eval_scores'] = self.trainer.eval_scores
        print('evaluation finished')

    def test_eval_set(self, dataset: BaseADDataset):
        if self.trainer is None:
            if dataset.dataset_name in ['cifar10', 'cifar100']:
                self.trainer = PvalueTrainer(n_jobs_dataloader=n_jobs_dataloader)
            elif dataset.dataset_name in ['clothing1M']:
                self.trainer = PvalueTrainerClothing1M(n_jobs_dataloader=n_jobs_dataloader)
            else:
                raise Exception(f'no such dataset {dataset.dataset_name} defined!')
        eval_set_acc = self.trainer.test_eval_set(dataset, self.net)
        print(f'eval-set_acc: {eval_set_acc}')
        # self.results['eval_scores'] = self.trainer.eval_scores
        # print('evaluation finished')

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""
        test_acc = 0.0
        if self.trainer is None:
            if dataset.dataset_name in ['cifar10', 'cifar100']:
                self.trainer = PvalueTrainer(n_jobs_dataloader=n_jobs_dataloader)
            elif dataset.dataset_name in ['clothing1M']:
                self.trainer = PvalueTrainerClothing1M(n_jobs_dataloader=n_jobs_dataloader)
            else:
                raise Exception(f'no such dataset {dataset.dataset_name}')
            test_acc = self.trainer.test(dataset, self.net)
            # Get results
            # self.results['test_auc'] = self.trainer.test_auc
            # self.results['test_time'] = self.trainer.test_time
            # self.results['test_scores'] = self.trainer.test_scores
        else:
            test_acc = self.trainer.test(dataset, self.net)
        return test_acc

    def save_model(self, export_model, out_dir):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        save_checkpoint({'net_dict': net_dict}, out_dir=out_dir, filename=export_model)
        # torch.save({'net_dict': net_dict}, export_model)

    def load_model(self, model_path, input_dir):
        """Load Deep SVDD model from model_path."""
        absoute_path = os.path.join(input_dir, model_path)
        model_dict = torch.load(absoute_path)
        # self.net.load_state_dict(model_dict['net_dict'])
        # if is_parallel:
        # self.net = torch.nn.DataParallel(self.net)
        import collections
        new_state_dict = collections.OrderedDict()
        for k, v in model_dict['net_dict'].items():
            # name = k.replace('module.module', 'module')
            name = k.replace('module.', '')

            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)


if __name__ == "__main__":
    print('start main')

    # Options ----------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    # for gpu
    parser.add_argument('--parallel', action='store_true')

    parser.add_argument('--dataset', type=str, help='cifar10, cifar100', default='cifar10')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # dataset split
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.6)
    parser.add_argument('--noise_type', type=str, default='instance')  # manual
    parser.add_argument('--leave_ratio', type=float, default=0.4)
    parser.add_argument('--l_train_ratio', type=float, default=0.5)
    parser.add_argument('--l_train_second_ratio', type=float, default=1.0)  # 10000 * 1.0
    parser.add_argument('--l_calib_second_ratio', type=float, default=1.0)  # 10000 * 1.0

    # net
    parser.add_argument('--net', default='resnet34')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)

    # load model
    parser.add_argument('--load_model_direct_score', default=False, action='store_true')
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str, default='final_model.tar')
    parser.add_argument('--test_acc', type=str)
    parser.add_argument('--load_model_to_train', action='store_true')
    parser.add_argument('--start_epoch', type=int, default=0)

    # draw figure
    parser.add_argument('--draw_score', action='store_true')

    # multiple testing
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--label_file_path', type=str, help='the path of noisy labels',
                        default='./data/noise_label_human.pt')
    parser.add_argument('--score_type', type=str, help='score function type', default='crossentropy')
    parser.add_argument('--score_evaluation', action='store_true')

    # train on selected clean
    parser.add_argument('--retrain_selectedset', action='store_true')
    parser.add_argument('--labeledsetpth', type=str, default='sel_clean.txt')
    parser.add_argument('--mini_batch_num', type=int, default=15000)

    # clothing1M
    parser.add_argument('--l_noisy_train_ratio', default=1.0, type=float)
    parser.add_argument('--lr_schedule', type=int, default=0)
    parser.add_argument('--training_sample_method', type=str, default='sample')

    parser.add_argument('--clean_train_key_list_txt', type=str, default='clean_train_key_list.txt')
    parser.add_argument('--noisy_train_key_list_txt', type=str, default='noisy_train_key_list.txt')
    parser.add_argument('--clean_val_key_list_txt', type=str, default='clean_val_key_list.txt')
    parser.add_argument('--clean_test_key_list_txt', type=str, default='clean_test_key_list.txt')

    dataset_name = 'cifar10'
    data_path = '~/data'

    # Setup --------------------------------------------------------------------------
    torch.multiprocessing.set_sharing_strategy('file_system')
    config = parser.parse_args()
    utility.set_bestgpu(parallel=config.parallel)
    is_parallel = config.parallel

    out_dir = utility.set_outdir(config)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'out_dir:{out_dir}')

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    assert config.noise_type in ['clean', 'instance', 'symmetric', 'pairflip']

    from data.utils import noisify, noisify_instance_v1


    class SplitCIFAR10Dataset(TorchvisionDataset):
        def __init__(self, root: str, leave_ratio=0.2, l_train_ratio=0.5, l_train_second_ratio=1.0,
                     l_calib_second_ratio=1.0, noise_type=None,
                     noise_rate=0.2,
                     data_name=None,
                     random_state=0):
            super(SplitCIFAR10Dataset, self).__init__(root, dataset_name=data_name)

            self.leave_ratio = leave_ratio
            self.l_train_ratio = l_train_ratio
            self.l_train_second_ratio = l_train_second_ratio
            self.l_calib_second_ratio = l_calib_second_ratio
            # transform1 = transforms.Compose([transforms.Lambda(lambda x: np.array(x))])
            if data_name == 'cifar10':
                train_set = MyCIFAR10(root=self.root, train=True, download=True, transform=None)
                test_set = MyCIFAR10(root=self.root, train=False, download=True, transform=None)
                self.nb_classes = 10
            elif data_name == 'cifar100':
                train_set = MyCIFAR100(root=self.root, train=True, download=True, transform=None)
                test_set = MyCIFAR100(root=self.root, train=False, download=True, transform=None)
                self.nb_classes = 100
            l_train_idx, l_calib_idx, eval_idx = self.__split_data(dataset=data_name, leave_ratio=self.leave_ratio,
                                                                   l_train_ratio=self.l_train_ratio,
                                                                   l_train_second_ratio=self.l_train_second_ratio,
                                                                   l_calib_second_ratio=self.l_calib_second_ratio)
            print(f'l_real_train:{len(l_train_idx)}')
            print(f'l_calib:{len(l_calib_idx)}')
            self.l_train_set = Subset(train_set, l_train_idx)
            self.l_calib_set = Subset(train_set, l_calib_idx)
            self.eval_set = Subset(train_set, eval_idx)
            self.test_set = test_set

            # corrupt eval_set
            self.noise_type = noise_type
            self.noise_rate = noise_rate
            self.noise_or_not = None
            self.actual_noise_rate = None

            self.corrupted_labels = None

            self.corrupted_labels, self.noise_or_not = self.__corrupt_evalset__()

        def __split_data(self, dataset: str, leave_ratio, l_train_ratio, l_train_second_ratio, l_calib_second_ratio):
            if dataset == 'cifar10' or dataset == 'cifar100':
                n_train = 50000
                indices = np.arange(n_train)
                random.shuffle(indices)
                n_leave = int(leave_ratio * n_train)  # trusted data num
                n_leave_train = int(l_train_ratio * n_leave)  # first split for train
                n_leave_calib = n_leave - n_leave_train  # first split for calib

                n_leave_train_real = int(l_train_second_ratio * n_leave_train)  # second_split for train
                n_leave_calib_real = int(l_calib_second_ratio * n_leave_calib)
                # l_train_idx = indices[:n_leave_train]
                l_train_idx = indices[:n_leave_train_real]
                l_calib_idx = indices[n_leave_train:n_leave_train + n_leave_calib_real]
                e_index = indices[n_leave:]
                return l_train_idx, l_calib_idx, e_index

        def __corrupt_evalset__(self):
            print(f'evaluation num:{len(self.eval_set)}')
            if len(self.eval_set) == 0:
                self.corrupted_labels = None
                self.noise_or_not = None
                return self.corrupted_labels, self.noise_or_not
            eval_dataloader = DataLoader(dataset=self.eval_set, batch_size=len(self.eval_set))
            eval_data, eval_labels, eval_idx = next(iter(eval_dataloader))
            idx_each_class_noisy = [[] for i in range(self.nb_classes)]
            print(f'noise_type:{self.noise_type}')
            print(f'noise_rate:{self.noise_rate}')

            if self.noise_type != 'clean':
                # corrupt the evaluation data
                if self.noise_type in ['symmetric', 'pairflip']:
                    real_labels = np.asarray([[eval_labels[i]] for i in range(len(eval_labels))])
                    corrupted_labels, self.actual_noise_rate = noisify(dataset='cifar10', train_labels=real_labels,
                                                                       noise_type=self.noise_type,
                                                                       noise_rate=self.noise_rate,
                                                                       random_state=random_state,
                                                                       nb_classes=self.nb_classes)
                    self.corrupted_labels = [i[0] for i in corrupted_labels]
                    _real_labels = [i[0] for i in real_labels]
                    for i in range(len(_real_labels)):
                        idx_each_class_noisy[self.corrupted_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]
                    self.noise_or_not = np.transpose(self.corrupted_labels) != np.transpose(_real_labels)
                    return self.corrupted_labels, self.noise_or_not
                elif self.noise_type == "instance":
                    self.corrupted_labels, self.actual_noise_rate = noisify_instance_v1(eval_data, eval_labels,
                                                                                        noise_rate=self.noise_rate)
                    real_labels = eval_labels.numpy()

                    print('over all noise rate is ', self.actual_noise_rate)
                    for i in range(len(eval_labels)):
                        idx_each_class_noisy[self.corrupted_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]

                    self.noise_or_not = np.transpose(self.corrupted_labels) != np.transpose(real_labels)
                    return self.corrupted_labels, self.noise_or_not
                else:
                    print('Noise_type not defined')
                    pass
            # I need to corrupt the primitive eval set


    class TargetTransformDataset(MyCIFAR10):
        def __init__(self, l_train_set, transform=None, target_transform=None):
            self.l_train_set = l_train_set
            self.transform = transform
            self.target_transform = target_transform
            l_train_set_dataloader = DataLoader(dataset=self.l_train_set, batch_size=len(self.l_train_set))
            self.train_data, self.train_labels, self.train_indices = next(iter(l_train_set_dataloader))
            print('Target Transform Dataset init finished! ')

        def __getitem__(self, index):
            img, target = self.train_data[index], self.train_labels[index]

            img = Image.fromarray(img.numpy())
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, index

        def __len__(self):
            return len(self.train_data)


    class TargetTransformEvalSet(MyCIFAR10):
        def __init__(self, eval_set=None, noise_or_not=None, noisy_labels=None, transform=None,
                     target_transform=None):
            self.noise_or_not = noise_or_not
            self.eval_set = eval_set
            self.transform = transform
            eval_set_loader = DataLoader(dataset=self.eval_set, batch_size=len(self.eval_set))
            self.eval_data, self.eval_labels, self.eval_indices = next(iter(eval_set_loader))
            self.eval_labels = noisy_labels
            print('TargetTransformEvalSet init finished ')

        def __getitem__(self, index):
            img, target = self.eval_data[index], self.eval_labels[index]
            # prim_index = self.eval_indices[index]

            img = Image.fromarray(img.numpy())
            if self.transform is not None:
                img = self.transform(img)

            # target = int(self.noise_or_not[index])
            # target = self.target_transform(target)

            return img, target, index

        def __len__(self):
            return len(self.eval_data)


    class TransformEvaltoTrain(Clothing1M):
        def __init__(self, eval_set: Clothing1M, transform=None):
            self.noisy_train_paths = eval_set.noisy_train_paths
            self.noisy_labels = eval_set.noisy_labels
            self.transform = transform

        def __getitem__(self, index):
            img_path = self.noisy_train_paths[index]
            target = self.noisy_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index

        def __len__(self):
            return len(self.noisy_train_paths)


    class SelectedEvalTrain(Clothing1M):
        def __init__(self, eval_set: Clothing1M, sel_clean_idx=[]):
            self.noisy_train_paths = np.array(eval_set.noisy_train_paths)[sel_clean_idx].tolist()
            self.noisy_labels = eval_set.noisy_labels
            self.num_classes = len(set(get_keys_from_dict(self.noisy_labels, self.noisy_train_paths)))
            self.transform = eval_set.transform

        def __getitem__(self, index):
            img_path = self.noisy_train_paths[index]
            target = self.noisy_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index

        def __len__(self):
            return len(self.noisy_train_paths)


    class RetrainClothing1MEval(TorchvisionDatasetClothing1M):
        def __init__(self, eval_set=None, test_set=None, sel_clean_idx=None, training_sample_method=None):
            mean = (0.485, 0.456, 0.406)  # (0.6959, 0.6537, 0.6371),
            std = (0.229, 0.224, 0.225)  # (0.3113, 0.3192, 0.3214)
            normalize = transforms.Normalize(mean=mean, std=std)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            eval_set = TransformEvaltoTrain(eval_set, transform=train_transform)
            print(f'len of sel_clean_idx {len(sel_clean_idx)}')
            retrain_eval_set = SelectedEvalTrain(eval_set, sel_clean_idx=sel_clean_idx)

            self.train_set = None
            self.original_retrain_set = retrain_eval_set
            self.calib_set = None
            self.eval_set = None
            self.test_set = test_set
            self.training_sample_method = training_sample_method
            # self.noise_or_not = noise_or_not
            print('sel eval_clothing1M init finihsed')


    class RetrainCIFAR10Eval(TorchvisionDataset):
        def __init__(self, split_cifar10: SplitCIFAR10Dataset = None, sel_clean_idx=None):
            train_transform = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.ToTensor()])
            retrain_eval_set = TargetTransformEvalSet(eval_set=split_cifar10.eval_set, noise_or_not=None,
                                                      noisy_labels=split_cifar10.corrupted_labels,
                                                      transform=train_transform, target_transform=None)
            retrain_eval_set = Subset(retrain_eval_set, indices=sel_clean_idx)

            test_set = TargetTransformDataset(l_train_set=split_cifar10.test_set, transform=test_transform,
                                              target_transform=None)

            self.train_set = None
            self.retrain_set = retrain_eval_set
            self.calib_set = None
            self.eval_set = None
            self.test_set = test_set

            self.prim_eval_set = None
            print('sel eval_cifar10 init finihsed')


    class BHClothing1M(TorchvisionDatasetClothing1M):
        def __init__(self, dataset_name, original_train_set, calib_set, eval_set, test_set, clean_train_set):
            self.dataset_name = dataset_name
            self.original_train_set = original_train_set
            self.clean_train_set = clean_train_set
            self.train_set = None
            self.calib_set = calib_set
            self.eval_set = eval_set
            self.test_set = test_set
            self.retrain_set = None
            print('BH clothing1M construction finished!')


    class BHCIFAR10(TorchvisionDataset):
        def __init__(self, split_cifar10: SplitCIFAR10Dataset = None):
            self.dataset_name = split_cifar10.dataset_name
            train_transform = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.ToTensor()])
            train_set = TargetTransformDataset(l_train_set=split_cifar10.l_train_set, transform=train_transform,
                                               target_transform=None)
            calib_set = None
            if len(split_cifar10.l_calib_set) != 0:
                calib_set = TargetTransformDataset(l_train_set=split_cifar10.l_calib_set, transform=test_transform,
                                                   target_transform=None)
            test_set = TargetTransformDataset(l_train_set=split_cifar10.test_set, transform=test_transform,
                                              target_transform=None)
            noise_or_not = split_cifar10.noise_or_not

            prim_eval_set = TargetTransformDataset(l_train_set=split_cifar10.eval_set, transform=test_transform,
                                                   target_transform=None)
            eval_set = None
            if len(split_cifar10.eval_set) != 0:
                eval_set = TargetTransformEvalSet(eval_set=split_cifar10.eval_set, noise_or_not=noise_or_not,
                                                  noisy_labels=split_cifar10.corrupted_labels, transform=test_transform)
            self.noise_or_not_normal = noise_or_not
            self.train_set = train_set
            self.retrain_set = None
            self.calib_set = calib_set
            self.eval_set = eval_set
            self.test_set = test_set
            self.noise_or_not = noise_or_not

            self.prim_eval_set = prim_eval_set
            print('BH cifar10 init finihsed')


    # main
    # obtain score function
    random_state = 0
    method_oneclass = 'overall_loss'
    if method_oneclass == 'overall_loss':
        nu = 0.1
        score_handler = ScoreFuncHandler(out_dir=out_dir)
        net_name = config.net
        score_handler.set_network(net_name=net_name)
        n_jobs_dataloader = 0

        if config.dataset in ['cifar10', 'cifar100']:
            split_cifar10 = SplitCIFAR10Dataset(root='~/data', leave_ratio=config.leave_ratio,
                                                l_train_ratio=config.l_train_ratio,
                                                l_train_second_ratio=config.l_train_second_ratio,
                                                l_calib_second_ratio=config.l_calib_second_ratio,
                                                noise_type=config.noise_type,
                                                noise_rate=config.noise_rate, data_name=config.dataset, random_state=0)
        elif config.dataset == 'clothing1M':
            clothing1M_dir = './data/clothing1M'
            time1 = time.time()
            # Follow 2022 C2D
            mean = (0.485, 0.456, 0.406)  # (0.6959, 0.6537, 0.6371),
            std = (0.229, 0.224, 0.225)  # (0.3113, 0.3192, 0.3214)
            normalize = transforms.Normalize(mean=mean, std=std)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

            clean_train_set = Clothing1M(root=clothing1M_dir, mode='clean_train', transform=train_transform,
                                         clean_train_key_list_txt=config.clean_train_key_list_txt)
            time2 = time.time()
            print(f'load clean train_set: {time2 - time1}')
            calib_set = Clothing1M(root=clothing1M_dir, mode='clean_val', transform=test_transform,
                                   clean_val_key_list_txt=config.clean_val_key_list_txt)
            print(f'load clean calib set time: {time.time() - time2}')
            time3 = time.time()
            l_noisy_train_ratio = config.l_noisy_train_ratio
            if config.retrain_selectedset:
                print('retrain at the selected set')
                eval_set = Clothing1M(root=clothing1M_dir, mode='noisy_train', transform=train_transform,
                                      l_noisy_train_ratio=l_noisy_train_ratio,
                                      noisy_train_key_list_txt=config.noisy_train_key_list_txt)
            else:
                eval_set = Clothing1M(root=clothing1M_dir, mode='noisy_train', transform=test_transform,
                                      l_noisy_train_ratio=l_noisy_train_ratio,
                                      noisy_train_key_list_txt=config.noisy_train_key_list_txt)

            print(f'load noisy train set time: {time.time() - time3}')
            time4 = time.time()
            test_set = Clothing1M(root=clothing1M_dir, mode='clean_test', transform=test_transform,
                                  clean_test_key_list_txt=config.clean_test_key_list_txt)
            print(f'load test set time :{time.time() - time4}')
            print(f'all clothing1M set finished: {time.time() - time1}')
        else:
            pass
        retrain_selectedset = config.retrain_selectedset
        if retrain_selectedset:
            print('retrain at the selected set')
            if config.input_dir is None:
                eval_set_len = len(eval_set)
                sel_clean_idx = list(range(eval_set_len))
            else:
                labeled_set_pth = os.path.join(config.input_dir, config.labeledsetpth)
                with open(labeled_set_pth, 'r') as label_file:
                    labeled_set_read = label_file.readline()
                    sel_clean_idx = intliststr_to_list(labeled_set_read)
                    print(f'len of sel_clean_idx: {len(sel_clean_idx)}')
            if config.dataset in ['cifar10', 'cifar100']:
                retrain_eval_set = RetrainCIFAR10Eval(split_cifar10=split_cifar10, sel_clean_idx=sel_clean_idx)
            elif config.dataset in ['clothing1M']:

                training_sample_method = config.training_sample_method
                retrain_eval_set = RetrainClothing1MEval(eval_set=eval_set, test_set=test_set,
                                                         sel_clean_idx=sel_clean_idx,
                                                         training_sample_method=training_sample_method)
            else:
                raise Exception(f'no such dataset {config.dataset}')
            score_handler = ScoreFuncHandler(out_dir=out_dir)
            net_name = config.net
            score_handler.set_network(net_name=net_name)
            n_jobs_dataloader = 0
            if config.dataset in ['cifar10', 'cifar100']:
                optimizer_name = 'sgd'
                lr = 0.1
                n_epochs = config.n_epochs
                # n_epochs = 2
                lr_milestone = [160]
                batch_size = 128
                weight_decay = 5e-4
                momentum = 0.9
                print('Training optimizer: %s' % optimizer_name)
                print('Training learning rate: %g' % lr)
                print('Training epochs: %d' % n_epochs)
                print('Training learning rate scheduler milestones: %s' % lr_milestone)
                print('Training batch size: %d' % batch_size)
                print('Training weight decay: %g' % weight_decay)
                print('Training momentum: %g' % momentum)
                score_handler.re_train(dataset=retrain_eval_set, optimizer_name=optimizer_name, lr=lr,
                                       n_epochs=n_epochs,
                                       lr_milestones=lr_milestone, batch_size=batch_size, weight_decay=weight_decay,
                                       n_jobs_dataloader=n_jobs_dataloader)
            elif config.dataset in ['clothing1M']:
                optimizer_name = 'sgd'
                lr = config.lr
                n_epochs = config.n_epochs
                batch_size = config.batch_size
                weight_decay = config.weight_decay
                momentum = config.momentum
                mini_batch_num = config.mini_batch_num
                lr_schedule = config.lr_schedule
                training_sample_method = config.training_sample_method

                if config.dataset in ['cifar10', 'cifar100']:
                    lr_milestone = [160]
                elif config.dataset == 'clothing1M':
                    if lr_schedule == 0:
                        lr_milestone = [40]
                    elif lr_schedule == 3:
                        lr_milestone = [30, 60, 90]
                    else:
                        pass
                else:
                    raise Exception(f'no dataset {config.dataset}')
                print('Training optimizer: %s' % optimizer_name)
                print('Training learning rate: %g' % lr)
                print('Training epochs: %d' % n_epochs)
                print('Training learning rate scheduler milestones: %s' % lr_milestone)
                print('Training batch size: %d' % batch_size)
                print('Training weight decay: %g' % weight_decay)
                print('Training momentum: %g' % momentum)
                print('Training mini_batch_num %d' % mini_batch_num)
                score_handler.retrainClothing1M(dataset=retrain_eval_set, optimizer_name=optimizer_name, lr=lr,
                                                n_epochs=n_epochs,
                                                lr_milestones=lr_milestone, batch_size=batch_size,
                                                weight_decay=weight_decay, n_jobs_dataloader=n_jobs_dataloader,
                                                momentum=momentum,
                                                mini_batch_num=mini_batch_num,
                                                training_sample_method=training_sample_method)
            else:
                raise Exception(f'no such dataset {config.dataset}')
        else:
            leave_train = None
            if config.dataset in ['cifar10', 'cifar100']:
                leave_train = BHCIFAR10(split_cifar10=split_cifar10)
            elif config.dataset == 'clothing1M':
                leave_train = BHClothing1M(dataset_name=config.dataset, original_train_set=eval_set,
                                           calib_set=calib_set, eval_set=eval_set,
                                           test_set=test_set, clean_train_set=clean_train_set)
            else:
                pass
            load_model_direct_score = config.load_model_direct_score
            print(f'load_model_direct_score {load_model_direct_score}')

            if not load_model_direct_score:
                print('train to get score function')
                optimizer_name = 'sgd'
                lr = config.lr
                n_epochs = config.n_epochs
                batch_size = config.batch_size
                weight_decay = config.weight_decay
                momentum = config.momentum
                training_sample_method = config.training_sample_method
                if config.dataset in ['cifar10', 'cifar100']:
                    lr_milestone = [160]
                elif config.dataset == 'clothing1M':
                    lr_milestone = [40]
                else:
                    raise Exception(f'no dataset {config.dataset}')
                print('Training optimizer: %s' % optimizer_name)
                print('Training learning rate: %g' % lr)
                print('Training epochs: %d' % n_epochs)
                print('Training learning rate scheduler milestones: %s' % lr_milestone)
                print('Training batch size: %d' % batch_size)
                print('Training weight decay: %g' % weight_decay)
                print('Training momentum: %g' % momentum)

                # Train  model on dataset
                if config.load_model_to_train:
                    print('load trained model to continue to train')
                    input_dir = config.input_dir
                    model_file = config.model_file
                    score_handler.load_model(model_path=model_file, input_dir=input_dir)
                    score_handler.train(dataset=leave_train, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                        lr_milestones=lr_milestone, batch_size=batch_size, weight_decay=weight_decay,
                                        n_jobs_dataloader=n_jobs_dataloader, start_epoch=config.start_epoch,
                                        is_parallel=is_parallel)
                    score_handler.save_model(export_model='final_model.tar', out_dir=out_dir)
                else:
                    # direct train
                    score_handler.train(dataset=leave_train, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                        lr_milestones=lr_milestone, batch_size=batch_size, weight_decay=weight_decay,
                                        n_jobs_dataloader=n_jobs_dataloader, start_epoch=config.start_epoch)
                    score_handler.save_model(export_model='final_model.tar', out_dir=out_dir)
            else:
                print('load trained model to get test acc directly,not necessarily to score the evaluation set')
                input_dir = config.input_dir
                model_file = config.model_file
                score_handler.load_model(model_path=model_file, input_dir=input_dir)
                test_acc = score_handler.test(dataset=leave_train)
                print(f'test_acc:{test_acc}')

            classifier = score_handler.net
            score_handler.calib(dataset=leave_train, score_type=config.score_type)
            print('calib finished')

            if config.score_evaluation:
                score_handler.evaluate(dataset=leave_train, score_type=config.score_type)
                # need to calculate calib scores
                indices_cal, labels_cal, scores_cal = zip(*score_handler.results['calib_scores'])
                indices_cal, labels_cal, scores_cal = np.array(indices_cal), np.array(labels_cal), np.array(scores_cal)

                indices_eval, labels_eval, scores_eval = zip(*score_handler.results['eval_scores'])
                indices_eval, labels_eval, scores_eval = np.array(indices_eval), np.array(labels_eval), np.array(
                    scores_eval)

                cc = Pvalues1(classifier, scores_cal=scores_cal, scores_eval=scores_eval, delta=0.05)

                pvals_one_class = cc.predict()  # continue noise prior

                # Evaluate performance
                alpha = config.alpha
                if config.dataset in ['cifar10', 'cifar100']:
                    # for have true label
                    is_nonnull = np.array(leave_train.noise_or_not_normal).astype(int)
                    res_fdr, res_reject = evaluate_all_methods(pvals_one_class, is_nonnull, alpha=alpha)
                elif config.dataset == 'clothing1M':
                    # for no true label
                    res_fdr, res_reject = write_sel_clean(pvals_one_class, alpha=alpha)
                else:
                    pass

                # print(res_reject)
                reject_result = res_reject['pvalue'][False]
                sel_clean_idx = indices_eval[np.where(reject_result == False)].tolist()
                print(f'len of sel_clean_idx: {len(sel_clean_idx)}')
                sel_clean_idx_file = os.path.join(out_dir, 'sel_clean.txt')
                with open(sel_clean_idx_file, 'w') as file:
                    file.write(str(sel_clean_idx))
                print(f'sel_clean output to {out_dir} sel_clean.txt')

                draw_score = config.draw_score
                if draw_score:
                    font = {'weight': 'normal', 'size': 20}
                    plt.rc('font', **font)
                    # draw scores histogram
                    print('min_scores_cal: {:.3f}, max_scores_cal: {:.3f}'.format(min(scores_cal), max(scores_cal)))
                    print('min_scores_eval:{:.3f}, max_scores_eval: {:.3f}'.format(min(scores_eval), max(scores_eval)))
                    fig, axs = plt.subplots(1, figsize=(6, 4))
                    axs.hist(scores_eval[is_nonnull == 0], bins=50, alpha=0.5, label='clean', density=True)
                    axs.hist(scores_eval[is_nonnull == 1], bins=50, alpha=0.5, label='corrupted', density=True)
                    axs.legend(loc='upper right')
                    axs.set_xlabel('scores', fontdict=font)
                    axs.set_ylabel('density', fontdict=font)
                    axs.title.set_text(f'test acc (%):{config.test_acc}')
                    axs.set_xlim([-25, 0])
                    axs.set_ylim([0, 1.3])

                    if config.score_type == 'crossentropy':
                        axs.set_xlim([-25, 0])
                        axs.set_ylim([0, 1.3])
                    elif config.score_type == 'softmax_norm':
                        axs.set_xlim([0.2, 1])
                        axs.set_ylim([0, 18])
                    else:
                        pass
                    fig.tight_layout()
                    plt.savefig(
                        f'scores_{config.l_train_second_ratio}_{config.dataset}_{config.net}_{config.score_type}.pdf',
                        dpi=600)
                    plt.show()
