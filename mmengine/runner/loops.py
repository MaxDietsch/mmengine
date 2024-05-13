import logging
import bisect
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from .amp import autocast
from .base_loop import BaseLoop
from .utils import calc_dynamic_intervals

# for usage of self made sampler
import numpy as np
# only EpochBasedTrainLoop works with DynamicSampler
from mmengine.dataset import DynamicSampler
import torch.nn.functional as F
import heapq
import math
from mmengine.utils.bmu import get_combo_loader

@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        
        # initialize idx array which specifies which classes should 
        # be included in the training
        #threshold = 0.3
        #cls = [i for i in range(self.runner.model.head.fc.out_features)]

        while self._epoch < self._max_epochs and not self.stop_training:
            # self.run_epoch(cls)
            self.run_epoch()
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                
                # get the metrics
                mtrcs = self.runner.val_loop.run()
            
            # update sampler
            if isinstance(self.dataloader.sampler, DynamicSampler):
                f1_scores = torch.tensor(mtrcs['single-label/f1-score_classwise']) / 100
                self.dataloader.sampler.update_sample_size(f1_scores)
        
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(self.dataloader):
            # only run iterations for classes that should be included 
            # if data_batch['data_samples'][0].gt_label.item() in cls:
                # self.run_iter(idx, data_batch)
            #print(data_batch)
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


class _InfiniteDataloaderIterator:
    """An infinite dataloader iterator wrapper for IterBasedTrainLoop.

    It resets the dataloader to continue iterating when the iterator has
    iterated over all the data. However, this approach is not efficient, as the
    workers need to be restarted every time the dataloader is reset. It is
    recommended to use `mmengine.dataset.InfiniteSampler` to enable the
    dataloader to iterate infinitely.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._iterator = iter(self._dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Sequence[dict]:
        try:
            data = next(self._iterator)
        except StopIteration:
            print_log(
                'Reach the end of the dataloader, it will be '
                'restarted and continue to iterate. It is '
                'recommended to use '
                '`mmengine.dataset.InfiniteSampler` to enable the '
                'dataloader to iterate infinitely.',
                logger='current',
                level=logging.WARNING)
            self._epoch += 1
            if hasattr(self._dataloader, 'sampler') and hasattr(
                    self._dataloader.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, 'batch_sampler') and hasattr(
                    self._dataloader.batch_sampler.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
        return data


@LOOPS.register_module()
class CoSenTrainLoop(BaseLoop):
    """Loop for epoch-based training based on:
        Cost-Sensitive Learning of Deep Feature Representations from Imbalanced Data by S.H. Khan, M. Hayat ...
        Requirements:
            model must be of type CoSenClassifier
            new (non-default) parameters in the init.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        s_freq (int): set the frequency how often S should be evaluated: f.e. 3 means S
            is calculated every 3 epochs
        s_samples_per_class (List[int]): set the number of samples per class which are 
            included in the process of calculating S (should be only a small set to be computationally efficient)
        samples_per_class (List[int]): set how many samples are present in the dataset per class
        mu1, mu2, s1, s2 (float): Hyperparameters of the CoSen algorithm 
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            s_freq: int, 
            s_samples_per_class: List[int],
            samples_per_class: List[int],
            mu1: float,
            mu2: float,
            s1: float,
            s2: float, 
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)


        # for CoSen
        from mmpretrain.models.classifiers import CoSenClassifier # this is not good style
        assert isinstance(self.runner.model, CoSenClassifier), 'The model should be of type CoSenClassifier when using CoSenTrainLoop'

        self.num_classes = len(self.dataloader.dataset.metainfo['classes'])
        self.s_freq = s_freq
        self.s_samples_per_class = s_samples_per_class
        
                # store distances
        self.d = torch.zeros((sum(self.s_samples_per_class), self.num_classes))

        # stores deep features
        self.v = [[] for _ in range(self.num_classes)]

        # store c2c separability
        self.c2c_sep = torch.zeros((self.num_classes, self.num_classes))

        # define H

        # to work with ROS sampler (in the other implementation of ROS for each epoch there was a slightly different dataset size)
        from mmengine.dataset.sampler import CoSenROSSampler
        if isinstance(self.dataloader.sampler, CoSenROSSampler):
            samples_per_class = self.dataloader.sampler.counts
            print(samples_per_class)
        samples_per_class = torch.tensor(samples_per_class)
        self.size_dataset = samples_per_class.sum()
        h1 = samples_per_class.view(-1, 1) / self.size_dataset
        h2 = samples_per_class.view(1, -1) / self.size_dataset
        self.h = torch.max(h1, h2).to(torch.device('cuda'))
        
        # for calculating confusion matrix store y_pred and y_true
        self.y_pred = torch.randint(self.num_classes, (self.size_dataset, ))
        self.y_true = torch.randint(self.num_classes, (self.size_dataset, ))
        
        # Hyperparameter
        self.mu1 = mu1 
        self.mu2 = mu2 
        self.s1 = s1
        self.s2 = s2
        
        # store best accurcy, used for updating learning rate of cosen matrix 
        self.best_acc = 0
        self.updated = False

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter


    def calc_c2c_separability(self):

        for i in range(self.num_classes):
            low_idx = sum(self.s_samples_per_class[ : i ])
            high_idx = sum(self.s_samples_per_class[  : i+1 ])
            for j in range(self.num_classes):

                # get sorted distances
                # row l contains distance of v[i][l] to each of v[j]
                sorted_distances = (torch.sort(torch.cdist(self.v[i], self.v[j]))[0]).to(torch.device('cpu'))
                # decide which element to take, the smallest (inter class) or the 2nd smallest (intra class)
                print(sorted_distances)
                entry_idx = 0 if i != j else 1
                self.d[low_idx : high_idx, j] += sorted_distances[ : , entry_idx]
        
        #print(self.d)

        # based on the distances, fill S(p, q)
        for i in range(self.num_classes):
            low_idx = sum(self.s_samples_per_class[ : i ])
            high_idx = sum(self.s_samples_per_class[ : i+1 ])
            
            for j in range(self.num_classes):

                ratio = torch.sum(self.d[low_idx : high_idx, i] / self.d[low_idx : high_idx , j])
                self.c2c_sep[i, j] = 1/self.s_samples_per_class[i] * ratio

        self.d.fill_(0)


    # could be done by import of torchmetrics
    def confusion_matrix(self, y_pred, y_true):
        conf_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            conf_matrix[t, p] += 1
        
        # to get probabilities
        return conf_matrix / conf_matrix.sum(1, keepdim = True)


    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            
            # update S only in specific epochs like in the paper
            with torch.no_grad():
                if self._epoch % self.s_freq == 0 and self._epoch != 0:
                    # reset updated lr 
                    self.updated = False

                    # get the deep features for each class
                    for idx, data_batch in enumerate(self.dataloader):
                        batch = self.runner.model.data_preprocessor(data_batch, True) 
                        inputs = batch['inputs']
                        data_samples = batch['data_samples']
                        labels = torch.cat([i.gt_label for i in data_samples])
                        outs = self.runner.model.extract_feat(inputs)
                        [self.v[label].append(outs[0][ix]) for ix, label in enumerate(labels) if len(self.v[label]) < self.s_samples_per_class[label]]
                        
                        if all( x >= y for x, y in zip([len(self.v[i]) for i in range(self.num_classes)], self.s_samples_per_class)):
                            for i in range(self.num_classes):
                                self.v[i] = torch.stack(self.v[i], dim = 0)
                            break

                    # calculate S 
                    self.calc_c2c_separability()
                    self.v = [[] for _ in range(self.num_classes)]
                    # print(self.c2c_sep)
                    
                    # calculate confusion matrix R (could be done with library torchmetrics)
                    r = self.confusion_matrix(self.y_pred, self.y_true)
                    # print(r)
                    
                    # calculate matrix T
                    t_temp = torch.mul(torch.exp( - (self.c2c_sep - self.mu1) ** 2 / (2 * self.s1 ** 2)), torch.exp( - (r - self.mu2) ** 2 / (2 * self.s2 ** 2)))
                    t = torch.mul(self.h, t_temp)
                    #print(t)
                    
                    # calculate gradient and update cost matrix 
                    # print(self.runner.model.head.loss_module.xi)
                    self.runner.model.head.loss_module.update_xi(t.view(-1, 1))
                    print(self.runner.model.head.loss_module.xi)


            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                metric = self.runner.val_loop.run()
            
            # if new accuracy is better than before update learning rate of cosen matrix
            if metric['accuracy/top1'] > self.best_acc and not self.updated:
                print('Update learning rate of CoSen matrix')
                new_lr = self.runner.model.head.loss_module.get_xi_lr() * 0.01
                self.runner.model.head.loss_module.update_xi(new_lr)
                self.updated = True


        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.

        # now returns the loss and the cls_score of the model as a tuple
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        # get true and predicted labels 
        true_labels = torch.cat([i.gt_label for i in data_batch['data_samples']])
        
        pred_scores = F.softmax(outputs[1], dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach().squeeze()

        # fill the y_pred and y_true
        batch_size = self.dataloader.batch_size
        low_idx = idx * batch_size
        high_idx = min((idx + 1) * batch_size, self.size_dataset)
        self.y_pred[low_idx : high_idx] = pred_labels
        self.y_true[low_idx: high_idx] = true_labels

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs[0])
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


# not tested
@LOOPS.register_module()
class BalancedMixUpTrainLoop(BaseLoop):
    """Loop for Balanced-MixUP training. Like in the paper: 
        Balanced-MixUp for Highly Imbalanced Medical Image Classification
    Requirements:

        new non-default arguments

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        samples_per_class (List[int]): array where a[i] defines how many samples of class i are in the training set 
        alpha (int): hyperparameter for the beta distribution (see paper )
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            samples_per_class: List[int], 
            alpha: int, 
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)


        # BalancedMixUp

        from mmpretrain.models.heads import BMUHead # this is not good style
        assert isinstance(self.runner.model.head, BMUHead), 'The model head should be of type BMUHead when using BMUTrainLoop'

        self.alpha = alpha
        self.n_classes = len(samples_per_class)
        # define the combo dataloader
        # combo loader has a list where first element is dataloader for instance based sampling
        #                               second element is dataloader for class based sampling
        self.combo_loader = get_combo_loader(self.dataloader, torch.tensor(samples_per_class))

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        
        while self._epoch < self._max_epochs and not self.stop_training:
            # self.run_epoch(cls)
            self.run_epoch()
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()
        
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(self.combo_loader):
            
            #print(data_batch)
            # get inputs from instance sampling 
            inputs = data_batch[0]['inputs']
            labels = torch.tensor([i.gt_label for i in data_batch[0]['data_samples']])
            #print(inputs)
            #print(labels)

            # get inputs from class sampling
            balanced_inputs = data_batch[1]['inputs']
            balanced_labels = torch.tensor([i.gt_label for i in data_batch[1]['data_samples']])

            # mix the inputs and labels
            lam = np.random.beta(self.alpha, 1) # lambda is the same for whole batch, maybe change it
            mixed_inputs = (1 - lam) * inputs + lam * balanced_inputs
            mixed_labels = (1 - lam) * F.one_hot(labels, self.n_classes) + lam * F.one_hot(balanced_labels, self.n_classes)
            mixed_labels = mixed_labels.to('cuda')

            data_batch = {'inputs': mixed_inputs, 'data_samples': mixed_labels}
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]



# not tested
@LOOPS.register_module()
class DOSTrainLoop(BaseLoop):
    """Loop for epoch-based training based on: 
        Deep Over-sampling Framework for Classifying Imbalanced Data by Shin Ando and Chun Yuan Huang
        Requirements: 
            shuffle of dataloader should be set to False
            requires as classifier model the DOSClassifier
            requires as model head the DOSHead
            requires as loss the DOSLoss
            requires BatchSize to be 1 
            requires DOSSampler as sampler of the dataloader
            new (non-default) parameter in the initializing (specified in train_cfg)

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        k (List[int]): set the overloading parameter of the algorithm for each class
        r (List[int]): like in the paper (How many weights should be sampled per overloading
            sample)
        samples_per_class (List[int]): set how many samples are present in the dataset per class
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            k: List[int],
            r: List[int],
            samples_per_class: List[int],
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

        # for DOS
        from mmpretrain.models.classifiers import DOSClassifier # this is not good style
        assert isinstance(self.runner.model, DOSClassifier), 'The model should be of type DOSClassifier when using DOSTrainLoop'
        
        from ..dataset.sampler import DOSSampler
        assert isinstance(self.dataloader.sampler, DOSSampler), 'The sampler of the dataloader should be of type DOSSampler when using DOSTrainLoop'

        #assert self.dataloader.batch_size == 1, 'The batch size should be set to 1 if you want to use DOS when using DOSTrainLoop'        

        self.num_classes = len(self.dataloader.dataset.metainfo['classes'])
        self.b_size = self.dataloader.batch_size

        # for generator of dataloader sampler
        self.seed = 0

        # set the overloading parameter k, set r and samples_per_class
        self.samples_per_class = samples_per_class 
        self.r = r #[0, 3, 2, 1]
        self.k = k #[0, 3, 2, 1]


        """Pytorchifying 
        # dimension of deep features
        in_dim = self.runner.model.head.in_channels

        # storde deep features 
        self.v = [torch.empty((samples, in_dim), device = torch.device("cuda")) for samples in self.samples_per_class]

        # store the batch number and position in each batch of each image
        self.batch_idx = [torch.zeros((samples, 2), dtype = torch.int) for samples in self.samples_per_class]
        
        # store mutual distance matrix
        self.d = [torch.zeros((i, i)) for i in self.samples_per_class]

        # store for each image, the deep features (at the right batch and location in the batch )
        #self.n = [torch.empty(self.samples_per_class[i], self.k[i], in_dim) for i in range(self.num_classes)]
        self.n = [torch.empty(0, 0) for _ in range(len(self.dataloader.dataset))]

        # do the same for the weights
        self.w = [torch.empty(0, 0) for _ in range(len(self.dataloader.dataset))]

        #"""

        in_dim = self.runner.model.head.in_channels

        # store mutual distance matrix
        self.d = [torch.zeros((i, i)) for i in self.samples_per_class]

        # for efficiency, so that idx of image in dataloader is stored and not whole image
        self.batch_idx = [[] for _ in range(self.num_classes)]
        self.batch_idx = [torch.zeros((samples,), device = torch.device("cuda"), dtype = torch.int) for samples in self.samples_per_class]


        # store deep features

        self.v = [[] for _ in range(self.num_classes)]
        self.v = [torch.empty((samples, in_dim), device = torch.device("cuda")) for samples in self.samples_per_class]
        self.n = {}
        self.w = {}
        # store the overloaded training samples
        self.z = {'image': [], 'n': [], 'w': []}
        #"""

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def calc_mutual_distance_matrix(self):
        """calculates mutual distances between every deep feature belonging to the same class"""
        """
        for h in range(self.num_classes):
            for i in range(self.samples_per_class[h]):
                for j in range(i, self.samples_per_class[h]):
                    if i == j:
                        self.d[h][i, j] = float('inf')
                        continue
                    self.d[h][i, j] = torch.norm(self.v[h][i] - self.v[h][j])
                    self.d[h][j, i] = self.d[h][i, j]
        #"""

        #"""Pytorchifiying: 
        for h in range(self.num_classes): 
            diff = self.v[h][ : , None, : ] - self.v[h][ None, : , : ]
            dist = torch.sqrt(torch.sum(diff ** 2, dim = 2))
            #dist.fill_diagonal_(float('inf'))
            self.d[h] = dist
        #"""


    def generate_overloaded_samples(self):
        """generates deep features like explained in the paper"""

        # get deep features
        with torch.no_grad():
            """
            for idx, data_batch in enumerate(self.dataloader):
                batch = self.runner.model.data_preprocessor(data_batch, True)
                input = batch['inputs']
                label = batch['data_samples'][0].gt_label
                
                self.v[label].append(self.runner.model.extract_feat(input)[0])
                self.batch_idx[label].append(idx)
            #"""

            #"""Pytorchifying
            counter = [0 for _ in range(self.num_classes)]

            for idx, data_batch in enumerate(self.dataloader):
                batch = self.runner.model.data_preprocessor(data_batch, True)
                input = batch['inputs']
                label = batch['data_samples'][0].gt_label
                
                self.v[label][counter[label]] = (self.runner.model.extract_feat(input)[0])
                self.batch_idx[label][counter[label]] = idx
                counter[label] += 1
            #"""



            """Pytorchifying:
            counter = [0 for _ in range(self.num_classes)]

            for idx, data_batch in enumerate(self.dataloader):

                batch = self.runner.model.data_preprocessor(data_batch, True)
                input = batch['inputs']
                labels = [i.gt_label for i in batch['data_samples']]
                #print(labels)
                
                feats = self.runner.model.extract_feat(input)[0]

                #self.v[labels[i]][counter[labels[i]]] = feats[i] for i in range(feat.shape[0])
                #self.batch_idx[labels][counter[labels]] = idx
                #counter[labels[i]] += 1 for i in range(labels.shape[0])


                #print(feats)
                for i, label in enumerate(labels): 
                    self.v[label][counter[label]] = feats[i]
                    self.batch_idx[label][counter[label]] = torch.Tensor([idx, i])
                    counter[label] += 1
                
                #"""
            #print(self.v)


        # get mutual distance matrix
        self.calc_mutual_distance_matrix()
        #print(self.d)
        #n_p, n_n= [], []

        for i in range(self.num_classes):

            #"""Pytorchifying
            if self.k[i] == 0:
                continue

            indices = [torch.topk(self.d[i][j], self.k[i], largest = False).indices for j in range(self.samples_per_class[i])]
            indices = [torch.topk(self.d[i][j], self.k[i] + 1, largest = False).indices for j in range(self.samples_per_class[i])]
            indices = torch.stack(indices, dim = 0)
            #print(indices.shape)

            n = self.v[i][indices]
            
            #print(n)
            #print(n.shape)
            #print(self.n[i].shape)
            
            w = (torch.abs(torch.randn(self.samples_per_class[i], self.r[i], self.k[i]))).to(torch.device("cuda"))
            w /= torch.linalg.norm(w, ord = 1, dim=2, keepdim = True)
            
            #print(self.batch_idx)
            for j, ind in enumerate(indices):
                #print(ind)
                #print(n[i][1 : ].shape)
                self.n[self.batch_idx[i][ind[0]].item()] = n[j][1 : ]
                self.w[self.batch_idx[i][ind[0]].item()] = w[j]

            #for j, pos in enumerate(self.batch_idx[i]):
                #self.n[pos[0] * self.b_size + pos[1]] = n[j]
                #self.w[pos[0] * self.b_size + pos[1]] = w[j]
            #
                        
        #print(self.n)

        """
            test_n = []
            for j in range(self.samples_per_class[i]):
                n = []
                # get deep features with shortest distance to feature vector with batch index of batch_idx[i][j]
                for x in torch.topk(self.d[i][j], self.k[i], largest = False).indices:
                    n.append(self.v[i][x])
                test_n.append(n)
                
                # sample weight vectors
                w = (torch.abs(torch.randn(self.r[i], self.k[i]))).to(torch.device("cuda"))
                w /= torch.linalg.norm(w, ord = 1, dim=1, keepdim = True)
                
                # define overloaded sample
                self.z['image'].append(self.batch_idx[i][j])
                self.z['n'].append(n)
                self.z['w'].append(w)
            n_n.append(test_n)
            
        #print(n_p)
        #print(n_n)
        
        assert len(n_p) == len(n_n), 'zeroth length not identical'
        for i in range(len(n_p)):
            assert len(n_p[i]) == len(n_n[i]), 'first length not identical'
            for j in range(len(n_p[i])):
                assert len(n_p[i][j]) == len(n_n[i][j]), 'second length not identical'
                for l in range(len(n_p[i][j])):
                    #print(torch.all(torch.eq(n_p[i][j][l], n_n[i][j][l])))
                    assert torch.all(torch.eq(n_p[i][j][l], n_n[i][j][l])), 'not identical elements'
                   
            #print (n2 == n)
        
        # zero out big variables for next iterations
        self.v = [[] for _ in range(self.num_classes)]
        self.batch_idx = [[] for _ in range(self.num_classes)]
        #"""



    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        
        while self._epoch < self._max_epochs and not self.stop_training:

            self.run_epoch()
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')

        # reset dataloader, so that the images are feed in the same permutation
        self.dataloader.sampler.reset_generator(self.seed, self._epoch)
        # get the overloaded samples
        self.generate_overloaded_samples()
        
        self.runner.model.train()
        self.dataloader.sampler.reset_generator(self.seed, self._epoch)
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        
        
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        
        """Pytorchifying
        outputs = self.runner.model.train_step(data_batch, 
                                              self.n[idx * self.b_size : (idx + 1) * self.b_size],
                                              self.w[idx * self.b_size : (idx + 1) * self.b_size],
                                              optim_wrapper = self.runner.optim_wrapper)
        #"""
        
        #"""Pytorchifying:
        label = data_batch['data_samples'][0].gt_label
        n_i = self.n[idx] if self.k[label] != 0 else torch.empty(0, 0)
        w_i = self.w[idx] if self.k[label] != 0 else torch.empty(0, 0)
        outputs = self.runner.model.train_step(
            data_batch, 
            n_i, w_i,
            optim_wrapper=self.runner.optim_wrapper) 
        #"""

        """
        outputs = self.runner.model.train_step(
            data_batch, 
            self.z['n'][idx], self.z['w'][idx],
            optim_wrapper=self.runner.optim_wrapper)
        #"""

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


# not tested 
@LOOPS.register_module()
class HardSamplingBasedTrainLoop(BaseLoop):
    """Loop based on epoch-based training based on: 
        Imbalanced Deep Learning by Minority Class Incremental Rectification by Qi Dong et al.
        (only class based sampling with relative comparison)
        Requirements: 
            new (non-default parameters in the init)

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        min_classes List(int): Determines the minority classes for the algorithm (only
            these will be used for mining hard samples, in the paper there is a criterion for that)
        min_thrs (float): The threshold below which an sample is count as hard positive (can be 1 because of top-k mining)
        max_thrs (float): The threshold above which an samples is count as hard negative (can be 0 since top-k mining)
        k (int): The k in top-k mining (how many hard positives and negatives will be mined)
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            min_classes: List[int],
            min_thrs: float,
            max_thrs: float,
            k: float, 
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

        # for Hard Sampling 
        self.num_classes = len(self.dataloader.dataset.metainfo['classes'])

        # minimal threshold, below is considered hard positive, conversely maximal threshold
        self.min_thrs = min_thrs
        self.max_thrs = max_thrs

        # used to say which classes should be used for hard sampling
        self.min_classes = set(min_classes)
        
        # the k in top k mining 
        self.k = k

        # store the output tensor of the hard samples, but not the position like in self.hard_samples
        self.out = torch.zeros(2, self.num_classes, self.k, self.num_classes)

        # safe position of hard samples, first dim is hard neg (0) or hard pos (1)
        # second dim is class of hard sample
        # for each class we have a min_heap and a max_heap so that only the top k samples for hard negative and hard positives are stored
        self.hard_samples = [[[] for _ in range(self.num_classes)] for _ in range(2)] #torch.empty((self.num_classes, ))

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter
    
    """
    def mine_hard_samples(self):
        for idx, data_batch in enumerate(self.dataloader):

            # get labels and predictions
            batch = self.runner.model.data_preprocessor(data_batch, True)
            inputs = batch['inputs']
            data_samples = batch['data_samples']
            labels = torch.cat([i.gt_label for i in data_samples])
            pred = self.runner.model.predict(inputs)
            pred = torch.stack([i.pred_score for i in pred])
            #print(pred)
            pred_labels = torch.argmax(pred, dim = 1)
            #print(pred_labels)

            ### MINE HARD SAMPLES

            # masks where the lables of min_classes are in the array-row
            min_labels_mask = torch.tensor([label.item() in self.min_classes for label in labels])
            # print(min_labels_mask)

            # get the indices in the batch of min_class samples
            true_ind = torch.nonzero(min_labels_mask).view((-1, )).to(torch.device("cuda"))
            # print(true_ind)
            
            # get the concrete labels of the min classes
            min_labels = labels[min_labels_mask]
            # print(min_labels)


            ## MINE HARD NEGATIVES

            # get the label for which the maximum prediction was made and the maximum prediction score
            max_pred_lab = pred[ min_labels_mask ].argmax(dim=1) 
            max_pred, max_pred_lab = pred[ min_labels_mask ].max(dim = 1)
            # print(max_pred)
            # print(max_pred_lab)
            
            # check if predictions are wrong and above the maximum threshold
            max_thrs_mask = torch.logical_and(torch.ne(max_pred_lab, min_labels), torch.gt(max_pred, self.max_thrs))
            # print(max_thrs_mask)
            
            # get indices where wrong prediction scores a above the threshold
            hard_neg_ind = torch.nonzero(max_thrs_mask)
            # print(hard_neg_ind)

            # get batch_indices of wrong prediction scores higher than the threshold
            neg_batch_ind = [true_ind[ind] for ind in hard_neg_ind]
            # print(neg_batch_ind)

            # write hard negative samples into self.hard_samples (need a min heap)
            for i, lab in enumerate(hard_neg_ind):
                if (len(self.hard_samples[0][max_pred_lab[lab]]) >= self.k)  and  (max_pred[lab] > self.hard_samples[0][max_pred_lab[lab]][0][0]):
                    heapq.heappop(self.hard_samples[0][max_pred_lab[lab]])
                    heapq.heappush(self.hard_samples[0][max_pred_lab[lab]], [max_pred[lab], idx, neg_batch_ind[i]])
                elif len(self.hard_samples[0][max_pred_lab[lab]]) < self.k:
                    heapq.heappush(self.hard_samples[0][max_pred_lab[lab]], [max_pred[lab], idx, neg_batch_ind[i]])
            # print(self.hard_samples)

            
            ## MINE HARD POSITIVES
            
            # get the prediction scores for min classes
            min_pred = pred[ min_labels_mask , min_labels ]
            # print(min_pred)

            # check if prediction scores are below a threshold
            min_thrs_mask = min_pred < self.min_thrs
            # print(min_thrs_mask)

            # get the indices where the prediction scores are below a threshold
            hard_pos_ind = torch.nonzero(min_thrs_mask)
            # print(indices)
            
            # get only the indices in the batch which have prediction score lower than the threshold
            pos_batch_ind = [true_ind[ind] for ind in hard_pos_ind]
            # print(batch_indices)

            # convert them into the original indices of the while dataset
            #original_indices = [idx, batch_indices]
            #print(original_indices)
            
            # write hard positives into self.hard_samples (need a max heap, -> minus before min_pred to convert to min_heap) 
            for i, lab in enumerate(hard_pos_ind):
                if (len(self.hard_samples[1][min_labels[lab]]) >= self.k)  and  ( - min_pred[lab] > self.hard_samples[1][min_labels[lab]][0][0]):
                    heapq.heappop(self.hard_samples[1][min_labels[lab]])
                    heapq.heappush(self.hard_samples[1][min_labels[lab]], [ - min_pred[lab], idx, pos_batch_ind[i]])
                elif len(self.hard_samples[1][min_labels[lab]]) < self.k:
                    heapq.heappush(self.hard_samples[1][min_labels[lab]], [ - min_pred[lab], idx, pos_batch_ind[i]])
            # print(self.hard_samples)
        """

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        

        while self._epoch < self._max_epochs and not self.stop_training:
            
            """
            with torch.no_grad(): 
                # mine hard samples
                self.mine_hard_samples()
                # print(self.hard_samples)
            """

            self.run_epoch()
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        

        """
        # first generate score for hard samples: 
        for idx, data_batch in enumerate(self.dataloader):
            for i in range(2):
                for j in range(self.num_classes):
                    for l, k in enumerate(self.hard_samples[i][j]):
                        if idx == k[1]:
                            # print(data_batch)
                            # print(data_batch['inputs'][k[2].cpu()])

                            batch = {}
                            batch['data_samples'] = None 
                            batch['inputs'] = data_batch['inputs'][k[2].item()].unsqueeze(0)
                            #print(data_batch)

                            batch = self.runner.model.data_preprocessor(batch, True)
                            # print(data_batch)
                            out = self.runner.model.predict(batch['inputs'])
                            print(out)
                            self.out[i, j, l] = out[0].pred_score
                            print(self.out)
        """





        for idx, data_batch in enumerate(self.dataloader):


            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


@LOOPS.register_module()
class IterBasedTrainLoop(BaseLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_iters: int,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_iters = int(max_iters)
        assert self._max_iters == max_iters, \
            f'`max_iters` should be a integer number, but get {max_iters}'
        self._max_epochs = 1  # for compatibility with EpochBasedTrainLoop
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)
        # get the iterator of the dataloader
        self.dataloader_iterator = _InfiniteDataloaderIterator(self.dataloader)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self._iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        super().__init__(runner, dataloader)

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


