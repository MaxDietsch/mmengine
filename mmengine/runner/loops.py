import logging
import bisect
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from .amp import autocast
from .base_loop import BaseLoop
from .utils import calc_dynamic_intervals

# for usage of self made sampler
import numpy as np
from mmengine.dataset import DynamicSampler 

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
        threshold = 0.3
        cls = [i for i in range(self.runner.model.head.fc.out_features)]

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
                f1_scores = np.array(mtrcs['single-label/f1-score_classwise']) / 100
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
class DOSTrainLoop(BaseLoop):
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
        self.num_classes = len(self.dataloader.dataset.metainfo['classes'])

        #TODO: make samples_per_class and r as parameter of __init__
        self.samples_per_class = samples_per_class #[0, 45, 132, 539]
        self.r = r #[0, 3, 2, 1]
        self.k = r #[0, 3, 2, 1]

        self.d = [torch.zeros((i, i)) for i in self.samples_per_class]
        self.batch_idx = [[] for _ in range(self.num_classes)]
        self.v = [[] for _ in range(self.num_classes)]
        
        self.z = {'image': [], 'n': [], 'w': []}


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
        for h in range(self.num_classes):
            for i in range(self.samples_per_class[h]):
                for j in range(i, self.samples_per_class[h]):
                    if i == j:
                        self.d[h][i, j] = 99999999
                    self.d[h][i, j] = torch.norm(self.v[h][i] - self.v[h][j])
                    self.d[h][j, i] = self.d[h][i, j]

    def generate_overloaded_samples(self):

        # get deep features
        with torch.no_grad():
            for idx, data_batch in enumerate(self.dataloader):
                batch = self.runner.model.data_preprocessor(data_batch, True)
                input = batch['inputs']
                label = batch['data_samples'][0].gt_label
                
                self.v[label].append(self.runner.model.extract_feat(input)[0])
                self.batch_idx[label].append(idx)

        # get mutual distance matrix
        self.calc_mutual_distance_matrix()

        for i in range(self.num_classes):
            for j in range(self.samples_per_class[i]):
                n = []
                
                # get deep features with shortest distance to feature vector with batch index of batch_idx[i][j]
                for x in torch.topk(self.d[i][j], self.k[i], largest = False).indices:
                    n.append(self.v[i][x])
                
                # sample weight vectors
                w = (torch.abs(torch.randn(self.r[i], self.k[i]))).to(torch.device("cuda"))
                w /= torch.norm(w, dim=1, keepdim = True)
                
                # define overloaded sample
                self.z['image'].append(self.batch_idx[i][j])
                self.z['n'].append(n)
                self.z['w'].append(w)
        
        # zero out big variables
        self.v = [[] for _ in range(self.num_classes)]
        self.batch_idx = [[] for _ in range(self.num_classes)]



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

        self.generate_overloaded_samples()

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
        outputs = self.runner.model.train_step(
            data_batch, 
            self.z['n'][idx], self.z['w'][idx],
            optim_wrapper=self.runner.optim_wrapper)

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
class CoSenTrainLoop(BaseLoop):
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
            s_freq: int, 
            s_samples_per_class: List[int],
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
        self.num_classes = len(self.dataloader.dataset.metainfo['classes'])
        self.s_freq = s_freq
        self.s_samples_per_class = s_samples_per_class
        self.d = torch.zeros((self.num_classes, self.num_classes))
        self.v = [[] for _ in range(self.num_classes)]




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

    # d = [[label]: [], [], [], [] ]

    def calc_mutual_distance_matrix(self):

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                dist = 0 
                if i == j:
                    distances = torch.cdist(self.v[j], self.v[j])
                    print(distances)
                    print(torch.sort(distances)[0])
                    d[i, j] = torch.sort(distances)[ : , 1 ].mean()
                    continue

                for k, t1 in enumerate(self.v[i]):

                    diff = self.v[j] - t1
                    dist += torch.min(torch.sqrt(torch.sum(diff**2, axis = -1)))


                    #dist += torch.min(torch.cdist(t1.unsqueeze(0), torch.stack(self.v[j])))
                self.d[i, j] = dist / self.s_samples_per_class[i]



    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        self.runner.model.train()

        while self._epoch < self._max_epochs and not self.stop_training:
            
            # update S only in specific epochs like in the paper
            
            with torch.no_grad():
                if self._epoch % self.s_freq == 0:

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

                    print(self.v[0].shape)


                    # calculate S 
                    self.calc_mutual_distance_matrix()
                    print(self.d)



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


