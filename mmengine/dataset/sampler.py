# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterator, Optional, Sized, List

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS

# for DynamicSampler
import numpy as np
import random 

@DATA_SAMPLERS.register_module()
class DefaultSampler(Sampler):
    """The default data sampler for both distributed and non-distributed
    environment.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


@DATA_SAMPLERS.register_module()
class DOSSampler(DefaultSampler):
    """
        The only difference from DefaultSampler is that we can set the generator to have more
        control over the randomness in the trainloop
    """

    def __init__(self, **kwargs):
        super(DOSSampler, self).__init__(**kwargs)
        self.g = torch.Generator()

    def reset_generator(self, seed, epoch):
        self.g.manual_seed(seed + epoch)


    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=self.g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)


# like DefaultSampler above but DynamicSampler which is capable of ROS
@DATA_SAMPLERS.register_module()
class DynamicSampler(Sampler):
    # num_classes should be the number of classes 

    def __init__(self,
                 dataset: Sized,
                 num_classes: int,
                 enable_ROS: bool = False, 
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        self.enable_ROS = enable_ROS

        # get the label of each element in the dataset
        data_list = self.dataset.load_data_list()
        self.labels = [item['gt_label'] for item in data_list]

        # initialize sample_size like in the paper
        self.num_classes = num_classes
        self.average_sample_size = len(self.labels) // self.num_classes
        self.sample_size = np.full(self.num_classes, self.average_sample_size)

        # get indices of specific labels
        self.label_indices = [[] for _ in range(0, self.num_classes)]
        for idx, label in enumerate(self.labels):
            self.label_indices[label].append(idx)

        # check if this is necessary, i think not, but normally it has to be done
        self.num_samples = math.ceil((np.sum(self.sample_size) - self.rank) / self.world_size)
        self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""

        # get indices of elements which should be included in training 
        counts = [0] * self.num_classes
        indices = []
        if self.enable_ROS:
            for label in range (0, self.num_classes):
                for _ in range(int(self.sample_size[label])):
                    indices.append(random.choice(self.label_indices[label]))
                    counts[label] += 1
                #indices.append(random.choice(self.label_indices[label]) for _ in range (int(self.sample_size[label])))
        else:
            for label in range(0, self.num_classes):
                random_elements = random.sample(self.label_indices[label], min(int(self.sample_size[label]), len(self.label_indices[label])))
                indices.extend(random_elements)
                counts[label] += len(random_elements)
            #for idx, label in enumerate(self.labels):
            #   if counts[label] <= self.sample_size[label]:
            #      indices.append(idx)
            #     counts[label] += 1
        print(f"current distribution of samples from the dataset is : {counts}")

        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            indices = np.array(indices)
            np.random.seed(self.seed + self.epoch)
            np.random.shuffle(indices)
        
        # update num_samples and total_size for correct printed output of train process
        self.num_samples = math.ceil((len(indices) - self.rank) / self.world_size)
        self.total_size = self.num_samples * self.world_size

        # self.round_up would not be useful, as it would change the sample size

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    
    # dynamically set the class sizes based on f1-scores 
    def update_sample_size(self, f1_scores: List[float]) -> None:
        self.sample_size = (1 - f1_scores) / (np.sum(1- f1_scores)) * self.average_sample_size

# like above but relative over-sampling (ros)
@DATA_SAMPLERS.register_module()
class ROSSampler(Sampler):
    # ros_pct: ratio between sizes of upsampled minority classes to majority class
    # any class smaller than majority class is a minority class 
    # rus_maj_pct: ratio of kept elements in comparison to unkept ones of majority class 
    # num_classes is the number of classes in the dataset 

    def __init__(self,
                 dataset: Sized,
                 num_classes: int, 
                 ros_pct: float = 1,
                 rus_maj_pct: float = 1, 
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        # get the initial count of each class
        self.num_classes = num_classes
        self.ros_pct = ros_pct
        self.rus_maj_pct = rus_maj_pct
        self.label_counts = np.full(self.num_classes, 0)
        data_list = self.dataset.load_data_list()
        self.labels = [item['gt_label'] for item in data_list]
        for item in data_list:
            self.label_counts[item['gt_label']] += 1

        # calculate how many samples need to be duplicated for each class
        self.factors = np.round(max(self.label_counts) * self.ros_pct * self.rus_maj_pct / self.label_counts, 2)
        
        # correctly set the number of samples of the majority class
        self.factors[np.argmax(self.label_counts)] = 1 * self.rus_maj_pct

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""

        # get indices of elements which should be included in training
        indices = []
        counts = [0] * self.num_classes
        for idx, label in enumerate(self.labels):
            prob = self.factors[label] - int(self.factors[label])
            replications =  int((np.ceil(self.factors[label]) if np.random.rand() < prob else np.floor(self.factors[label]))) 
            indices += [idx] * replications
            counts[label] += replications
        print(f"current distribution of samples from the dataset is : {counts}")

        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            indices = np.array(indices)
            np.random.seed(self.seed + self.epoch)
            np.random.shuffle(indices)
        
        # update num_samples and total_size for correct printed output of train process
        self.num_samples = math.ceil((len(indices) - self.rank) / self.world_size)
        self.total_size = self.num_samples * self.world_size

        # self.round_up would not be useful, as it would change the sample size

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
 
# like above but relative-under-sampling (rus)
@DATA_SAMPLERS.register_module()
class RUSSampler(Sampler):
    # rus_pct should be ratio between sizes of undersampled majority classes to minority class
    # any class bigger than minority class is a majority class 
    # ros_min_pct: array in which element i in the ratio of kept elements in comparison to unkept ones of class i 
    # num_classes is the number of classes in the dataset 

    def __init__(self,
                 dataset: Sized,
                 num_classes: int, 
                 rus_pct: float = 1,
                 ros_min_pct: float = 1, 
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        # get the initial count of each class
        self.num_classes = num_classes
        self.rus_pct = rus_pct
        self.ros_min_pct = ros_min_pct
        self.label_counts = np.full(self.num_classes, 0)
        data_list = self.dataset.load_data_list()
        self.labels = [item['gt_label'] for item in data_list]
        for item in data_list:
            self.label_counts[item['gt_label']] += 1

        # calculate how many samples need to be duplicated for each class
        self.factors = np.round(min(self.label_counts) * self.rus_pct * self.ros_min_pct / self.label_counts, 2)
        
        # correctly set the number of samples of the minority class
        self.factors[np.argmin(self.label_counts)] = 1 * self.ros_min_pct

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""

        # get indices of elements which should be included in training
        indices = []
        counts = [0] * self.num_classes
        for idx, label in enumerate(self.labels):
            prob = self.factors[label] - int(self.factors[label])
            replications =  int((np.ceil(self.factors[label]) if np.random.rand() < prob else np.floor(self.factors[label]))) 
            indices += [idx] * replications
            counts[label] += replications
        print(f"current distribution of samples from the dataset is : {counts}")

        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            indices = np.array(indices)
            np.random.seed(self.seed + self.epoch)
            np.random.shuffle(indices)
        
        # update num_samples and total_size for correct printed output of train process
        self.num_samples = math.ceil((len(indices) - self.rank) / self.world_size)
        self.total_size = self.num_samples * self.world_size

        # self.round_up would not be useful, as it would change the sample size

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
 


@DATA_SAMPLERS.register_module()
class InfiniteSampler(Sampler):
    """It's designed for iteration-based runner and yields a mini-batch indices
    each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.world_size)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        yield from self.indices

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        pass
