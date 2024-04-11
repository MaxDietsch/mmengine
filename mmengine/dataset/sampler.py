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
    """ implements dynamic sampling like stated in the paper: 
        Dynamic Sampling in Convolutional Neural Networks for Imbalanced Data Classification by Pouyanfar et al.

        Args: 
            enable_ROS (bool): if set to true then oversampling is applied if samples_size is greater 
                        than the actual samples of that class
    """

    def __init__(self,
                 dataset: Sized,
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

        # for Dynamic Sampling
        # if set to true then oversampling is applied if samples_size is greater than the actual samples of that class
        self.enable_ROS = enable_ROS

        # get the label of each element in the dataset
        data_list = self.dataset.load_data_list()
        self.labels = torch.tensor([item['gt_label'] for item in data_list])
        #numpy: self.labels = [item['gt_label'] for item in data_list]

        # initialize sample_size like in the paper
        self.num_classes = len(self.dataset.metainfo['classes'])
        #self.average_sample_size = len(self.labels) // self.num_classes
        
        # exclude majority class: 
        element_counts = self.labels.bincount()
        most_common_element = element_counts.argmax()
        count = element_counts[most_common_element].item()
        self.average_sample_size = (len(self.labels) - count) // (self.num_classes - 1)

        self.sample_size = torch.full((self.num_classes, ), self.average_sample_size, dtype = torch.int)
        # numpy: self.sample_size = np.full(self.num_classes, self.average_sample_size)

        # get indices of specific labels
       
        self.label_indices = [torch.tensor([], dtype=torch.int) for _ in range(self.num_classes)]
        for class_idx in range(self.num_classes):
            # Find indices where labels match the current class index
            class_indices = (self.labels == class_idx).nonzero().squeeze()
            self.label_indices[class_idx] = class_indices.float()
        #print(f'label_indices: {self.label_indices}')
        
        # numpy: 
        """
        self.label_indices = [[] for _ in range(0, self.num_classes)]
        for idx, label in enumerate(self.labels):
            self.label_indices[label].append(idx)
        """

        # check if this is necessary, i think not, but normally it has to be done
        self.num_samples = math.ceil((torch.sum(self.sample_size) - self.rank) / self.world_size)
        self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""

        # get indices of elements which should be included in training 
        counts = torch.zeros(self.num_classes, dtype = torch.int)
        indices = torch.empty((self.sample_size.int().sum(), ), dtype=torch.long)
        start_idx = 0
        if self.enable_ROS:
            for label in range(0, self.num_classes): 
                num_samples = int(self.sample_size[label])
                end_idx = start_idx + num_samples
                if num_samples > 0 and len(self.label_indices[label]) > 0:
                    sampled_indices = self.label_indices[label][torch.multinomial(self.label_indices[label], num_samples, replacement=True)]
                    indices[start_idx:end_idx] = sampled_indices
                    counts[label] += num_samples
                start_idx = end_idx
        else: 
            for label in range(0, self.num_classes): 
                num_samples = int(self.sample_size[label])
                end_idx = start_idx + num_samples
                if num_samples > 0 and len(self.label_indices[label]) > 0:
                    sampled_indices = self_label_indices[label][torch.multinomial(self.label_indices[label], min(num_samples, len(self.label_indices[label])), replacement=True)]
                    indices[start_idx:end_idx] = sampled_indices
                    counts[label] += num_samples
                start_idx = end_idx

        print(f"current distribution of samples from the dataset is : {counts}")
        #print(f'current samples size is: {self.sample_size}')

        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = indices[torch.randperm(indices.size(0), generator=g)].tolist()

        # numpy:
        """
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

        if self.shuffle:
            indices = np.array(indices)
            np.random.seed(self.seed + self.epoch)
            np.random.shuffle(indices)
        """
        
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
    def update_sample_size(self, f1_scores: torch.Tensor) -> None:
        self.sample_size = (1 - f1_scores) / (torch.sum(1 - f1_scores)) * self.average_sample_size


# like above but relative over-sampling (ros)
@DATA_SAMPLERS.register_module()
class ROSSampler(Sampler):
    """ implements random over-sampling of minority classes
        every class, that is not as big as the biggest class is a minority class: 
        Args:
            ros_pct (float): ratio between sizes of upsampled minority classes to majority class
                             any class smaller than majority class is a minority class
            rus_maj_pct: ratio of kept elements in comparison to unkept ones of majority class
    """

    def __init__(self,
                 dataset: Sized,
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
        
        # for ROS
        # get the initial count of each class
        self.num_classes = len(self.dataset.metainfo['classes'])
        self.ros_pct = ros_pct
        self.rus_maj_pct = rus_maj_pct
        data_list = self.dataset.load_data_list()

        self.labels = torch.tensor([item['gt_label'] for item in data_list])

        self.label_counts = torch.bincount(self.labels, minlength = self.num_classes)
        # print(f'label counts: {self.label_counts}')

        # calculate how often a sample needs to be duplicated for each class
        self.factors = torch.round(self.label_counts.max() * self.ros_pct * self.rus_maj_pct / self.label_counts, decimals = 2)

        # correctly set the number of samples of the majority class
        self.factors[torch.argmax(self.label_counts)] = 1 * self.rus_maj_pct
        # print(f'factors: {self.factors}')

        """
        # numpy:
        self.label_counts = np.full(self.num_classes, 0)
        self.labels = [item['gt_label'] for item in data_list]
        for item in data_list:
            self.label_counts[item['gt_label']] += 1

        # calculate how many samples need to be duplicated for each class
        self.factors = np.round(max(self.label_counts) * self.ros_pct * self.rus_maj_pct / self.label_counts, 2)
        
        # correctly set the number of samples of the majority class
        self.factors[np.argmax(self.label_counts)] = 1 * self.rus_maj_pct
        """

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

        #print(f'wolrd_size: {world_size}')
        #print(f'rank: {rank}')
        #print(f'num samples: {self.num_samples}')
        #print(f'total size: {self.total_size}')

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""

        # get indices of elements which should be included in training
        indices = []
        counts = torch.zeros(self.num_classes, dtype=torch.int32)

        for idx, label in enumerate(self.labels):
            # Probability part of the factor
            prob = self.factors[label] - int(self.factors[label])  
            # Determine replications based on probability
            replications = int(torch.ceil(self.factors[label]) if torch.rand(1) < prob else torch.floor(self.factors[label]))
            # Add index 'replications' times
            indices.extend([idx] * replications)
            counts[label] += replications 

        print(f"current distribution of samples from the dataset is : {counts}")

        # deterministically shuffle based on epoch and seed
        #print(f'indices before shuffle: {indices}')
        #print(f'length of indices: {len(indices)}')
        if self.shuffle:
            indices = torch.tensor(indices)
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = indices[torch.randperm(indices.size(0), generator=g)].tolist()

        # numpy
        """
        counts = [0] * self.num_classes
        for idx, label in enumerate(self.labels):
            prob = self.factors[label] - int(self.factors[label])
            replications =  int((np.ceil(self.factors[label]) if np.random.rand() < prob else np.floor(self.factors[label]))) 
            indices += [idx] * replications
            counts[label] += replications

        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            indices = np.array(indices)
            np.random.seed(self.seed + self.epoch)
            np.random.shuffle(indices)
        """

        # update num_samples and total_size for correct printed output of train process
        self.num_samples = math.ceil((len(indices) - self.rank) / self.world_size)
        self.total_size = self.num_samples * self.world_size

        # self.round_up would not be useful, as it would change the sample size

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        #print(f'indices after rank, world_size... : {indices}')
        #print(f'new length of indices: {len(indices)}')
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
    """ implements random under-sampling of majority classes
        every class, that is not as small as the smallest class is a majority class: 
        Args:
            rus_pct (float): should be ratio between sizes of undersampled majority classes to minority class
                             any class bigger than minority class is a majority class
            ros_min_pct (float): ratio of upsampled minority class to normal minority class
    """

    def __init__(self,
                 dataset: Sized,
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
        self.num_classes = len(self.dataset.metainfo['classes'])
        self.rus_pct = rus_pct
        self.ros_min_pct = ros_min_pct
        data_list = self.dataset.load_data_list()

        self.labels = torch.tensor([item['gt_label'] for item in data_list])
        
        self.label_counts = torch.bincount(self.labels, minlength = self.num_classes)
        # print(f'label counts: {self.label_counts}')

        # calculate how often a sample needs to be duplicated for each class
        self.factors = torch.round(self.label_counts.min() * self.rus_pct * self.ros_min_pct / self.label_counts, decimals = 2)

        # correctly set the number of samples of the majority class
        self.factors[torch.argmin(self.label_counts)] = 1 * self.ros_min_pct
        # print(f'factors: {self.factors}')

        """
        # numpy: 
        self.label_counts = np.full(self.num_classes, 0)
        self.labels = [item['gt_label'] for item in data_list]

        for item in data_list:
            self.label_counts[item['gt_label']] += 1

        # calculate how many samples need to be duplicated for each class
        self.factors = np.round(min(self.label_counts) * self.rus_pct * self.ros_min_pct / self.label_counts, 2)
        
        # correctly set the number of samples of the minority class
        self.factors[np.argmin(self.label_counts)] = 1 * self.ros_min_pct
        """

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
        counts = torch.zeros(self.num_classes, dtype=torch.int32)

        
        for idx, label in enumerate(self.labels):
            # Probability part of the factor
            prob = self.factors[label] - int(self.factors[label])  
            # Determine replications based on probability
            replications = int(torch.ceil(self.factors[label]) if torch.rand(1) < prob else torch.floor(self.factors[label]))
            # Add index 'replications' times
            indices.extend([idx] * replications)
            counts[label] += replications 

        print(f"current distribution of samples from the dataset is : {counts}")

        # deterministically shuffle based on epoch and seed
        #print(f'indices before shuffle: {indices}')
        #print(f'length of indices: {len(indices)}')
        if self.shuffle:
            indices = torch.tensor(indices)
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = indices[torch.randperm(indices.size(0), generator=g)].tolist()

        # numpy:
        """
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
        """

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
