
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        next(self.loader_iters[0])
        next(self.loader_iters[1])
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches

# get sampling probabilities for each class
def get_sampling_probabilities(class_count, mode='instance'):
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)
    return sampling_probabilities

 # modify dataloader so that it samples based on probabilities
def modify_loader(loader, samples_per_class, mode):
    class_count = samples_per_class
    sampling_probs = get_sampling_probabilities(class_count, mode=mode)
    
    dr = []
    for count, value in enumerate(class_count):
        dr.append(torch.full((value,), count))  
    dr = torch.cat(dr)

    sample_weights = sampling_probs[dr]

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    print(loader.dataset)
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    print(mod_loader.dataset)
    return mod_loader


# get combo loader consisting of instance based sampling and class based sampling 
def get_combo_loader(loader, samples_per_class):
    imbalanced_loader = loader
    balanced_loader = modify_loader(loader, samples_per_class, mode = 'class')

    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader

