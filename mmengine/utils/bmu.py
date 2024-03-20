
from mmpretrain import datasets
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
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
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

 # modify dataloader so that it samples based on probabilities
def modify_loader(loader, mode, ep=None, n_eps=None):
    class_count = np.unique(loader.dataset.dr, return_counts=True)[1]
    sampling_probs = get_sampling_probabilities(class_count, mode=mode)
    sample_weights = sampling_probs[loader.dataset.dr]

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    return mod_loader


# get combo loader consisting of instance based sampling and class based sampling 
def get_combo_loader(loader):
    imbalanced_loader = loader
    balanced_loader = modify_loader(loader, mode='class')

    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader


def get_train_datasets(data_root, ann_file, data_prefix, with_label, classes, pipeline, see_classes):

    train_dataset = CustomDataset(data_root=data_root,
                                  ann_file=ann_file,
                                  data_prefix=data_prefix,
                                  with_label=with_label,
                                  classes=classes,
                                  pipeline=pipeline
                                  )
    if see_classes:
        print(20 * '*')
        for c in range(len(np.unique(train_dataset.dr))):
            exs_train = np.count_nonzero(train_dataset.dr== c)
            print('Found {:d} train examples of class {}'.format(exs_train, c))

    return train_dataset


def get_train_loader(batch_size, num_workers, data_root, ann_file, data_prefix, with_label, classes, pipeline, see_classes=True):

    train_dataset = get_train_dataset(data_root, ann_file, data_prefix, with_label, classes, pipeline, see_classes)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True, drop_last=True)
    return train_loader
