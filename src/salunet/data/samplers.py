import torch


class RepeatedAugmentationSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_repeats: int = 1, shuffle: bool = True):
        self.data_source = data_source
        self.num_repeats = num_repeats
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.arange(n).repeat(self.num_repeats)
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.data_source) * self.num_repeats
