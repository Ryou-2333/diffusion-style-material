import torch.utils.data as data
import pytorch_lightning as pl
import numpy.random


class RandomNoiseDataSet(data.Dataset):
    def __init__(self, seed, count, dim):
        '''
            Dummy Dataset
        '''
        super().__init__()
        self.seed = seed
        self.count = count
        #numpy.random.RandomState(seed)
        self.datalist = [numpy.random.normal(0, 1, dim) for _ in range(count)]
    
    def __getitem__(self, index):
        return self.datalist[index]
    
    def __len__(self):
        return self.count

class RandomNoiseDataLoader(pl.LightningDataModule):
    def __init__(self, opt, cfg, device_num):
        super().__init__()
        self.seed = cfg["params"]["seed"]
        self.count = cfg["params"]["count"]
        self.dim = cfg["params"]["dim"]
        self.dataset = RandomNoiseDataSet(self.seed, self.count, self.dim)
        self.dataLoader = data.DataLoader(
            dataset     = self.dataset,
            batch_size  = opt.batch_size,
            shuffle     = False,
            num_workers = opt.num_threads,
            drop_last   = device_num > 1,
        )

    def train_dataloader(self):
        return self.dataLoader

    def test_dataloader(self):
        return self.dataLoader

    def get_data_size(self):
        return len(self.dataset)

    def __iter__(self):
        for d in self.dataLoader:
            yield d

    def __len__(self):
        return len(self.dataLoader)