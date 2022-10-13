from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from cv_preprocess import PreProcessor
import torchvision
import torch


class TrainDataset(Dataset):
    def __init__(self) -> None:
        self.datas = torchvision.datasets.MNIST('../data/', train=True, download=True)
        self.pre_processor = PreProcessor()

    def __getitem__(self, index):
        x, y = self.datas[index]
        x = self.pre_processor(x)
        return x, y
    
    def __len__(self):
        return len(self.datas)

    

def get_train_loader(configer):
    dataset = TrainDataset()
    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    b_s = configer.params['b_s']
    train_dataloader = DataLoader(train_set,
                            batch_size=b_s,
                            shuffle=True,
                            num_workers=configer.params['num_workers'])

    valid_dataloader = DataLoader(val_set,
                            batch_size=b_s,
                            shuffle=False,
                            num_workers=configer.params['num_workers'])

    info_str = '\nGetting Train DataLoader\n'
    info_str += f'\ttraining set num samples: {len(train_set)}\n'
    info_str += f'\ttraining set batch size: {b_s}\n'
    info_str += f'\ttraining set len dataloder: {len(train_dataloader)}\n\n'
    info_str += f'\tvalid set num samples: {len(val_set)}\n'
    info_str += f'\tvalid set batch size: {b_s}\n'
    info_str += f'\tvalid set len dataloder: {len(valid_dataloader)}\n'
    print(info_str)
    return train_dataloader, valid_dataloader
