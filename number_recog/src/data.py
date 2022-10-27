from distutils.log import INFO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.cv_preprocess import PreProcessor
import torchvision
import torch
from PIL import ImageOps

class TrainDataset(Dataset):
    def __init__(self) -> None:
        self.datas = torchvision.datasets.ImageFolder(root='/Users/starfish/Desktop/python_proj/ML-funny-projects/rasp/data_gen')
        self.pre_processor = PreProcessor()

    def __getitem__(self, index):
        return self._apply_preprocess(index, self.pre_processor)
    
    def __len__(self):
        return len(self.datas)

    def _apply_preprocess(self, index, pre_processor):
        x, y = self.datas[index]
        x = ImageOps.grayscale(x)
        x = pre_processor(x)
        return x, y

class TestDataset(Dataset):
    def __init__(self) -> None:
        self.datas = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        self.pre_processor = PreProcessor()

    def __getitem__(self, index):
        return self._apply_preprocess(index, self.pre_processor)

    def __len__(self):
        return len(self.datas)

    def _apply_preprocess(self, index, pre_processor):
        x, y = self.datas[index]
        x = ImageOps.grayscale(x)
        x = pre_processor(x)
        return x, y
    

def get_train_loader(configer, train_size=0.7, valid_b_s=32):
    dataset = TrainDataset()
    N = len(dataset)
    N_train = int(N * train_size)
    N_valid = N - N_train
    train_data, valid_data = torch.utils.data.random_split(dataset, [N_train, N_valid])    

    b_s = configer.params['b_s']
    train_loader = DataLoader(train_data,
                            batch_size=b_s,
                            shuffle=True,
                            num_workers=configer.params['num_workers'])
    valid_loader = DataLoader(valid_data,
                            batch_size=valid_b_s,
                            shuffle=False,
                            num_workers=configer.params['num_workers'])
    info_str = '\nGetting Train DataLoader\n'
    info_str += f'\ttraining set num samples: {len(train_data)}\n'
    info_str += f'\ttraining set batch size: {b_s}\n'
    info_str += f'\ttraining set len dataloder: {len(train_loader)}\n'
    info_str += f'\tvalid set num samples: {len(valid_data)}\n'
    info_str += f'\tvalid set batch size: {valid_b_s}\n'
    info_str += f'\tvalid set len dataloder: {len(valid_loader)}\n'
    print(info_str)
    return train_loader, valid_loader

def get_test_loader(configer):
    dataset = TestDataset()
    b_s = configer.params['b_s']
    dataloader = DataLoader(dataset,
                            batch_size=b_s,
                            shuffle=True,
                            num_workers=configer.params['num_workers'])
    info_str = '\nGetting Test DataLoader\n'
    info_str += f'\tTesting set num samples: {len(dataset)}\n'
    info_str += f'\tTesting set batch size: {b_s}\n'
    info_str += f'\ttesting set len dataloder: {len(dataloader)}\n'
    print(info_str)
    return dataloader