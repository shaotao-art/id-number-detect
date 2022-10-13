import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class TestDataset(Dataset):
    def __init__(self) -> None:
        self.datas = torchvision.datasets.MNIST('../data/', train=False, download=True)
        self.pre_processor = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        x, y = self.datas[index]
        x = self.pre_processor(x)
        return x, y

    def __len__(self):
        return len(self.datas)


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

def test(mode, dataloader):
    pass

def infer(model, x):
    pass