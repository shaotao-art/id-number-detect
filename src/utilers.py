import pathlib
import numpy as np
import torch


class Utiler:
    def __init__(self, model_save_path='./ckp/model.ckp', ckp_path='./ckp/', seed=2022) -> None:
        self.ckp_path = ckp_path
        self.model_save_path = model_save_path
        self.seed = seed
    
    def apply(self):
        self._mkdir()
        self._set_seed()

    def load_ckp(self, model, optimizer, device):
        checkpoint = torch.load(self.model_save_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f'Loading ckp from {self.model_save_path}...\nTraining will start from epoch {start_epoch}')
        return start_epoch

    def save_ckp(self, model, optimizer, epoch):
        torch.save({'optimizer': optimizer.state_dict(), 
            'model': model.state_dict(), 
            'epoch': epoch + 1}, 
            self.model_save_path)
        print(f'Ckp to {self.model_save_path}...\nCheckpointing epoch {epoch}')

    def _mkdir(self):
        pathlib.Path(self.ckp_path).mkdir(parents=True, exist_ok=True)
        print(f'Making fold for ckp\nThe path is {self.ckp_path}')

    def _set_seed(self):
        """
        set random seed
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        print(f'Setting seed to {self.seed}')