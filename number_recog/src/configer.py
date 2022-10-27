import torch

class Configer:
    def __init__(self) -> None:
        self.params = {
            'device': torch.device('mps'),
            'num_epoch': 100,
            'l_r': 1e-4,
            'b_s': 64,
            'num_workers': 2,
            'ckp_path': './ckp',
            'model_save_path': './ckp/model.ckp'
        }    

    def __str__(self) -> str:
        params = self.params.items()
        out = ''
        out += '\nTraining Configs\n'
        for k, v in params:
            out += f"\t{k}:  {v}\n"
        out +='\n'
        return out