from src.configer import Configer
from src.data import get_train_loader
from src.data import get_test_loader
from src.model import NNet
from src.trainer import Trainer
from src.utilers import Utiler
import os
from torch import nn
from torch import optim
from torch.optim import lr_scheduler


def main():
    utiler = Utiler()
    utiler.apply()

    configer = Configer()
    device = configer.params['device']
    print(configer)


    model = NNet()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=configer.params['l_r'])
    criterion = nn.CrossEntropyLoss()
    train_dataloader, valid_dataloader = get_train_loader(configer)

    if os.path.exists(configer.params['model_save_path']):
        start_epoch = utiler.load_ckp(model, optimizer, device)
    else:
        start_epoch = 0
        model.to(device)
        print(f'No ckp found\ntraining will start from srcatch')

    trainer = Trainer(configer.params['num_epoch'])
    trainer.run(train_dataloader,
                valid_dataloader,
                model,
                optimizer,
                criterion,
                start_epoch,
                utiler,
                device)

if __name__ == "__main__":
    main()






