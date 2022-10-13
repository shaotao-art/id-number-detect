from configer import Configer
from data import get_train_loader
from model import NNet
from trainer import Trainer
from utilers import Utiler
import os
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

def main():
    # utiler and configer
    utiler = Utiler()
    utiler.apply()
    print(utiler)
    configer = Configer()
    device = configer.params['device']
    print(configer)

    # model
    model = NNet()
    print(model)

    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=configer.params['l_r'])
    criterion = nn.CrossEntropyLoss()

    # get dataloader
    train_dataloader, valid_dataloader = get_train_loader(configer)


    # load ckp
    if os.path.exists(configer.params['model_save_path']):
        start_epoch = utiler.load_ckp(model, optimizer, device)
    else:
        start_epoch = 0

    # trainer
    trainer = Trainer(configer.params['num_epoch'])
    trainer.run(train_dataloader, valid_dataloader, model, optimizer, criterion, start_epoch, utiler, device)

if __name__ == '__main__':
    main()







