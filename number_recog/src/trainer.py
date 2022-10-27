import time
from src.loops import train_step, valid
import torch

class Trainer():
    def __init__(self, num_epoch) -> None:
        self.num_epoch = num_epoch

    def _train_step(self, model, batch, criterion, device):
        return train_step(model, batch, criterion, device)

    def _validation_steps(self, dataloader, model, device):
        return valid(dataloader, model, device)
    
    def run(self, train_dataloader, valid_dataloader, model, optimizer, criterion, start_epoch, utiler, device):
        for epoch in range(start_epoch, self.num_epoch):
            self._train_one_epoch(train_dataloader, model, optimizer, criterion, epoch, device)
            self._validation(valid_dataloader, model, device)
            utiler.save_ckp(model, optimizer, epoch)

    def _train_one_epoch(self, dataloader, model, optimizer, criterion, cur_epoch, device):
        print('\nTraining...')

        model.train()
        model.to(device)
        start = time.time()
        loss_lst = []
        for i, batch in enumerate(dataloader):
            loss = self._train_step(model, batch, criterion, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_lst.append(loss.item())
            if i % (len(dataloader) // 10) == 0:
                now = time.time()
                time_used = int(now - start)
                print(f'epoch:[{cur_epoch:>3d}/{self.num_epoch:>3d}], batch:[{i:>3d}/{len(dataloader):>3d}], time used: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec, loss:[{loss:>4f}]')

        end = time.time()
        time_used = int(end - start)

        print(f'\nepoch:[{cur_epoch:>3d} done!, avg loss: {torch.tensor(loss_lst).mean().item()}, time to run this epoch: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec')


    def _validation(self, dataloader, model, device):
        print('\nValidation...')

        model.eval()
        model.to(device)
        start = time.time()
        score = self._validation_steps(dataloader, model, device)
        end = time.time()
        time_used = int(end - start)
        print(f'score: {score}, time to run validation: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec.')

