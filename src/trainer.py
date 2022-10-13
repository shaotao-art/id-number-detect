import time
from matplotlib.pyplot import axis
from tqdm import tqdm
import torch

class Trainer():
    def __init__(self, num_epoch) -> None:
        self.num_epoch = num_epoch

    def _train_step(self, model, batch, criterion, device):
        x, y = batch
        x, y = x.squeeze().to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        return loss


    def _validation_steps(self, dataloader, model, device):
        with torch.no_grad():
            # loop = tqdm(dataloader)
            num_right = 0
            num_sample = 0
            for i, (x, y) in enumerate(dataloader):
                num_sample += len(y)
                x, y = x.to(device), y.to(device)
                pred = model(x)
                pred = torch.argmax(pred, dim=1)
                num_right += torch.sum(pred == y)
            return num_right / num_sample


    def run(self, train_dataloader, valid_dataloader, model, optimizer, criterion, start_epoch, utiler, device):
        for _ in range(start_epoch, self.num_epoch):
            cur_epoch = self._train_one_epoch(train_dataloader, model, optimizer, criterion, start_epoch, device)
            self._validation(valid_dataloader, model, device)
            utiler.save_ckp(model, optimizer, cur_epoch)

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
                print(f'epoch:[{cur_epoch:>3d}/{self.num_epoch:>3d}],\
                        batch:[{i:>3d}/{len(dataloader):>3d}],\
                        time used: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec, \
                        loss:[{loss:>4f}]')
        end = time.time()
        time_used = int(end - start)

        print(f'epoch:[{cur_epoch:>3d} done!,\
                        avg loss: {.1}, \
                        time to run this epoch: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec')

        return cur_epoch + 1


    def _validation(self, dataloader, model, device):
        print('\nValidation...')
        model.eval()
        model.to(device)
        start = time.time()
        score = self._validation_steps(dataloader, model, device)
        end = time.time()
        time_used = int(end - start)
        print(f'score: {score}, time to run validation: {(time_used // 60):>2d} min {(time_used % 60):>2d} sec.')

