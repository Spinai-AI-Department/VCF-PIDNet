import torch
from tqdm import tqdm
from time import time
try:
    from .utils import noise_injection
except:
    from utils import noise_injection
noise_mean = 0.0
noise_std = 0.01

class Trainer():

    def __init__( self, model, optimizer, scheduler, metric_func, loss_func, device, logger):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_func = metric_func
        self.loss_func = loss_func
        self.device = device
        self.logger = logger

        self.loss = 0
        self.sem_loss = 0
        self.bd_loss = 0
        self.metric = dict()
        for func_name in self.metric_func.keys():
            self.metric[func_name] = 0

    def train(self, dataloader, epoch_id=0):

        self.model.train()

        for batch_id, (x, y, bd) in enumerate(tqdm(dataloader)):


            # Load Data
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.long)
            bd = bd.to(self.device, dtype=torch.float32)


            # Prediction
            # loss, y_pred, acc, loss_list = self.model(x, y, bd)
            y_pred_model = self.model(x)

            # Loss

            #loss = self.loss_func(y_pred, y)
            loss, y_pred, acc, loss_list = self.loss_func(y_pred_model, y, bd)


            # Loss

            #loss = self.loss_func(y_pred, y)


            # Update
            self.optimizer.zero_grad()
            loss.backward()

            #noise_injection(self.model, "weight", noise_mean, noise_std)

            self.optimizer.step()

            # record
            for key, func in self.metric_func.items():

                self.metric[key] += func(y_pred, y)
            self.loss += loss.item()
            self.sem_loss += loss_list[0].item()
            self.bd_loss += loss_list[1].item()
            self.logger.debug(f"TRAINER | train epoch: {epoch_id}, batch: {batch_id}/{len(dataloader)-1}, loss: {loss.item()}, sem_loss: {loss_list[0].item()}, bd_loss: {loss_list[1].item()}")

        self.scheduler.step()
        for key, value in self.metric.items():

            self.metric[key] = float(torch.round(value/len(dataloader), decimals=4))

        self.loss = round(self.loss/len(dataloader), 4)
        self.sem_loss = round(self.sem_loss/len(dataloader), 4)
        self.bd_loss = round(self.bd_loss/len(dataloader), 4)
        log_message = f"train loss: {self.loss} metric"
        for key, value in self.metric.items():
            log_message += f' {key}: {value}'
        self.logger.info(log_message)

    def validate(self, dataloader, epoch_id=0):
        self.model.eval()
        with torch.no_grad():
            for batch_id, (x, y, bd) in enumerate(tqdm(dataloader)):


                # Load Data
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.long)
                bd = bd.to(self.device, dtype=torch.float32)



                # Prediction
                y_pred_model = self.model(x)

                # Loss

                loss, y_pred, acc, loss_list = self.loss_func(y_pred_model, y, bd)

                # record
                for key, func in self.metric_func.items():
                    self.metric[key] += func(y_pred, y)
                self.loss += loss.item()
                self.sem_loss += loss_list[0].item()
                self.bd_loss += loss_list[1].item()
                self.logger.debug(f"TRAINER | val epoch: {epoch_id}, batch: {batch_id}/{len(dataloader)-1}, loss: {loss.item()}, sem_loss: {loss_list[0].item()}, bd_loss: {loss_list[1].item()}")

        for key, value in self.metric.items():
            self.metric[key] = float(torch.round(value/len(dataloader), decimals=4))
        self.loss = round(self.loss/len(dataloader), 4)
        self.sem_loss = round(self.sem_loss/len(dataloader), 4)
        self.bd_loss = round(self.bd_loss/len(dataloader), 4)

        log_message = f"val loss: {self.loss} metric"
        for key, value in self.metric.items():
            log_message += f' {key}: {value}'
        self.logger.info(log_message)

    def clear_history(self):
        torch.cuda.empty_cache()
        self.loss = 0
        self.sem_loss = 0
        self.bd_loss = 0
        for func_name in self.metric_func.keys():
            self.metric[func_name] = 0
        clear_message = f"TRAINER | Clear history, loss: {self.loss},"
        for key, value in self.metric.items():
            clear_message += f" {key}: {value}"
        self.logger.debug(clear_message)

