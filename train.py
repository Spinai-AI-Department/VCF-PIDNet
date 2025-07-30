import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import yaml
from time import time
import numpy as np
import random
from glob import glob

from utils.dataset import load_data, Custom_Dataset
from utils.metrics import get_metric
from utils.models import get_model
from utils.recorder import Recorder
from utils.utils import get_logger, EarlyStopper
from utils.trainer import Trainer
from utils.losses import get_loss


def train_func(config, train_paths, valid_paths):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device is {device}")
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_result_path = os.path.join(root_path, "results", 'dataset', 
                                        config["model_name"],
                                        "ex_{}_{}_{}".format(config['model_parameters']['p3'], config['model_parameters']['p4'],
                                         config['model_parameters']['p5'],),
                                        train_serial)
    os.makedirs(train_result_path, exist_ok=True)

    logger = get_logger(name='train',
                        file_path=os.path.join(train_result_path, 'log.log'),
                        level='info')

    train_dataset = Custom_Dataset(
         data_path=train_paths, size=(config['height'], config['width']), mode='train', edge_pad=True,
    )   

    valid_dataset = Custom_Dataset(
         data_path=valid_paths, size=(config['height'], config['width']), mode='valid', edge_pad=True,
    )
    dataloader = {
        "train":DataLoader(
            dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False
        ),
        "valid":DataLoader(
            dataset=valid_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False
        )
    }
    logger.info(f"Load dataset, train: {len(dataloader['train'])}, val: {len(dataloader['valid'])}")
    # model setting

                                    

    model = get_model(config['model_name'])
    model = model(**config['model_parameters']).to(device)

    # optimizer setting
    optimizer = optim.AdamW(
        params=model.parameters(), lr=config['training']["initial_learning_rate"]
    )
    # scheduler setting
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=config["restart_epoch"], eta_min=config['training']["minimum_learning_rate"]
    )

    metric_func = dict()
    for each_func in config['metirc_func']:
        metric_func[each_func] = get_metric(each_func)

    loss_func = get_loss(**config['loss_config'])

    earlystopper = EarlyStopper(
        patience=config['earlystopping_patience'],
        logger=logger
    )
    record = Recorder(
        record_dir = train_result_path,
        model = model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_func=metric_func,
        loss_func=loss_func,
        device=device,
        logger=logger
    )
    # config file save
    with open(os.path.join(record.record_dir, 'setting.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    for epoch_id in range(config['epoch_num']):
        logger.info(f"Epoch {epoch_id}/{config['epoch_num']}")
        row = dict()
        row['epoch_id'] = epoch_id
        row['train_serial'] = train_serial
        row['lr'] = trainer.scheduler.get_last_lr()

        # Train
        print(f"Epoch {epoch_id}/{config['epoch_num']} Train..")
        start = time()
        trainer.train(dataloader=dataloader["train"], epoch_id=epoch_id)
        end = time()-start
        row['train_loss'] = trainer.loss
        row['train_sem_loss'] = trainer.sem_loss
        row['train_bd_loss'] = trainer.bd_loss
        for key, value in trainer.metric.items():
            row['train_{}'.format(key)] = value
        row['train_elapsed_time'] = round(end, 3)


        trainer.clear_history()

        # Validation
        print(f"Epoch {epoch_id}/{config['epoch_num']} Validation..")
        start = time()
        trainer.validate(dataloader=dataloader["valid"], epoch_id=epoch_id)
        end = time()-start
        row['val_loss'] = trainer.loss
        row['val_sem_loss'] = trainer.sem_loss
        row['val_bd_loss'] = trainer.bd_loss
        for key, value in trainer.metric.items():
            row['val_{}'.format(key)] = value
        row['val_elapsed_time'] = round(end, 3)

        trainer.clear_history()
        # Log
        record.add_row(row)
        record.save_plot(config['plot'])


        # Check early stopping
        earlystopper.check_early_stopping(row[config['earlystopping_target']])
        if earlystopper.patience_counter == 0:
            record.save_weight(epoch=epoch_id)

        if earlystopper.stop:
            print(f"Epoch {epoch_id}/{config['epoch_num']}, Stopped counter {earlystopper.patience_counter}/{config['earlystopping_patience']}")
            logger.info(f"Epoch {epoch_id}/{config['epoch_num']}, Stopped counter {earlystopper.patience_counter}/{config['earlystopping_patience']}")
            break


    print("END TRAINING")
    logger.info("END TRAINING")
    return




if __name__=="__main__":

    # Set random seed, deterministic
    seed = 67
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    root_path = os.path.dirname(__file__)
    config_path = os.path.join(root_path, 'hyper_tuning.yaml')
    with open(config_path, 'r', encoding='UTF8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    train_paths, valid_paths = load_data(config['dataset_path'])

    max_cnt = 14
    cnt = 0
    print("train path len : ",len(train_paths))
    print("valid path len : ",len(valid_paths))
    for enc_rate in range(0 if config['tuning'] else config['p3'], config['p3']+1):
        for skip_rate in range(0 if config['tuning'] else config['p4'], config['p4']+1):
            for dec_rate in range(0 if config['tuning'] else config['p5'], config['p5'] + 1):
                if (enc_rate < skip_rate and skip_rate < dec_rate) or (enc_rate==0 and skip_rate==0 and dec_rate == 0) or (enc_rate==0 and skip_rate==0 and skip_rate < dec_rate) or (enc_rate==0 and enc_rate < skip_rate and skip_rate==dec_rate):
                    for lr in config['initial_learning_rate']:
                        for min_lr in config['minimum_learning_rate_relative_to_iterative']:
                            train_func(
                                config={
                                    **config,
                                    'training':{
                                        'initial_learning_rate': lr,
                                        'minimum_learning_rate': min_lr*lr
                                    },
                                    'model_parameters':{
                                        'name':config['model_name'],
                                        'num_classes':config['class_num'],
                                        'p3':enc_rate,
                                        'p4':skip_rate,
                                        'p5':dec_rate,
                                    }
                                },
                                train_paths=train_paths, valid_paths=valid_paths,
                            )