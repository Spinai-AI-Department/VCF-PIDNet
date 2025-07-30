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

from dataset import load_data, Custom_Dataset
from models import get_model
from metrics import get_metric
from recorder import Recorder
from utils import get_logger, EarlyStopper
from trainer import Trainer
from losses import get_loss

root_path = r'C:\Users\taesh\cwc'
size = (384, 384)


def train_func(config, train_path, test_path):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device is {device}")
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_result_path = os.path.join(root_path, "results", 'dataset_1700', 
                                        config["model_name"],  "drop_expanded_rotate_discrete_noise_005",
                                        "ex_{}_{}_{}".format(config['model_parameters']['enc_use_drop'], config['model_parameters']['use_skip_drop'],
                                         config['model_parameters']['use_dec_drop'],),
                                        train_serial)
    os.makedirs(train_result_path, exist_ok=True)

    logger = get_logger(name='train',
                        file_path=os.path.join(train_result_path, 'log.log'),
                        level='info')

    train_dataset = Custom_Dataset(
         data_path=train_path[0], label_path=train_path[1], size=size, mode='train',
    )   

    test_dataset = Custom_Dataset(
         data_path=test_path[0], label_path=test_path[1], size=size, mode='test',
    )
    dataloader = {
        "train":DataLoader(
            dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False
        ),
        "test":DataLoader(
            dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False
        )
    }
    logger.info(f"Load dataset, train: {len(dataloader['train'])}, val: {len(dataloader['test'])}")
    # model setting

                                    

    model = get_model(config['model_name'])
    model = model(**config['model_parameters']).to(device)


    model_load_path = glob(os.path.join(root_path, "results", 'dataset_1700', 
                                        config["model_name"],  "drop_expanded_rotate_discrete_noise_005",
                                        "{}_{}_{}".format(config['model_parameters']['enc_use_drop'], config['model_parameters']['use_skip_drop'],
                                         config['model_parameters']['use_dec_drop'],), '*'))[0] 
    if os.path.isdir(model_load_path):
        model.load_state_dict(torch.load(os.path.join(model_load_path, 'model.pt'))['model'])

    print("encoder drop out", model.encoder.blocks[0][4])
    print("skip drop out", model.decoder.blocks[0]['skip_blocks'][0][4])
    print("decoder drop out", model.decoder.blocks[0]['merge_block'][3])
    #print("decoder drop out", model.decoder.blocks[0]['block'][3])
    # optimizer setting
    optimizer = optim.AdamW(
        params=model.parameters(), lr=config["initial_learning_rate"]
    )
    # scheduler setting
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=config["restart_epoch"], eta_min=config["minimum_learning_rate"]
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
        for key, value in trainer.metric.items():
            row['train_{}'.format(key)] = value
        row['train_elapsed_time'] = round(end, 3)


        trainer.clear_history()

        # Validation
        print(f"Epoch {epoch_id}/{config['epoch_num']} Validation..")
        start = time()
        trainer.validate(dataloader=dataloader["test"], epoch_id=epoch_id)
        end = time()-start
        row['val_loss'] = trainer.loss
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

    config_path = os.path.join(root_path, 'unetSeries_torch', 'hyper_tuning.yaml')
    with open(config_path, 'r', encoding='UTF8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    train_path, test_path = load_data(os.path.join(root_path, "dataset", "new_dataset"))

    train_fixed_config = config['train_fixed']
    model_fixed_config = config['model_fixed']
    max_cnt = 14
    cnt = 0
    print("train path len : ",len(train_path[0]))
    print("test path len : ",len(test_path[0]))
    for enc_rate in range(config['enc_use_drop']+1):
        for skip_rate in range(config['use_skip_drop']+1):
            for dec_rate in range(config['use_dec_drop'] + 1):
                #if enc_rate < dec_rate or (enc_rate==0 and dec_rate ==0):
                if (enc_rate < skip_rate and skip_rate < dec_rate) or (enc_rate==0 and skip_rate==0 and dec_rate == 0) or (enc_rate==0 and skip_rate==0 and skip_rate < dec_rate) or (enc_rate==0 and enc_rate < skip_rate and skip_rate==dec_rate):
                    for lr in config['initial_learning_rate']:
                        if skip_rate == 0:
                            continue
                        for min_lr in config['minimum_learning_rate_relative_to_iterative']:
                            train_unfixed_config = {
                                'initial_learning_rate': lr,
                                'minimum_learning_rate': min_lr*lr
                            }
                            model_unfixed_config = {
                                'enc_use_drop': enc_rate,
                                #'dec_use_drop': dec_rate,
                                'use_skip_drop':skip_rate,
                                'use_dec_drop': dec_rate
                            }
                            train_func(
                                config={
                                    **train_fixed_config, **train_unfixed_config,
                                    'model_parameters':{
                                        **model_fixed_config,
                                        **model_unfixed_config
                                    }
                                },
                                train_path=train_path, test_path=test_path,
                            )