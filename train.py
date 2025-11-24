import os
import torch
import time
import logging

from matplotlib import pyplot as plt

from torch import nn
from torch import optim
from transformers import get_linear_schedule_with_warmup

from models.TSSCD import *
from utils import *
from data_loader import *
from metrics import Evaluator, SpatialChangeDetectScore, TemporalChangeDetectScore
from torch.nn.utils import clip_grad_norm_

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(legacy='1.25')

class Diceloss(nn.Module):
    def __init__(self, smooth=1.):
        super(Diceloss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=0).sum(dim=0)
        loss = (1 - ((2. * intersection + self.smooth) / (
                pred.sum(dim=0).sum(dim=0) + target.sum(dim=0).sum(dim=0) + self.smooth)))
        return loss.mean()

def plot_train_metrics(model_metrics_data):
    pass
            
def trainModel(model, train_dl, test_dl, 
               model_name='TSSCD_Unet', iter_num=200, fold='1000',
               is_opt_only=False,
               is_early_stopping=True):
    # log setting
    if is_opt_only:
        model_idx = str(fold)[:4] + '_opt_only'
    else:
        model_idx = str(fold)[:4]
    log_filename = f'models\\model_data\\log\\{model_name}\\{model_idx}\\{fold}.log'
    logger = logging.getLogger(f'logger_{fold}')
    logger.setLevel(logging.INFO)
    
    if os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            f.truncate()
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # load data
    loss_fn = nn.CrossEntropyLoss()  # classification loss function
    loss_ch_noch = Diceloss()  # changed loss function

    total_steps = len(train_dl) * iter_num
    warmup_steps = int(0.1 * total_steps)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Start training
    early_stopping = EarlyStopping(patience=32)
    best_acc, best_spatialscore, best_temporalscore = 0, 0, 0
    
    model_saved_times, last_saved_epoch = 0, 0
    model_metrics_data = dict()
    
    model.train()
    for epoch in range(iter_num):
        train_tqdm = tqdm(iterable=train_dl, total=len(train_dl))
        train_tqdm.set_description_str(f'Train epoch: {epoch}')
        
        train_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        train_loss1_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        train_loss2_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        
        for train_data, train_labels in train_tqdm:
            train_data, train_labels = train_data.to(device), train_labels.to(device)
            pred = model(train_data.float())
            
            pre_label = torch.argmax(input=pred, dim=1)
            # time series has changed or not
            pre_No_change = pre_label.max(dim=1).values == pre_label.min(dim=1).values
            label_No_change = train_labels.max(dim=1).values == train_labels.min(dim=1).values

            loss1 = loss_fn(pred, train_labels.long())
            loss2 = loss_ch_noch(pre_No_change, label_No_change)
            loss = loss1
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            with torch.no_grad():
                train_loss1_sum = torch.cat([train_loss1_sum, torch.unsqueeze(input=loss1, dim=-1)], dim=-1)
                train_loss2_sum = torch.cat([train_loss2_sum, torch.unsqueeze(input=loss2, dim=-1)], dim=-1)
                train_loss_sum = torch.cat([train_loss_sum, torch.unsqueeze(input=loss, dim=-1)], dim=-1)
                train_tqdm.set_postfix(
                    {'train loss': train_loss_sum.mean().item(), 'train loss1': train_loss1_sum.mean().item(),
                     'train loss2': train_loss2_sum.mean().item()})
        
        # lr_scheduler.step()
        
        logger.info(f'Epoch {epoch}, Train loss: {train_loss_sum.mean().item()}')
        train_tqdm.close()
        
        valid_loss_sum, best_acc, best_spatialscore, best_temporalscore, \
        model_saved_times, last_saved_epoch,\
        model_metrics = validModel( test_dl, model, device, logger,
                                    True, best_acc, best_spatialscore, best_temporalscore,
                                    epoch, last_saved_epoch,
                                    model_saved_times,
                                    model_name, fold=fold, is_opt_only=is_opt_only)
        model_metrics_data[epoch] = model_metrics

        logger.info(f'model saved {model_saved_times} times, last saved epoch: {last_saved_epoch}.\n')
        print(f'model saved {model_saved_times} times, last saved epoch: {last_saved_epoch}.\n')
        print(f'Current model: {model_name}, fold: {fold}; early stop ref: valid_loss_sum; loss: loss1 + 0.5 * loss2')
        
        if is_early_stopping:
            early_stopping(valid_loss_sum)
            # early_stopping(best_acc)
            if early_stopping.early_stop:
                break
    
    logger.removeHandler(file_handler)
    file_handler.close()
    return model_metrics_data

def validModel(test_dl, model, device, logger, saveModel=True,
               best_acc=0, best_spatialscore=0, best_temporalscore=0,
               epoch=0, last_saved_epoch=0, model_saved_times=0, 
               model_name='TSSCD_FCN', fold='1000', is_opt_only=False):
    evaluator = Evaluator(5)
    loss_fn = nn.CrossEntropyLoss()
    loss_ch_noch = Diceloss()
    model.eval()
    with torch.no_grad():
        valid_tqdm = tqdm(iterable=test_dl, total=len(test_dl))
        valid_tqdm.set_description_str('Valid : ')
        valid_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        evaluator.reset()
        spatialscore = SpatialChangeDetectScore()
        temporalscore = TemporalChangeDetectScore(series_length=60, error_rate=1)
        for valid_data, valid_labels in valid_tqdm:
            valid_data, valid_labels = valid_data.to(device), valid_labels.to(device)
            valid_pred = model(valid_data.float())
            
            pre_label = torch.argmax(input=valid_pred, dim=1)
            pre_No_change = pre_label.max(dim=1).values == pre_label.min(dim=1).values
            label_No_change = valid_labels.max(dim=1).values == valid_labels.min(dim=1).values
            # Loss function
            loss1 = loss_fn(valid_pred, valid_labels.long())
            loss2 = loss_ch_noch(pre_No_change, label_No_change)
            valid_loss = loss1

            evaluator.add_batch(valid_labels.cpu().numpy(), torch.argmax(input=valid_pred, dim=1).cpu().numpy())

            valid_loss_sum = torch.cat([valid_loss_sum, torch.unsqueeze(input=valid_loss, dim=-1)], dim=-1)
            valid_tqdm.set_postfix({'valid loss': valid_loss_sum.mean().item()})

            predList = torch.argmax(input=valid_pred, dim=1).cpu().numpy()
            labelList = valid_labels.cpu().numpy()

            for pre, label in zip(predList, labelList):
                pre, label = pre[None, :], label[None, :]
                _, prechangepoints, pretypes = FilteringSeries(pre, method='Majority', window_size=3)
                _, labchangepoints, labtypes = FilteringSeries(label, method='NoFilter')
                spatialscore.addValue(labchangepoints[0], prechangepoints[0])
                spatialscore.addLccValue(pretypes[0], labtypes[0])
                temporalscore.addValue(labchangepoints[0], prechangepoints[0])
                
        valid_tqdm.close()

        # Evaluation Accuracy
        Acc = evaluator.Pixel_Accuracy()
        Acc_class, Acc_mean = evaluator.Class_Accuracy()
        print('OA:', round(Acc, 4))
        print('AA:', round(Acc_mean, 4), '; Acc_class:', [round(i, 4) for i in Acc_class])
        F1 = evaluator.F1()
        print('F1:', round(F1, 4))
        Kappa = evaluator.Kappa()
        print('Kappa:', round(Kappa, 4))
        mIoU = evaluator.Mean_Intersection_over_Union()
        print(f'mIoU:', f'{round(mIoU, 4)} ({round(best_acc, 4)})')
        # Spaital metrics
        spatialscore.getScore()
        spatial_f1 = spatialscore.spatial_f1
        print('spatial_LccAccuracy: ', f'{round(spatialscore.getLccScore(), 4)} ({round(best_spatialscore, 4)})')
        print(f'spatial_PA: {round(spatialscore.spatial_pa, 4)}; spatial_UA: {round(spatialscore.spatial_ua, 4)}; spatial_f1: {round(spatial_f1, 4)}')
        # Temporal metrics
        temporalscore.getScore()
        print('temporal_CdAccuracy: ', f'{round(temporalscore.getCDScore(), 4)} ({round(best_temporalscore, 4)})')
        print(f'temporal_PA: {round(temporalscore.temporal_pa, 4)}; temporal_UA: {round(temporalscore.temporal_ua, 4)}; temporal_f1: {round(temporalscore.temporal_f1, 4)}')
        
        logger.info(f'mIoU: {round(mIoU, 4)}; OA: {round(Acc, 4)}; AA: {round(Acc_mean, 4)}; F1: {round(F1, 4)}; Kappa: {round(Kappa, 4)};')
        logger.info(f'spatial_LccAccuracy: {round(spatialscore.getLccScore(), 4)}; spatial_PA: {round(spatialscore.spatial_pa, 4)}; spatial_UA: {round(spatialscore.spatial_ua, 4)}; spatial_F1: {round(spatial_f1, 4)}')
        logger.info(f'temporal_CdAccuracy: {round(temporalscore.getCDScore(), 4)}; temporal_PA: {round(temporalscore.temporal_pa, 4)}; temporal_UA: {round(temporalscore.temporal_ua, 4)}; temporal_F1: {round(temporalscore.temporal_f1, 4)}')
        
        confusion_matrix = evaluator.confusion_matrix  # current epoch's confusion matrix
        confusion_matrix_str = np.array2string(
            confusion_matrix, 
            precision=4,
            suppress_small=True,
            separator='\t'
        )
        logger.info(f'Confusion Matrix\n {confusion_matrix_str[1:-1]}')
        
        if saveModel:
            model_idx = (str(fold)[:4] + '_opt_only') if is_opt_only else str(fold)[:4]
            # if mIoU >= best_acc and spatialscore.getLccScore() >= best_spatialscore:
            if mIoU >= best_acc:
                torch.save(model.state_dict(), os.path.join(f'models\\model_data\\{model_name}\\{model_idx}', f'{fold}.pth'))
                logger.info(f'Epoch {epoch} saved.')
                best_acc = mIoU
                best_spatialscore = spatialscore.getLccScore()
                best_temporalscore = temporalscore.getCDScore()
                
                model_saved_times += 1
                last_saved_epoch = epoch
            return valid_loss_sum.mean().item(), best_acc, best_spatialscore, best_temporalscore,\
                   model_saved_times, last_saved_epoch,\
                   { # Finally saved model's metrics
                       'mIoU': round(mIoU, 4),
                       'spatial_LccAccuracy': round(spatialscore.getLccScore(), 4),
                       'temporal_CdAccuracy': round(temporalscore.getCDScore(), 4),
                       
                       'OA': round(Acc, 4),
                       'AA': round(Acc_mean, 4),
                       'Acc_class': [round(i, 4) for i in Acc_class],
                       'F1': round(F1, 4),
                       'Kappa': round(Kappa, 4),
                       'spatial_PA': round(spatialscore.spatial_pa, 4),
                       'spatial_UA': round(spatialscore.spatial_ua, 4),
                       'spatial_F1': round(spatial_f1, 4),
                       'temporal_PA': round(temporalscore.temporal_pa, 4),
                       'temporal_UA': round(temporalscore.temporal_ua, 4),
                       'temporal_F1': round(temporalscore.temporal_f1, 4)
                   }
        else:
            return

if __name__ == '__main__':
    iter_num, batch_size = 800, 64
    k_random_permutation = 5
    def training_set(model_name):
        '''
            continue (pass) when return True

        '''
        return False
        if model_name == 'TSSCD_TransEncoder':
            return True
        else:
            return False
        
    # load dataset
    model_idx = 1037
    model_save_name = str(model_idx)
    confirm_model_idx = input(f'Current model index is {model_save_name}. Continue? (y/n)\n')
    if confirm_model_idx == 'y':
        print('Start training...')
    else:
        print(f'Exit...({confirm_model_idx})')
        exit()

    # tralid = np.load(os.path.join('./models/model_data/dataset', str(model_idx), 'tralid.npy'))
    # test = np.load(os.path.join('./models/model_data/dataset', str(model_idx), 'test.npy'))
    
    # tralid_opt_only = np.load(os.path.join('./models/model_data/dataset', str(model_idx), 'tralid_opt_only.npy'))
    # test_opt_only = np.load(os.path.join('./models/model_data/dataset', str(model_idx), 'test_opt_only.npy'))
    
    # # _ds, _ds_opt_only = np.vstack((tralid, test)), np.vstack((tralid_opt_only, test_opt_only))
    # _ds, _ds_opt_only = tralid, tralid_opt_only
    split_rate = 0.8
    
    # Valid dataset —— "tralid" will split into train and valid
    
    for fold in range(k_random_permutation):
        fold += 1
        tralid = np.load(os.path.join('./models/model_data/dataset', str(model_idx), str(fold), 'tralid.npy'))
        test = np.load(os.path.join('./models/model_data/dataset', str(model_idx), str(fold), 'test.npy'))
        train_dl, valid_dl = make_dataloader(tralid, type='train', is_shuffle=True, batch_size=64), \
                             make_dataloader(test, type='test', is_shuffle=False, batch_size=64)
    # for fold, (train_dl, valid_dl) in enumerate(random_permutation(_ds, n_split=5, split_rate=split_rate, batch_size=64)):
        for model_name, model in generate_model_instances(is_opt_only=False, model_idx=model_idx):
            if training_set(model_name):  continue
            model = model.to(device=device)
            model_metrics = trainModel(
                model=model, 
                train_dl=train_dl, test_dl=valid_dl,
                model_name=model_name, 
                iter_num=iter_num, 
                fold=model_save_name + '_' + str(fold),
                is_opt_only=False,
                is_early_stopping=True
            )
    # Opt Only
    for fold in range(k_random_permutation):
        fold += 1
        tralid = np.load(os.path.join('./models/model_data/dataset', str(model_idx), str(fold), 'tralid_opt_only.npy'))
        test = np.load(os.path.join('./models/model_data/dataset', str(model_idx), str(fold), 'test_opt_only.npy'))
        train_dl, valid_dl = make_dataloader(tralid, type='train', is_shuffle=True, batch_size=64), \
                             make_dataloader(test, type='test', is_shuffle=False, batch_size=64)
    # for fold, (train_dl, valid_dl) in enumerate(random_permutation(_ds_opt_only, n_split=5, split_rate=split_rate, batch_size=64)):
        for model_name, model in generate_model_instances(is_opt_only=True, model_idx=model_idx):
            if training_set(model_name):  continue
            model = model.to(device=device)
            model_metrics = trainModel(
                model=model, 
                train_dl=train_dl, test_dl=valid_dl,
                model_name=model_name, 
                iter_num=iter_num, 
                fold=model_save_name + '_' + str(fold) + '_opt_only',
                is_opt_only=True,
                is_early_stopping=True
            )
            
    # # Test dataset —— traid for training & test for validation
    # for model_name, model in generate_model_instances(is_opt_only=False, model_idx=model_idx):
    #     if training_set(model_name):  continue
    #     model = model.to(device=device)
    #     model_metrics = trainModel(
    #         model=model,
    #         train_dl=make_dataloader(tralid, type='train', is_shuffle=True, batch_size=batch_size), 
    #         test_dl=make_dataloader(test, type='test', is_shuffle=False, batch_size=batch_size),
    #         model_name=model_name,
    #         iter_num=iter_num,
    #         fold=model_save_name,
    #         is_opt_only=False,
    #         is_early_stopping=True
    #     )
    # # Opt Only
    # for model_name, model in generate_model_instances(is_opt_only=True, model_idx=model_idx):
    #     if training_set(model_name):  continue
    #     model = model.to(device=device)
    #     model_metrics = trainModel(
    #         model=model,
    #         train_dl=make_dataloader(tralid_opt_only, type='train', is_shuffle=True, batch_size=batch_size), 
    #         test_dl=make_dataloader(test_opt_only, type='test', is_shuffle=False, batch_size=batch_size),
    #         model_name=model_name,
    #         iter_num=iter_num,
    #         fold=model_save_name + '_opt_only',
    #         is_opt_only=True,
    #         is_early_stopping=True
    #     )


        

