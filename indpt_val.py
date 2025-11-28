import os
import torch
import time
import logging

from matplotlib import pyplot as plt

from torch import nn

from models.TSSCD import *
from utils import *
from data_loader import *
from metrics import Evaluator, SpatialChangeDetectScore, TemporalChangeDetectScore, ChangeTypeAccuracyMatrix

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

def validModel(test_dl, model, device, logger,
               best_acc=0, best_spatialscore=0, best_temporalscore=0,
               epoch=1, last_saved_epoch=1, model_saved_times=0):
    loss_fn = nn.CrossEntropyLoss()
    loss_ch_noch = Diceloss()
    model.eval()
    with torch.no_grad():
        valid_tqdm = tqdm(iterable=test_dl, total=len(test_dl))
        valid_tqdm.set_description_str('Valid : ')
        valid_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        
        evaluator = Evaluator(5)
        change_type_eval = ChangeTypeAccuracyMatrix(num_classes=5, tol=1)
        evaluator.reset()
        change_type_eval.reset()
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
                change_type_eval.add_sequence(labtypes[0], pretypes[0])
                
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
        
        logger.info(f'Epoch {epoch}, Train loss: 0.0')
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
        change_type_acc = change_type_eval.get_accuracy_matrix()
        change_type_acc_str = np.array2string(
            change_type_acc, 
            precision=4,
            suppress_small=True,
            separator='\t'
        )
        logger.info(f'Change Type Accuracy Matrix\n {change_type_acc_str[1:-1]}')
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
def independent_evaluate_main_model(model, model_name='TSSCD_Unet', model_idx=1036, is_opt_only=False, k=1):
    """
    Regional accuracy evaluation function - performs single accuracy evaluation for each region
    MODIFICATION: Replaced training logic with regional evaluation
    """
    # MODIFICATION: Create separate logger for each province
    model_idx_str = str(model_idx)
    postfix = '' if not is_opt_only else '_opt_only'
    logger_name = f'logger_{model_idx_str}'
    log_filename = os.path.join('./models/model_data/log', model_name, f'{model_idx_str}{postfix}', f'{model_idx_str}_{k}{postfix}.log')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    # Create province-specific logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing log handlers and file
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            f.truncate()
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Load regional test data
    regional_npy = f'test.npy' if not is_opt_only else f'test_opt_only.npy'
    test_data = np.load(os.path.join('./models/model_data/dataset', str(model_idx), str(k), regional_npy))
    print(f'Loading test data from: {os.path.join("./models/model_data/dataset", str(model_idx), str(k), regional_npy)}')
    # Create data loader for the region
    test_dl = make_dataloader(test_data, type='test', is_shuffle=False, batch_size=64)
    
    # Perform single accuracy evaluation (epoch=1, no training)
    _, _, _, _, _, _, _ = validModel(
        test_dl=test_dl,
        model=model,
        device=device,
        logger=logger,  # Use province-specific logger
        best_acc=0,
        best_spatialscore=0,
        best_temporalscore=0,
        epoch=0,  # Fixed to 1 for single evaluation
        last_saved_epoch=0,
        model_saved_times=0,
    )
    # Clean up province logger
    logger.removeHandler(file_handler)
    file_handler.close()

def evaluateRegionalAccuracy(model, model_name='TSSCD_Unet', model_idx=1036, is_opt_only=False, k=1):
    """
    Regional accuracy evaluation function - performs single accuracy evaluation for each region
    MODIFICATION: Replaced training logic with regional evaluation
    """
    # Define region list based on actual file naming
    provinces = ['FJ', 'GDGX', 'JS', 'SD', 'SH', 'ZJ']
    
    # Evaluate each region
    for province in provinces:
        # MODIFICATION: Create separate logger for each province
        model_idx_str = str(model_idx) if not is_opt_only else f'{model_idx}_opt_only'
        logger_name = f'{province}_logger_{model_idx_str}'
        log_filename = f'models\\model_data\\log\\{model_name}\\{model_idx_str}\\{str(k)}\\{province}_{model_idx_str}.log'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        
        # Create province-specific logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing log handlers and file
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        if os.path.exists(log_filename):
            with open(log_filename, 'w') as f:
                f.truncate()
        
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Load regional test data
        regional_npy = f'{province}_test.npy' if not is_opt_only else f'{province}_test_opt_only.npy'
        test_data = np.load(os.path.join('./models/model_data/dataset', str(model_idx), str(k), regional_npy))
        print(f'Loading {province} test data from: {os.path.join("./models/model_data/dataset", str(model_idx), str(k), regional_npy)}')
        # Create data loader for the region
        test_dl = make_dataloader(test_data, type='test', is_shuffle=False, batch_size=64)
        
        # Perform single accuracy evaluation (epoch=1, no training)
        _, _, _, _, _, _, _ = validModel(
            test_dl=test_dl,
            model=model,
            device=device,
            logger=logger,  # Use province-specific logger
            best_acc=0,
            best_spatialscore=0,
            best_temporalscore=0,
            epoch=0,  # Fixed to 1 for single evaluation
            last_saved_epoch=0,
            model_saved_times=0,
        )
        # Clean up province logger
        logger.removeHandler(file_handler)
        file_handler.close()
# =============================================================================
# MODIFIED MAIN PROGRAM: Regional Evaluation Mode
# =============================================================================
if __name__ == '__main__':
    # MODIFICATION: Changed to regional evaluation mode
    model_idx = 1037
    model_save_name = str(model_idx)
    k_random_permutation = 5
    
    # MODIFICATION: User confirmation for regional evaluation
    confirm_eval = input(f'Start regional accuracy evaluation, model index: {model_save_name}. Continue? (y/n)\n')
    if confirm_eval != 'y':
        exit(print(f'Exiting...({confirm_eval})'))
    
    print('Starting regional accuracy evaluation...')
    
    # MODIFICATION: Evaluate each model architecture (normal mode)
    for model_name, model in generate_model_instances(is_opt_only=False, model_idx=model_idx):
        for k in range(1, k_random_permutation + 1):
            model = model.to(device=device)
            
            model_ = f'{model_idx}'
            model_path = os.path.join(f'models\\model_data\\{model_name}\\{model_}', f'{model_}_{k}.pth')
            model_state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
            model.load_state_dict(model_state_dict)
            model.eval()
            
            print(f'Evaluating model: {model_name}')
            independent_evaluate_main_model(
                model=model,
                model_name=model_name,
                model_idx=model_idx,
                is_opt_only=False,
                k=k
            )
            evaluateRegionalAccuracy(
                model=model,
                model_name=model_name,
                model_idx=model_idx,
                is_opt_only=False,
                k=k
            )
    
    # MODIFICATION: Evaluate each model architecture (opt_only mode)
    for model_name, model in generate_model_instances(is_opt_only=True, model_idx=model_idx):
        for k in range(1, k_random_permutation + 1):
            model = model.to(device=device)
            
            model_ = f'{model_idx}_opt_only'
            model_path = os.path.join(f'models\\model_data\\{model_name}\\{model_}', f'{model_idx}_{k}_opt_only.pth')
            print(f'Loading model parameters from: {model_path}')
            model_state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
            model.load_state_dict(model_state_dict)
            model.eval()
            
            print(f'Evaluating model: {model_name}')
            independent_evaluate_main_model(
                model=model,
                model_name=model_name,
                model_idx=model_idx,
                is_opt_only=True,
                k=k
            )
            evaluateRegionalAccuracy(
                model=model,
                model_name=model_name,
                model_idx=model_idx,
                is_opt_only=True,
                k=k
            )
    
    print('Regional accuracy evaluation completed!')