import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import tqdm
from model import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

_, config = load_config_yaml('config.yaml')
config['device'] = torch.device('cuda:'+ config['gpu'])

if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):     # test, not exists
    os.makedirs(config['ckpt_path'])
if config['phase'] == 'train':
    save_config_yaml(config['ckpt_path'], config)
print(config)

# define dataset
Data = LongitudinalData(config['dataset_name'], config['data_path'], img_file_name=config['img_file_name'],
            noimg_file_name=config['noimg_file_name'], subj_list_postfix=config['subj_list_postfix'],
            data_type=config['data_type'], batch_size=config['batch_size'], num_fold=config['num_fold'],
            aug=config['aug'], fold=config['fold'], shuffle=config['shuffle'])
trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
testDataLoader = Data.testLoader

# define model
if config['model_name'] in ['CLS']:
    model = CLS(latent_size=config['latent_size'], use_feature=config['use_feature'], dropout=(config['froze_encoder']==False), gpu=config['device']).to(config['device'])
else:
    raise ValueError('Not support other models yet!')

# froze encoder
if config['froze_encoder']:
    for param in model.encoder.parameters():
        param.requires_grad = False

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4, amsgrad=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    print('starting lr:', optimizer.param_groups[0]['lr'])
else:
    start_epoch = -1

def train():
    global_iter = 0
    monitor_metric_best = -1
    start_time = time.time()

    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        loss_all_dict = {'all': 0, 'recon': 0., 'dis': 0., 'dir': 0., 'cls': 0.}
        global_iter0 = global_iter

        pred_list = []
        label_list = []
        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            img1 = sample['img'].to(config['device'], dtype=torch.float).unsqueeze(1)
            if config['subj_list_postfix'] == 'C_single':
                label = sample['age'].to(config['device'], dtype=torch.float)
            else:
                label = sample['label'].to(config['device'], dtype=torch.float)

            if img1.shape[0] <= config['batch_size'] // 2:
                break

            # run model
            pred = model.forward_single(img1)

            loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(config['pos_weight']), config['subj_list_postfix'])
            loss = config['lambda_cls'] * loss_cls
            loss_all_dict['cls'] += loss_cls.item()

            pred_list.append(pred_sig.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            optimizer.step()
            optimizer.zero_grad()

            if global_iter % 1 == 0:
                # pdb.set_trace()
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], cls=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_cls.item()))

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)

        pred_list = np.concatenate(pred_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        compute_classification_metrics(label_list, pred_list, config['subj_list_postfix'])

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['bacc']
        scheduler.step(monitor_metric)
        save_result_stat(stat, config, info='val')
        print(stat)

        # save ckp
        is_best = False
        if monitor_metric >= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        print(optimizer.param_groups[0]['lr'])
        save_checkpoint(state, is_best, config['ckpt_path'])


def evaluate(phase='val', set='val', save_res=True, info=''):
    model.eval()
    if phase == 'val':
        loader = valDataLoader
    else:
        if set == 'train':
            loader = trainDataLoader
        elif set == 'val':
            loader = valDataLoader
        elif set == 'test':
            loader = testDataLoader
        else:
            raise ValueError('Undefined loader')

    zs_file_path = os.path.join(config['ckpt_path'], 'result_train', 'results_allbatch.h5')
    if info == 'dataset':
        if not os.path.exists(zs_file_path):
            raise ValueError('Not existing zs for training batches!')
        else:
            zs_file = h5py.File(zs_file_path, 'r')
            zs_all = [torch.tensor(zs_file['z1']).to(config['device'], dtype=torch.float), torch.tensor(zs_file['z2']).to(config['device'], dtype=torch.float)]
            interval_all = torch.tensor(zs_file['interval']).to(config['device'], dtype=torch.float)
    elif info == 'batch' and set == 'train' and os.path.exists(zs_file_path):
        return

    res_path = os.path.join(config['ckpt_path'], 'result_'+set)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        # raise ValueError('Exist results')
        os.remove(path)

    loss_all_dict = {'all': 0, 'recon': 0., 'dir': 0., 'dis': 0., 'cls': 0.}
    img1_list = []
    label_list = []
    recon1_list = []
    z1_list = []
    age_list = []
    pred_list = []

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            img1 = sample['img'].to(config['device'], dtype=torch.float).unsqueeze(1)
            if config['subj_list_postfix'] == 'C_single':
                label = sample['age'].to(config['device'], dtype=torch.float)
            else:
                label = sample['label'].to(config['device'], dtype=torch.float)

            # run model
            pred = model.forward_single(img1)

            loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(config['pos_weight']), config['subj_list_postfix'])
            loss = config['lambda_cls'] * loss_cls
            loss_all_dict['cls'] += loss_cls.item()

            pred_list.append(pred_sig.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

            if phase == 'test' and save_res:
                img1_list.append(img1.detach().cpu().numpy())
                age_list.append(sample['age'].numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        pred_list = np.concatenate(pred_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        bacc = compute_classification_metrics(label_list, pred_list, config['subj_list_postfix'])
        loss_all_dict['bacc'] = bacc

        if phase == 'test' and save_res:
            img1_list = np.concatenate(img1_list, axis=0)
            age_list = np.concatenate(age_list, axis=0)
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('img1', data=img1_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('age', data=age_list)
            h5_file.create_dataset('pred', data=pred_list)


    return loss_all_dict

if config['phase'] == 'train':
    train()
else:
    stat = evaluate(phase='test', set='test', save_res=True)
