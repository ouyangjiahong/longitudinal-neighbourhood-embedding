import os
import time
import pdb
from glob import glob
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.misc as sci
import scipy.ndimage
import shutil
from skimage.measure import compare_psnr, compare_ssim
import sklearn.metrics
import matplotlib as mpl
import nibabel as nib
import h5py
import pandas as pd
import yaml
import copy

# define dataloader
class CrossSectionalDataset(Dataset):
    def __init__(self, dataset_name, data_img, data_noimg, subj_id_list, case_id_list, aug=False, is_label_tp=False):
        self.dataset_name = dataset_name
        self.data_img = data_img
        self.data_noimg = data_noimg
        self.subj_id_list = subj_id_list
        self.case_id_list = case_id_list
        self.aug = aug
        self.is_label_tp = is_label_tp

    def __len__(self):
        # return len(self.subj_id_list)*3
        return len(self.subj_id_list)

    def __getitem__(self, idx):
        # idx = idx // 3
        subj_id = self.subj_id_list[idx]
        case_id = self.case_id_list[0][idx]
        case_order = self.case_id_list[1][idx]

        if self.aug:
            rand_idx = np.random.randint(0, 10)
            img = np.array(self.data_img[subj_id][case_id][rand_idx])
        else:
            img = np.array(self.data_img[subj_id][case_id])
            if len(img.shape) != 3:
                img = np.array(self.data_img[subj_id][case_id][0])
        img = np.nan_to_num(img, nan=0.0, copy=False)
        img[np.isinf(img)] = 0

        # img = np.array(self.data_img[subj_id][case_id])
        if self.is_label_tp:
            label = np.array(self.data_noimg[subj_id]['label_all'][case_order])
        else:
            label = np.array(self.data_noimg[subj_id]['label'])
        age = np.array(self.data_noimg[subj_id]['age'] + self.data_noimg[subj_id]['date_interval'][case_order])
        if self.dataset_name == 'LAB':
            age = (age - 47.3) / 17.6
        if self.dataset_name == 'NCANDA':
            age = (age - 19.5) / 3.4
        return {'img': img, 'label': label, 'age': age}

class LongitudinalPairDataset(Dataset):
    def __init__(self, dataset_name, data_img, data_noimg, subj_id_list, case_id_list, aug=False, is_label_tp=False):
        self.dataset_name = dataset_name
        self.data_img = data_img
        self.data_noimg = data_noimg
        self.subj_id_list = subj_id_list
        self.case_id_list = case_id_list
        self.aug = aug
        self.is_label_tp = is_label_tp
        self.cluster_centers_list = None
        self.cluster_ids_list = None
        self.sample_idx_list = None

    def __len__(self):
        return len(self.subj_id_list)
        # return 140

    def init_kmeans(self, N_km=[120,60,30]):
        self.N_km = N_km
        # self.cluster_centers_list = [np.zeros((n_km, z_dim)) for n_km in self.N_km]
        self.cluster_ids_list = [np.zeros(self.__len__()) for n_km in self.N_km]

    def update_kmeans(self, cluster_ids_list):
        # self.cluster_centers_list = cluster_centers_list
        self.cluster_ids_list = cluster_ids_list

    def minimatch_sampling_strategy(self, cluster_centers_list, cluster_ids_list, bs=64):
        # compute distance between clusters
        cluster_dis_ids_list = []

        for m in range(len(cluster_centers_list)):
            cluster_centers = cluster_centers_list[m]
            n_km = cluster_centers.shape[0]
            cluster_dis_ids = np.zeros((n_km, n_km))
            for i in range(n_km):
                dis_cn = np.sqrt(np.sum((cluster_centers[i].reshape(1,-1) - cluster_centers)**2, 1))
                cluster_dis_ids[i] = np.argsort(dis_cn)
            cluster_dis_ids_list.append(cluster_dis_ids)

        n_batch = np.ceil(self.__len__() / bs).astype(int)
        sample_idx_list = []
        for nb in range(n_batch):
            m_idx = np.random.choice(len(cluster_centers_list))         # select round of kmeans
            c_idx = np.random.choice(cluster_centers_list[m_idx].shape[0])  # select a cluster
            sample_idx_batch = []
            n_s_b = 0
            for c_idx_sel in cluster_dis_ids_list[m_idx][c_idx]:        # get nbr clusters given distance to selected cluster c_idx
                sample_idx = np.where(cluster_ids_list[m_idx] == c_idx_sel)[0]
                if n_s_b + sample_idx.shape[0] >= bs:
                    sample_idx_batch.append(np.random.choice(sample_idx, bs-n_s_b, replace=False))
                    break
                else:
                    sample_idx_batch.append(sample_idx)
                    n_s_b += sample_idx.shape[0]

            sample_idx_batch = np.concatenate(sample_idx_batch, 0)
            sample_idx_list.append(sample_idx_batch)

        sample_idx_list = np.concatenate(sample_idx_list, 0)
        self.sample_idx_list = sample_idx_list[:self.__len__()]

    def __getitem__(self, given_idx):
        if self.sample_idx_list is None:
            idx = given_idx
        else:
            idx = self.sample_idx_list[given_idx]

        subj_id = self.subj_id_list[idx]
        case_id_1 = self.case_id_list[0][idx]
        case_id_2 = self.case_id_list[1][idx]
        case_order_1 = self.case_id_list[2][idx]
        case_order_2 = self.case_id_list[3][idx]
        if self.is_label_tp:
            label = np.array(self.data_noimg[subj_id]['label_all'][case_order_2])
        else:
            label = np.array(self.data_noimg[subj_id]['label'])
        # label_all = np.array(self.data_noimg[subj_id]['label_all'])[[case_order_1, case_order_2]]
        interval = np.array(self.data_noimg[subj_id]['date_interval'][case_order_2] - self.data_noimg[subj_id]['date_interval'][case_order_1])
        age = np.array(self.data_noimg[subj_id]['age'] + self.data_noimg[subj_id]['date_interval'][case_order_1])

        if self.aug:
            rand_idx = np.random.randint(0, 10)
            img1 = np.array(self.data_img[subj_id][case_id_1][rand_idx])
            img2 = np.array(self.data_img[subj_id][case_id_2][rand_idx])
        else:
            img1 = np.array(self.data_img[subj_id][case_id_1])
            img2 = np.array(self.data_img[subj_id][case_id_2])
            if len(img1.shape) != 3:
                img1 = np.array(self.data_img[subj_id][case_id_1][0])
                img2 = np.array(self.data_img[subj_id][case_id_2][0])
        if not self.cluster_ids_list:
            return {'img1': img1, 'img2': img2, 'label': label, 'interval': interval, 'age': age}
        else:
            cluster_ids = [self.cluster_ids_list[i][idx] for i in range(len(self.cluster_ids_list))]
            return {'img1': img1, 'img2': img2, 'label': label, 'interval': interval, 'age': age, 'cluster_ids': np.array(cluster_ids)}

class LongitudinalData(object):
    def __init__(self, dataset_name, data_path, img_file_name='ADNI_longitudinal_img.h5',
                noimg_file_name='ADNI_longitudinal_noimg.h5', subj_list_postfix='NC_AD', data_type='single',
                aug=False, batch_size=16, num_fold=5, fold=0, shuffle=True, num_workers=0):
        if dataset_name == 'ADNI' or dataset_name == 'LAB' or dataset_name == 'NCANDA':
            data_img = h5py.File(os.path.join(data_path, img_file_name), 'r')
            data_noimg = h5py.File(os.path.join(data_path, noimg_file_name), 'r')

            if data_type == 'single':
                subj_id_list_train, case_id_list_train = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_train_'+subj_list_postfix+'_single.txt'), 'single')
                subj_id_list_val, case_id_list_val = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_val_'+subj_list_postfix+'_single.txt'), 'single')
                subj_id_list_test, case_id_list_test = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_test_'+subj_list_postfix+'_single.txt'), 'single')
            else:
                subj_id_list_train, case_id_list_train = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_train_'+subj_list_postfix+'.txt'), 'pair')
                subj_id_list_val, case_id_list_val = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_val_'+subj_list_postfix+'.txt'), 'pair')
                subj_id_list_test, case_id_list_test = self.load_idx_list(os.path.join(data_path, 'fold'+str(fold)+'_test_'+subj_list_postfix+'.txt'), 'pair')

            if dataset_name == 'NCANDA':
                is_label_tp = True
            else:
                is_label_tp = False
            if data_type == 'single':
                self.train_dataset = CrossSectionalDataset(dataset_name, data_img, data_noimg, subj_id_list_train, case_id_list_train, aug=aug, is_label_tp=is_label_tp)
                self.val_dataset = CrossSectionalDataset(dataset_name, data_img, data_noimg, subj_id_list_val, case_id_list_val, aug=False, is_label_tp=is_label_tp)
                self.test_dataset = CrossSectionalDataset(dataset_name, data_img, data_noimg, subj_id_list_test, case_id_list_test, aug=False, is_label_tp=is_label_tp)
            elif data_type == 'pair':
                self.train_dataset = LongitudinalPairDataset(dataset_name, data_img, data_noimg, subj_id_list_train, case_id_list_train, aug=aug, is_label_tp=is_label_tp)
                self.val_dataset = LongitudinalPairDataset(dataset_name, data_img, data_noimg, subj_id_list_val, case_id_list_val, aug=False, is_label_tp=is_label_tp)
                self.test_dataset = LongitudinalPairDataset(dataset_name, data_img, data_noimg, subj_id_list_test, case_id_list_test, aug=False, is_label_tp=is_label_tp)
            else:
                raise ValueError('Did not support pair or sequential data yet')

        else:
            raise ValueError('Not support this dataset!')

        self.trainLoader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.valLoader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # self.valLoader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.testLoader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def load_idx_list(self, file_path, data_type):
        lines = pd.read_csv(file_path, sep=" ", header=None)
        if data_type == 'single':
            return np.array(lines.iloc[:,0]), [np.array(lines.iloc[:,1]), np.array(lines.iloc[:,2])]
        elif data_type == 'pair':
            return np.array(lines.iloc[:,0]), [np.array(lines.iloc[:,1]),np.array(lines.iloc[:,2]),np.array(lines.iloc[:,3]),np.array(lines.iloc[:,4])]
        else:
            raise ValueError('Not support sequential data type')


# load config file from ckpt
def load_config_yaml(yaml_path):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return True, config
    else:
        return False, None

# save config file at the beginning of the training
def save_config_yaml(ckpt_path, config):
    yaml_path = os.path.join(ckpt_path, 'config.yaml')
    remove_key = []
    for key in config.keys():
        if isinstance(config[key], int) or isinstance(config[key], float) or isinstance(config[key], str) or isinstance(config[key], list)  or isinstance(config[key], dict):
            continue
        remove_key.append(key)
    config_copy = copy.deepcopy(config)
    for key in remove_key:
        config_copy.pop(key, None)
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(config_copy, file)
    print('Saved yaml file')

# load model/scheduler
def load_checkpoint_by_key(values, checkpoint_dir, keys, device, ckpt_name='model_best.pth.tar'):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = os.path.join(checkpoint_dir, ckpt_name)
    print(filename)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            try:
                if key == 'model':
                    values[i] = load_checkpoint_model(values[i], checkpoint[key])
                else:
                    values[i].load_state_dict(checkpoint[key])
                print('loading ' + key + ' success!')
            except:
                print('loading ' + key + ' failed!')
        print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(filename, \
                epoch, checkpoint['monitor_metric']))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch

# load each part of the model
def load_checkpoint_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

# save results statistics
def save_result_stat(stat, config, info='Default'):
    stat_path = os.path.join(config['ckpt_path'], 'stat.csv')
    columns=['info',] + sorted(stat.keys())
    if not os.path.exists(stat_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(stat_path, mode='a', header=True)

    stat['info'] = info
    for key, value in stat.items():
        stat[key] = [value]
    df = pd.DataFrame.from_dict(stat)
    df = df[columns]
    df.to_csv(stat_path, mode='a', header=False)

def save_checkpoint(state, is_best, checkpoint_dir):
    print("save checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch']).zfill(3)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir+'/model_best.pth.tar')
        print('save best')

def compute_classification_metrics(label, pred, dataset_name='ADNI', postfix='NC_AD', task='classification'):
    if task == 'age':
        r2 = sklearn.metrics.r2_score(label, pred)
        if dataset_name == 'LAB':
            label = label * 17.6 + 47.3
            pred = pred * 17.6 + 47.3
        if dataset_name == 'NCANDA':
            label = label * 3.4 + 19.5
            pred = pred * 3.4 + 19.5
        mse = sklearn.metrics.mean_squared_error(label, pred, squared=False)
        mae = np.abs(pred - label).mean()
        print(mse, r2, mae)
        return r2
    else:
        pred_bi = (pred>0.5).squeeze(1)
        if dataset_name == 'ADNI':
            if 'NC_AD' in postfix:
                classes = [0,2]
            elif 'pMCI_sMCI' in postfix:
                classes = [3,4]
        elif dataset_name == 'LAB':
            if 'C_E_HE' in postfix:
                label = (label > 0)
                classes = [0,1]
        elif dataset_name == 'NCANDA':
            label = (label > 0)
            classes = [0,1]
        tp = np.sum(np.logical_and(label==classes[1], pred_bi==1))
        fp = np.sum(np.logical_and(label==classes[0], pred_bi==1))
        tn = np.sum(np.logical_and(label==classes[0], pred_bi==0))
        fn = np.sum(np.logical_and(label==classes[1], pred_bi==0))
        auc = sklearn.metrics.roc_auc_score(label==classes[1], pred.squeeze(1))
        sen = tp/(tp+fn)
        spe = tn/(tn+fp)
        bacc = 0.5 * (sen + spe)
        print(auc, bacc, sen, spe)
        return bacc
