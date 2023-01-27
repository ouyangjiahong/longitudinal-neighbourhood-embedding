
import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import random
import pdb
import skimage.transform

seed = 10
np.random.seed(seed)


# preprocess subject label and data
csv_path = '/data/jiahong/data/NCANDA/2019-10-02_cahalan.csv'      # label
csv_path_raw = '/data/jiahong/data/NCANDA/demographics.csv'  # demo
data_path = '/data/jiahong/data/NCANDA/FA/'
df = pd.read_csv(csv_path, usecols=['subject', 'visit', 'cahalan'])
df_raw = pd.read_csv(csv_path_raw, usecols=['subject', 'visit', 'visit_age', 'sex', 'arm'])


# load label, age, image paths
'''
struct subj_data

age: baseline age,
label: label for the subject, NAN
label_all: list of labels for each timestep, 0 - control, 1 - moderate, 2 - heavy, 3 - heavy with binging
date_interval: list of intervals, in year
img_paths: list of image paths
'''

img_paths = glob.glob(data_path+'*.nii.gz')
img_paths = sorted(img_paths)
subj_data = {}
label_dict = {'control': 0, 'moderate': 1, 'heavy': 2, 'heavy_with_binging': 3}
nan_label_count = 0
nan_idx_list = []
# pdb.set_trace()
for img_path in img_paths:
    subj_id = 'NCANDA_'+os.path.basename(img_path).split('_')[1]
    if 'baseline' in img_path:
        visit_id = 0
        visit_name = os.path.basename(img_path).split('_')[2].split('.')[0]
    else:
        visit_id = int(os.path.basename(img_path).split('_')[3][0])
        visit_name = 'followup_' + os.path.basename(img_path).split('_')[3].split('.')[0]
    print(subj_id, visit_id, visit_name)

    rows = df.loc[(df['subject'] == subj_id) & (df['visit'] == visit_id)]
    rows_raw = df_raw.loc[(df_raw['subject'] == subj_id) & (df_raw['visit'] == visit_name) & (df_raw['arm'] == 'standard')]
    if rows.shape[0] == 0 or rows_raw.shape[0] == 0:
        print('Missing label for', subj_id, visit_id, visit_name)
    else:
        # build dict
        if pd.isnull(rows.iloc[0]['cahalan']):
            continue
        if subj_id not in subj_data:
            subj_data[subj_id] = {'age': rows_raw.iloc[0]['visit_age'], 'sex': (0 if rows_raw.iloc[0]['sex']=='F' else 1), 'label_all': [], 'date_interval': [], 'img_paths': []}

        subj_data[subj_id]['date_interval'].append(rows_raw.iloc[0]['visit_age'] - subj_data[subj_id]['age'])
        if rows_raw.iloc[0]['visit_age'] - subj_data[subj_id]['age'] < 0:
            pdb.set_trace()
        subj_data[subj_id]['img_paths'].append(os.path.basename(img_path))
        subj_data[subj_id]['label_all'].append(label_dict[rows.iloc[0]['cahalan']])


# pdb.set_trace()
subj_id_list = []
num_ts_0 = 0
num_ts_1 = 0
num_ts_2 = 0
num_ts_3 = 0
for subj_id in subj_data.keys():
    subj_id_list.append(subj_id)
    label_all = np.array(subj_data[subj_id]['label_all'])
    num_ts_0 += (label_all==0).sum()
    num_ts_1 += (label_all==1).sum()
    num_ts_2 += (label_all==2).sum()
    num_ts_3 += (label_all==3).sum()
print('Number of timesteps, control/moderate/heavy/heavy_with_binging:', num_ts_0, num_ts_1, num_ts_2, num_ts_3)

np.save('/data/jiahong/data/NCANDA/NCANDA_longitudinal_subj.npy', subj_id_list)

# save subj_data to h5
# pdb.set_trace()
h5_noimg_path = '/data/jiahong/data/NCANDA/NCANDA_longitudinal_noimg_0219.h5'
if not os.path.exists(h5_noimg_path):
    f_noimg = h5py.File(h5_noimg_path, 'a')
    for i, subj_id in enumerate(subj_data.keys()):
        subj_noimg = f_noimg.create_group(subj_id)
        # subj_noimg.create_dataset('label', data=subj_data[subj_id]['label'])
        subj_noimg.create_dataset('label_all', data=subj_data[subj_id]['label_all'])
        subj_noimg.create_dataset('date_interval', data=subj_data[subj_id]['date_interval'])
        subj_noimg.create_dataset('age', data=subj_data[subj_id]['age'])
        subj_noimg.create_dataset('sex', data=subj_data[subj_id]['sex'])
        # subj_noimg.create_dataset('img_paths', data=subj_data[subj_id]['img_paths'])

# save images to h5
# pdb.set_trace()
# h5_img_path = '/data/jiahong/data/NCANDA/NCANDA_longitudinal_img.h5'
# if not os.path.exists(h5_img_path):
#     f_img = h5py.File(h5_img_path, 'a')
#     for i, subj_id in enumerate(subj_data.keys()):
#         subj_img = f_img.create_group(subj_id)
#         img_paths = subj_data[subj_id]['img_paths']
#         for img_path in img_paths:
#             img_nib = nib.load(os.path.join(data_path,img_path))
#             img = img_nib.get_fdata()
#             img = skimage.transform.resize(img, (64,64,64))
#             img = (img - np.mean(img)) / np.std(img)
#             subj_img.create_dataset(os.path.basename(img_path), data=img)
#         print(i, subj_id)

def augment_image(img, rotate, shift, flip):
    # pdb.set_trace()
    img = scipy.ndimage.interpolation.rotate(img, rotate[0], axes=(1,0), reshape=False)
    img = scipy.ndimage.interpolation.rotate(img, rotate[1], axes=(0,2), reshape=False)
    img = scipy.ndimage.interpolation.rotate(img, rotate[2], axes=(1,2), reshape=False)
    img = scipy.ndimage.shift(img, shift[0])
    if flip[0] == 1:
        img = np.flip(img, 0) - np.zeros_like(img)
    return img

# h5_img_path = '/data/jiahong/data/NCANDA/NCANDA_longitudinal_img_aug.h5'
# aug_size = 10
# if not os.path.exists(h5_img_path):
#     f_img = h5py.File(h5_img_path, 'a')
#     for i, subj_id in enumerate(subj_data.keys()):
#         subj_img = f_img.create_group(subj_id)
#         img_paths = subj_data[subj_id]['img_paths']
#         rotate_list = np.random.uniform(-2, 2, (aug_size-1, 3))
#         shift_list =  np.random.uniform(-2, 2, (aug_size-1, 1))
#         flip_list =  np.random.randint(0, 2, (aug_size-1, 1))
#         for img_path in img_paths:
#             img_nib = nib.load(os.path.join(data_path,img_path))
#             img = img_nib.get_fdata()
#             img = skimage.transform.resize(img, (64,64,64))
#             img = (img - np.mean(img)) / np.std(img)
#             imgs = [img]
#             for j in range(aug_size-1):
#                 imgs.append(augment_image(img, rotate_list[j], shift_list[j], flip_list[j]))
#             imgs = np.stack(imgs, 0)
#             subj_img.create_dataset(os.path.basename(img_path), data=imgs)
#         print(i, subj_id)

def save_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id+'\n')

# save txt, subj_id, case_id, case_number, case_id, case_number
def save_pair_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id[0]+' '+case_id[1]+' '+str(case_id[2])+' '+str(case_id[3])+'\n')

def save_single_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id[0]+' '+str(case_id[1])+'\n')

def get_subj_pair_case_id_list(subj_data, subj_id_list):
    subj_id_list_full = []
    case_id_list_full = []
    for subj_id in subj_id_list:
        case_id_list = subj_data[subj_id]['img_paths']
        for i in range(len(case_id_list)):
            for j in range(i+1, len(case_id_list)):
                subj_id_list_full.append(subj_id)
                case_id_list_full.append([case_id_list[i],case_id_list[j],i,j])

                # pdb.set_trace()
                # filter out pairs that are too close
                # if subj_data[subj_id]['date_interval'][j] - subj_data[subj_id]['date_interval'][i] >= 2:
                #     subj_id_list_full.append(subj_id)
                #     case_id_list_full.append([case_id_list[i],case_id_list[j],i,j])
    return subj_id_list_full, case_id_list_full

def get_subj_single_case_id_list(subj_data, subj_id_list):
    subj_id_list_full = []
    case_id_list_full = []
    for subj_id in subj_id_list:
        case_id_list = subj_data[subj_id]['img_paths']
        for i in range(len(case_id_list)):
            subj_id_list_full.append(subj_id)
            case_id_list_full.append([case_id_list[i], i])
    return subj_id_list_full, case_id_list_full

pdb.set_trace()
subj_list_postfix = 'all_single'
# subj_id_all = np.load('/data/jiahong/data/NCANDA/NCANDA_longitudinal_subj.npy', allow_pickle=True)
subj_id_all = subj_id_list

subj_list = []
subj_test_list = []
subj_val_list = []
subj_train_list = []

for fold in range(5):
    num_subj = len(subj_id_all)
    subj_test_list = subj_id_all[fold*int(0.2*num_subj):(fold+1)*int(0.2*num_subj)]
    class_train_val = subj_id_all[:fold*int(0.2*num_subj)] + subj_id_all[(fold+1)*int(0.2*num_subj):]
    subj_val_list = class_train_val[:int(0.1*len(class_train_val))]
    subj_train_list = class_train_val[int(0.1*len(class_train_val)):]

    if 'single' in subj_list_postfix:
        subj_id_list_train, case_id_list_train = get_subj_single_case_id_list(subj_data, subj_train_list)
        subj_id_list_val, case_id_list_val = get_subj_single_case_id_list(subj_data, subj_val_list)
        subj_id_list_test, case_id_list_test = get_subj_single_case_id_list(subj_data, subj_test_list)

        save_single_data_txt('../data/NCANDA/fold'+str(fold)+'_train_' + subj_list_postfix + '.txt', subj_id_list_train, case_id_list_train)
        save_single_data_txt('../data/NCANDA/fold'+str(fold)+'_val_' + subj_list_postfix + '.txt', subj_id_list_val, case_id_list_val)
        save_single_data_txt('../data/NCANDA/fold'+str(fold)+'_test_' + subj_list_postfix + '.txt', subj_id_list_test, case_id_list_test)
    else:
        subj_id_list_train, case_id_list_train = get_subj_pair_case_id_list(subj_data, subj_train_list)
        subj_id_list_val, case_id_list_val = get_subj_pair_case_id_list(subj_data, subj_val_list)
        subj_id_list_test, case_id_list_test = get_subj_pair_case_id_list(subj_data, subj_test_list)

        save_pair_data_txt('../data/NCANDA/fold'+str(fold)+'_train_' + subj_list_postfix + '.txt', subj_id_list_train, case_id_list_train)
        save_pair_data_txt('../data/NCANDA/fold'+str(fold)+'_val_' + subj_list_postfix + '.txt', subj_id_list_val, case_id_list_val)
        save_pair_data_txt('../data/NCANDA/fold'+str(fold)+'_test_' + subj_list_postfix + '.txt', subj_id_list_test, case_id_list_test)
