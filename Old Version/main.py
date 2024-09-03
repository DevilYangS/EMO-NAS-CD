import logging

import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

from NASCD import NASCD
import random
import torch.backends.cudnn as cudnn




DECSPACE = []

DECSPACE.append( [0, 1, 10] )
DECSPACE.append([0, 2, 10, 1, 2, 11, 3, 4, 10] )
DECSPACE.append([1, 2, 10, 3, 0, 6, 0, 2, 11, 0, 5, 11, 4, 6, 10] )
DECSPACE.append([0, 1, 10, 0, 2, 10, 3, 4, 11, 5, 0, 13] )
DECSPACE.append([1, 2, 10, 0, 2, 10, 3, 4, 11, 5, 0, 13, 6, 0, 6] )
DECSPACE.append([2, 0, 3, 1, 3, 10, 0, 4, 11, 5, 0, 13, 6, 0, 6] )
DECSPACE.append( [2, 0, 1, 1, 3, 10, 0, 4, 11, 5, 0, 13, 6, 0, 0, 7, 0, 6] )


DECSPACE.append([0, 1, 10]  )
DECSPACE.append(  [0, 1, 10, 3, 0, 7] )
DECSPACE.append(  [0, 1, 10, 3, 0, 2, 0, 0, 3, 4, 5, 10] )
DECSPACE.append( [1, 0, 13, 0, 0, 13, 3, 4, 10, 5, 0, 6] )
DECSPACE.append( [0, 1, 10, 2, 3, 11, 4, 0, 13, 5, 0, 6] )
DECSPACE.append( [1, 1, 12, 1, 3, 11, 0, 0, 9, 5, 2, 11, 4, 6, 10, 7, 0, 6, 8, 0, 13] )
DECSPACE.append( [1, 1, 12, 1, 3, 11, 0, 0, 9, 5, 2, 11, 4, 6, 10, 0, 0, 9, 8, 0, 0, 9, 0, 4, 7, 10, 10, 11, 0, 6, 12, 0, 13] )




from sklearn.utils import shuffle
from tqdm import tqdm
import os
import numpy as np
def split_dataset(train_data,valid_data,test_data,split=0.5):
    train_data = pd.concat([train_data,valid_data,test_data])
    train_set,test_set = pd.DataFrame(),pd.DataFrame()
    for index, value in enumerate(tqdm(train_data.groupby('user_id'))):
        train_size = int(len(value[1]) * split)+1

        item = value[1]
        item = shuffle(item)

        train_set = pd.concat([train_set,item[:train_size]])
        test_set = pd.concat([test_set,item[train_size:]])

    train_set = train_set.reset_index()
    test_set = test_set.reset_index()
    return train_set,test_set,test_set





train_data = pd.read_csv("../comparison/EduCDM-main/EduCDM-main/data/a0910/train.csv")
valid_data = pd.read_csv("../comparison/EduCDM-main/EduCDM-main/data/a0910/valid.csv")
test_data = pd.read_csv("../comparison/EduCDM-main/EduCDM-main/data/a0910/test.csv")
df_item = pd.read_csv("../comparison/EduCDM-main/EduCDM-main/data/a0910/item.csv")

item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)


user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
knowledge_n = np.max(list(knowledge_set))


def transform(user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)




batch_size = 128


for names in ['ASSIST','SLP']:
# for names in ['SLP']:

    list_ratio = [0.5,0.6,0.7,0.8] if names=='SLP' else [0.6,0.7,0.8]

    for split_ratio in list_ratio:
        print(split_ratio)

        if names=='ASSIST':

            train_data1,valid_data1,test_data1 = split_dataset(train_data,valid_data,test_data,split=split_ratio)
            train_set, valid_set, test_set = [
                transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
                for data in [train_data1, valid_data1, test_data1]
            ]
        else:
            from utils_dataset_train import  get_dataset
            train_set, valid_set, test_set ,user_n,item_n,knowledge_n = get_dataset(name='slp',ratio= split_ratio)



        for archi_i, NASDEC in enumerate(DECSPACE):

            result_1 =[]
            result_2 =[]

            for run_i in range(30):

                logging.getLogger().setLevel(logging.INFO)
                cdm = NASCD(knowledge_n, item_n, user_n,dec=NASDEC)

                best_result, last_result =cdm.train(train_set, test_set, epoch=50, device="cuda",lr=0.002) # SLP epoch 20: 0.8442

                del cdm
                result_1.append(best_result)
                result_2.append(last_result)


            root_path_1 ='experiment/train/NAS_CD_archi_{}_{}_{}_best_result_.txt'.format(archi_i,names,split_ratio)
            root_path_2 ='experiment/train/NAS_CD_archi_{}_{}_{}_last_result_.txt'.format(archi_i,names,split_ratio)

            result_1 = np.array(result_1)
            result_2 = np.array(result_2)
            result_1 = np.vstack([result_1, np.mean(result_1,axis=0 )])
            result_2 = np.vstack([result_2, np.mean(result_2,axis=0 )])

            np.savetxt(root_path_1,np.array(result_1),delimiter=' ')
            np.savetxt(root_path_2,np.array(result_2),delimiter=' ')




