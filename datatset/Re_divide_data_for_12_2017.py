import json
import random
import numpy as np
from collections import Counter


min_log = 15


def redivide_data(name='assist12',ratio=0.7):
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set, val_set and test_set (0.7:0.1:0.2)
    :return:
    '''
    with open('./data/{}/{}}_train_set.json'.format(name,name), encoding='utf8') as i_f:
        stus = json.load(i_f)
    with open('.//data/{}/{}_val_set.json'.format(name,name), encoding='utf8') as i_f:
        val = json.load(i_f)
    with open('./data/{}/{}_test_set.json'.format(name,name), encoding='utf8') as i_f:
        test = json.load(i_f)
    

    # # 2. divide dataset into train_set, and test_set
    train_slice, train_set, val_set, test_set = [], [], [], []
    val_slice,test_slice = [],[]
    all_user=[]
    all_user.extend(stus)
    all_user.extend(val)
    all_user.extend(test)
    train_size=int(len(all_user)*(ratio+0.1))
    train_set=all_user[:train_size]
    test_set=all_user[train_size:]
    


    if ratio==0.7:

        random.shuffle(train_set)
        with open('./data/{}/{}_train_set.json'.format(name,name), 'w', encoding='utf8') as output_file:
            json.dump(train_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set

        random.shuffle(test_set)
        with open('./data/{}/{}_test_set.json'.format(name,name), 'w', encoding='utf8') as output_file:
            json.dump(test_set, output_file, indent=4, ensure_ascii=False)
    else:
        random.shuffle(train_set)
        with open('./data/{}/{}_train_set_{}.json'.format(name,name,int(ratio*10+1)), 'w', encoding='utf8') as output_file:
            json.dump(train_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set

        random.shuffle(test_set)
        with open('./data/{}/{}_test_set_{}.json'.format(name,name,int(ratio*10+1)), 'w', encoding='utf8') as output_file:
            json.dump(test_set, output_file, indent=4, ensure_ascii=False)





if __name__ == '__main__':
    # RE-SPLIT THE dataset assist12 and assist2017
    redivide_data(name='assist12',ratio=0.6)

