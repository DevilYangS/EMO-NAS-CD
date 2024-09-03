import json
import random
import numpy as np



min_log = 15


def divide_data(name='assist', ratio=0.6):
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set, val_set and test_set (0.6:0.1:0.2)
    :return:
    '''
    with open('{}_log_data.json'.format(name,name), encoding='utf8') as i_f:
        stus = json.load(i_f)

    # get the knowledge matrix
    Logs =[]
    for stu in stus:
        logs = []
        for log in stu['logs']:
            logs.append(log)
        Logs.extend(logs)
    Exer_list = []
    Concept=[]
    for item in Logs:
        if item['exer_id'] not in Exer_list:
            Exer_list.append(item['exer_id'])
            Concept.append(item['knowledge_code'])
    A = 1
    Concept_k = []
    for x in Concept:
        Concept_k.extend(x)
    Concept_k = np.unique(Concept_k)

    Matrix_item2knowledge = np.zeros([len(Exer_list),len(Concept_k)])
    Matrix_item2knowledge = np.zeros([np.max(Exer_list),len(Concept_k)])
    for item1,item2 in zip(Exer_list,Concept):
        Matrix_item2knowledge[item1-1,np.array(item2)-1] = 1
    np.savetxt('data/{}/{}_item2knowledge.txt'.format(name,name),Matrix_item2knowledge.astype(int),delimiter=' ')





    # 1. delete students who have fewer than min_log response logs
    stu_i = 0
    count_i=0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
            count_i+=1
        stu_i += 1
    # 2. divide dataset into train_set, val_set and test_set
    train_slice, train_set, val_set, test_set = [], [], [], []
    val_slice,test_slice = [],[]


    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_val = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * ratio)
        val_size = int(stu['log_num'] * 0.1)
        test_size = stu['log_num'] - train_size - val_size
        logs = []
        for log in stu['logs']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_val['log_num'] = val_size
        stu_val['logs'] = logs[train_size:train_size+val_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[-test_size:]
        train_slice.append(stu_train)

        val_slice.append(stu_val)
        test_slice.append(stu_test)




        # val_set.append(stu_val)
        # test_set.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
        for log in stu_val['logs']:
            val_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                            'knowledge_code': log['knowledge_code']})
        for log in stu_test['logs']:
            test_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                             'knowledge_code': log['knowledge_code']})





    random.shuffle(train_set)
    # with open('data/junyi_train_slice.json', 'w', encoding='utf8') as output_file:
    #     json.dump(train_slice, output_file, indent=4, ensure_ascii=False)

    if ratio==0.7:
        with open('data/{}/{}_train_set.json'.format(name,name), 'w', encoding='utf8') as output_file:
            json.dump(train_set, output_file, indent=4, ensure_ascii=False)

        random.shuffle(val_set)
        with open('data/{}/{}_val_set.json'.format(name,name), 'w', encoding='utf8') as output_file:
            json.dump(val_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set

        random.shuffle(test_set)
        with open('data/{}/{}_test_set.json'.format(name,name), 'w', encoding='utf8') as output_file:
            json.dump(test_set, output_file, indent=4, ensure_ascii=False)

    else:
        with open('data/{}/{}_train_set_{}.json'.format(name,name,int(ratio*10+1)), 'w', encoding='utf8') as output_file:
            json.dump(train_set, output_file, indent=4, ensure_ascii=False)

        random.shuffle(val_set)
        with open('data/{}/{}_val_set_{}.json'.format(name,name,int(ratio*10+1)), 'w', encoding='utf8') as output_file:
            json.dump(val_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set

        random.shuffle(test_set)
        with open('data/{}/{}_test_set_{}.json'.format(name,name,int(ratio*10+1)), 'w', encoding='utf8') as output_file:
            json.dump(test_set, output_file, indent=4, ensure_ascii=False)






if __name__ == '__main__':
    # ratio ==0.6, then named _7; if  ratio ==0.7, it is named train_set, test_set without  _8

  # ratio from 0.4 to 0.7 [0.4 0.5 0.6 0.7] executed
    divide_data(name='junyi',ratio=0.6) # could be used for ASSIST09 (e.g. assist), SLP and JUNYI, but here used for SLP and JUNYI

