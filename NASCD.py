import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from utils.CDM import CDM
from Models.NCDMNet import Net as NCDNet
# from Models.NASCDNet import NASCDNet
from Models.NASCDNetV2 import NASCDNet

class NASCD(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n):
        super(NASCD, self).__init__()
        # self.ncdm_net = NCDNet(knowledge_n, exer_n, student_n)
        args={'n_student': student_n, 'n_exercise': exer_n, 'n_concept':knowledge_n, 'dim':knowledge_n}

        # self.ncdm_net = NASCDNet(args,  NAS_dec=[0,1,5,  1,1,6, 4,0,5,  4,0,0, 0,6,7]) # Wrong NCD but better performance
        # [0, 1, 6, 1, 1, 9, 4, 0, 6, 4, 0, 0, 0, 6, 10] # under Genotype_mapping
        # self.ncdm_net = NASCDNet(args,  NAS_dec=[0,0,5,  1,0,5, 1,0,6, 5,0,5,   4,0,0, 0,7,7,  6,8,8]) # NCD  based on MixedOp

        # self.ncdm_net = NASCDNet(args,  NAS_dec=[1,0,8,  1,0,10, 0,3,12,  5,0,9, 6,0,0, 4,7,11, 8,0,7]) # MIRT wrong version but similar peformance
        # self.ncdm_net = NASCDNet(args,  NAS_dec=[1,0,8,  1,0,10, 0,3,12,  5,0,9, 4,0,0, 6,7,11, 8,0,7]) # MIRT

        # self.ncdm_net = NASCDNet(args,  NAS_dec=[1,0,10,  1,0,10, 1,0,10,  3,0,8, 5,0,7, 4,0,0,
        #                                          0,8,11, 6,9,12, 10,0,7, 7,11,12, 11,12,11, 7,13,11]) #IRT wrong version: theta dim is 128 but should be 1

        # self.ncdm_net = NASCDNet(args,  NAS_dec=[0,1,13,  3,0,7]) # MCD net version-1 with similar performance: concatLinear [256,1]
        self.ncdm_net = NASCDNet(args,  NAS_dec=[0,1,13, 3,0,10, 4,0,7]) # MCD net: concatLinear + FFN
        # [0,1,12, 3,0,9, 4,0,6] # under Genotype_mapping

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                # pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                pred: torch.Tensor = self.ncdm_net([user_id, item_id, knowledge_emb])
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net([user_id, item_id, knowledge_emb])
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)

