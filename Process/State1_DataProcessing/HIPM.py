import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import torch.optim as opt
from torch.autograd import Variable

class HIPM_1:
    def __init__(self, config, module, device):
        self.module = module
        self.numFactor = config["numFactor"]
        self.input_length = config["input_length"]
        self.familiar_user_num = config["familiar_user_num"]
        self.device = device

    def get_parameters(self):
        para = []
        return para

    def hipm(self, feed_dict, user_em):
        item_batch = feed_dict["input_seq_batch"]
        return self.module["itemEmbedding"][item_batch]


class HIPM_2:
    def __init__(self, config, module, device):
        self.module = module
        self.numFactor = config["numFactor"]
        self.input_length = config["input_length"]
        self.familiar_user_num = config["familiar_user_num"]
        self.device = device
        self.itemEmbedding = module["itemEmbedding"]

        user_weight = np.zeros([1, self.numFactor])
        for e in range(self.numFactor):
            user_weight[0][e] = 0.5
        self.user_weight = Variable(torch.tensor(user_weight, dtype=torch.float32).to(self.device),
                                    requires_grad=True)

    def get_parameters(self):
        para = []
        para.append(self.user_weight)
        optimize_dict = [{'params': para}]
        return optimize_dict

    def get_user_memory_embedding(self, feed_dict):
        user_batch = feed_dict["user_batch"]
        input_seq_batch = feed_dict["input_seq_batch"]
        input_user_seq_batch = torch.tensor(feed_dict["input_user_seq_batch"]).to(self.device)

        # user_embedding = user_em
        user_memory_embedding = self.module["userEmbedding"][input_user_seq_batch]
        return user_memory_embedding

    def hipm(self, feed_dict, user_em):
        """
        return shape: (train_batch, input_size, numFactor)
        """

        user_memory_embedding = self.get_user_memory_embedding(feed_dict)
        user_embedding = torch.reshape(user_em, (-1, 1, self.numFactor))

        user_memory_embedding = torch.reshape(user_memory_embedding,
                                              (-1, self.input_length * self.familiar_user_num, self.numFactor))

        weight = torch.reshape(torch.div(torch.matmul(user_memory_embedding, user_embedding.transpose(1,2)),
                                         torch.sqrt(torch.tensor(self.numFactor).float())),
                               (-1, self.input_length, self.familiar_user_num))

        Softmax = nn.Softmax(dim=2)

        attention = torch.unsqueeze(Softmax(weight), dim=3)

        out = torch.mean(torch.mul(
            torch.reshape(user_memory_embedding, (-1, self.input_length, self.familiar_user_num, self.numFactor)),
            attention), dim=2)

        item_em = self.itemEmbedding[feed_dict["input_seq_batch"]]
        item_pre_embedding = torch.add(item_em, torch.mul(torch.clamp(self.user_weight, 0.1, 1.0),
                                                          out))

        return item_pre_embedding


class HIPM_3(nn.Module):
    def __init__(self, config, module, device):
        super(HIPM_3, self).__init__()
        self.module = module
        self.numFactor = config["numFactor"]
        self.input_length = config["input_length"]
        self.feature_gate_item = nn.Linear(self.numFactor, self.numFactor).to(device)
        self.feature_gate_user = nn.Linear(self.numFactor, self.numFactor).to(device)

    def get_parameters(self):
        para = list(self.parameters())
        optimize_dict = [{'params': para}]
        return optimize_dict

    def forward(self, feed_dict, user_em):
        item_em = self.module["itemEmbedding"][feed_dict["input_seq_batch"]]
        gate = torch.sigmoid(self.feature_gate_item(item_em) + self.feature_gate_user(user_em).unsqueeze(1))
        gated_item = item_em * gate

        return gated_item

    def hipm(self, feed_dict, user_em):
        return self.forward(feed_dict, user_em)


