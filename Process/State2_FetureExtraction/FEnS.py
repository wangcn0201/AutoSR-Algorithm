import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class FEnS_1:
    def __init__(self, config, module, device, label):
        self.config = config
        self.module = module
        self.n_linear_layers = config[label+"_"+"linear_layers"]
        self.numFactor = config["numFactor"]

        self.dropout = nn.Dropout(config["dropout"])

        self.linear_weights = []
        self.linear_biases = []
        for i in range(self.n_linear_layers):
            weight = Variable(torch.randn((self.numFactor, self.numFactor), dtype=torch.float32).to(device),
                     requires_grad=True)
            bias = Variable(torch.randn((self.numFactor), dtype=torch.float32).to(device),
                     requires_grad=True)
            self.linear_weights.append(weight)
            self.linear_biases.append(bias)

    def get_parameters(self):
        para = []
        para += self.linear_weights + self.linear_biases
        optimize_dict = [{'params': para}]
        return optimize_dict

    def res_nn(self, aggre_embs):
        out = aggre_embs
        pre = out

        # hid_units = [None for i in range(self.n_linear_layers)]

        for layer in range(self.n_linear_layers):
            dropout_layer = self.dropout(out)
            out = torch.add(torch.matmul(dropout_layer, self.linear_weights[layer]), self.linear_biases[layer])
            out = torch.add(out, pre)

            out = torch.tanh(out)

            pre = out

        return out

    def fens(self, extracted_feture):
        uihid_agg_emb = extracted_feture
        aggre_embs = self.res_nn(uihid_agg_emb)
        return aggre_embs

class FEnS_2:
    def __init__(self, config, module, device, label):
        self.config = config
        self.module = module
        self.device = device

    def get_parameters(self):
        para = []
        return para

    def fens(self, extracted_feture):
        batchsize = self.config["batchSize"]
        numFactor = self.config["numFactor"]
        #enhanced_feture = Variable(torch.zeros((batchsize, numFactor)).to(self.device),
        #                           requires_grad=True)
        enhanced_feture = extracted_feture.mul(0.)
        return enhanced_feture