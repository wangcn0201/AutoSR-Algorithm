import torch
import torch.nn as nn
from torch.autograd import Variable

class UVPM_1:
    def __init__(self, config, module, device):
        self.config = config
        self.module = module
        self.numFactor = self.config["numFactor"]
        self.dropout = nn.Dropout(config["dropout"])

        self.softmax = nn.Softmax(dim=2)

        self.item_weight = Variable(torch.randn((self.numFactor, self.numFactor), dtype=torch.float32).to(device),
                                    requires_grad=True)
        self.user_weight = Variable(torch.randn((self.numFactor, self.numFactor), dtype=torch.float32).to(device),
                                    requires_grad=True)
        self.att_bias = Variable(torch.randn((self.numFactor), dtype=torch.float32).to(device),
                                 requires_grad=True)
        self.out_weight = Variable(torch.randn((self.numFactor, 1), dtype=torch.float32).to(device),
                                    requires_grad=True)


    def get_parameters(self):
        para = []
        para.append(self.item_weight)
        para.append(self.user_weight)
        para.append(self.att_bias)
        para.append(self.out_weight)
        optimize_dict = [{'params': para}]
        return optimize_dict

    def uvpm(self, feed_dict):
        item_embs = self.module["itemEmbedding"][feed_dict["input_seq_batch"]]
        "(batchsize, input_length, numFactor)"
        user_embs = self.module["userEmbedding"][feed_dict["user_batch"]]
        "(batchsize, numFactor)"

        drop_item_embs = self.dropout(item_embs)
        wi = torch.matmul(torch.reshape(drop_item_embs, (-1, self.numFactor)), self.item_weight)
        wi = torch.reshape(wi, (-1, self.config["input_length"], self.numFactor))

        drop_user_embs = self.dropout(user_embs)
        wu = torch.matmul(drop_user_embs, self.user_weight)
        wu = torch.reshape(wu, (-1, 1, self.numFactor))

        w = torch.tanh(torch.add(torch.add(wi, wu), self.att_bias))
        w = torch.reshape(w, (-1, self.numFactor))

        outs = torch.sigmoid(torch.matmul(w, self.out_weight))
        outs = torch.reshape(outs, (-1, self.config["input_length"], 1))

        outs = self.softmax(outs.transpose(1, 2))
        outs = torch.matmul(outs, item_embs)

        outs = torch.add(outs, torch.reshape(item_embs[:,-1,:], (-1, 1, self.numFactor)))
        outs = torch.reshape(outs, (-1, self.numFactor))

        return outs

class UVPM_2:
    def __init__(self, config, module, device):
        self.config = config
        self.module = module
        self.device = device

    def get_parameters(self):
        para = []
        return para

    def uvpm(self, feed_dict):
        """
        :return: (batchsize, numFactor)
        """

        user_batch = torch.tensor(feed_dict["user_batch"]).to(self.device)
        user_embedding = self.module["userEmbedding"][user_batch]
        return user_embedding

