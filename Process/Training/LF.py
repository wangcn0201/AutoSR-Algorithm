import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LF_1:
    def __init__(self, config, module, device):
        self.config = config
        self.module = module
        self.batchsize = config["batchSize"]
        self.ws = torch.ones((self.batchsize)).to(device)
        self.input_length = config["input_length"]
        self.target_length = config["targetlength"]
        self.neg_length = config["neglength"]
        self.reg = config["reg"]

    def get_parameters(self):
        para = []
        return para

    def lf(self, user_em, item_batch, item_em, prediction, extra_info):

        self.L2_emb = extra_info

        pos_prediction = prediction[:,:self.target_length]
        neg_prediction = prediction[:,-self.neg_length:]
        diff = torch.sum(torch.sub(pos_prediction, neg_prediction), dim=1)

        self.Loss_0 = - torch.log(torch.sigmoid(diff)+1e-8)
        self.Loss_0 = torch.add(self.Loss_0, self.reg * self.L2_emb)
        self.Loss_0 = self.Loss_0 * self.ws
        self.error_emb = torch.sum(self.Loss_0)
        self.error_nn = self.error_emb / self.batchsize / self.input_length
        return self.error_nn

class LF_2:
    def __init__(self, config, module, device):
        self.config = config
        self.module = module
        self.trainBatchSize = config["batchSize"]
        self.numK = config["numK"]
        self.numFactor = config["numFactor"]
        self.target_length = config["targetlength"]
        self.neg_num = config["neglength"]
        self.target_weight = 0.5

        labels_vector1 = Variable(torch.full([self.trainBatchSize, self.numK, 1], 1.0).to(device),
                                  requires_grad=False)
        labels_vector2 = Variable(torch.full([self.trainBatchSize, self.numK, self.neg_num], 0.0).to(device),
                                  requires_grad=False)
        self.labels2 = torch.cat([labels_vector1, labels_vector2], dim=2)

        self.softmax = nn.Softmax(dim=2)

    def get_parameters(self):
        para = []
        # para.append(self.labels2)
        return para

    def lf(self, user_em, item_batch, item_em, prediction, extra_info):
        element_wise_mul = extra_info[0]
        prior_weight = extra_info[1]
        "(batchsize, numk, 1)"
        sig_weight = torch.sigmoid(prior_weight)

        mse_log = torch.abs(element_wise_mul - sig_weight)
        mse_t = (self.target_weight + element_wise_mul) - self.target_weight * (mse_log + element_wise_mul)
        mse_p = torch.log(mse_t + 1e-7)
        mse_n = torch.log((1 - mse_t) + 1e-7)
        mse_loss = torch.reshape(torch.mean(torch.sum(self.labels2 * (mse_n - mse_p) - mse_n,
                                                      dim=2), dim=1), [-1, 1])
        return mse_loss

class LF_3:
    def __init__(self, config, module, device):
        self.config = config
        self.module = module
        self.trainBatchSize = config["batchSize"]
        self.targetlength = config["targetlength"]
        self.device = device

    def get_parameters(self):
        para = []
        return para

    def lf(self, user_em, item_batch, item_em, prediction, extra_info):
        pos_item = torch.tensor([0] * self.trainBatchSize).to(self.device)
        crossentropyloss = nn.CrossEntropyLoss()
        loss = crossentropyloss(prediction, pos_item.squeeze())
        return loss

class LF_4:
    def __init__(self, config, module, device):
        self.config = config
        self.module = module
        self.trainBatchSize = config["batchSize"]

    def get_parameters(self):
        para = []
        return para

    def lf(self, user_em, item_batch, item_em, prediction, extra_info):
        output = prediction
        # target = item_batch[0]
        # target_scores = torch.squeeze(torch.gather(output, 1, target.view(-1, 1)))
        target_scores = prediction[:,0]
        below = -torch.log(torch.sum(torch.exp(output), 1))
        log_pl = target_scores + below
        neg_like = -log_pl
        return neg_like

class LF_5:
    def __init__(self, config, module, device):
        self.config = config
        self.module = module
        self.batchsize = config["batchSize"]
        self.input_length = config["input_length"]
        self.target_length = config["targetlength"]
        self.neg_length = config["neglength"]


    def get_parameters(self):
        para = []
        return para

    def lf(self, user_em, item_batch, item_em, prediction, extra_info):
        predictions = prediction
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        # loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
        neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        loss = F.softplus(-(pos_pred - neg_pred)).mean()
        # ↑ For numerical stability, we use 'softplus(-x)' instead of '-log_sigmoid(x)'
        return loss
