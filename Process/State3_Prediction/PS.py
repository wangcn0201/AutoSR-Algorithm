import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class PS_1:
    def __init__(self, config, module, device):
        self.module = module
        self.reg = config["reg"]
        self.numItem = module["numItem"]
        self.target_length = config["targetlength"]
        self.config = config
        # self.neg_length = config["neglength"]

    def get_parameters(self):
        para = []
        return para

    def ps(self, feed_dict, user_em, history_item_em, item_batch, item_em, extracted_feture, enhanced_feature, mode):
        neg_length = None
        if mode == "test" or mode == "val_test":
            neg_length = self.config["eval_item_num"]
        else :
            neg_length = self.config["neglength"]


        ori_item_em = self.module["itemEmbedding"][item_batch]

        self.neu_embs = extracted_feture + enhanced_feature
        "(batchsize, numFactor)"
        prediction1 = torch.matmul(self.neu_embs.unsqueeze(1), item_em.transpose(1, 2))
        "(batchsize, numFactor) * (batchsize, numFactor, 2) = (batchsize, 2)"
        prediction2 = torch.matmul(user_em.unsqueeze(1), ori_item_em.transpose(1, 2))
        "(batchsize, 2)"
        self.prediction = (prediction1 + prediction2).squeeze()

        # pos_prediction = self.prediction[:][:self.target_length]
        # neg_prediction = self.prediction[:][-self.neg_length:]
        # diff = torch.sum(torch.sub(pos_prediction, neg_prediction), dim=1)
        if mode == "test" or mode == "val_test":
            extra_info = None
        else:
            pos_item_em = item_em[:,:self.target_length].squeeze()
            neg_item_em = item_em[:,-neg_length:].squeeze()
            pos_ori_em = ori_item_em[:,:self.target_length].squeeze()
            neg_ori_em = ori_item_em[:,-neg_length:].squeeze()
            self.L2_emb = torch.sum(torch.mul(user_em, user_em), dim=1) \
                          + torch.sum(torch.mul(pos_item_em, pos_item_em), dim=1) \
                          + torch.sum(torch.mul(neg_item_em, neg_item_em), dim=1) \
                          + torch.sum(torch.mul(pos_ori_em, pos_ori_em), dim=1) \
                          + torch.sum(torch.mul(neg_ori_em, neg_ori_em), dim=1) \
                          + torch.sum(torch.mul(extracted_feture, extracted_feture), dim=1) \
                          + torch.sum(torch.mul(enhanced_feature, enhanced_feature), dim=1)

            # self.Loss_0 = - torch.log(torch.sigmoid(diff))
            # out = torch.add(self.Loss_0, self.reg * self.L2_emb)

            extra_info = self.L2_emb
        return self.prediction, extra_info

class PS_2:
    def __init__(self, config, module, device):
        self.module = module
        self.device = device
        self.config = config

        self.numK = config["numK"]
        self.numFactor = config["numFactor"]
        self.hidden_units = 2 * self.numFactor
        self.numItem = module["numItem"]
        self.trainBatchSize = config["batchSize"]
        self.targetlength = config["targetlength"]
        self.neglength = config["neglength"]

        dropout = config["dropout"]
        self.drop = nn.Dropout(dropout)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

        self.itemEmbedding = module["itemEmbedding"]

        self.user_bias = Variable(torch.randn((2 * self.numFactor), dtype=torch.float32).to(device),
                                  requires_grad=True)
        self.denserlayer = Variable(torch.clamp(torch.randn(self.numK * self.numFactor, self.hidden_units),
                                                -1 / torch.sqrt(torch.tensor(self.numFactor).float()),
                                                1 / torch.sqrt(torch.tensor(self.numFactor).float())).to(device),
                                    requires_grad=True)
        self.prior_weight = Variable(
            torch.clamp(torch.randn((self.numK, self.hidden_units)), -1 / torch.sqrt(torch.tensor(self.numFactor).float()),
                        1 / torch.sqrt(torch.tensor(self.numFactor).float())).to(device), requires_grad=True)
        denseBias = torch.randn((self.numK * self.numFactor), dtype=torch.float32).to(device)
        self.denseBias = Variable(torch.clamp(denseBias, 0.1, 1).to(device), requires_grad=True)

    def get_parameters(self):
        out = []
        out.append(self.user_bias)
        out.append(self.denserlayer)
        out.append(self.prior_weight)
        out.append(self.denseBias)
        optimize_dict = [{'params': out}]
        return optimize_dict

    def ps(self, feed_dict, user_em, history_item_em, item_batch, item_em, extracted_feture, enhanced_feature, mode):
        neg_length = None
        if mode == "test" or mode == "val_test":
            neg_length = self.config["eval_item_num"]
        else:
            neg_length = self.config["neglength"]

        extracted_feture = extracted_feture + enhanced_feature

        merged = torch.tanh(torch.cat([user_em, extracted_feture], dim=1) + self.user_bias)
        "(batchsize, 2*numFactor)"
        k_user_embedding = torch.matmul(merged, self.denserlayer.transpose(0, 1))
        "(batchsize, numk*numFactor)"
        user_embedding_new = torch.tanh(k_user_embedding + self.denseBias)
        user_embedding_drop = self.drop(user_embedding_new)
        user_embedding_drop = torch.reshape(user_embedding_drop,
                                            [-1, self.numK, self.numFactor])  # (batchsize, numk, nunFactor)

        # pos_embedding = item_em[:, :self.targetlength, :]  # (batchsize, tarlen, numFactor)
        # neg_embedding = item_em[:, -self.neglength:, :]
        # element_pos = torch.matmul(user_embedding_drop, pos_embedding.transpose(1, 2))  # (batchsize, numk, tarlen)
        # element_neg = torch.matmul(user_embedding_drop, neg_embedding.transpose(1, 2))
        element = torch.matmul(user_embedding_drop, item_em.transpose(1, 2))  #(batchsize, numk, tar+neg)
        element_wise_mul = self.softmax2(element)
        # element_wise_mul = self.softmax2(
        #     torch.cat([element_pos, element_neg], dim=2))  # (batchsize, numk, tarlen+neglen)

        prior_weight = torch.reshape(torch.matmul(merged, self.prior_weight.transpose(0, 1)),
                                     [-1, self.numK, 1])  # beta
        soft_weight = self.softmax1(prior_weight)  # wik    #(batchsize, numk, 1)

        prediction = torch.sum(torch.mul(element_wise_mul, soft_weight), dim=1).squeeze()  # (batchsize, tarlen+neglen)

        extra_info = [element_wise_mul, prior_weight]
        return prediction, extra_info

    
class PS_3(nn.Module):
    def __init__(self, config, module, device):
        super(PS_3, self).__init__()
        self.numItem = module["numItem"]
        self.numFactor = config["numFactor"]
        self.batchsize = config["batchSize"]
        self.totallength = config["targetlength"] + config["neglength"]
        # self.targetlength + self.neglength
        # self.totallength =

        # self.decoder = nn.Linear(module.numFactor, module.numItem).to(device)
        '''
        self.weight = Variable(torch.randn((self.numItem, self.numFactor)).to(device),
                                requires_grad=True)
        '''
        self.bias = Variable(torch.randn((self.numItem, 1)).to(device), requires_grad=True)
        # self.decoder = nn.Linear(self.batchsize*self.totallength, self.numFactor).to(device)
        #self.decoder = nn.Linear(self.totallength, self.totallength).to(device)

    def get_parameters(self):
        para = []
        #para += list(self.parameters())
        #para.append(self.weight)
        para.append(self.bias)
        optimize_dict = [{'params': para}]
        return optimize_dict

    def ps(self, feed_dict, user_em, history_item_em, item_batch, item_em, extracted_feture, enhanced_feature, mode):
        "(batchsize, 1, numFactor)"
        "(batchsize, 2, numFactor)"
        #weights = self.weight[item_batch]
        weights = item_em
        biases = self.bias[item_batch].squeeze()
        # decoded_extracted_feature = self.decoder(extracted_feture)
        extracted_feture = extracted_feture + enhanced_feature
        out = torch.add(torch.matmul(extracted_feture.unsqueeze(1), weights.transpose(1, 2)).squeeze(), biases)
        #out = torch.matmul(extracted_feture.unsqueeze(1), weights.transpose(1, 2)).squeeze()
        #(batchsize, numFactor) * (batchsize, 100, numFactor) = (batchsize, 100)
        #out = self.decoder(out).squeeze(1)
        extra_info = []
        return out, extra_info

class PS_4(nn.Module):
    def __init__(self, config, module, device):
        super(PS_4, self).__init__()
        self.module = module
        self.numItem = module["numItem"]
        self.numFactor = config["numFactor"]
        self.batchSize = config["batchSize"]
        #self.itemEmbedding = Variable(torch.randn((self.numItem, self.numFactor)).to(device), requires_grad=True)

        #self.items_to_pred = torch.tensor([[items for items in range(self.numItem)] * self.batchSize]).\
        #    view(self.batchSize,self.numItem).to(device)
        #"(batchsize, numItem)"

        #self.W2 = nn.Embedding(self.numItem, self.numFactor, padding_idx=0).to(device)
        self.b2 = nn.Embedding(self.numItem, 1, padding_idx=0).to(device)
        #self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def get_parameters(self):
        para = []
        para += list(self.parameters())
        #para.append(self.itemEmbedding)
        optimize_dict = [{'params': para}]
        return optimize_dict

    def ps(self, feed_dict, user_em, history_item_em, item_batch, item_em, extracted_feture, enhanced_feature, mode):
        ori_item_em = self.module["itemEmbedding"][feed_dict["input_seq_batch"]]
        # w2 = self.W2(self.items_to_pred)
        # b2 = self.b2(self.items_to_pred)
        w2 = item_em
        b2 = self.b2(item_batch)
        "(batchsize, 2, numFactor)"
        "(batchsize, 2)"
        res = torch.baddbmm(b2, w2, user_em.unsqueeze(2)).squeeze()
        "user_em.unsqueeze(2): (batchsize, numFactor, 1)"
        "(batchsize, 2)"
        # union-level
        extracted_feture = extracted_feture + enhanced_feature
        res += torch.bmm(extracted_feture.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
        "(batchsize, 1, numFactor)   (batchsize, numFactor, 2)"
        "(batchsize, 2)"

        # item-item product
        rel_score = ori_item_em.bmm(w2.permute(0, 2, 1))
        "(batchsize, 2, numFactor) * (batchsize, numFactor, 2)"

        rel_score = torch.sum(rel_score, dim=1)
        res += rel_score
        return res, []

class PS_5(nn.Module):
    def __init__(self, config, module, device):
        super(PS_5, self).__init__()
        self.numItem = module["numItem"]
        self.numUser = module["numUser"]
        self.numFactor = config["numFactor"]
        self.batchSize = config["batchSize"]

        self.user_bias = Variable(torch.randn((self.numUser, 1)).to(device), requires_grad=True)
        self.item_bias = Variable(torch.randn((self.numItem, 1)).to(device), requires_grad=True)

    def get_parameters(self):
        para = []
        para.append(self.user_bias)
        para.append(self.item_bias)
        optimize_dict = [{'params': para}]
        return optimize_dict

    def ps(self, feed_dict, user_em, history_item_em, item_batch, item_em, extracted_feture, enhanced_feature, mode):
        u_ids = feed_dict["user_batch"]    #(32, 1)
        i_ids = item_batch  #(32, 2)
        # i_ids = i_ids.repeat(self.batchSize).reshape(self.batchSize, self.numItem)  #(32, numItem)
        u_bias = self.user_bias[u_ids]   #(32, 1)
        i_bias = self.item_bias[i_ids].squeeze(-1)   #(32, 2)
        extracted_feture = extracted_feture + enhanced_feature
        prediction = (user_em[:,None,:] * item_em).sum(-1)   #(32)
        prediction = prediction + (extracted_feture[:,None,:] * item_em).sum(-1)
        prediction = prediction + u_bias + i_bias
        return prediction.view((self.batchSize, -1)), []

class PS_6(nn.Module):
    def __init__(self, config, module, device):
        super(PS_6, self).__init__()
        self.numItem = module["numItem"]
        self.numUser = module["numUser"]
        self.numFactor = config["numFactor"]
        self.batchSize = config["batchSize"]

        self.prediction = nn.Linear(self.numFactor, 1, bias=False)

    def get_parameters(self):
        paras = list(self.parameters())
        optimize_dict = [{'params': paras}]
        return optimize_dict

    def ps(self, feed_dict, user_em, history_item_em, item_batch, item_em, extracted_feture, enhanced_feature, mode):
        extracted_feture = extracted_feture + enhanced_feature
        prediction = user_em[:, None, :] * item_em
        prediction = prediction + (extracted_feture[:,None,:] * item_em)
        prediction = self.prediction(prediction)
        return prediction.view((self.batchSize, -1)), []
