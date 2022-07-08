import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class FExS_1:
    def __init__(self, config, module, device, label):
        self.config = config
        self.att_layers = self.config[label+"_"+"attention_layers"]
        self.device = device

        self.module = module
        self.numFactor = self.config["numFactor"]
        self.input_length = self.config["input_length"]

        self.dropout = nn.Dropout(self.config["dropout"])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

        #init weights
        self.user_weights = []
        self.user_biases = []
        self.item_weights = []
        self.item_biases = []
        self.user_hidatt_rel_weights = []
        self.item_hidatt_rel_weights = []
        self.hidatt_rel_biases = []
        self.out_rel_weights = []
        for i in range(config[label+"_"+"attention_layers"]):
            user_weight = Variable(torch.randn((self.numFactor, self.numFactor), dtype=torch.float32).to(self.device),
                     requires_grad=True)
            self.user_weights.append(user_weight)
            user_bias = Variable(torch.randn((self.numFactor), dtype=torch.float32).to(self.device),
                     requires_grad=True)
            self.user_biases.append(user_bias)

            item_weight = Variable(torch.randn((self.numFactor, self.numFactor), dtype=torch.float32).to(self.device),
                     requires_grad=True)
            self.item_weights.append(item_weight)
            item_bias = Variable(torch.randn((self.numFactor), dtype=torch.float32).to(self.device),
                     requires_grad=True)
            self.item_biases.append(item_bias)

            user_hidatt_rel_weight = Variable(torch.randn((self.numFactor, self.numFactor),
                                                          dtype=torch.float32).to(self.device), requires_grad=True)
            item_hidatt_rel_weight = Variable(torch.randn((self.numFactor, self.numFactor),
                                                          dtype=torch.float32).to(self.device), requires_grad=True)
            out_rel_weight = Variable(torch.randn((self.numFactor, 1),
                                                  dtype=torch.float32).to(self.device), requires_grad=True)
            hidatt_rel_bias = Variable(torch.randn((self.numFactor),
                                                   dtype=torch.float32).to(self.device), requires_grad=True)
            self.user_hidatt_rel_weights.append(user_hidatt_rel_weight)
            self.item_hidatt_rel_weights.append(item_hidatt_rel_weight)
            self.out_rel_weights.append(out_rel_weight)
            self.hidatt_rel_biases.append(hidatt_rel_bias)

        self.hidatt_weight = Variable(torch.randn((self.numFactor, self.numFactor), dtype=torch.float32).to(self.device),
                     requires_grad=True)
        self.hidatt_bias = Variable(torch.randn((self.numFactor), dtype=torch.float32).to(self.device),
                     requires_grad=True)
        self.hidatt_out = Variable(torch.randn((self.numFactor, 1), dtype=torch.float32).to(self.device),
                     requires_grad=True)

    def get_parameters(self):
        para = []
        para = para + self.user_weights + self.user_biases + self.item_weights + self.item_biases
        para += self.user_hidatt_rel_weights + self.hidatt_rel_biases + \
                self.item_hidatt_rel_weights + self.out_rel_weights
        para.append(self.hidatt_weight)
        para.append(self.hidatt_bias)
        para.append(self.hidatt_out)
        optimize_dict = [{'params': para}]
        return optimize_dict

    def ures_nn(self, aggre_embs):
        out = aggre_embs

        hid_units = [None for i in range(self.att_layers)]
        for layer in range(self.att_layers):
            pre = out

            dropout_layer = self.dropout(out)
            out = torch.add(torch.matmul(dropout_layer, self.user_weights[layer]), self.user_biases[layer])
            out = torch.add(out, pre)

            out = self.relu(out)

            hid_units[layer] = out

        return hid_units

    def ires_nn(self, aggre_embs):
        pre = aggre_embs
        hid_units = []

        for layer in range(self.att_layers):
            pre = torch.reshape(pre, (-1, self.numFactor))

            dropout_layer = self.dropout(pre)
            out = torch.add(torch.matmul(dropout_layer, self.item_weights[layer]), self.item_biases[layer])
            out = torch.add(out, pre)
            pre = torch.reshape(self.relu(out), (-1, self.config["input_length"], self.numFactor))

            hid_units.append(pre)

        return hid_units

    def multRelAtt(self, uhid_units, ihid_units):
        hid_atts = []

        for i in range(self.att_layers):

            drop_item_embs = self.dropout(ihid_units[i])
            wi = torch.matmul(torch.reshape(drop_item_embs, (-1, self.numFactor)),
                           self.item_hidatt_rel_weights[i])
            wi = torch.reshape(wi, (-1, self.input_length, self.numFactor))

            drop_user_embs = self.dropout(uhid_units[i])
            wu = torch.matmul(drop_user_embs, self.user_hidatt_rel_weights[i])
            wu = torch.reshape(wu, [-1, 1, self.numFactor])



            w = torch.tanh(torch.add(torch.add(wi, wu), self.hidatt_rel_biases[i]))


            w = torch.reshape(w, (-1, self.numFactor))
            outs = torch.sigmoid(torch.matmul(w, self.out_rel_weights[i]))
            outs = torch.reshape(outs, [-1, self.input_length, 1])

            # tensor shape: batch,1,row
            outs = self.softmax(outs.transpose(1, 2))
            outs = torch.matmul(outs, ihid_units[i])

            outs = torch.reshape(outs, [-1, 1, self.numFactor])

            hid_atts.append(outs)

        return hid_atts

    def hidAtt(self, hidden_embs):
        """
        	input hidden_embs: [batch, layers, dim]
        """
        drop_hid_embs = self.dropout(hidden_embs)
        wh = torch.matmul(torch.reshape(drop_hid_embs, (-1, self.numFactor)), self.hidatt_weight)
        "(batchsize * layers, numFactor)"
        wh = torch.reshape(wh, (-1, self.att_layers + 1, self.numFactor))
        "(batchsize, layers, numFactor)"



        w = torch.tanh(torch.add(wh, self.hidatt_bias))

        w = torch.reshape(w, (-1, self.numFactor))

        outs = torch.sigmoid(torch.matmul(w, self.hidatt_out))
        outs = torch.reshape(outs, (-1, self.att_layers + 1, 1))
        "(batchsize, layers, 1)"

        # tensor shape: batch,1,row
        outs = self.softmax(outs.transpose(1, 2))
        "(batchsize, 1, layers)"
        outs = torch.reshape(torch.matmul(outs, hidden_embs), (-1, self.numFactor))
        "(batchsize, numFactor)"

        return outs

    def fexs(self, feed_dict, history_item_em, user_em):
        ori_user_batch = feed_dict["user_batch"]
        ori_item_batch = feed_dict["input_seq_batch"]

        ori_user_emb = self.module["userEmbedding"][ori_user_batch]
        ori_item_emb = history_item_em


        uhid_units = self.ures_nn(ori_user_emb)
        ihid_units = self.ires_nn(ori_item_emb)

        uihid_atts = self.multRelAtt(uhid_units, ihid_units)

        uihid_atts.append(torch.reshape(user_em, [-1, 1, self.numFactor]))
        uihid_att_embs = torch.cat((uihid_atts), 1)

        uihid_agg_emb = self.hidAtt(uihid_att_embs)

        return uihid_agg_emb

class FExS_2(nn.Module):
    def __init__(self, config, module, device, label):
        super(FExS_2, self).__init__()
        self.config = config
        self.module = module

        self.numItem = module["numItem"]
        self.numFactor = config["numFactor"]
        self.numHid = config[label+"_"+"hidden_layer"]
        self.numLayer = config["GRU_layers"]
        self.batchsize = config["batchSize"]

        self.hidden = Variable(torch.zeros((self.numLayer, self.batchsize, self.numHid)).to(device))

        dropout = config["dropout"]
        self.drop = nn.Dropout(dropout)

        self.decoder = nn.Linear(self.numHid, self.numFactor).to(device)

        self.gru = nn.GRU(self.numFactor, self.numHid, self.numLayer, dropout=dropout).to(device)

    def get_parameters(self):
        para = []
        para += list(self.parameters())
        optimize_dict = [{'params': para}]
        return optimize_dict

    def forward(self, feed_dict, history_item_em, user_em):
        emb = history_item_em
        emb = self.drop(emb)
        emb = torch.reshape(emb, (self.config["input_length"], self.batchsize, -1))
        output, _ = self.gru(emb, self.hidden)
        out = self.decoder(output)
        out = torch.reshape(out, (self.batchsize, self.config["input_length"], -1))
        "####"
        out = torch.mean(out, dim=1)
        "####"
        return out

    def fexs(self, feed_dict, history_item_em, user_em):
        return self.forward(feed_dict, history_item_em, user_em)


class FExS_3:
    def __init__(self, config, module, device, label):
        self.module = module
        self.device = device
        self.numFactor = config["numFactor"]

        self.itemEmbedding = module["itemEmbedding"]

    def get_parameters(self):
        out = []
        return out

    def fexs(self, feed_dict, history_item_em, user_em):
        item_em = self.itemEmbedding[feed_dict["input_seq_batch"]]
        # item_em = feed_dict["input_seq_batch"]
        # item_pre_embedding = torch.add(item_em, torch.mul(torch.clamp(self.user_weight, 0.1, 1.0),
        #                                                   history_item_em))
        # "(batchsize, input_length, numFactor)"
        item_pre_embedding = history_item_em
        weight = torch.div(torch.matmul(item_pre_embedding, torch.unsqueeze(user_em, dim=2)),
                           torch.sqrt(torch.tensor(self.numFactor).float()))
        "(batchsize, input_length, 1)"
        Softmax = nn.Softmax(dim=1)
        attention = Softmax(weight)
        "(batchsize, input_length, 1)"
        out = torch.mean(torch.mul(item_em, attention), dim=1)
        "(batchsize, numFactor)"
        return out

class FExS_4(nn.Module):
    def __init__(self, config, module, device, label):
        super(FExS_4, self).__init__()
        self.pooling = config[label+"_"+"Pooling_method"]
        self.batchsize = config["batchSize"]
        self.numFactor = config["numFactor"]
        self.input_length = config["input_length"]

        self.instance_gate_item = Variable(torch.zeros(self.numFactor, 1).type(torch.FloatTensor).to(device),
                                           requires_grad=True)
        self.instance_gate_user = Variable(torch.zeros(self.numFactor,
                                                       self.input_length).type(torch.FloatTensor).to(device),
                                           requires_grad=True)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)

    def get_parameters(self):
        para = []
        para.append(self.instance_gate_item)
        para.append(self.instance_gate_user)
        optimize_dict = [{'params': para}]
        return optimize_dict

    def forward(self, history_item_em, user_em):
        instance_score = torch.sigmoid(torch.matmul(history_item_em, self.instance_gate_item.unsqueeze(0)).squeeze() +
                                       user_em.mm(self.instance_gate_user))
        "(batchsize, input_length)"
        union_out = history_item_em * instance_score.unsqueeze(2)

        if self.pooling == "Avg":
            union_out = torch.sum(union_out, dim=1)
            union_out = union_out / torch.sum(instance_score, dim=1).unsqueeze(1)
            "(batchsize, numFactor)"
        elif self.pooling == "Max":
            _, index = torch.max(instance_score, 1)
            index = torch.reshape(index, (-1, 1))
            index = index.repeat(1, 1, self.numFactor).reshape((self.batchsize, 1, self.numFactor))
            union_out = torch.gather(union_out, 1, index).squeeze()
        else :
            raise Exception("no such method")
        return union_out

    def fexs(self, feed_dict, history_item_em, user_em):
        return self.forward(history_item_em, user_em)

class FExS_5:
    def __init__(self, config, module, device, label):
        self.config = config
        self.module = module
        self.device = device

    def get_parameters(self):
        para = []
        return para

    def fexs(self, feed_dict, history_item_em, user_em):
        batchsize = self.config["batchSize"]
        numFactor = self.config["numFactor"]
        out = user_em.mul(0.)
        #Variable(torch.zeros((batchsize, numFactor)).to(self.device), requires_grad=True)
        return out
        