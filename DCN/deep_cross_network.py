'''
Author: dqdallen 1986360994@qq.com
Date: 2022-06-15 10:38:25
LastEditors: dqdallen 1986360994@qq.com
LastEditTime: 2022-06-30 09:30:57
FilePath: \recommendation\DCN\deep_cross_network.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn


class DCN(nn.Module):

    def __init__(self, hidden_dims, cross_num, emb_dim, dropouts,
                 fea_nums_dict, numb_dim):
        """_summary_

        Args:
            hidden_dims (list): the dims of each hidden layer
            cross_num (int): the number of features cross interaction
            emb_dim (int): the dim of embedding
            dropouts (list): the dropout rates
            fea_nums_dict (dict): the number of each category feature
            numb_dim (int): the number of number feature
        """
        super(DCN, self).__init__()
        self.hidden_dims = hidden_dims
        self.cross_num = cross_num
        self.emb_dim = emb_dim
        self.dropouts = dropouts + [0]
        self.fea_nums_dict = fea_nums_dict
        # 不同特征的embedding层
        self.emb_layer = nn.ModuleDict()
        for fea_name in self.fea_nums_dict.keys():
            self.emb_layer[fea_name] = nn.Embedding(
                self.fea_nums_dict[fea_name], emb_dim)

        # DNNs部分
        input_dim = len(self.fea_nums_dict.values()) * emb_dim + numb_dim
        hidden_dims = [input_dim] + hidden_dims
        self.dnns = nn.ModuleList([nn.BatchNorm1d(input_dim)])
        for i in range(len(hidden_dims) - 1):
            self.dnns.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.dnns.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.dnns.append(nn.Dropout(self.dropouts[i]))
        # 特征交叉交互部分
        self.corss_layer = nn.ModuleList()
        for _ in range(cross_num):
            self.corss_layer.append(nn.Linear(input_dim - numb_dim, 1))

        self.predict = nn.Linear(
            hidden_dims[-1] + emb_dim * len(self.fea_nums_dict.values()), 1)
        self.sigmoid = nn.Sigmoid()

    def coss_layer(self, features):
        # 获得embedding
        emb_arr = []
        for fea_name in features:
            emb_tmp = self.emb_layer[fea_name](features[fea_name])
            emb_arr.append(emb_tmp)
        self.embedding = torch.cat(emb_arr, 1)
        emb = self.embedding
        # 特征进行cross interaction
        for i in range(self.cross_num):
            emb_tmp = torch.bmm(torch.transpose(emb.unsqueeze(1), 1, 2),
                            emb.unsqueeze(1))
            emb_tmp = self.corss_layer[i](emb_tmp)
            emb = emb_tmp.transpose(1, 2).squeeze(1) + emb
        return emb

    def deep_layer(self, numb_features):
        deep_emb = torch.cat([self.embedding, numb_features], 1)
        for layer in self.dnns:
            deep_emb = layer(deep_emb)
        return deep_emb

    def forward(self, numb_features, features):
        cross_result = self.coss_layer(features)
        dnn_result = self.deep_layer(numb_features)
        output = torch.cat([cross_result, dnn_result], 1)
        output = self.predict(output)
        output = self.sigmoid(output)
        return output
