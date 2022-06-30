"""
Author: dqdallen 1986360994@qq.com
Date: 2022-06-13 16:01:11
LastEditors: dqdallen 1986360994@qq.com
LastEditTime: 2022-06-15 15:22:23
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, hidden_dims, emb_dim, dropouts, fea_nums_dict, numb_dim):
        """deepfm模型

        Args:
            hidden_dims (list): the hidden dims of dnn
            emb_dim (int): the dim of embedding
            dropouts (list): the rates of dropouts in dnn
            fea_nums_dict (dict): each category's feature nums
            numb_dim (int): the dim of numerical features
        """
        super(DeepFM, self).__init__()
        self.hidden_dims = hidden_dims
        self.emb_dim = emb_dim
        self.dropouts = dropouts + [0]
        self.fea_nums_dict = fea_nums_dict
        self.emb_layer = nn.ModuleDict()
        for fea_name in self.fea_nums_dict.keys():
            self.emb_layer[fea_name] = nn.Embedding(
                self.fea_nums_dict[fea_name], emb_dim
            )

        self.fm_first = nn.ModuleDict()
        for fea_name in self.fea_nums_dict.keys():
            self.fm_first[fea_name] = nn.Embedding(
                self.fea_nums_dict[fea_name], 1)

        input_dim = len(self.fea_nums_dict.values()) * emb_dim + numb_dim
        hidden_dims = [input_dim] + hidden_dims
        self.dnns = nn.ModuleList([nn.BatchNorm1d(input_dim)])
        for i in range(len(hidden_dims) - 1):
            self.dnns.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.dnns.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.dnns.append(nn.Dropout(self.dropouts[i]))
        self.predict = nn.Linear(
            hidden_dims[-1] + emb_dim + len(self.fea_nums_dict.values()), 1
        )
        self.sigmoid = nn.Sigmoid()

    def fm_layer(self, features):
        # 一阶交互
        emb_arr = []
        for fea_name in features:
            emb_tmp = self.fm_first[fea_name](features[fea_name])
            emb_arr.append(emb_tmp)
        first_order = torch.cat(emb_arr, 1)

        emb_arr = []
        for fea_name in features:
            emb_tmp = self.emb_layer[fea_name](features[fea_name])
            emb_arr.append(emb_tmp)
        self.embedding = torch.cat(emb_arr, 1)
        # 二阶交互
        sum_square = sum(emb_arr) ** 2
        square_sum = sum([emb * emb for emb in emb_arr])
        second_order = 0.5 * (sum_square - square_sum)
        # 拼接一阶和二阶交互
        fm_result = torch.cat([first_order, second_order], 1)
        return fm_result

    def deep_layer(self, numb_features):
        """dnns层

        Args:
            numb_features (tensor): numerical features

        Returns:
            tensor: dnn_result
        """
        # 拼接类别特征的embedding和数值特征
        deep_emb = torch.cat([self.embedding, numb_features], 1)
        for layer in self.dnns:
            deep_emb = layer(deep_emb)
        return deep_emb

    def forward(self, numb_features, features):
        fm_result = self.fm_layer(features)
        dnn_result = self.deep_layer(numb_features)
        output = torch.cat([fm_result, dnn_result], 1)
        output = self.predict(output)
        output = self.sigmoid(output)
        return output
