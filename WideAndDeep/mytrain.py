import sys

# sys.path.append("/content/dien/script/")

import numpy
from data_iterator import DataIterator
import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from wide_deep import WideAndDeep

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0


def prepare_data(input, target, maxlen=None, return_neg=False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros(
        (n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros(
        (n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)


def eval(test_data, model, model_path):

    loss_sum = 0.
    nums = 0
    scores = []
    targets = []
    with torch.no_grad():
        for src, tgt in test_data:
            nums += 1
            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
                src, tgt, return_neg=True)
            mids = torch.tensor(mids).cuda()
            uids = torch.tensor(uids).cuda()
            cats = torch.tensor(cats).cuda()
            mid_his = torch.tensor(mid_his).cuda()
            cat_his = torch.tensor(cat_his).cuda()
            target = torch.tensor(target).cuda()
            features = {'uid': uids, 'mid': mids, 'cat': cats,
                        'mid_his': mid_his, 'cat_his': cat_his}
            output = model(features)
            # loss = nn.BCELoss()(output, target.float())
            loss = nn.CrossEntropyLoss()(output, target.float())
            loss_sum += loss
            prob = output[:, 0].tolist()
            target = target[:, 0].tolist()
            scores += prob
            targets += target
    test_auc = roc_auc_score(targets, scores)
    loss_sum = loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        torch.save(model.state_dict(), model_path)
    return test_auc, loss_sum


def train(
        train_file="local_train_splitByUser",
        test_file="local_test_splitByUser",
        uid_voc="uid_voc.pkl",
        mid_voc="mid_voc.pkl",
        cat_voc="cat_voc.pkl",
        batch_size=512,
        maxlen=100,
        test_iter=1000,
        save_iter=1000,
        model_type='Wide',
        seed=2,
):
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)

    train_data = DataIterator(train_file, uid_voc, mid_voc,
                              cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
    test_data = DataIterator(test_file, uid_voc, mid_voc,
                             cat_voc, batch_size, maxlen)
    n_uid, n_mid, n_cat = train_data.get_n()
    # bce_loss = nn.BCELoss()
    bce_loss = nn.CrossEntropyLoss()
    if model_type == 'Wide':

        cat_features = {'uid': n_uid, 'mid': n_mid, 'cat': n_cat}
        embed_dims = [256, 64, 2]
        encode_dim = 32
        model = WideAndDeep(cat_features, embed_dims, encode_dim, 5, 6).cuda()
        optim = torch.optim.Adam(model.parameters(), lr=0.001)

    iter = 0
    for itr in range(5):
        loss_sum = 0.0
        for src, tgt in train_data:
            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
                src, tgt, maxlen, return_neg=True)
            mids = torch.tensor(mids).cuda()
            uids = torch.tensor(uids).cuda()
            cats = torch.tensor(cats).cuda()
            mid_his = torch.tensor(mid_his).cuda()
            cat_his = torch.tensor(cat_his).cuda()
            target = torch.tensor(target).cuda()
            features = {'uid': uids, 'mid': mids, 'cat': cats,
                        'cat_his': cat_his, 'mid_his': mid_his}
            output = model(features)
            loss = bce_loss(output, target.float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss
            iter += 1
            if iter % 1000 == 0:
                print(f'epoch: {itr}, iter: {iter}, total_loss: {loss_sum / iter}')
            if iter % test_iter == 0:
                test_auc, test_loss = eval(test_data, model, best_model_path)
                print(f'test_auc: {test_auc}, test_loss: {test_loss}')
            if iter % save_iter == 0:
                torch.save(model.state_dict(), model_path)
        if itr == 3:
            for params in optim.param_groups:
                params['lr'] = 0.0005


if __name__ == '__main__':
    SEED = 2022
    torch.manual_seed(SEED)  # cpu
    torch.cuda.manual_seed(SEED)  # gpu
    numpy.random.seed(SEED)
    random.seed(SEED)
    train(seed=SEED)
