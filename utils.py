# -*- coding: utf-8 -*- # 
# @Time    : 2021-05-02 19:24
# @Email   : zhilishen@smail.nju.edu.cn
# @Author  : Zhili Shen
# @File    : utils.py
# @Notice  :
import os
import re
import time
import torch
import collections
import numpy as np
import pandas as pd
import torchtext.vocab as Vocab
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from config import get_argument

opt = get_argument().parse_args()
trainDataRoot = opt.train_data_root
testDataRoot = opt.test_data_root
cacheDir = opt.vocab_dir
vocab = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenized(text):
    return [tok.lower() for tok in text.split(' ')]


def get_tokenized_product_fit(data):
    return [tokenized(review) for review in data]


def get_vocab_product_fit(data):
    tokenize_data = get_tokenized_product_fit(data)
    counter = collections.Counter([tk for st in tokenize_data for tk in st])
    return Vocab.Vocab(counter, min_freq=opt.min_frequency)


def pre_process_product_fit(words):
    max_l = opt.max_length_sentence

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenize_words = tokenized(words)
    features = pad([vocab.stoi[word] for word in tokenize_words])
    return features


def bust_size2vec(bust_size):
    alpha_bust_sizes = ['aa', 'a', 'b', 'c', 'd', 'd+', 'dd', 'ddd/e', 'f', 'g', 'h', 'i', 'j']
    numeric_bust_sizes = [num for num in range(len(alpha_bust_sizes))]
    bust_size2num_dict = {alpha_bust_size: numeric_bust_size for alpha_bust_size, numeric_bust_size in
                          zip(alpha_bust_sizes, numeric_bust_sizes)}
    bust_size_a = int(bust_size[:2])
    bust_size_b = bust_size2num_dict[bust_size[2:]]
    return bust_size_a, bust_size_b


def height2num(height):
    height_pattern = re.compile(r"(\d)' (\d{1,2})")
    height_results = height_pattern.search(height)
    feet_num = height_results.group(1)
    inch_num = height_results.group(2)
    metres = int(feet_num) * 0.3048 + int(inch_num) * 0.0254
    return metres


def weight2num(weight):
    weight_pattern = re.compile(r"(\d{1,4})")
    weight_results = weight_pattern.search(weight)
    weight_num = weight_results.group(1)
    metres = int(weight_num)
    return metres


def transform(data):
    data[['bust size a', 'bust size b']] = data['bust size'].apply(bust_size2vec).apply(pd.Series)
    data['height'] = data['height'].apply(height2num)
    data['weight'] = data['weight'].apply(weight2num)
    data['review'] = data['review_summary'] + data['review_text']
    data['review'] = data['review'].apply(pre_process_product_fit)
    data.drop(['bust size', 'review_date', 'review_summary', 'review_text'], axis=1, inplace=True)


def get_vocab(data):
    global vocab
    vocab = get_vocab_product_fit(data['review'])


def get_attr_num(data):
    attr_cols = data.select_dtypes(include=['int64'])
    attr_num_dict = {}
    for col in attr_cols:
        attr_num_dict[col] = len(data[col].unique())
    return attr_num_dict


def load_pretrained_embedding(words, pretrained_vocab):
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 0
    if oov_count > 0:
        print("Oops! there are {:d} oov words".format(oov_count))
    return embed


def train(train_iter, valid_iter, net, loss, optimizer, num_epochs):
    net = net.to(device)
    print("training on", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for x, y in train_iter:
            for k, v in x.items():
                if torch.is_tensor(v):
                    x[k] = v.to(device)
            y = y.to(device)
            y_hat, pred_probs = net(x)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        valid_acc, f1_score, auc = evaluate_accuracy(valid_iter, net)
        torch.save(net.state_dict(),
                   os.path.join(opt.saved_models_dir,
                                '{:s}-f1_score{:.3f}-epoch{:d}.pt'.format(opt.TextCNNorBiRNN, f1_score, epoch+1)))
        print('epoch: {:d}, loss: {:.5f}, train accuracy: {:.3f}, valid accuracy: {:.3f}, valid f1 score: {:.3f}, '
              'test auc: {:.3f}, time: {:.1f} '
              'sec.'.format(
                epoch + 1,
                train_l_sum / batch_count,
                train_acc_sum / n, valid_acc, f1_score, auc,
                time.time() - start))


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    preds = []
    target = []
    pred_probs = []
    with torch.no_grad():
        for x, y in data_iter:
            for k, v in x.items():
                if torch.is_tensor(v):
                    x[k] = v.to(device)
            net.eval()
            y = y.to(device)
            y_hat, pred_prob = net(x)
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
            preds.append(y_hat.argmax(dim=1).cpu().numpy())
            pred_probs.append(pred_prob.cpu().data.numpy())
            target.append(y.cpu().numpy())
            net.train()
    pred_tracker = np.stack(preds[:-1]).reshape(-1)
    target_tracker = np.stack(target[:-1]).reshape(-1)
    pred_probs_tracker = np.stack(pred_probs[:-1], axis=0).reshape(-1, 3)
    f1_score = metrics.f1_score(target_tracker, pred_tracker, average="macro")
    auc = metrics.roc_auc_score(target_tracker, pred_probs_tracker, average="macro", multi_class="ovr")
    return acc_sum / n, f1_score, auc


def predict(test_iter, net, num2fit):
    preds = []
    with torch.no_grad():
        for x in test_iter:
            for k, v in x.items():
                if torch.is_tensor(v):
                    x[k] = v.to(device)
            net.eval()
            y_hat, pred_prob = net(x)
            preds.append(y_hat.argmax(dim=1).cpu().numpy())
            net.train()
    preds = np.stack(preds[:]).reshape(-1)
    preds = [num2fit[i] for i in preds]
    preds = [line + "\n" for line in preds[:-1]] + [preds[-1]]
    with open(opt.output_data_root, "w") as f:
        f.writelines(preds)


def process_data(train_data_root, test_data_root):
    train_data = pd.read_csv(train_data_root)
    test_data = pd.read_csv(test_data_root)

    numeric_col = train_data.select_dtypes(include=['float64', 'int64'])
    numeric_col_with_null = [col_name for col_name in numeric_col if train_data[col_name].isnull().any()]
    string_col = train_data.select_dtypes(include=['object'])
    string_col_with_null = [col_name for col_name in string_col if train_data[col_name].isnull().any()]
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    frequent_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for col in numeric_col_with_null:
        median_imputer.fit(np.array(train_data[col]).reshape(-1, 1))
        train_data[col] = median_imputer.transform(np.array(train_data[col]).reshape(-1, 1)).squeeze()
        test_data[col] = median_imputer.transform(np.array(test_data[col]).reshape(-1, 1)).squeeze()
    for col in string_col_with_null:
        frequent_imputer.fit(np.array(train_data[col]).reshape(-1, 1))
        train_data[col] = frequent_imputer.transform(np.array(train_data[col]).reshape(-1, 1)).squeeze()
        test_data[col] = frequent_imputer.transform(np.array(test_data[col]).reshape(-1, 1)).squeeze()

    train_test_data = pd.concat([train_data, test_data])
    train_test_data['review'] = train_test_data['review_summary'] + train_test_data['review_text']
    get_vocab(train_test_data)

    transform(train_data)
    transform(test_data)

    numeric_col = train_data.select_dtypes(include=['float64', 'int64'])
    numeric_col_names = [col_name for col_name in numeric_col if col_name != 'item_id' and col_name != 'user_id']
    string_col = train_data.select_dtypes(include=['object'])
    string_col_names = [col_name for col_name in string_col if col_name != 'review' and col_name != 'fit']
    string_col_names.extend(['user_id', 'item_id'])
    train_test_data = pd.concat([train_data, test_data])
    scaler = StandardScaler()
    scaler.fit(train_test_data.loc[:, numeric_col_names])
    train_data_numeric = pd.DataFrame(scaler.transform(train_data.loc[:, numeric_col_names]),
                                      columns=numeric_col_names)
    test_data_numeric = pd.DataFrame(scaler.transform(test_data.loc[:, numeric_col_names]),
                                     columns=numeric_col_names)

    ordinal_enc = OrdinalEncoder()
    ordinal_enc.fit(train_test_data.loc[:, string_col_names])
    train_data_categorical = pd.DataFrame(np.array(ordinal_enc.transform(train_data.loc[:, string_col_names]),
                                                   dtype=np.int64), columns=string_col_names)
    test_data_categorical = pd.DataFrame(np.array(ordinal_enc.transform(test_data.loc[:, string_col_names]),
                                                  dtype=np.int64), columns=string_col_names)

    ordinal_enc.fit(np.array(train_data['fit']).reshape(-1, 1))
    train_data['fit'] = np.array(ordinal_enc.transform(np.array(train_data['fit']).reshape(-1, 1)).squeeze(),
                                 dtype=np.int64)
    num2fit = {num: fit for num, fit in
               zip([0, 1, 2], ordinal_enc.inverse_transform([[0], [1], [2]]).squeeze())}

    processed_train_data = pd.concat(
        [train_data_numeric, train_data_categorical, train_data[['review', 'fit']]], axis=1)
    processed_test_data = pd.concat(
        [test_data_numeric, test_data_categorical, test_data[['review']]], axis=1)
    global vocab
    return processed_train_data, processed_test_data, num2fit, vocab


if __name__ == '__main__':
    pass
