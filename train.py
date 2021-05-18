# -*- coding: utf-8 -*- # 
# @Time    : 2021-05-04 12:38
# @Email   : zhilishen@smail.nju.edu.cn
# @Author  : Zhili Shen
# @File    : train.py
# @Notice  :
import os
import torch
import numpy as np
import torch.nn as nn
import torchtext.vocab as Vocab
from torch.utils.data import DataLoader, random_split
from model import SiZeLet
from dataset import ProductFitDataset
from utils import device, train, predict, process_data, load_pretrained_embedding

from config import get_argument

opt = get_argument().parse_args()
trainDataRoot = opt.train_data_root
testDataRoot = opt.test_data_root

if __name__ == '__main__':
    glove_vocab = Vocab.GloVe(name=opt.vocab_name, dim=opt.vocab_embedding_dim, cache=opt.vocab_dir)
    train_data, test_data, num2fit, vocab = process_data(trainDataRoot, testDataRoot)
    product_fit_labeled_data = ProductFitDataset(train_data)
    product_fit_unlabeled_data = ProductFitDataset(test_data, train=False)
    labeled_dataset_size = len(product_fit_labeled_data)
    train_size = int(np.floor(opt.train_test_proportion * labeled_dataset_size))
    valid_size = labeled_dataset_size - train_size
    train_dataset, valid_dataset = random_split(product_fit_labeled_data, [train_size, valid_size])

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True
    )
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False
    )
    test_data_loader = DataLoader(
        dataset=product_fit_unlabeled_data,
        batch_size=1,
        shuffle=False,
    )

    net = SiZeLet(vocab)

    if opt.use_pretrained_model:
        saved_models = os.listdir(opt.saved_models_dir)
        if len(saved_models) == 0:
            raise Exception("Oops, the saved models dir is empty, please train the model from scratch.")
        else:
            net = net.to(device)
            net.load_state_dict(torch.load(os.path.join(opt.saved_models_dir, saved_models[-1])))
            print("We have trained the model: {:s}!".format(saved_models[-1]))
    else:
        net.review_transform_blocks.embedding.weight.data.copy_(
            load_pretrained_embedding(vocab.itos, glove_vocab)
        )
        if opt.TextCNNorBiRNN == 'TextCNN':
            net.review_transform_blocks.constant_embedding.weight.data.copy_(
                load_pretrained_embedding(vocab.itos, glove_vocab)
            )
            net.review_transform_blocks.constant_embedding.weight.requires_grad = False
        else:
            net.review_transform_blocks.embedding.weight.requires_grad = False

        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr)
        loss = nn.CrossEntropyLoss()

        train(train_data_loader, valid_data_loader, net, loss, optimizer, opt.num_epochs)
        print("We have trained the model!")

    predict(test_data_loader, net, num2fit)
    print("We have written outputs to file!")
