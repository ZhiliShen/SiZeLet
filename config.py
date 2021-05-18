# -*- coding: utf-8 -*- # 
# @Time    : 2021-05-10 11:40
# @Email   : zhilishen@smail.nju.edu.cn
# @Author  : Zhili Shen
# @File    : config.py
# @Notice  :

import argparse


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument('--train_data_root', type=str, default='./product_fit/train.txt')
    parser.add_argument('--test_data_root', type=str, default='./product_fit/test.txt')
    parser.add_argument('--output_data_root', type=str, default='./output/output.txt')
    parser.add_argument('--vocab_dir', type=str, default='./cache')
    parser.add_argument('--saved_models_dir', type=str, default='./saved_models')

    # Model parameters
    parser.add_argument('--body_type', type=int, default=7)
    parser.add_argument('--category', type=int, default=68)
    parser.add_argument('--rented_for', type=int, default=9)
    parser.add_argument('--user_id', type=int, default=105571)
    parser.add_argument('--item_id', type=int, default=5850)
    parser.add_argument('--user_embedding_dim', type=int, default=10)
    parser.add_argument('--item_embedding_dim', type=int, default=10)
    parser.add_argument('--review_embedding_dim', type=int, default=100)
    parser.add_argument('--kernel_sizes', type=list, default=[3, 4, 5])
    parser.add_argument('--num_channels', type=list, default=[100, 100, 100])
    parser.add_argument('--num_hidden', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--user_numeric_num', type=int, default=5)
    parser.add_argument('--item_numeric_num', type=int, default=2)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_targets', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--vocab_name', type=str, default='twitter.27B')
    parser.add_argument('--vocab_embedding_dim', type=int, default=100)
    parser.add_argument('--train_test_proportion', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--user_item_embedding_flag', type=bool, default=True)
    parser.add_argument('--TextCNNorBiRNN', type=str, default='TextCNN')
    parser.add_argument('--max_length_sentence', type=int, default=300)
    parser.add_argument('--min_frequency', type=int, default=400)
    parser.add_argument('--use_pretrained_model', type=bool, default=False)
    return parser
