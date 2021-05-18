# -*- coding: utf-8 -*- # 
# @Time    : 2021-05-03 23:37
# @Email   : zhilishen@smail.nju.edu.cn
# @Author  : Zhili Shen
# @File    : model.py
# @Notice  :
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_argument
opt = get_argument().parse_args()


class SiZeLet(nn.Module):
    def __init__(self, vocab):
        super(SiZeLet, self).__init__()
        self.vocab = vocab

        self.body_type_embedding = nn.Embedding(
            num_embeddings=opt.body_type,
            embedding_dim=opt.user_embedding_dim,
            max_norm=opt.max_norm
        )

        self.category_embedding = nn.Embedding(
            num_embeddings=opt.category,
            embedding_dim=opt.item_embedding_dim,
            max_norm=opt.max_norm
        )

        self.rented_for_embedding = nn.Embedding(
            num_embeddings=opt.rented_for,
            embedding_dim=opt.item_embedding_dim,
            max_norm=opt.max_norm
        )

        self.item_id_embedding = nn.Embedding(
            num_embeddings=opt.item_id,
            embedding_dim=opt.item_embedding_dim,
            max_norm=opt.max_norm
        )

        user_feature = 1 * opt.user_embedding_dim + opt.user_numeric_num
        self.user_transform_way = [user_feature, 64, 32]
        self.user_transform_blocks = []
        for i in range(1, len(self.user_transform_way)):
            self.user_transform_blocks.append(
                ResBlock(
                    self.user_transform_way[i - 1],
                    self.user_transform_way[i],
                    opt.activation,
                )
            )
            self.user_transform_blocks.append(nn.Dropout(p=opt.dropout))
        self.user_transform_blocks = nn.Sequential(*self.user_transform_blocks)

        item_feature = 3 * opt.item_embedding_dim + opt.item_numeric_num
        self.item_transform_way = [item_feature, 64, 32]
        self.item_transform_blocks = []
        for i in range(1, len(self.item_transform_way)):
            self.item_transform_blocks.append(
                ResBlock(
                    self.item_transform_way[i - 1],
                    self.item_transform_way[i],
                    opt.activation,
                )
            )
            self.item_transform_blocks.append(nn.Dropout(p=opt.dropout))
        self.item_transform_blocks = nn.Sequential(*self.item_transform_blocks)

        if opt.TextCNNorBiRNN == 'TextCNN':
            self.review_transform_blocks = TextCNN(self.vocab, opt.review_embedding_dim,
                                                   opt.kernel_sizes, opt.num_channels)
        else:
            self.review_transform_blocks = BiRNN(self.vocab, opt.review_embedding_dim, opt.num_hidden,
                                                 opt.num_layers)

        if opt.user_item_embedding_flag:
            combined_layer_input_size = 32 + 32 + 64
            self.combined_transform_way = [combined_layer_input_size, 64, 32]
        else:
            combined_layer_input_size = 64
            self.combined_transform_way = [combined_layer_input_size, 32, 16]
        self.combined_blocks = []
        for i in range(1, len(self.combined_transform_way)):
            self.combined_blocks.append(
                ResBlock(
                    self.combined_transform_way[i - 1],
                    self.combined_transform_way[i],
                    opt.activation,
                )
            )
            self.combined_blocks.append(nn.Dropout(p=opt.dropout))
        self.combined_blocks = nn.Sequential(*self.combined_blocks)

        self.hidden2output = nn.Linear(self.combined_transform_way[-1], opt.num_targets)

    def forward(self, batch_input):

        # User Transform Way
        body_type_emb = self.body_type_embedding(batch_input["body type"])

        user_representation = torch.cat(
            [body_type_emb, batch_input["user_numeric"]], dim=-1
        )
        user_representation = self.user_transform_blocks(user_representation)

        # Item Transform Way
        rented_for_emb = self.rented_for_embedding(batch_input["rented for"])
        item_id_emb = self.item_id_embedding(batch_input["item_id"])
        category_emb = self.category_embedding(batch_input["category"])
        item_representation = torch.cat(
            [category_emb, rented_for_emb, item_id_emb, batch_input["item_numeric"]], dim=-1
        )
        item_representation = self.item_transform_blocks(item_representation)

        # Review Transform Way
        review_representation = self.review_transform_blocks(batch_input["review"])

        # Combine the Transform Ways
        if opt.user_item_embedding_flag:
            combined_representation = torch.cat(
                [user_representation, item_representation, review_representation], dim=-1
            )
        else:
            combined_representation = review_representation
        combined_representation = self.combined_blocks(combined_representation)

        # Output layer of logit and pred_prob
        logit = self.hidden2output(combined_representation)
        pred_prob = F.softmax(logit, dim=-1)

        return logit, pred_prob


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(ResBlock, self).__init__()
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh

        self.inp_transform = nn.Linear(input_dim, output_dim)
        self.out_transform = nn.Linear(output_dim, output_dim)
        self.inp_projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y = self.activation(self.inp_transform(x))
        z = self.activation(self.out_transform(y) + self.inp_projection(x))
        return z


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.decoder = nn.Linear(sum(num_channels), 64)
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=2 * embed_size,
                                        out_channels=c,
                                        kernel_size=k))

    def forward(self, inputs):
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hidden, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hidden,
                               num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hidden, 64)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[1]), -1)
        outputs = self.decoder(encoding)
        return outputs
