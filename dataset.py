# -*- coding: utf-8 -*- # 
# @Time    : 2021-05-04 18:47
# @Email   : zhilishen@smail.nju.edu.cn
# @Author  : Zhili Shen
# @File    : dataset.py
# @Notice  :
import numpy as np
from torch.utils.data import Dataset


class ProductFitDataset(Dataset):
    def __init__(self, data, train=True):
        self.data = data
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        features = {
            "body type": np.asarray(self.data["body type"][item], dtype=np.int64),
            "user_numeric": np.asarray(
                [self.data[feature][item] for feature in
                 ["age", "height", "weight", "bust size a", "bust size b"]], dtype=np.float32),
            "category": np.asarray(self.data["category"][item], dtype=np.int64),
            "rented for": np.asarray(self.data["rented for"][item], dtype=np.int64),
            "item_id": np.asarray(self.data["item_id"][item], dtype=np.int64),
            "item_numeric": np.asarray(
                [self.data[feature][item] for feature in
                 ["rating", "size"]], dtype=np.float32),
            "review": np.asarray(self.data["review"][item], dtype=np.int64)
        }
        if self.train:
            label = np.asarray(self.data["fit"][item], dtype=np.int64)
            return features, label
        else:
            return features


if __name__ == "__main__":
    pass
