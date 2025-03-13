import json
import os
from torch.utils.data import DataLoader
import numpy as np

# Todo: not sure if needs data aug
class SiamDataset:
    def __init__(self, pairs_save_path):
        # 1. 读取 JSON 配置文件，获取 pairs_save_path
        self.pairs_save_path = pairs_save_path

        self.data_list = []

        # 2. 遍历 pairs_save_path 下的所有子文件夹（pairs_save_key_path）
        for key in os.listdir(self.pairs_save_path):
            subfolder = os.path.join(self.pairs_save_path, key)
            for file in os.listdir(subfolder):
                npy_path = os.path.join(subfolder, file)
                data = np.load(npy_path)
                for sample in data:
                    self.data_list.append(sample)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 5. 从数据列表中取出一个样本
        sample = self.data_list[idx]
        x1 = sample[0]
        x2 = sample[1]
        return x1, x2


# Demo
if __name__ == '__main__':
    dataset = SiamDataset("pre_train/pre_train_setting.json")
    print("数据集大小:", len(dataset))
    x1, x2 = dataset[0]
    print("x1 shape:", x1.shape)
    print("x2 shape:", x2.shape)