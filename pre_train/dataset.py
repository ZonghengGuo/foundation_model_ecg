import matplotlib.pyplot as plt
import h5py
import os
from torch.utils.data import DataLoader

# Todo: not sure if needs data aug
class SiamDataset:
    def __init__(self, h5_file):
        # open H5 database, get data pairs and quality difference
        self.f = h5py.File(h5_file, 'r')
        self.seg1 = self.f['seg1']
        self.seg2 = self.f['seg2']
        self.diff = self.f['diff']

        # load order index
        self.sorted_indices = self.f['sorted_indices'][:]

    def __len__(self):
        return len(self.sorted_indices)

    def __getitem__(self, idx):
        # Accroading to order index to get pairs.
        sorted_idx = self.sorted_indices[idx]
        sample1 = self.seg1[sorted_idx]
        sample2 = self.seg2[sorted_idx]
        label = self.diff[sorted_idx]

        # In pretraing stage, label is non-used, only get sample1 and sample2
        return (sample1, sample2), label

    def close(self):
        self.f.close()


# Demo
if __name__ == '__main__':
    save_path = "saved_data/pair_segments"
    h5_file = os.path.join(save_path, 'datasets.h5')
    batch_size = 32

    dataset = SiamDataset(h5_file)
    print("Total numbers of pairs:", len(dataset))

    # for i in range(len(dataset)):
    #     (s1, s2), diff_value = dataset[i]
    #     print(f"Sample {i}: diff = {diff_value}, seg1 shape = {s1.shape}, seg2 shape = {s2.shape}")

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch_idx, ((s1, s2), diff) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}: diff shape = {diff.shape}, seg1 shape = {s1.shape}, seg2 shape = {s2.shape}")

    dataset.close()