import torch
import torch.nn as nn
import resnet


class DirectFwd(nn.Module):
    def __init__(self):
        super(DirectFwd, self).__init__()

    def forward(self, x):
        return x


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, dim=256, pred_dim=64, projector=True, predictor=True, single_source_mode=False,
                 encoder='ResNet18'):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = getattr(resnet, encoder, None)(num_classes=dim)
        # print(f"encoder is {self.encoder}")
        self.single_source_mode = single_source_mode

        if projector:
            # build a 3-layer projector
            prev_dim = self.encoder.fc.weight.shape[1]

            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),  # second layer
                                            self.encoder.fc,
                                            nn.BatchNorm1d(dim, affine=False))  # output layer
            self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        if predictor:
            # build a 2-layer predictor

            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),  # hidden layer
                                           nn.Linear(pred_dim, dim))  # output layer
        else:
            self.predictor = DirectFwd()

    def forward(self, x1, x2):
        """
        Input:
            PPG_G: first views of images
            PPG_B: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        if not self.single_source_mode:
            # compute features for one view
            z1, features1 = self.encoder(x1)  # NxC
            z2, features2 = self.encoder(x2)  # NxC

            p1 = self.predictor(z1)  # NxC
            p2 = self.predictor(z2)  # NxC

            return p1, p2, z1.detach(), z2.detach()

# 测试代码
if __name__ == "__main__":
    batch_size = 8
    seq_length = 1200  # 设定一个固定的序列长度

    model = SimSiam()

    # 生成两个不同的数据视图（模拟数据增强）
    x1 = torch.randn(batch_size, seq_length).unsqueeze(dim=1)
    x2 = torch.randn(batch_size, seq_length).unsqueeze(dim=1)

    p1, p2, z1, z2 = model(x1, x2)

    print("p1 shape:", p1.shape)
    print("p2 shape:", p2.shape)
    print("z1 shape:", z1.shape)
    print("z2 shape:", z2.shape)
