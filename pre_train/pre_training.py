import torch
from model import SimSiam
import os
from dataset import SiamDataset
from torch.utils.data import random_split, DataLoader
import losses
import numpy as np
from tqdm import tqdm
import json

device = 'cuda'

# Setting
with open("pre_train/pre_train_setting.json", "r") as f:
    setting = json.load(f)
batch_size = setting["batch_size"]
backbone = setting["backbone"]
pair_data_path = setting["pairs_save_path"]
lr = setting["lr"]
epochs = setting["epochs"]
ratio_train_val = setting["ratio_train_val"]
model_save_path = setting["model_save_path"]

model = SimSiam(encoder = backbone,projector=True)
model = model.to(device)

dataset = SiamDataset(pair_data_path)

train_size = int(ratio_train_val * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("Total numbers of training pairs:", len(train_dataset))
print("Total numbers of validation pairs:", len(val_dataset))
print("Using the model of:", backbone)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

criterion = losses.NegativeCosineSimLoss().to(device)

param_groups = [
    {'params': list(set(model.parameters()))}
]

opt = torch.optim.Adam(param_groups, lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

losses_list = []

for epoch in range(0, epochs):
    model.train()
    losses_per_epoch = []

    pbar = tqdm(enumerate(train_dataloader))

    for batch_idx, (x1, x2) in tqdm(enumerate(train_dataloader)):
        x1, x2 = x1.to("cuda", dtype=torch.float32), x2.to("cuda", dtype=torch.float32)
        p1, p2, z1, z2 = model(x1.unsqueeze(dim=1), x2.unsqueeze(dim=1))
        loss = criterion(p1, p2, z1, z2)

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.cpu().data.numpy())
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(train_dataloader),
                100. * batch_idx / len(train_dataloader),
                loss.item()))

    losses_list.append(np.mean(losses_per_epoch))
    print("losses_list: ", losses_list)
    # wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()

    # Validation
    model_is_training = model.training
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        losses_val = []
        pbar = tqdm(enumerate(val_dataloader))
        for batch_idx, (x1, x2) in pbar:
            x1, x2 = x1.to(device, dtype=torch.float32), x2.to(device, dtype=torch.float32)
            p1, p2, z1, z2 = model(x1.unsqueeze(dim=1), x2.unsqueeze(dim=1))
            loss = criterion(p1, p2, z1, z2)
            losses_val.append(loss.cpu().data.numpy())
        print(f"Validation loss {np.mean(losses_val)}")

    model.train()
    model.train(model_is_training)

    if losses_list[-1] == min(losses_list):
        print("Model is going to save")
        print(f"last loss: {losses_list[-1]} | min loss: {min(losses_list)}")
        # if not os.path.exists('{}'.format(LOG_DIR)):
        #     os.makedirs('{}'.format(LOG_DIR))
        torch.save({'model_state_dict':model.state_dict()}, '{}/{}_.pth'.format(model_save_path, backbone))
