import torch
from model import SimSiam
import os
from dataset import SiamDataset
from torch.utils.data import DataLoader
import losses
import numpy as np
from tqdm import tqdm
import json

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Setting
with open("../setting.json", "r") as f:
    setting = json.load(f)
batch_size = setting["batch_size"]
backbone = setting["backbone"]
h5_data_path = setting["h5_data_path"]
lr = setting["lr"]
epochs = setting["epochs"]


model = SimSiam(encoder = backbone,projector=True)
model = model.to(device)

h5_file = os.path.join(h5_data_path, 'datasets.h5')
train_dataset = SiamDataset(h5_file)
print("Total numbers of training pairs:", len(train_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

criterion = losses.NegativeCosineSimLoss().to(device)

param_groups = [
    {'params': list(set(model.parameters()))}
]
opt = torch.optim.Adam(param_groups, lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(opt)

losses_list = []

for epoch in range(0, epochs):
    model.train()
    losses_per_epoch = []

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
        pbar = tqdm(enumerate(dataloader_val))
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
        torch.save({'model_state_dict':model.state_dict()}, '{}/{}_{}_{}_{}_{}.pth'.format(LOG_DIR, args.baseline, args.model, args.sz_embedding, args.loss,epoch))
