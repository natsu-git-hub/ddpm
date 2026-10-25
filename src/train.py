import torch
import einops
import os

import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from utils import loss_fn, get_beta_schedule, get_alpha_schedule, get_alpha_bar_schedule
from logger import get_logger
from dataset import CustomDataset

logger = get_logger(__name__)

MAX_EPOCHS = 100000
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
INFERENCES_INTERVAL = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TIME_STEPS = 1000

def train(train_dataloader, device):
    
    model = UNet()
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    betas = get_beta_schedule(TIME_STEPS, device)
    alphas = get_alpha_schedule(betas, device)
    alpha_bars = get_alpha_bar_schedule(alphas, device)

    losses = []
    
    for epoch in range(MAX_EPOCHS):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}: "):
            # x0 ~ q(x0)
            x0 = batch[0]
            x0 = x0.to(device)

            # t ∼ Uniform({1, . . . , T})
            t = torch.randint(0, TIME_STEPS, (x0.shape[0],))
            t = t.to(device)

            # sample noise ε ∼ N (0, I)
            noise = torch.randn_like(x0)    
            
            # rearrange alpha_bars to match the batch size
            alpha_bar_t = einops.rearrange(alpha_bars[t], 'b -> b 1 1 1')
            
            inp = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

            e = model(inp, t)
            
            # loss: ||ε - εθ(√α¯tx0 + √1 − α¯tε, t)||^2
            
            loss = loss_fn(e, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        if epoch % PRINT_INTERVAL == 0:
            logger.info(f"Epoch {epoch}, Loss: {sum(losses) / len(losses)}")
            losses = []

        if epoch % SAVE_INTERVAL == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), f"models/model_{epoch}.pth")

        if epoch % INFERENCES_INTERVAL == 0:
            # TODO: Generate images
            pass

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_data = datasets.MNIST(root='../data', train=True, download=True, transform=None)
    train_dataset = CustomDataset(train_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    train(train_dataloader, device)
    