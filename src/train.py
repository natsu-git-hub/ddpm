import torch
import einops
import os
import glob

import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from utils import loss_fn, get_beta_schedule, get_alpha_schedule, get_alpha_bar_schedule, sample, save_samples, save_loss_plot
from logger import get_logger
from dataset import CustomDataset

logger = get_logger(__name__)

MAX_EPOCHS = 60
SAVE_INTERVAL = 5
PRINT_INTERVAL = 10
INFERENCES_INTERVAL = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TIME_STEPS = 1000
NUM_OF_SAMPLES = 16

def load_latest_checkpoint(model, optimizer, device):
    paths = glob.glob("models/model_*.pth")
    if not paths:
        return 0, []
    path = max(paths, key=lambda p: int(p.split("_")[-1].split(".")[0]))
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    logger.info(f"Resumed from {path}")
    return checkpoint["epoch"] + 1, checkpoint["history"]

def train(train_dataloader, device):
    
    model = UNet()
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    betas = get_beta_schedule(TIME_STEPS, device)
    alphas = get_alpha_schedule(betas, device)
    alpha_bars = get_alpha_bar_schedule(alphas, device)

    start_epoch, history = load_latest_checkpoint(model, optimizer, device)
    losses = []
    
    for epoch in range(start_epoch, MAX_EPOCHS):
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
            history.append(loss.item())
            
        save_loss_plot(history, "loss.png")

        if epoch % PRINT_INTERVAL == 0:
            logger.info(f"Epoch {epoch}, Loss: {sum(losses) / len(losses)}")
            losses = []

        if epoch % SAVE_INTERVAL == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
            }
            torch.save(checkpoint, f"models/model_{epoch}.pth")

        if epoch % INFERENCES_INTERVAL == 0:
            model.eval()
            samples = sample(model, device, TIME_STEPS, NUM_OF_SAMPLES, betas, alphas, alpha_bars)
            logger.info(f"Generated {NUM_OF_SAMPLES} samples")
            # save samples
            if not os.path.exists("samples"):
                os.makedirs("samples")
            save_samples(samples, "samples", epoch)
            model.train()

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=None)
    train_dataset = CustomDataset(train_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    train(train_dataloader, device)
    