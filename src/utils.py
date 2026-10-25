import torch
from torchvision.utils import save_image
from matplotlib.figure import Figure

from logger import get_logger

logger = get_logger(__name__)

# loss: ||ε - εθ(√α¯tx0 + √1 − α¯tε, t)||^2
def loss_fn(pred_noise, target_noise):
    loss = torch.mean((target_noise - pred_noise) ** 2)
    return loss

def get_beta_schedule(time_steps, device):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, time_steps).to(device)

def get_alpha_schedule(beta_schedule, device):
    return (1 - beta_schedule).to(device)

def get_alpha_bar_schedule(alpha_schedule, device):
    return torch.cumprod(alpha_schedule, dim=0).to(device)

def save_samples(samples, folder, epoch):
    # map values from [-1, 1] to [0, 1]
    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, f"{folder}/{epoch}.png", nrow=4)

def save_loss_plot(losses, path):
    fig = Figure()
    ax = fig.subplots()
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    fig.savefig(path)

@torch.no_grad()
def sample(model, device, time_steps, samples, betas, alphas, alpha_bars):
    # sample from noise
    xT = torch.randn(samples, 1, 28, 28).to(device)
    
    for t in reversed(range(time_steps)):
        # z ∼ N (0, I) if t > 0, else z = 0
        z = torch.randn_like(xT) if t > 0 else torch.zeros_like(xT)
        
        t_tensor = torch.full((samples,), t, device=device, dtype=torch.long)
        e_theta = model(xT, t_tensor)

        # xt−1 = 1/√αt (xt − (1−αt)/√(1−α¯t) εθ(xt, t)) + σt z
        xT = (1 / torch.sqrt(alphas[t])) * (xT - (1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t]) * e_theta) + torch.sqrt(betas[t]) * z

    return xT

if __name__ == "__main__":
    beta_schedule = get_beta_schedule(1000, "cpu")
    alpha_schedule = get_alpha_schedule(beta_schedule)
    alpha_bar_schedule = get_alpha_bar_schedule(alpha_schedule)
    logger.info(f"Beta schedule start 10: {beta_schedule[:10]}")
    logger.info(f"Alpha schedule start 10: {alpha_schedule[:10]}")
    logger.info(f"Alpha bar schedule start 10: {alpha_bar_schedule[:10]}")
    logger.info(f"Beta schedule end 10: {beta_schedule[-10:]}")
    logger.info(f"Alpha schedule end 10: {alpha_schedule[-10:]}")
    logger.info(f"Alpha bar schedule end 10: {alpha_bar_schedule[-10:]}")