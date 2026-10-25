import torch

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