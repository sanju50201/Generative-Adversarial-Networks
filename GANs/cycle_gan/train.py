import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, seed_everything
from dataset import FinancialDataset
from discriminator import TabularDiscriminator
from generator import TabularGenerator


def train_fn(disc, gen, loader, opt_disc, opt_gen, criterion, z_dim, device):
    loop = tqdm(loader, leave=True)

    for batch_idx, real_samples in enumerate(loop):
        real_samples = real_samples.to(device)
        batch_size = real_samples.shape[0]

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake_samples = gen(noise)
        disc_real = disc(real_samples).view(-1)
        disc_fake = disc(fake_samples).view(-1)
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (disc_loss_real + disc_loss_fake) / 2

        opt_disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        output = disc(fake_samples).view(-1)
        gen_loss = criterion(output, torch.ones_like(output))

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        loop.set_postfix(disc_loss=disc_loss.item(), gen_loss=gen_loss.item())


def main():
    seed_everything()
    device = torch.device(config.DEVICE)
    z_dim = config.NOISE_DIM
    lr = config.LEARNING_RATE

    disc = TabularDiscriminator(config.NUM_FEATURES).to(device)
    gen = TabularGenerator(z_dim, config.NUM_FEATURES).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, lr)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, lr)

    dataset = FinancialDataset("../fraud.csv")
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                        shuffle=True, num_workers=2)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, loader, opt_disc,
                 opt_gen, criterion, z_dim, device)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
