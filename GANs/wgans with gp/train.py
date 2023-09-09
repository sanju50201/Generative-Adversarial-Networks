import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Generator, Discriminator
from data import fraud_data
from utils import gradient_penalty, initialize_weights
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-5
batch_size = 64
num_epochs = 50
# channels_img = 1
z_dim = 100
features_critic = 16
features_gen = 16
critic_iterations = 5
lambda_gp = 10


fraud_tensors = torch.tensor(fraud_data.values).float().to(device)
fraud_df = TensorDataset(fraud_tensors)
loader = DataLoader(fraud_df, batch_size=batch_size, shuffle=True)

input_dim = fraud_tensors.shape[1]

generator = Generator(z_dim, input_dim, features_gen).to(device)
discriminator = Discriminator(input_dim, features_critic).to(device)
initialize_weights(generator)
initialize_weights(discriminator)

opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

generator.train()
discriminator.train()


for epoch in range(num_epochs):
    for batch_idx, (real,) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Generator max E[D(x)] - E[D(G(z))]
        for _ in range(critic_iterations):
            noise = torch.randn(cur_batch_size, z_dim).to(device)
            fake = generator(noise)
            disc_real = discriminator(real).reshape(-1)
            disc_fake = discriminator(fake).reshape(-1)
            gp = gradient_penalty(discriminator, real, fake, device=device)
            loss_disc = (
                -(torch.mean(disc_real) + torch.mean(disc_fake)) + lambda_gp * gp
            )
            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

        # Discriminator

        gen_fake = discriminator(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        generator.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )
