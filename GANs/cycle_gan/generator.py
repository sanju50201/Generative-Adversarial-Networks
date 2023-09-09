import torch.nn as nn


class TabularGenerator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(TabularGenerator, self).__init__()

        # Define the network layers
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, output_dim),
            # Tanh is used to make sure the outputs are between -1 and 1, which can then be rescaled or normalized as needed.
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


# Test
# Length of the noise vector, a common choice is 100 but can be adjusted.
# noise_dim = 100
# output_dim = 30  # For example, if you have 30 features in your dataset
# generator = TabularGenerator(noise_dim=noise_dim, output_dim=output_dim)
# print(generator)
