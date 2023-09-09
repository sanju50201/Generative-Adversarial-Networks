import torch.nn as nn


class TabularDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(TabularDiscriminator, self).__init__()

        # Define the network layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Test
# input_dim = 30  # For example, if you have 30 features in your dataset
# discriminator = TabularDiscriminator(input_dim=input_dim)
# print(discriminator)
