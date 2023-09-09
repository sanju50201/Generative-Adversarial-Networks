import torch
import pandas as pd
import config
from generator import TabularGenerator
from utils import load_checkpoint


def generate_samples(generator, num_samples, z_dim, device):
    z = torch.randn(num_samples, z_dim).to(device)
    with torch.no_grad():
        synthetic_data = generator(z).cpu().numpy()
    synthetic_df = pd.DataFrame(synthetic_data, columns=[
                                "amount", "oldBalanceOrig", "newBalanceOrig", "oldBalanceDest", "newBalanceDest"])
    return synthetic_df


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = config.NOISE_DIM
    generator = TabularGenerator(z_dim, config.NUM_FEATURES).to(device)
    load_checkpoint(config.CHECKPOINT_GEN, generator, None, None)
    num_samples_to_generate = 10000
    synthetic_data = generate_samples(
        generator, num_samples_to_generate, z_dim, device)

    gen_df = pd.DataFrame(synthetic_data.cpu().numpy(), columns=[
                          "amount", "oldBalanceOrig", "newBalanceOrig", "oldBalanceDest", "newBalanceDest"])
    print(gen_df)


if __name__ == "__main__":
    main()
