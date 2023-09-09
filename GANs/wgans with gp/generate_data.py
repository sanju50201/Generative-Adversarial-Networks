import torch
import numpy
import pandas as pd
import numpy as np
from train import z_dim, device, generator

num_samples = 100

z = torch.randn(num_samples, z_dim).to(device)

with torch.no_grad():
    synthetic_data = generator(z).cpu().numpy()

print(synthetic_data)


synthetic_df = pd.DataFrame(synthetic_data, columns=[
                            "type", "amount", "oldBalanceOrig", "newBalanceOrig", "oldBalanceDest", "newBalanceDest", "isFraud"])

print(synthetic_df[:5])

synthetic_df.to_csv("./fraud_synthetic_data.csv", index=False)
