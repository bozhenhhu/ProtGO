import numpy as np
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T


def losses(out, latent, model, data, train_dataset, optimizer, loss_mse, device, loss_fn, balance=0.1):
    
        out, latent, ori = model(data)

        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)
        loss_ce = loss_fn(out.sigmoid(), y)

        std, mean = torch.std_mean(latent, dim=0)
        GO_mean =  train_dataset.GO_mean.to(device)
        GO_std = train_dataset.GO_std.to(device)
        loss_kl = loss_mse(mean, GO_mean) + loss_mse(std, GO_std)
        loss = loss_ce + balance * loss_kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        loss_ces += loss_ce
        loss_kls += loss_kl
        step += 1
        


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='ProteinSSA')

    parser.add_argument('--embedding_file', default='embeddings/three_type_embeddings.npy', type=str, help='path where to save checkpoint') 
    parser.add_argument('--balance', default=1., type=float)
    args = parser.parse_args()
    return args
