import numpy as np
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn.functional as F


import torch_geometric.transforms as T
from datasets import GODataset_three
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from models import  Model_teacher




if __name__ == '__main__':
    args = parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = GODataset_three(root=args.data_dir, random_seed=args.seed, level=args.level, split='train')


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)

    model = Model_teacher().to(device)
    
    # model.load_state_dict(torch.load('./ckpt/go/go_3type.pt'))
    model.load_state_dict(torch.load('./ckpt/go/go_3teacher.pt'))
    model.eval()
    # Iterate over the validation data.

    three_type_embeddings= []
    probs = []
    labels = []
    for data in train_loader:
        data = data.to(device)
        with torch.no_grad():
            x, pos = (model.embedding(data.x), data.pos)
            y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)

            for i, layer in enumerate(model.layers):
                x = layer(x, pos)
                if i == len(model.layers) - 1:
                    x = global_mean_pool(x)
                elif i % 2 == 1:
                    x, pos = model.local_mean_pool(x, pos)
                    latent =  x
            # out = model.classifier(x)

            # prob = out.sigmoid().detach().cpu().numpy()
            # y = np.stack(data.y, axis=0)
            # probs.append(prob)
            # labels.append(y)
            label_emd = model.label_encoder(y) 
            x = x + label_emd
            three_type_embeddings.append(x.detach().cpu().numpy())



    # print(three_type_embeddings)
    np.save('./embeddings/three_teacher_embeddings.npy', three_type_embeddings)


