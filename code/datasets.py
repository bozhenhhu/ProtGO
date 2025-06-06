import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

from utils import orientation

# AA Letter to id
aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}


resname_to_pc7_dict = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
                'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
                'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
                'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
                'F': [ 0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
                'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
                'H': [ 0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
                'I': [ 0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
                'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
                'L': [ 0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
                'M': [ 0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
                'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
                'P': [ 0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
                'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
                'R': [ 0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
                'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
                'T': [ 0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
                'V': [ 0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
                'W': [ 0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
                'Y': [ 0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476],
                'X': [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]}


class GODataset_KD(Dataset):

    def __init__(self, root='/tmp/protein-data/go', level='mf', percent=30, random_seed=0, split='train', embedding_file='embeddings'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        
        GO_embedding = np.load(embedding_file)
        GO_embedding = GO_embedding.reshape((-1, 2048))
        self.GO_mean = torch.from_numpy(np.mean(GO_embedding, axis=0))
        self.GO_std = torch.from_numpy(np.std(GO_embedding, axis=0))


        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(root, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class GODataset_KeAP(Dataset):

    def __init__(self, root='/tmp/protein-data/go', level='mf', percent=30, random_seed=0, split='train', embedding_path='embeddings/GO'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        KeAP_embedding = np.load(os.path.join(embedding_path, 'KeAP_'+split+'.npy'))

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        i = 0
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int), KeAP_embedding[i]))
            i += 1

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(root, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino, KeAP_embedding = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        embed = KeAP_embedding[np.newaxis, :]
        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos),    # [num_nodes, num_dimensions]
                    embed = torch.from_numpy(embed))

        return data

class GODataset_three(Dataset):
    
    def __init__(self, root='/tmp/protein-data/go', level='mf', percent=30, random_seed=0, split='train'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}

        for level in ['mf', 'bp', 'cc']:
            with open(os.path.join(root, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
                for idx, line in enumerate(f):
                    if idx == 1 and level == "mf":
                        level_idx = 1
                        arr = line.rstrip().split('\t')
                        for go in arr:
                            go_annotations[go] = go_cnt
                            go_num[go] = 0
                            go_cnt += 1
                    elif idx == 5 and level == "bp":
                        level_idx = 2
                        arr = line.rstrip().split('\t')
                        for go in arr:
                            go_annotations[go] = go_cnt
                            go_num[go] = 0
                            go_cnt += 1
                    elif idx == 9 and level == "cc":
                        level_idx = 3
                        arr = line.rstrip().split('\t')
                        for go in arr:
                            go_annotations[go] = go_cnt
                            go_num[go] = 0
                            go_cnt += 1
                    elif idx > 12:
                        arr = line.rstrip().split('\t')
                        protein_labels = []
                        if len(arr) > level_idx:
                            protein_go_list = arr[level_idx]
                            protein_go_list = protein_go_list.split(',')
                            for go in protein_go_list:
                                if len(go) > 0:
                                    protein_labels.append(go_annotations[go])
                                    go_num[go] += 1
                        if level == "mf":
                            self.labels[arr[0]] = np.array(protein_labels, dtype=int)
                        else:
                            protein_labels = np.concatenate([self.labels[arr[0]], np.array(protein_labels, dtype=int)])
                            self.labels[arr[0]] = protein_labels

        self.num_classes = len(go_annotations)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            for go_idx in self.labels[protein_name]:
                # print(protein_name)
                # print(go_idx)
                label[go_idx] = 1.0
            # label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class ECDataset_KD(Dataset):

    def __init__(self, root='/tmp/protein-data/ec', percent=30, random_seed=0, split='train', embedding_file='embeddings'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        GO_embedding = np.load(embedding_file)
        GO_embedding = GO_embedding.reshape((-1, 2048))
        self.GO_mean = torch.from_numpy(np.mean(GO_embedding, axis=0))
        self.GO_std = torch.from_numpy(np.std(GO_embedding, axis=0))
        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))

        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(root, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels)/ec_num[ec]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class ECDataset_KeAP(Dataset):

    def __init__(self, root='/tmp/protein-data/ec', percent=30, random_seed=0, split='train', embedding_path='embeddings/ec'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        KeAP_embedding = np.load(os.path.join(embedding_path, 'KeAP_'+split+'.npy'))

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        i=0
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int), KeAP_embedding[i]))
            i+=1

        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(root, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels)/ec_num[ec]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino, KeAP_embedding = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        embed = KeAP_embedding[np.newaxis, :]
        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos),    # [num_nodes, num_dimensions]
                    embed = torch.from_numpy(embed)      #[1,1024]
        )

        return data