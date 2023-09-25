import os

import numpy as np
import torch
from .Utils import normalize

class Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_list_path, pair_dir, transform=None):
        self.pair_list = np.loadtxt(pair_list_path).astype(np.int)
        self.transform = transform
        self.pair_dir = pair_dir
        self.preload()

    def preload(self):
        self.dataset = {}
        for c_no, s, ed, es in self.pair_list:
            ed_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, ed, s)))
            es_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, es, s)))
            if c_no not in self.dataset:
                self.dataset[c_no] = {}
            if s not in self.dataset[c_no]:
                self.dataset[c_no][s] = {
                    ed: {
                        'img': torch.tensor(normalize(ed_unit['img'].astype(np.float))).unsqueeze(0).unsqueeze(0).float(),
                        'seg': torch.tensor(ed_unit['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float(),
                    },
                    es: {
                        'img': torch.tensor(normalize(es_unit['img'].astype(np.float))).unsqueeze(0).unsqueeze(0).float(),
                        'seg': torch.tensor(es_unit['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float()
                    },
                }
            else:
                self.dataset[c_no][s][ed] = {
                        'img': torch.tensor(normalize(ed_unit['img'].astype(np.float))).unsqueeze(0).unsqueeze(0).float(),
                        'seg': torch.tensor(ed_unit['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float(),
                    }
                self.dataset[c_no][s][es] = {
                        'img': torch.tensor(normalize(es_unit['img'].astype(np.float))).unsqueeze(0).unsqueeze(0).float(),
                        'seg': torch.tensor(es_unit['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float()
                    }

    def __len__(self):
        return self.pair_list.shape[0]

    def __getitem__(self, idx):
        c_no, s, ed, es = self.pair_list[idx]

        output = {
            'src': self.dataset[c_no][s][ed],
            'tgt': self.dataset[c_no][s][es]
        }

        if self.transform:
            output = self.transform(output)

        return output


class Collate(object):
    def __call__(self, batch):
        output = {
            'src': {
                'img': torch.cat([d['src']['img'] for d in batch], 0),
                'seg': torch.cat([d['src']['seg'] for d in batch], 0),
            },
            'tgt': {
                'img': torch.cat([d['tgt']['img'] for d in batch], 0),
                'seg': torch.cat([d['tgt']['seg'] for d in batch], 0),
            }
        }

        return output


class CollateGPU(object):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def collate(self, batch):
        output = {
            'src': {
                'img': torch.cat([d['src']['img'] for d in batch], 0).cuda(),
                'seg': torch.cat([d['src']['seg'] for d in batch], 0).cuda(),
            },
            'tgt': {
                'img': torch.cat([d['tgt']['img'] for d in batch], 0).cuda(),
                'seg': torch.cat([d['tgt']['seg'] for d in batch], 0).cuda(),
            }
        }
        return output

    def __call__(self, batch):
        batch = self.collate(batch)
        if self.transforms:
            batch = self.transforms(batch)
        return batch
