"""
File: test.py
Original Author: Yuqin Yuan
Date: 2024-08-28
"""
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy import signal
from sklearn import preprocessing

def Radar_HP(data):
    low = 0.5 / 100
    b, a = signal.butter(5, low, btype='highpass')
    data_HP = signal.filtfilt(b, a, data, axis=1) 
    return data_HP

class RadarDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv,transform=False):
        super(RadarDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        self.data_dir = data_dir
        self.labels = df
        self.classes = ['AF']
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        subject_id = row['subject_id']
        path = os.path.join(self.data_dir, subject_id+'.mat')
        data = loadmat(path)   
        radardata = data['Radar_data']
        Radardata = radardata.reshape(radardata.shape[0],radardata.shape[1]*radardata.shape[2])
        Radar_data = preprocessing.normalize(Radar_HP(Radardata.T))
        Radar_data = Radar_data.astype('<f8').T
        if self.label_dict.get(subject_id):
            labels = self.label_dict.get(subject_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[subject_id] = labels   
        sample = {'Radar': torch.from_numpy(Radar_data.transpose()).float(), 'label': torch.from_numpy(labels).float(), 'subject_id': subject_id}
        return sample
    def __len__(self):
        return len(self.labels)
