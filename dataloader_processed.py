import torch
import pickle 
import numpy as np
import os


class Dataset(torch.utils.data.Dataset):
  
    'Characterizes a dataset for PyTorch'
    def __init__(self, sets = 'train', data_dir = '/home/mk2299/Mice_experiments/Sinz2018_NIPS/Sinz2018_NIPS_data/cache', mouse_id = None, behavioral = True, channel_input = True, seq_length = 150):
        'Initialization'
        self.sets = sets
        self.behavioral = behavioral
        self.channel_input = channel_input
        self.seq_length = seq_length
        with open('%s/%s.pkl' % (data_dir, mouse_id), 'rb') as pkl_file:
            self.data_dict = pickle.load(pkl_file)
        self.list_IDs = np.arange(len(self.data_dict['stims_' + sets]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        seq_length = self.seq_length
        # Load data and get label
        X = self.data_dict['stims_' + self.sets][ID]
        y = self.data_dict['responses_' + self.sets][ID]
        bh = self.data_dict['behav_' + self.sets][ID]
        eye = self.data_dict['eye_' + self.sets][ID]
        start_idx = np.random.randint(X.shape[1]-seq_length-1)
        if self.behavioral:  
            if self.channel_input:
                bh = np.tile((bh.T)[:,:,np.newaxis,np.newaxis], (1,1,36,64))
                eye = np.tile((eye.T)[:,:,np.newaxis,np.newaxis], (1,1,36,64))
                X = np.vstack((X, eye, bh))
                
                return X[:,start_idx : start_idx + seq_length], y[start_idx:start_idx + seq_length] 
            else: 
                return X[np.newaxis, start_idx : start_idx + seq_length], bh[start_idx:start_idx + seq_length], eye[start_idx:start_idx + seq_length], y[:,start_idx:start_idx + seq_length]
        else:
            return X[:,start_idx:start_idx + seq_length], y[start_idx:start_idx + seq_length]
