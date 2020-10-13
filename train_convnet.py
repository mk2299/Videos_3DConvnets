import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
import numpy as np
import h5py
sys.path.append('/home/mk2299/Mice_experiments/attorch')
sys.path.append('/home/mk2299/Mice_experiments/Sinz2018_NIPS/')
sys.path.append('/home/mk2299/Mice_experiments/Sinz2018_NIPS/my_exp/')
from nips2018.architectures.shifters import StaticAffineShifter
from nips2018.architectures.modulators import GateGRUModulator 
from scipy import stats
import hashlib
import inspect
import random
from tensorflow import losses
from numpy import pi
from base_models import Readout, smoothness_regularizer_2d, group_sparsity_regularizer_2d
from nips2018.architectures.readouts import SpatialTransformerPooled3dReadout, ST3dSharedGridStopGradientReadout, FullyConnectedReadout
from nips2018.architectures.cores import StackedFeatureGRUCore, Stacked3dCore, Stacked2dCore
from nips2018.architectures.base import CorePlusReadout3d
from attorch.layers import elu1, Elu1
import torch.nn as nn
from torch.nn import functional as F
import torch
import pickle
from itertools import count
from tqdm import tqdm
from collections import namedtuple,  OrderedDict
from attorch.train import early_stopping, cycle_datasets
from itertools import chain, repeat
from nips2018.utils.measures import corr
from scipy.stats import pearsonr
from models_attention import TimeDistributed, ConvLSTM
#from models.resnet import generate_model
from models_resnet import generate_model
import argparse
from dataloader_processed import Dataset

params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 6}


def ResNet_core(freeze = False, pretrained = False, input_channels = 6):
    core = generate_model(model_depth = 18, input_channels = input_channels) 
    if pretrained:
        pretrain_mdl = torch.load('./pretrained_weights/r3d18_K_200ep.pth')
        core.load_state_dict(pretrain_mdl['state_dict'])
    if freeze:
        for param in core.parameters():
            param.requires_grad = False
    return core

class PoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        #_assert_no_grad(target)
        lag = target.size(1) - output.size(1)
        loss =  (output - target[:, lag:, :] * torch.log(output + self.bias))
        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)
        
        

def full_objective(model, readout_key, inputs, targets, beh = None, eye_pos = None):
    if eye_pos and beh:
        outputs = model(inputs, '0', eye_pos=eye_pos, behavior=beh)
    else:
        outputs = model(inputs, '0')
    return (criterion(outputs, targets)
            + model.readout.regularizer(readout_key).cuda(0)
           + (model.shifter.regularizer(readout_key) if model.shift else 0)
            + (model.modulator.regularizer(readout_key) if model.modulate else 0)) / acc

def compute_scores(y, y_hat, axis=0):
    pearson = corr(y, y_hat, axis=axis)
    return PerformanceScores(pearson=pearson)

def compute_predictions(loader, model, case_num, readout_key = '0', reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for data_batch in loader:
        
        
        if case_num!=2:
            x_val, y_val = data_batch
            y_mod = model(x_val.cuda().float(), readout_key).data.cpu().numpy()   
        else:
            x_val, behav_val, eye_val, y_val = data_batch
            y_mod = model(x_val.cuda().float(), readout_key, eye_pos=eye_val.cuda().float(), behavior=behav_val.cuda().float()).data.cpu().numpy()     
        
        
        neurons = y_val.size(-1)
        lag = y_val.shape[1] - y_mod.shape[1]
        if reshape:
            y.append(y_val[:, lag:, :].numpy().reshape((-1, neurons)))
            y_hat.append(y_mod.reshape((-1, neurons)))
        else:
            y.append(y_val[:, lag:, :].numpy())
            y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat)
    if not return_lag:
        return y, y_hat
    else:
        return y, y_hat, lag
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='3D Convnet model')
    parser.add_argument('--mouse_id', default='17358-5-3', type=str, help='Mouse ID')
    parser.add_argument('--behavioral', action="store_false",  help='If set, include behavioral variables')
    parser.add_argument('--channel_input', action="store_false", help='If set, include behavioral variables as channels')
    parser.add_argument('--n_channels', default=6, type=int)
    parser.add_argument('--shifter', action="store_true")
    parser.add_argument('--modulator', action="store_true")
    parser.add_argument('--data_path', default = "/home/mk2299/Mice_experiments/Sinz2018_NIPS/Sinz2018_NIPS_data/cache", type = str, help = 'Path to data')
    parser.add_argument('--model_file', default = None, help = 'Location for saving model')
    #parser.add_argument('--gpu', default = "0", type = str, help = 'GPU device')
    
    args = parser.parse_args()
    n_neurons = OrderedDict([('0',1740)])
    shifter = StaticAffineShifter(n_neurons, input_channels=2, hidden_channels=2, bias=True, gamma_shifter=0.001) if args.shifter else None
    modulator = GateGRUModulator(n_neurons, gamma_modulator=0.0, hidden_channels=50, offset=1, bias=True) if args.modulator else None
    PerformanceScores = namedtuple('PerformanceScores', ['pearson'])

    core = ResNet_core(input_channels = args.n_channels)
    readout = ST3dSharedGridStopGradientReadout(torch.Size([128, 150,18,32]),  
                                               n_neurons, 
                                               positive=False,  
                                               gamma_features=1., 
                                               pool_steps=2,
                                                kernel_size=4,
                                                stride=4,
                                            gradient_pass_mod=3
                                           )
    
    model = CorePlusReadout3d(core, readout, nonlinearity=Elu1(), 
                        shifter = shifter, modulator = modulator, burn_in=0)
    
   

    # Generators
    training_set = Dataset(sets = 'train', data_dir = args.data_path, mouse_id = args.mouse_id, behavioral = args.behavioral, channel_input = args.channel_input)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset(sets = 'val', data_dir = args.data_path, mouse_id = args.mouse_id, behavioral = args.behavioral, channel_input = args.channel_input)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    
    criterion = PoissonLoss3d()
    n_datasets = training_set.__len__() 
    acc = 1 # accumulate gradient over this many steps


    # --- setup objective
    grad_passes = 0
    for ro in model.readout.values():
        grad_passes += int(not ro.stop_grad)


    mu_dict = OrderedDict([
        (k, torch.Tensor(np.load('%s/%s_mean.npy' % (args.data_path, args.mouse_id) ))) for k in ['0']
    ])
    eye_dict = OrderedDict([
        (k, torch.Tensor(np.load('%s/%s_mean_eye.npy' % (args.data_path, args.mouse_id)))) for k in ['0']
    ])

    model.readout.initialize(mu_dict)
  
    if model.shifter is not None:
        biases = eye_dict
        model.shifter.initialize(bias=biases)
    if model.modulator is not None:
        model.modulator.initialize()
    model = model.cuda()
    
    model.train()
    accumulate_gradient=1
    schedule = [0.005, 0.001] 
    n_epochs = 10

    if args.behavioral and args.channel_input: 
        case_num = 1
    if args.behavioral and not args.channel_input: 
        case_num = 2
    if not args.behavioral: 
        case_num = 3
    print('Behavioral: ', args.behavioral)
    print('Running case number: ', case_num)
        
    for opt, lr in zip(repeat(torch.optim.Adam), schedule):
        print('Training with learning rate', lr)
        optimizer = opt(model.parameters(), lr=lr)
        optimizer.zero_grad()
        iteration = 0
        assert accumulate_gradient > 0, 'accumulate_gradient needs to be > 0'
        for epoch in range(n_epochs):
            print('Epoch done')
            for data_batch in training_generator:
                if case_num!=2:
                    x_batch, y_batch = data_batch
                    obj = full_objective(model, '0', x_batch.float().cuda(), y_batch.float().cuda())
                else:
                    x_batch, behav_batch, eye_batch, y_batch = data_batch
                    obj = full_objective(model, '0', x_batch.float().cuda(), behav_batch.float().cuda(), eye_batch.float().cuda(), y_batch.float().cuda())
                #x_batch = x_batch.to(device=device, dtype=torch.float)
                
                obj.backward()
                if iteration % accumulate_gradient == accumulate_gradient - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                iteration += 1 
                if iteration % 100 == 0:
                    model.eval()
                    true, preds = compute_predictions(validation_generator, model, case_num)
                    print('Val correlation:', np.mean([pearsonr(true[:,i], preds[:,i])[0] for i in range(true.shape[1])]))
                    model.train()
                    
    torch.save(model, "saved_models/%s" % args.model_file)

    
    

    

    
    

