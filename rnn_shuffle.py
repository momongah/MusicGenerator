import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import init
import os

class loader(Dataset):
    def __init__(self, txt_file):
        content = open(txt_file, 'r').read()

        self.data = []
        
        #for songs with <100 sized last chunks
        #append λ for remaining chars to denote EOS
        while content:
            start = content.find('<start>')
            end = content.find('<end>')+5
            x_i = content[start:end+1]

            pad_len = 100 - len(x_i) % 100 + 1
            x_i = x_i + 'λ' * pad_len

            self.data.append(x_i)
            content = content[end+1:]

    def __len__(self):
        return len(self.data)
        
    #return the idx-th chunk and target
    def __getitem__(self, idx):
        return self.data[idx]


class RNnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, weight, num_layers=1):
        super(RNnet, self).__init__()        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        #onehot encoding of the characters
        self.char_embeddings = nn.Embedding.from_pretrained(weight) 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)     
        self.hidden2out = nn.Linear(hidden_size, output_size)
               
    
    def forward(self, sequence, states):
        
        #get the one hot encoding of chars for forward pass
        embeds = self.char_embeddings(sequence)
        embeds = torch.transpose(embeds, 0, 1)
    
        lstmout, states = self.lstm(embeds, states)
        
        output = self.hidden2out(lstmout)
        output = output.view(-1, self.output_size)
               
        return output, (states[0].detach(), states[1].detach())
    