#!/usr/bin/env python3

'''
    This file contains the summarizer and the implementation
'''

import datetime
from utils import *
import torch.nn as nn
import torch
import math


class Summarizer(nn.Module):
    '''
        This class is the summarizer
    '''

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, vocab:Vocab, dropout=0.5,):
        '''
            This function initializes the summarizer
            arg:
                ntoken: the number of tokens
                ninp: the number of input
                nhead: the number of heads
                nhid: the number of hidden layers
                nlayers: the number of layers
                vocab: the vocab
                dropout: the dropout rate
        '''
        
        super(Summarizer, self).__init__()
        
        self.model_type = 'Transformer'
        self.src_mask = None
        self.vocab = vocab

        self.pos = PositionalEncoding(ninp, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)

        self.ninp = ninp
        self.n_token = ntoken
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    

    def init_weights(self):
        '''
            This function initializes the weights
        '''

        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        '''
            This function is the forward function
            arg:
                src: the source
            return:
                output: the output
        '''
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def summarize(self, content, date, decoding_strategy="greedy", max_length=1000):
        '''
            This function takes in the content and date and returns the summary
            arg:
                content: the content
                date: the date
            return:
                summary: the summary
        '''

        # TODO fix this
        content_str = "".join([str(o.thread) for o in content])
        content_list = content_str.split()

        # trim content list
        if len(content_list) > max_length:
            content_list = content_list[:max_length]


        content_tensor = torch.tensor([self.vocab.numberize(word) for word in content_list], dtype=torch.long)            
        date_tensor = torch.tensor(content[-1].timestamp, dtype=torch.long)

        # concatenate the date and content
        # content_tensor = torch.cat((date_tensor, content_tensor), 0)

        output = self.forward(content_tensor)
        
        # tokens = []
        # if decoding_strategy == "greedy":
        #     for val in output:            
        #         item = torch.argmax(val)
        #         tokens.append(self.vocab.denumberize(item))


        
        return output
    

    def __summarize_baseline(self, emailList:list[Email]) -> str:
        '''
            This function summarizes the data using the baseline method.
            In this case we are defining a "good summary" to be at least 35% of the original text.
        '''
        message_parts = []
        for item in emailList:
            length = len(item.thread)
            length_to_extract = int(length * 0.35)
            message_parts.append(item.thread[:length_to_extract])
        
        message = " ".join(message_parts)


        return message

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
