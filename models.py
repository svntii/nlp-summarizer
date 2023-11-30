#!/usr/bin/env python3

'''
    This file contains the summarizer and the implementation
'''

import datetime
from utils import *
import torch.nn as nn
import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Summarizer(nn.Module):
    
    def __init__(self, input_dim:int, output_dim:int, num_heads:int, num_layers:int, vocab:Vocab, emb_dim=256):
        super(Summarizer, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)   
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(emb_dim, output_dim)
        self.vocab = vocab

    
    def forward(self, content, date, summary):
        '''
            This function takes in text and returns the summary
            arg: 
                data: the text to be summarized
            return:
                summary: the summary of the text
        '''

        content_embedding = self.embedding(content)
        target_embedding = self.embedding(summary)        
        date_embedding = self.embedding(date)

        # input_vector = torch.cat((content_embedding, date_embedding), dim=1)

        output = self.transformer(content_embedding, target_embedding)
        output = self.fc(output)

        return output

    def summarize(self, content, date, summary) -> str:
        '''
            This function takes in text and returns the summary
            arg: 
                data: the text to be summarized
            return:
                summary: the summary of the text
        '''
        o = self(content, date, summary)
        return o
        
        # return self.__summarize_baseline(content)
    
    

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
    

    def __train(self, data):
        pass