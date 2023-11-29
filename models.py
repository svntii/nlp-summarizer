#!/usr/bin/env python3

'''
    This file contains the summarizer and the implementation
'''


from utils import *
import torch.nn as nn
import torch



class Summarizer(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_heads, num_layers, emb_dim=256):
        super(Summarizer, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)   
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(emb_dim, output_dim)

    
    def forward(self, content, date):
        
        content_embedding = self.embedding(content)
        date_embedding = self.embedding(date)
        
        input_vector = torch.cat((content_embedding, date_embedding), dim=1)

        output = self.transformer(input_vector)
        output = self.fc(output)

        return output

    def summarize(self, content) -> str:
        '''
            This function takes in text and returns the summary
            arg: 
                data: the text to be summarized
            return:
                summary: the summary of the text
        '''
        summary = self(content)
        
        
        return self.__summarize_baseline(content)
    
    

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