#!/usr/bin/env python3

'''
    This file contains the summarizer and the implementation
'''


from utils import *

class Summarizer():
    
    def __init__(self):
        pass

    def summarize(self, article) -> str:
        '''
            This function takes in text and returns the summary
            arg: 
                data: the text to be summarized
            return:
                summary: the summary of the text
        '''
        return self.__summarize_baseline(article)

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