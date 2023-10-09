#!/usr/bin/env python3

'''
    This file contains the summarizer and the implementation
'''


from utils import *

class Summarizer():
    
    def __init__(self):
        pass

    def summarize(self, article: Article) -> str:
        '''
            This function takes in text and returns the summary
            arg: 
                data: the text to be summarized
            return:
                summary: the summary of the text
        '''
        return self.__summarize_baseline(article)
    
    def __summarize_baseline(self, article:Article) -> str:
        '''
            This function summarizes the data using the baseline method.
            In this case we are defining a "good summary" to be at least 35% of the original text.
        '''

        length = len(article.content)
        length = int(length * .35)
       
        return article.content[:length]
    





    def __train(self, data):
        pass