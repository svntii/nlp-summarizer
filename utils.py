#!/usr/bin/env python3


'''
    This file contains all the utility functions used in the project.
'''


import json
import collections.abc
import enum



class objects(enum.Enum):
    '''
        This class contains all the objects used in the project
    '''
    EMAIL_DETAILS = 1
    EMAIL_SUMMARIES = 2


class Email():

    kThreadID = 'thread_id'
    kSubject = 'subject'
    kTimestamp = 'timestamp'
    kFrom = 'from'
    kTo = 'to'
    kBody = 'body'

    def __init__(self, id, subject, timestamp, source, to, thread):
        self.id = id
        self.subject = subject
        self.timestamp = timestamp # example 2002-01-29 11:23:42 or 1012303422000 
        self.source = source
        self.to = to
        self.thread = thread
        self.message = self.__clean_thread(thread)
    
    def __clean_thread(self, thread):
        '''
            This function cleans the thread and returns a list of sentences
        '''
        message_list = []
        temp = thread.split("Subject:")
        
        return temp

class EmailSummaries():
    kThreadID = 'thread_id'
    kSummary = 'summary'

    def __init__(self, id, summary):
        self.id = id
        self.summary = summary
    
    def __size__(self):
        return len(self.summary)


class Utils():

    @staticmethod
    def read_csv(filename, asObject=False, objectType: objects = objects.EMAIL_SUMMARIES):
        '''
            This function reads the json file and returns the data
            
            ARGS:
                filename: the name of the json file
            RETURN:
                data: the data read from the file 
                    - as a dictionary (id, [Emails])
                    - as a list [objects]
        '''
        
        if asObject:
            data = {}
        else:
            data = []

        with open(file=filename, mode='r') as file:
            f = json.load(file)

            for item in f:
                if asObject:
                    if objectType == objects.EMAIL_DETAILS:
                        a = Email(item[Email.kThreadID], item[Email.kSubject], item[Email.kTimestamp], item[Email.kFrom], item[Email.kTo], item[Email.kBody])
                        data[a.id] = data.get(a.id, []) + [a]
                    elif objectType == objects.EMAIL_SUMMARIES:
                        a = EmailSummaries(item[EmailSummaries.kThreadID], item[EmailSummaries.kSummary])
                        data[a.id] = a
                else:
                    a = item
                    data.append(a)   
        return data
    @staticmethod
    def build_vocab(data):
        '''
            This function builds the vocabulary from the data
            ARGS:
                data: the data to build the vocabulary from ([Email], EmailSummaries)
            RETURN:
                vocab: the vocabulary
        '''
        vocab = Vocab()
        for email_list, summary in data:
            for email in email_list:
                for word in email.thread.split():
                    vocab.add(word)
            
            for word in summary.summary.split():
                vocab.add(word)
        
        return vocab

class Vocab(collections.abc.MutableSet):
    """
        Set-like data structure that can change words into numbers and back.
        From Prof. David Chiang Code
    """
    def __init__(self):
        words = {'<BOS>', '<EOS>', '<UNK>'}
        self.num_to_word = list(words)
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(self, word):
        raise NotImplementedError()
    def update(self, words):
        self |= words
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else:
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]