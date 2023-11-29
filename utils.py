#!/usr/bin/env python3


'''
    This file contains all the utility functions used in the project.
'''


import json
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
        self.timestamp = timestamp
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
                    elif objectType == objects.EMAIL_SUMMARIES:
                        a = EmailSummaries(item[EmailSummaries.kThreadID], item[EmailSummaries.kSummary])
                    data[a.id] = data.get(a.id, []) + [a]
                else:
                    a = item
                    data.append(a)   
        return data 


