#!/usr/bin/env python3


'''
    This file contains all the utility functions used in the project.
'''


import csv
csv.field_size_limit(10 * 1024 * 1024)




class Article():
    '''
    ['', 'id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content']
    '''
    def __init__(self, id, title, publication, author, date, year, month, url, content):
        self.id = id
        self.title = title
        self.publication = publication
        self.author = author
        self.date = date
        self.year = year
        self.month = month
        self.url = url
        self.content = content




class Evaluator():
    '''
        This class contains the methods to evaluate the summarizer
    '''
    def __init__(self):
        pass

    def evaluate(self, data):
        '''
            This function evaluates the summarizer
        '''
        pass



def read_csv(filename, asObject=False):
    '''
        This function reads the csv file and returns the data
        
        ARGS:
            filename: the name of the csv file
    '''
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if asObject:
                a = Article(row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
            else:
                a = row
            data.append(a)
    return data[1:]


