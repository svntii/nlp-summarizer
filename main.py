#!/usr/bin/env python3

ARTICLE1 = './data/articles1.csv'
ARTICLE2 = './data/articles2.csv'
ARTICLE3 = './data/articles3.csv'


from utils import *
from summarizer import *


def test_baseline(a1, s):
    for article in a1:
        print(article.content)
        print('-------------------')
        print(s.summarize(article.content))
        break


def main():
    a1 = read_csv(ARTICLE1, True)
    s = Summarizer()

    test_baseline(a1, s)



if __name__ == '__main__':
    main()