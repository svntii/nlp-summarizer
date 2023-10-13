#!/usr/bin/env python3


EMAIL_DETAILS = "data/email_thread_details.json"
EMAIL_SUMMARIES = "data/email_thread_summaries.json"


from utils import *
from summarizer import *


def test_baseline(emailList, s):
    for key, value in emailList.items():
        print('-------------------')
        print(s.summarize(value))
        break


def main():

    details = Utils.read_csv(EMAIL_DETAILS, asObject=True, objectType=objects.EMAIL_DETAILS)    
    summaries = Utils.read_csv(EMAIL_SUMMARIES, asObject=True, objectType=objects.EMAIL_SUMMARIES)

    s = Summarizer()

    test_baseline(details, s)



if __name__ == '__main__':
    main()