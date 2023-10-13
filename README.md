# NLP-Summarizer 
This is a Text Summarizer that uses techniques covered in CSE 40657/60657 to effectively capture to meaning and potientially act on text passed in.

Authors: Santiago Rodriguez | svntiiago@gmail.com



## Data Set

The data used to train this model was from Marawan Mamdouh posted on [kaggle](https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset)

Will add more data sets as the project progresses

---
### Data Format / Objects

Emails are stored in a json format with the following fields

`Email()`

- `thread_id`: The id of the email thread
- `subject`: The subject of the email
- `timestamp`: The timestamp of the email
- `from`: The sender of the email
- `to`: The recipient of the email
- `body`: The body of the email

---
`EmailSummaries()`

The summaries are stored in a json format with the following fields

- `thread_id`: The id of the email thread
- `summary`: The summary of the email thread

## PM2 - Baseline

The baseline for this project is a simple extractive summarizer that uses the first 35%  of the text of each email. This is a very simple approach that does not take into account the meaning of the text. This is a good starting point to compare the performance of the model to.

I created and setup various interfaces to be used and added as the project progresses

- Summarizer Interface      | *summarizer.py*
- Utility Interface         | *utils.py*
- Layers for Model Training | *layers.py*









