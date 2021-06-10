with open('/content/drive/MyDrive/END2.0/SentimentClassification/datasetSentences.txt') as f:
    lines = f.readlines()
# for line in lines:
#   line.split('\t')
dataset = [line.split('\t') for line in lines]  
dataset = pd.DataFrame(dataset, columns = ['num', 'text']) 
dataset = dataset.drop([0]) # dropping first row which is 'sentence_index" and "sentence" 
dataset = dataset.astype({'num': 'int64', 'text': 'string'})
# dataset = dataset.set_index('num')

with open('/content/drive/MyDrive/END2.0/SentimentClassification/sentiment_labels.txt') as f:
    lines = f.readlines()
dataset_label = [line.rstrip("\n").split("|") for line in lines]  
dataset_label = pd.DataFrame(dataset_label, columns = ['num', 'label'])
dataset_label = dataset_label.drop([0]) # dropping first row which is 'phrase ids" and "sentiment values" 
dataset_label = dataset_label.astype({'num': 'int64', 'label': 'float64'})
dataset_label.dtypes

with open('/content/drive/MyDrive/END2.0/SentimentClassification/datasetSplit.txt') as f:
    lines = f.readlines()
import pandas as pd

dataset_split = [line.rstrip("\n").split(",") for line in lines]  
dataset_split = pd.DataFrame(dataset_split, columns = ['num', 'split'])
dataset_split = dataset_split.drop([0]) # dropping first row which is 'sentence_index' and "splitset_label" 
dataset_split = dataset_split.astype({'num': 'int64', 'split': 'int32'})
dataset_split.dtypes


dataset = dataset.merge(dataset_label, on='num',how='inner')
dataset = dataset.merge(dataset_split, on='num',how='inner')
dataset.head()


def binify(row):
  # print(row)
  bin = 5
  for i in range(bin):
    if i/bin <= row < (i+1)/bin:
      return i
    elif row == 1.0:
      return bin - 1


dataset['label_distint'] = dataset['label'].apply(binify)

dataset.label_distint.value_counts()

dataset[dataset.label_distint.isna()]