# Session 5 : Sentiment Analysis using LSTM RNN

About Dataset: Sentiment Treebank includes fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences and it presents new challenges for sentiment compositionality. It is based on  paper: **Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank**

# Data Preparation: 

### Unify Dataset sentences and Labels

      
        with open('/content/drive/MyDrive/END2.0/SentimentClassification/datasetSentences.txt') as f:
            lines = f.readlines()
    
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
     

**Data Distribution**

    [6]   dataset.label_distint.value_counts()
      
    2    6240
    1    2175
    3    2139
    4     663
    0     638
    Name: label_distint, dtype: int64
**Text Augmentation:** 
I am using `nlpaug` library for text augmentation
1. **Synonym Augmentation**

	    aug = naw.SynonymAug(aug_src='wordnet')
	    augmented_text = aug.augment(text)
	    print("Original:")
	    print(text)
	    print("Augmented Text:")
	    print(augmented_text)

	**Original**: The quick brown fox jumps over the lazy dog . 
	**Augmented** Text: The ready brown slyboots climb up over the lazy dog.
2. **Random Swap**
			
		aug = naw.RandomWordAug(action="swap")
		augmented_text = aug.augment(text)
		print("Original:")
		print(text)
		print("Augmented Text:")
		print(augmented_text)
	**Original**: The quick brown fox jumps over the lazy dog .
	**Augmented Text:** Quick the fox brown over jumps the lazy dog.

 **3. Transalation**
		

	    text = 'i am sleeping on the floor'
		back_translation_aug = naw.BackTranslationAug(
		from_model_name='transformer.wmt19.en-de',
		to_model_name='transformer.wmt19.de-en')
		back_translation_aug.augment(text)


**Original**: i am sleeping on the floor
**Augmented Text:** I sleep on the floor

## Network Architecture

    classifier(
      (embedding): Embedding(16929, 300)
      (encoder): LSTM(300, 16, batch_first=True, dropout=0.5, bidirectional=True)
      (fc): Linear(in_features=16, out_features=5, bias=True)
    )
    The model has 5,119,489 trainable parameters

----------
##  Using Pre-trained glove embedding

    Text.build_vocab(train,
    max_size = MAX_VOCAB_SIZE,
    vectors = "glove.840B.300d")
Assigning the weights

	model.embedding.weight.data.copy_(pretrained_embeddings)
	> tensor([[ 0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000], [ 0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000], [ 0.0120, 0.2075, -0.1258, ..., 0.1387, -0.3605, -0.0350], ..., [ 0.0164, -0.1268, 0.1124, ..., -0.0723, 0.4662, -0.3872], [-0.0592, 0.1091, -0.2313, ..., 0.0914, 0.6806, -0.4423], [-0.3185, -0.0888, 0.0675, ..., -0.2871, 0.6534, -0.5551]])

## Loss: Categorical Loss function and F1 Score
**Categorical Accuracy**

	def  categorical_accuracy(preds, y):
		"""
		Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
		"""
		top_pred = preds.argmax(1, keepdim = True)
		correct = top_pred.eq(y.view_as(top_pred)).sum()
		acc = correct.float() / y.shape[0]
		return acc

**F1 Score**

    true_y = F.one_hot(batch.labels, 5).to('cpu')
    f1_scr = f1_score(true_y, predictions.data.to('cpu') > 0.5, average="samples")

##  Training Logs

    0% [██████████████████████████████] 100% | ETA: 00:00:00
Total time elapsed: 00:00:00

	EPOCH: 0
	Train Loss: 1.576 | Train Acc: 37.20% |  F1 Score: 0.003498134328358209
	 Val. Loss: 1.527 |  Val. Acc: 45.62% |  F1 Score: 0.15872061965811965

0% [██████████████████████████████] 100% | ETA: 00:00:00
Total time elapsed: 00:00:00

	EPOCH: 1
	Train Loss: 1.442 | Train Acc: 53.46% |  F1 Score: 0.49125466417910446
	 Val. Loss: 1.465 |  Val. Acc: 45.71% |  F1 Score: 0.45185630341880345

0% [██████████████████████████████] 100% | ETA: 00:00:00
Total time elapsed: 00:00:00

	EPOCH: 2
	Train Loss: 1.395 | Train Acc: 53.59% |  F1 Score: 0.5324160447761194
	 Val. Loss: 1.455 |  Val. Acc: 45.71% |  F1 Score: 0.4561965811965812

0% [██████████████████████████████] 100% | ETA: 00:00:00
Total time elapsed: 00:00:00

	EPOCH: 3
	Train Loss: 1.382 | Train Acc: 53.58% |  F1 Score: 0.5346315298507462
	 Val. Loss: 1.452 |  Val. Acc: 45.71% |  F1 Score: 0.45706463675213677

0% [██████████████████████████████] 100% | ETA: 00:00:00
Total time elapsed: 00:00:00

	EPOCH: 4
	Train Loss: 1.377 | Train Acc: 53.64% |  F1 Score: 0.5360307835820896
	 Val. Loss: 1.450 |  Val. Acc: 45.71% |  F1 Score: 0.45706463675213677

## Training and Vaidation Loss Graph

![Overfitting](https://github.com/puevigreven/END2.0/blob/main/overfit.png)



# Test Sentence 

    Sentence: Pretentious editing ruins a potentially terrific flick . 
    True Label: neutral 
    Predicted: neutral 
    ------------------------- 
    Sentence: It could change America , not only because it is full of necessary discussion points , but because it is so accessible that it makes complex politics understandable to viewers looking for nothing but energetic entertainment . 
    True Label: neutral 
    Predicted: neutral 
    ------------------------- 
    Sentence: A journey through memory , a celebration of living , and a sobering rumination on fatality , classism , and ignorance . 
    True Label: neutral 
    Predicted: neutral 
    ------------------------- 
    Sentence: Long after you leave Justine , you 'll be wondering what will happen to her and wishing her the best -- whatever that might mean . 
    True Label: very positive 
    Predicted: neutral 
    ------------------------- 
    Sentence: So I just did . 
    True Label: negative 
    Predicted: neutral
     ------------------------- 
     Sentence: It is , by conventional standards , a fairly terrible movie ... but it is also weirdly fascinating , a ready-made Eurotrash cult object . 
     True Label: neutral 
     Predicted: neutral 
     ------------------------- 
     Sentence: Grant gets to display his cadness to perfection , but also to show acting range that may surprise some who thought light-hearted comedy was his forte . 
     True Label: positive 
     Predicted: neutral 
     ------------------------- 
     Sentence: Haneke challenges us to confront the reality of sexual aberration . 
     True Label: neutral
      Predicted: neutral 
     ------------------------- 
     Sentence: The story bogs down in a mess of purposeless violence . 
     True Label: positive 
     Predicted: neutral 
     ------------------------- 
     Sentence: As if to prove a female director can make a movie with no soft edges , Kathryn Bigelow offers no sugar-coating or interludes of lightness . 
     True Label: neutral 
     Predicted: neutral 
     -------------------------
