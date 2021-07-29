# Session 12

### 1. Dataset 

##### 1  - Multi30K German to English
Multi30K is a dataset to stimulate multilingual multimodal research for English-German. 

    train
     (en) 29000 sentences, 377534 words, 13.0 words/sent
     (de) 29000 sentences, 360706 words, 12.4 words/sent

    val
     (en) 1014 sentences, 13308 words, 13.1 words/sent
     (de) 1014 sentences, 12828 words, 12.7 words/sent

###### This task focuses on Machine Translation

### Output of training for 10 epochs:
    Epoch: 01 | Time: 1m 23s
    	Train Loss: 5.791 | Train PPL: 327.313
    	 Val. Loss: 5.674 |  Val. PPL: 291.200
    Epoch: 02 | Time: 1m 22s
    	Train Loss: 5.627 | Train PPL: 277.848
    	 Val. Loss: 5.683 |  Val. PPL: 293.689
    Epoch: 03 | Time: 1m 21s
    	Train Loss: 5.618 | Train PPL: 275.270
    	 Val. Loss: 5.691 |  Val. PPL: 296.139
    Epoch: 04 | Time: 1m 21s
    	Train Loss: 5.602 | Train PPL: 270.859
    	 Val. Loss: 5.696 |  Val. PPL: 297.636
    Epoch: 05 | Time: 1m 21s
    	Train Loss: 5.583 | Train PPL: 265.944
    	 Val. Loss: 5.717 |  Val. PPL: 303.990
    Epoch: 06 | Time: 1m 21s
    	Train Loss: 5.568 | Train PPL: 261.955
    	 Val. Loss: 5.730 |  Val. PPL: 308.110
    Epoch: 07 | Time: 1m 21s
    	Train Loss: 5.554 | Train PPL: 258.261
    	 Val. Loss: 5.735 |  Val. PPL: 309.635
    Epoch: 08 | Time: 1m 23s
    	Train Loss: 5.545 | Train PPL: 255.888
    	 Val. Loss: 5.759 |  Val. PPL: 317.042
    Epoch: 09 | Time: 1m 23s
    	Train Loss: 5.545 | Train PPL: 255.849
    	 Val. Loss: 5.764 |  Val. PPL: 318.554
    Epoch: 10 | Time: 1m 23s
    	Train Loss: 5.552 | Train PPL: 257.665
    	 Val. Loss: 5.762 |  Val. PPL: 317.934

##### 2. SQuAD 
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
######  This task is focuses on the task of question answering

Note - I was not able to successfully run the Transformer Model on this dataset.


