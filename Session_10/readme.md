# Session 10

## What is Teacher Forcing
Teacher forcing is the technique where the target word is passed as the next input to the decoder.

Let us assume we want to train an Machine Translation model(Chinese to English), and the ground truth is “Donald Trump was a President”. Our model makes a mistake in predicting the 2nd word and we have “Donald” and “Duck” for the 1st and 2nd prediction respectively.

Without Teacher Forcing, we would feed “Duck” back to our RNN to predict the 3rd word. Let’s say the 3rd prediction is “cartoon”. Even though it makes sense for our model to predict “cartoon” given the input is “duck”, it is different from the ground truth.


![alt Without Teacher Forcing](https://github.com/puevigreven/END2.0/blob/main/Session_10/images/without_teacher_forcing.png)
On the other hand, if we use Teacher Forcing, we would feed “trump” to our RNN for the 3rd prediction, after computing and recording the loss for the 2nd prediction.


![alt With Teacher forcing](https://github.com/puevigreven/END2.0/blob/main/Session_10/images/with_teacher_forcing.png)

## What is decoder attention actually doing?

In decoder attention mechanism we are using previous hidden output with embedded input to generate attention weights and use these attention weights with encoder outputs to create input for GRU. 

Step 1. Concat previous hidden with input embedded vector
``` 
self.attn(torch.cat((embedded[0], hidden[0]), 1))
```
Step 2. Pass the output of previous output to softmax layer, so the sum of weights equals 1 
``` 
attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
```
Step 3. Performs a batch matrix-matrix product of matrices stored in attn_weights and Encoder outputs.
```
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
```
Step 4. Concat Input embedded and attn_applied and pass it to linear layer
```
output = torch.cat((embedded[0], attn_applied[0]), 1)
output = self.attn_combine(output).unsqueeze(0)
```
Step 5. Pass this output as input to GRU
```
output = F.relu(output)
output, hidden = self.gru(output, hidden)
```

## Suggestions to improve this model 
#### 1. Glove embedding for French sentences
#### 2. Local Attention Mechanism : 
Local attention mechanism selectively focuses on a small window of context and is differentiable. This approach has an advantage of avoiding the expensive computation incurred in the soft attention and at the same time, is easier to train than the hard attention approach.

![alt Local Attention](https://github.com/puevigreven/END2.0/blob/main/Session_10/images/local_attention.png)
### Comparision with Classroom code
The Glove embedding approach was expected to perform better than classroom code, but I have observed the opposite (May be because of the a bug). 

Following are log comparision for first 5 steps:

Without Glove
```
1m 38s (- 23m 1s) (5000 6%) 3.3963
3m 13s (- 20m 59s) (10000 13%) 2.7843
4m 49s (- 19m 16s) (15000 20%) 2.4587
6m 24s (- 17m 36s) (20000 26%) 2.1692
8m 0s (- 16m 0s) (25000 33%) 1.9805
9m 37s (- 14m 25s) (30000 40%) 1.7411
```

With Glove
```
1m 28s (- 50m 3s) (5000 2%) 3.6348
2m 48s (- 46m 25s) (10000 5%) 3.0894
4m 9s (- 44m 16s) (15000 8%) 2.8698
5m 29s (- 42m 30s) (20000 11%) 2.6976
6m 49s (- 40m 58s) (25000 14%) 2.5395
8m 10s (- 39m 28s) (30000 17%) 2.4437
9m 31s (- 38m 4s) (35000 20%) 2.3016
```
