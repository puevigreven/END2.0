# Session 9

## Recall, Precision, and F1 Score


### Precision -
Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.

$$Precision = TP/(TP+FP)$$
### Recall (Sensitivity) -
 Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? We have got recall of 0.631 which is good for this model as it’s above 0.5.

$$Recall = TP/(TP+FN )$$

### F1 score - 
F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

$$F1Score = 2*(Recall * Precision) / (Recall + Precision)$$$


## Bleu Score
The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric for evaluating a generated sentence to a reference sentence.

$$\text{BLEU} = \text{BP} \cdot \exp \bigg( \sum_{n=1}^{N} w_n \log p_n \bigg)$$

where $$p_n$$ is the modified precision for gram, the base of $$e, w_n$$ is the natural base, is weight between 0 and 1 for $$\log p_n$$ and $$\sum_{n=1}^{N} w_n = 1$$, and BP is the brevity penalty to penalize short machine translations.

$$\text{BP} = 
\begin{cases} 
    1 & \text{if } c > r \\
    \exp \big(1-\frac{r}{c}\big) & \text{if } c \leq r
\end{cases}$$

where $$c$$ is the number of unigrams (length) in all the candidate sentences, and  is the best match lengths for each candidate sentence in the corpus. Here the best match length is the closest reference sentence length to the candidate sentences. 
```
>>> import nltk
>>> reference_1 = "the cat is on the mat".split()
>>> reference_2 = "there is a cat on the mat".split()
>>> candidate = "the cat the cat on the mat".split()
>>> bleu = nltk.translate.bleu_score.sentence_bleu(references=[reference_1, reference_2], hypothesis=candidate, weights=(0.25,0.25,0.25,0.25))
>>> print(bleu)
0.4671379777282001
```


### Perplexity

Intuitively, perplexity can be understood as a measure of uncertainty. The perplexity of a language model can be seen as the level of perplexity when predicting the following symbol. Consider a language model with an entropy of three bits, in which each bit encodes two possible outcomes of equal probability. This means that when predicting the next symbol, that language model has to choose among $$2^3 = 8$$ possible options. Thus, we can argue that this language model has a perplexity of 8.

Mathematically, the perplexity of a language model is defined as:

$$\textrm{PPL}(P, Q) = 2^{\textrm{H}(P, Q)}$$

Less entropy (or less disordered system) is favorable over more entropy. Because predictable results are preferred over randomness. This is why people say low perplexity is good and high perplexity is bad since the perplexity is the exponentiation of the entropy (and you can safely think of the concept of perplexity as entropy).
/image


### Bert Score

BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language generation tasks.
