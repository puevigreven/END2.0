# Session 14

## BERT (BERT Tutorial- How To Build a Question Answering Bot)

### Introduction from Paper 
BERT, or Bidirectional Encoder Representations from Transformers, improves upon standard Transformers by removing the unidirectionality constraint by using a masked language model (MLM) pre-training objective. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, BERT uses a next sentence prediction task that jointly pre-trains text-pair representations.

There are two steps in BERT: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-training tasks. For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters.
### How BERT works

BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. The detailed workings of Transformer are described in a paper by Google.
As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).
The chart below is a high-level description of the Transformer encoder. The input is a sequence of tokens, which are first embedded into vectors and then processed in the neural network. The output is a sequence of vectors of size H, in which each vector corresponds to an input token with the same index.
When training language models, there is a challenge of defining a prediction goal. Many models predict the next word in a sequence (e.g. “The child came home from ___”), a directional approach which inherently limits context learning. To overcome this challenge, BERT uses two training strategies:

### Training Logs :
    Evaluating:   0%|          | 0/425 [00:00<?, ?it/s]***** Running evaluation *****
    Num examples = 13600
    Batch size = 32

    Evaluating:   0%|          | 0/425 [00:10<?, ?it/s]
    Evaluating:   9%|▉         | 38/425 [00:10<01:48,  3.58it/s]
    Evaluating:  13%|█▎        | 56/425 [00:15<01:43,  3.58it/s]
    Evaluating:  17%|█▋        | 74/425 [00:20<01:38,  3.58it/s]
    Evaluating:  22%|██▏       | 92/425 [00:25<01:32,  3.58it/s]
    Evaluating:  26%|██▌       | 110/425 [00:30<01:27,  3.58it/s]
    Evaluating:  30%|███       | 128/425 [00:35<01:22,  3.59it/s]
    Evaluating:  34%|███▍      | 146/425 [00:40<01:17,  3.58it/s]
    Evaluating:  39%|███▊      | 164/425 [00:45<01:12,  3.58it/s]
    Evaluating:  43%|████▎     | 182/425 [00:50<01:07,  3.58it/s]
    Evaluating:  47%|████▋     | 200/425 [00:55<01:02,  3.58it/s]
    Evaluating:  51%|█████▏    | 218/425 [01:00<00:57,  3.58it/s]
    Evaluating:  56%|█████▌    | 236/425 [01:05<00:52,  3.58it/s]
    Evaluating:  60%|█████▉    | 254/425 [01:10<00:47,  3.58it/s]
    Evaluating:  64%|██████▍   | 272/425 [01:15<00:42,  3.58it/s]
    Evaluating:  68%|██████▊   | 290/425 [01:20<00:37,  3.59it/s]
    Evaluating:  72%|███████▏  | 308/425 [01:25<00:32,  3.59it/s]
    Evaluating:  77%|███████▋  | 326/425 [01:30<00:27,  3.59it/s]
    Evaluating:  81%|████████  | 344/425 [01:35<00:22,  3.59it/s]
    Evaluating:  85%|████████▌ | 362/425 [01:40<00:17,  3.58it/s]
    Evaluating:  89%|████████▉ | 380/425 [01:46<00:12,  3.58it/s]
    Evaluating:  94%|█████████▎| 398/425 [01:51<00:07,  3.58it/s]
    Evaluating: 100%|██████████| 425/425 [01:58<00:00,  3.58it/s]
    {
    "exact": 70.26025435862883,
    "f1": 73.4429145868012,
    "total": 11873,
    "HasAns_exact": 67.27395411605939,
    "HasAns_f1": 73.64840163446155,
    "HasAns_total": 5928,
    "NoAns_exact": 73.23801513877208,
    "NoAns_f1": 73.23801513877208,
    "NoAns_total": 5945,
    "best_exact": 70.62242061820939,
    "best_exact_thresh": -0.4445462226867676,
    "best_f1": 73.65357441378065,
    "best_f1_thresh": -0.4445462226867676
    }

### Example Sentences : 
Sentence 1: 

    "question": "In what country is Normandy located?",
    "id": "56ddde6b9a695914005b9628",
    "answers":
        {
            "text": "France",
            "answer_start": 159
        },
    "predicted Answer": 
        {
            "56ddde6b9a695914005b9628": "France"
        }
Sentence 2: 

    "question": "When were the Normans in Normandy?",
    "id": "56ddde6b9a695914005b9629",
    "answers":
        {
            "text": "10th and 11th centuries",
            "answer_start": 94
        }
    "predicted Answer": 
        {
        "56ddde6b9a695914005b9629": "10th and 11th centuries",
        }
Sentence 3:

    "question": "From which countries did the Norse originate?",
    "id": "56ddde6b9a695914005b962a",
    "answers":
        {
            "text": "Denmark, Iceland and Norway",
            "answer_start": 256
        }
    "predicted Answer": 
        {
            "56ddde6b9a695914005b962a": ""
        }
    
Sentence 4:    

    "question": "Who was the Norse leader?",
    "id": "56ddde6b9a695914005b962b",
    "answers":
        {
            "text": "Rollo",
            "answer_start": 308
        }
    "predicted Answer": 
        {
            "56ddde6b9a695914005b962b": "Rollo",
        }

Sentence 5:    

    "question": "What century did the Normans first gain their separate identity?",
    "id": "56ddde6b9a695914005b962c",
    "answers":
        {
            "text": "10th century",
            "answer_start": 671
        },
    "predicted Answer": 
        {
            "56ddde6b9a695914005b962c": "10th",
        }


## BERT Fine-Tuning Sentence Classification
### Training Logs :

    ======== Epoch 1 / 4 ========
    Training...
    Batch    40  of    241.    Elapsed: 0:00:09.
    Batch    80  of    241.    Elapsed: 0:00:17.
    Batch   120  of    241.    Elapsed: 0:00:25.
    Batch   160  of    241.    Elapsed: 0:00:34.
    Batch   200  of    241.    Elapsed: 0:00:42.
    Batch   240  of    241.    Elapsed: 0:00:51.

    Average training loss: 0.50
    Training epcoh took: 0:00:51

    Running Validation...
    Accuracy: 0.83
    Validation Loss: 0.43
    Validation took: 0:00:02

    ======== Epoch 2 / 4 ========
    Training...
    Batch    40  of    241.    Elapsed: 0:00:08.
    Batch    80  of    241.    Elapsed: 0:00:17.
    Batch   120  of    241.    Elapsed: 0:00:25.
    Batch   160  of    241.    Elapsed: 0:00:34.
    Batch   200  of    241.    Elapsed: 0:00:42.
    Batch   240  of    241.    Elapsed: 0:00:51.

    Average training loss: 0.31
    Training epcoh took: 0:00:51

    Running Validation...
    Accuracy: 0.85
    Validation Loss: 0.42
    Validation took: 0:00:02

    ======== Epoch 3 / 4 ========
    Training...
    Batch    40  of    241.    Elapsed: 0:00:08.
    Batch    80  of    241.    Elapsed: 0:00:17.
    Batch   120  of    241.    Elapsed: 0:00:25.
    Batch   160  of    241.    Elapsed: 0:00:34.
    Batch   200  of    241.    Elapsed: 0:00:42.
    Batch   240  of    241.    Elapsed: 0:00:51.

    Average training loss: 0.20
    Training epcoh took: 0:00:51

    Running Validation...
    Accuracy: 0.85
    Validation Loss: 0.44
    Validation took: 0:00:02

    ======== Epoch 4 / 4 ========
    Training...
    Batch    40  of    241.    Elapsed: 0:00:08.
    Batch    80  of    241.    Elapsed: 0:00:17.
    Batch   120  of    241.    Elapsed: 0:00:25.
    Batch   160  of    241.    Elapsed: 0:00:34.
    Batch   200  of    241.    Elapsed: 0:00:42.
    Batch   240  of    241.    Elapsed: 0:00:51.

    Average training loss: 0.14
    Training epcoh took: 0:00:51

    Running Validation...
    Accuracy: 0.85
    Validation Loss: 0.55
    Validation took: 0:00:02

    Training complete!
    Total training took 0:03:30 (h:mm:ss)


### Example Sentences : 

### BART 

BART is a denoising autoencoder for pretraining sequence-to-sequence models. 
It is trained by 

    (1) corrupting text with an arbitrary noising function, and 

    (2) learning a model to reconstruct the original text. 
It uses a standard Transformer-based neural machine translation architecture. It uses a standard seq2seq/NMT architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). This means the encoder's attention mask is fully visible, like BERT, and the decoder's attention mask is causal, like GPT2.


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

  