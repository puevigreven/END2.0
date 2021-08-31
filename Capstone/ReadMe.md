### Dense Passage Retrieval
Dense Passage Retrieval (DPR) for ODQA was introduced in 2020 as an alternative to the traditional TF-IDF and BM25 techniques for passage retrieval.

The abstract from the paper is the following:

> Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks.

DPR is able to outperform the traditional sparse retrieval methods for two key reasons:

- Semantically similar words (“hey”, “hello”, “hey”) will not be viewed as a match by TF. DPR uses dense vectors encoded with semantic meaning (so “hey”, “hello”, and “hey” will closely match).
- Sparse retrievers are not trainable. DPR uses embedding functions that we can train and fine-tune for specific tasks.

### The Capstone Project consists of two part
1. Retriever
2. Generator

### 1. Retriever

DPR works by using two unique BERT encoder models. One of those models — Eᴘ — encodes passages of text into an encoded passage vector (we store context vectors in our document store).

The other model — EQ — maps a question into an encoded question vector.

During training, we feed a question-context pair into our DPR model, and the model weights will be optimized to maximize the dot product between two respective Eᴘ/EQ model outputs:

![alt Retriever](https://github.com/puevigreven/END2.0/blob/main/Capstone/qna.png)
A high-level view of the flow of data through a DPR model during training.
Source : https://towardsdatascience.com/how-to-create-an-answer-from-a-question-with-dpr-d76e29cc5d60



The dot product value between the two model outputs Eᴘ(p) and EQ(q) measures the similarity between both vectors. A higher dot product correlates to a higher similarity — because the closer two vectors are to each other, the larger the dot product.

By training the two models to output the same vector, we are training the context encoder and question encoder to output very similar vectors for related question-context pairs.

At Runtime
Once the model (or two models) have been trained, we can begin using them for Q&A indexing and retrieval.

When we first build our document store, we need to encode the data we store in there using the Eᴘ encoder — so during document store initialization (or when adding new documents) — we run every passage of text through the Eᴘ encoder and store the output vectors in our document store.

For real-time Q&A, we only need the EQ encoder. When we ask a question, it will be sent to the EQ encoder which then outputs our EQ vector EQ(q).

Next, the EQ(q) vector is compared against the already indexed Eᴘ(p) vectors in our document store — where we filter for the vectors which return the highest similarity score:

sim(q,p) = EQ(q)ᵀ Eᴘ(p)

And that’s it! Our retriever has identified the most relevant contexts for our question.


### 2. Generator
Generator, after getting the documents, along with the query, generates the answer to the query, maximizing the probability p(y|x,z) or minimizing the log-likelihood of this probability.
![alt Retriever](https://github.com/puevigreven/END2.0/blob/main/Capstone/bart.png)


BART is used as the generator model. This takes as input the documents (passed on from retriever) concatenated together, pre-pended with the query, generates the answer token by token, minimizing the log-likelihood of p(y|x,z). So this BART generator works as described below: So e.g. three documents, z1, z2 and z3 get selected given the query x, maximizing the probability p(z|x). So now we have three latent documents for the query x : (x,z1), (x,z2) and (x,z3). Now, lets say BART model generates sequence for each of the latent documents, say, y11, y12 for z1, y21, y22 for z2 and y31, y32 for z3. Now we have 6 hypotheses to test. So we calculate the probability for each of these for each of the documents. So for example, for y11,  This is done for each yij, i=1,2,3 and j=1,2 to get the probability p(yij|x). Maximum value of this is returned.


Why FAISS?
Faiss is a library — developed by Facebook AI — that enables efficient similarity search.

So, given a set of vectors, we can index them using Faiss — then using another vector (the query vector), we search for the most similar vectors within the index.

Now, Faiss not only allows us to build an index and search — but it also speeds up search times to ludicrous performance levels. Lets first what all indexing options are available and what they do. a) IndexFlatL2 measures the L2 (or Euclidean) distance between all given points between our query vector, and the vectors loaded into the index. It’s simple, very accurate, but not too fast.

Faiss allows us to add multiple steps that can optimize our search using many different methods. A popular approach is to partition the index into Voronoi cells.




