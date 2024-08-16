# nlp-with-disaster-tweets

----------------------
## 1. Description
### About
In times of emergencies, Twitter has become an essential platform for real-time communication. The widespread use of smartphones allows individuals to report emergencies as they occur. As a result, various organizations, such as disaster relief agencies and news outlets, are increasingly interested in automatically monitoring Twitter. However, it can be challenging to determine whether a tweet genuinely reports a disaster or if it is using language figuratively.

### Objective
The primary objective is to train a model that can accurately differentiate between genuine disaster-related tweets and others that may be misleading or metaphorical. This process will begin with an exploration of text mining techniques and gradually introduce more complex concepts, ranging from N-grams to word embeddings and transformers.

----------------------

The size of train data: 7613
The size of test data: 3263
Target counts:
0|4342
1|3271
## 2. EDA
![alt text](./image/image.png)
![alt text](./image/image-2.png)
![alt text](./image/image-3.png)
- Compare the distributions of text lengths between the train and test datasets.
![alt text](./image/image-1.png)
### Missing 
- The location variable have significant missing values.
![alt text](./image/image-4.png)
![alt text](./image/image-5.png)
### Text preprocessing
- Text preprocessing is a crucial step in preparing raw text data. This process involves several steps, such as converting text to lowercase, removing URLs, mentions, hashtags, and punctuation, and tokenizing the text. 
- visualize word frequencies and analyze bigrams (pairs of consecutive words) and trigrams (triplets of consecutive words). These visualizations help in identifying common terms and phrases that might be important for distinguishing disaster-related tweets from non-disaster-related ones.
![alt text](./image/image-6.png)
![alt text](./image/image-7.png)
![alt text](./image/image-8.png)

## 3. Architecture
### Choosing a Word Embedding Method
There are several methods for converting text to word embeddings: TF-IDF, GloVe, and Word2Vec. Let's briefly compare these methods and their suitability for this particular problem.
#### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
- Description: TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It is calculated by multiplying the term frequency (TF) of a word by its inverse document frequency (IDF).
- Advantages:
  - Simple and easy to implement.
  - Effective for tasks where the frequency of terms is important.
- Disadvantages:
  - Creates sparse vectors, which can be computationally expensive.
  - Does not capture the semantic meaning of words.
#### 2. Word2Vec
- Description: Word2Vec is a neural network-based approach that learns dense vector representations for words by predicting neighboring words in a context window. It uses two models: Continuous Bag of Words (CBOW) and Skip-Gram.
- Advantages:
  - Produces dense, low-dimensional vectors.
  - Captures semantic relationships between words.
  - Efficient for large datasets.
- Disadvantages:
  - Requires significant computational resources for training on large corpora.
  - Context size is fixed, which can limit its ability to capture long-range dependencies.
#### 3. GloVe (Global Vectors for Word Representation)
- Description: GloVe is a count-based method that generates word embeddings by factorizing a word co-occurrence matrix. It combines the advantages of both word count statistics and prediction-based models like Word2Vec.
- Advantages:
  - Produces dense, low-dimensional vectors.
  - Captures both local and global statistical information.
  - Pre-trained models are available, which saves computational resources.
- Disadvantages:
  - Requires a large corpus to generate high-quality embeddings.
  - Fixed-size vectors may not capture nuanced context as well as dynamic models like BERT.
Use Case: Suitable for tasks where capturing both local and global context is important. Pre-trained models make it a good choice for projects with limited computational resources.
#### Conclusion
I'll use GloVe embeddings. GloVe strikes a good balance between capturing semantic relationships and computational efficiency. Pre-trained GloVe embeddings can be easily integrated into your model, allowing me to leverage high-quality word vectors without the need for extensive computational resources.

### Choosing the Model Architecture
#### LSTM
- Handling Sequential Data:
  - Tweets are sequences of words, and LSTMs are designed to handle sequential data effectively, capturing dependencies between words.
- Long-Term Dependencies:
  - LSTMs are capable of learning long-term dependencies, which are crucial in understanding the context of a tweet. And LSTMs can capture such relationships better than traditional RNNs.
- Avoiding the Vanishing Gradient Problem:
  - Traditional RNNs suffer from the vanishing gradient problem, where gradients diminish as they are backpropagated through time, leading to poor learning of long-range dependencies. LSTMs address this issue with gating mechanisms that control the flow of information, making them more effective for learning from longer sequences.
- Handling Variable-Length Sequences:
  - Tweets can vary in length, and LSTMs can handle variable-length sequences efficiently. The gating mechanism in LSTMs helps decide what information to keep or discard, making them robust to variations in input length.
![alt text](./image/image-9.png)

## 4. Results and Analysis
### Hyperparameter Bayesian Optimization
I employed Bayesian Optimization to fine-tune the hyperparameters, particularly learning rate and batch size. The optimization process aimed to maximize the F1 score on the validation set.
![alt text](./image/image-10.png)
### Classification Report
![alt text](./image/image-11.png)
### Best Hyperparameters
![alt text](./image/image-12.png)

## 5. Conclusion
### In this notebook:
- Performed exploratory data analysis (EDA) to understand the dataset.
- Preprocessed the text data, including cleaning and tokenization.
- Created word embeddings using GloVe.
- Defined and trained an LSTM model for binary classification.
- Evaluated the model using F1 score, confusion matrix, and accuracy.
### Next steps:
- Explore more advanced architectures such as transformer-based models.
- Use more sophisticated text augmentation techniques to improve the model.

## 6. Submission
![alt text](./image/image-13.png)