# SMS-Spam-Detection
This model detects SMS spams using Naive Bayes. It is based on the Bayes theorem and it learns parameters by observing each feature independently, regardless of all the other features in the data set and derive statistics from each class for each class. Multinomial Naive Bayes classifier uses a multinomial distribution for each one of the features generated on data. This system uses Multinomial Naive Bayes classifier for spam detection.

# Python libraries used
Pandas, numpy, matplotlib, seaborn, nltk and sklearn.

# Dataset Used
The dataset is taken from Kaggle. The dataset consists of 5572 tuples. Each tuple has two fields 
1. Label indicating ham or spam 
2. The message.

| Label | Count |
| ------------- | ------------- |
| Ham | 4825 |
| Spam | 747  |
| Total | 5572 |


# Data Preprocessing
The different data preprocessing operations performed are:
1. Punctuation removal: : <br />
All punctuations are removed from the messages. Punctuations marks are not required for classifying a message as spam or not. Hence, they must be removed to reduce the processing.

2. Stop word removal: <br />
Stop words refers to the most common words in a language. They do not play any significant role in analysis and hence they must be removed. Eg: "a", "and", "but", "how", "or", "what", etc. 

3. Stemming: <br />
Stemming algorithms work by cutting off the end or the beginning of the word, taking into account a list of common prefixes and suffixes that can be found in an inflected word. Stemming is used to get the root word of a particular. Root word can be used further for frequency analysis of the corpus.

# Vector Transformation
The classification algorithms need some sort of numerical feature vector in order to perform the classification task. There are actually many methods to convert a corpus to a vector format. The simplest is the the bag-of-words approach, where each unique word in a text will be represented by one number. Vector Transformation converts the message into 2-D matrix. One dimension represents the document while the other dimension represents each unique word in message corpus. Here TF-IDF vectorizer is used for this transformation.
TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. <br />
<br />
Term Frequency, TF(t) = Number of times term t appears in document p / Total number of terms in document p <br />
Inverse Document Frequency, IDF(t) = loge(Total number of documents / Number of documents with term t in it) <br />
TF-IDF(t) = TF(t) * IDF(t) <br />

# Result
Word cloud for Spam messages <br />
<img src="https://user-images.githubusercontent.com/30766392/78825891-4c7aed00-79fe-11ea-93f8-e4edc533f60a.png" width=50%/>

Confusion matrix for the model <br />
<img src="https://user-images.githubusercontent.com/30766392/78826193-b8f5ec00-79fe-11ea-9a4f-8900e29f5b85.png"/>

For model evaluation SciKit Learn's built-in classification report is used, which returns precision, recall, f1-score, and a column for support (meaning how many cases supported that classification). <br />
<img src="https://user-images.githubusercontent.com/30766392/78826443-1db14680-79ff-11ea-8040-06edb5a8c530.png"/>

For the model total number of test cases are 1115 and the number of wrong of predictions made are 39.

