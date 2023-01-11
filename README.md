### Summary

 Natural Language Processing (NLP) is the subfield of Artificial Intelligence (AI) wich combines Linguistics and Computer Science. It gives computers the ability to understand text and spoken words in much the same way as human beings. Also referred to as Computational Linguistics, its implementation involves the combination of Statisctics, Machine Learning an Deep Learning. In real life, it has many applications including sentiment analysis, language translation,  voice-operated GPS systems, digital assistants, speech-to-text dictation software, customer service chatbots and many more. In this project, I used Naive Bayes model (a machine learning algorithm) to classify spam messages. The model is 95% accurate. It was developped in **Python**. 
 


```python
import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
```


```python
## Reading the given dataset
spam = pd.read_csv("attachment_SMSSpamCollection_lyst7328.txt", sep = "\t", names=["label", "message"])
```


```python
print(spam.head())
```

      label                                            message
    0   ham  Go until jurong point, crazy.. Available only ...
    1   ham                      Ok lar... Joking wif u oni...
    2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
    3   ham  U dun say so early hor... U c already then say...
    4   ham  Nah I don't think he goes to usf, he lives aro...
    


```python
## Converting the read dataset into a list of tuples, each tuple(row) containing the message and it's label
data_set = []
for index,row in spam.iterrows():
    data_set.append((row['message'], row['label']))
```


```python
#print(data_set[:5])
```

    [('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...', 'ham'), ('Ok lar... Joking wif u oni...', 'ham'), ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", 'spam'), ('U dun say so early hor... U c already then say...', 'ham'), ("Nah I don't think he goes to usf, he lives around here though", 'ham')]
    


```python
print(len(data_set))
```

    5572
    

### Preprocessing


```python
## initialise the inbuilt Stemmer and the Lemmatizer
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
```


```python
def preprocess(document, stem=True):
    'changes document to lower case, removes stopwords and lemmatizes/stems the remainder of the sentence'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    if stem:
        words = [stemmer.stem(word) for word in words]
    else:
        words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

    # join words to make sentence
    document = " ".join(words)

    return document
```


```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\hleon\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\hleon\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
## - Performing the preprocessing steps on all messages
messages_set = []
for (message, label) in data_set:
    words_filtered = [e.lower() for e in preprocess(message, stem=False).split() if len(e) >= 3]
    messages_set.append((words_filtered, label))
```


```python
#print(messages_set[:5])
```

    [(['jurong', 'point', 'crazy', 'available', 'bugis', 'great', 'world', 'buffet', '...', 'cine', 'get', 'amore', 'wat', '...'], 'ham'), (['lar', '...', 'joke', 'wif', 'oni', '...'], 'ham'), (['free', 'entry', 'wkly', 'comp', 'win', 'cup', 'final', 'tkts', '21st', 'may', '2005.', 'text', '87121', 'receive', 'entry', 'question', 'std', 'txt', 'rate', 'apply', '08452810075over18'], 'spam'), (['dun', 'say', 'early', 'hor', '...', 'already', 'say', '...'], 'ham'), (['nah', "n't", 'think', 'usf', 'live', 'around', 'though'], 'ham')]
    

### Preparing to create features


```python
## - creating a single list of all words in the entire dataset for feature list creation

def get_words_in_messages(messages):
    all_words = []
    for (message, label) in messages:
      all_words.extend(message)
    return all_words
```


```python
## - creating a final feature list using an intuitive FreqDist, to eliminate all the duplicate words
## Note : we can use the Frequency Distribution of the entire dataset to calculate Tf-Idf scores like we did earlier.

def get_word_features(wordlist):

    #print(wordlist[:10])
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features
```


```python
## - creating the word features for the entire dataset
word_features = get_word_features(get_words_in_messages(messages_set))
print(len(word_features))
```

    8003
    

### Preparing to create a train and test set


```python
## - creating slicing index at 80% threshold
sliceIndex = int((len(messages_set)*.8))
```


```python
## - shuffle the pack to create a random and unbiased split of the dataset
random.shuffle(messages_set)
```


```python
train_messages, test_messages = messages_set[:sliceIndex], messages_set[sliceIndex:]
```


```python
len(train_messages)

```




    4457




```python
len(test_messages)
```




    1115



### Preparing to create feature maps for train and test data


```python
## creating a LazyMap of feature presence for each of the 8K+ features with respect to each of the SMS messages
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
```


```python
## - creating the feature map of train and test data

training_set = nltk.classify.apply_features(extract_features, train_messages)
testing_set = nltk.classify.apply_features(extract_features, test_messages)
```


```python
print('Training set size : ', len(training_set))
print('Test set size : ', len(testing_set))
```

    Training set size :  4457
    Test set size :  1115
    

### Training


```python
## Training the classifier with NaiveBayes algorithm
spamClassifier = nltk.NaiveBayesClassifier.train(training_set)
```

### Evaluation


```python
## - Analyzing the accuracy of the test set
print(nltk.classify.accuracy(spamClassifier, training_set))
```

    0.9921471842046219
    


```python
## Analyzing the accuracy of the test set
print(nltk.classify.accuracy(spamClassifier, testing_set))
```

    0.9775784753363229
    


```python
## Testing a example message with our newly trained classifier
m = 'CONGRATULATIONS!! As a valued account holder you have been selected to receive a £900 prize reward! Valid 12 hours only.'
print('Classification result : ', spamClassifier.classify(extract_features(m.split())))
```

    Classification result :  spam
    


```python
## Priting the most informative features in the classifier
print(spamClassifier.show_most_informative_features(50))
```

    Most Informative Features
             contains(award) = True             spam : ham    =    215.4 : 1.0
           contains(service) = True             spam : ham    =    113.8 : 1.0
             contains(nokia) = True             spam : ham    =    104.1 : 1.0
             contains(await) = True             spam : ham    =     98.3 : 1.0
            contains(urgent) = True             spam : ham    =     94.1 : 1.0
              contains(code) = True             spam : ham    =     89.9 : 1.0
               contains(txt) = True             spam : ham    =     70.8 : 1.0
          contains(delivery) = True             spam : ham    =     69.0 : 1.0
          contains(landline) = True             spam : ham    =     66.5 : 1.0
              contains(club) = True             spam : ham    =     64.8 : 1.0
           contains(private) = True             spam : ham    =     60.6 : 1.0
              contains(quiz) = True             spam : ham    =     56.5 : 1.0
           contains(attempt) = True             spam : ham    =     54.0 : 1.0
               contains(100) = True             spam : ham    =     52.3 : 1.0
            contains(reveal) = True             spam : ham    =     52.3 : 1.0
         contains(statement) = True             spam : ham    =     52.3 : 1.0
            contains(camera) = True             spam : ham    =     48.9 : 1.0
               contains(opt) = True             spam : ham    =     48.1 : 1.0
            contains(latest) = True             spam : ham    =     47.5 : 1.0
              contains(rate) = True             spam : ham    =     46.4 : 1.0
            contains(caller) = True             spam : ham    =     43.9 : 1.0
            contains(orange) = True             spam : ham    =     42.1 : 1.0
             contains(final) = True             spam : ham    =     41.4 : 1.0
            contains(mobile) = True             spam : ham    =     39.7 : 1.0
            contains(txting) = True             spam : ham    =     39.7 : 1.0
           contains(voucher) = True             spam : ham    =     39.7 : 1.0
              contains(draw) = True             spam : ham    =     36.9 : 1.0
             contains(apply) = True             spam : ham    =     35.6 : 1.0
          contains(sunshine) = True             spam : ham    =     35.6 : 1.0
           contains(receive) = True             spam : ham    =     34.8 : 1.0
              contains(info) = True             spam : ham    =     33.9 : 1.0
               contains(mat) = True             spam : ham    =     33.9 : 1.0
               contains(box) = True             spam : ham    =     33.7 : 1.0
             contains(music) = True             spam : ham    =     33.2 : 1.0
          contains(customer) = True             spam : ham    =     33.0 : 1.0
            contains(select) = True             spam : ham    =     32.8 : 1.0
              contains(comp) = True             spam : ham    =     31.4 : 1.0
            contains(flight) = True             spam : ham    =     31.4 : 1.0
              contains(user) = True             spam : ham    =     31.4 : 1.0
               contains(top) = True             spam : ham    =     29.6 : 1.0
          contains(motorola) = True             spam : ham    =     28.9 : 1.0
            contains(player) = True             spam : ham    =     28.9 : 1.0
              contains(cash) = True             spam : ham    =     28.1 : 1.0
           contains(contact) = True             spam : ham    =     27.6 : 1.0
              contains(line) = True             spam : ham    =     27.5 : 1.0
               contains(dvd) = True             spam : ham    =     27.2 : 1.0
             contains(pound) = True             spam : ham    =     27.2 : 1.0
            contains(summer) = True             spam : ham    =     27.2 : 1.0
           contains(network) = True             spam : ham    =     25.6 : 1.0
               contains(win) = True             spam : ham    =     24.0 : 1.0
    None
    


```python
## storing the classifier on disk for later usage
import pickle
f = open('nb_spam_classifier.pickle', 'wb')
pickle.dump(spamClassifier,f)
print('Classifier stored at ', f.name)
f.close()
```

    Classifier stored at  nb_spam_classifier.pickle
    
