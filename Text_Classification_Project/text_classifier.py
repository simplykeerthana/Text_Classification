# -*- coding: utf-8 -*-

import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

#importing the datasets 
reviews = load_files('txt_sentoken/')
X,y = reviews.data, reviews.target

# 0 is for negative, 1 is for positive


# presisting the dataset..... pickiling and unpickling the dataset


with open('X.pickle', 'wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle', 'wb') as f:
    pickle.dump(y,f)
    

X_in = open('X.pickle', 'rb')
y_in = open('y.pickle', 'rb')
X = pickle.load(X_in)
y = pickle.load(y_in)


# creating the corpus

corpus = []

for i in range(0, 2000):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)  
    
print(review)
    

#creating the bag of words model
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

print(X)

# Creating the Tf-Idf model

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

print(X)

# Creating the Tf-Idf model directly
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

print(X)

#training the data set
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#training our classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train, sent_train)

#testitng model performance 
sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)

print(cm)

#saving our classifier 
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
#saving the Tf-Idf model

with open('tdidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
#unpickling the classfier and vectorizer
    
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open('tdidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
sample = ["you are a nice person man, have a good life"]
sample = tfidf.transform(sample).toarray()


if clf.predict(sample) == 1:
    print('positive')
else:
    print('negative')

