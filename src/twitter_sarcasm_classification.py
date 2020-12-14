import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#Initialize Lemmatizer
lm = WordNetLemmatizer()

#Process Training Data
train_data = pd.read_json('data/train.jsonl', lines = True)
train_responses = train_data['response']
train_labels = train_data['label']
train_responses_updated = []

for train_response in train_responses:
    train_response = train_response.replace('@USER','')
    train_response = ' '.join([lm.lemmatize(y) for y in train_response.split()])
    train_response = train_response.lstrip().rstrip()
    #print(train_response)
    train_responses_updated.append(train_response)


#Initialize the TfidfVectorizer with appropriate parameters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
stop_words_list = text.ENGLISH_STOP_WORDS
tv = TfidfVectorizer(max_features=50, max_df=0.1, strip_accents = 'ascii', sublinear_tf= True, ngram_range=(2,3), stop_words=set(stop_words_list))

#Convert the responses into the format required for the fit function
train_X = tv.fit_transform(train_responses_updated).toarray()

#lr = LogisticRegression()
#lr.fit(train_X, train_labels)

#Train the model using the fit function
gnb = GaussianNB()
gnb.fit(train_X,train_labels)

#Process Test Data
test_data = pd.read_json('data/test.jsonl', lines = True)
test_responses = test_data['response']
test_ids = test_data['id']
test_responses_updated = []

for test_response in test_responses:
    test_response = test_response.replace('@USER','')
    test_response = ' '.join([lm.lemmatize(y) for y in test_response.split()])
    test_response = test_response.lstrip().rstrip()
    #print(test_response)
    test_responses_updated.append(test_response)

#Convert the responses into the format required for the fit function
test_X = tv.fit_transform(test_responses_updated).toarray()
#test_labels = lr.predict(test_X)

#Make the predictions using the predict function
test_labels = gnb.predict(test_X)

# Write the predictions into the output file
i = 0
answer = open('data/answer.txt', "w")
for id in test_ids:
    answer.write(str(id))
    answer.write(",")
    answer.write(str(test_labels[i]))
    answer.write("\n")
    i = i+1