# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:42:54 2022

@author: stimp
"""

#Making necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Reading the data
st = pd.read_csv("Fake.csv")

#Formatting the data
st.shape
st.head()

#Labelling to specify data easily
labels=st.label
labels.head()

#Splitting the dataset
x_train,x_test,y_train,y_test=train_test_split(st['text'], labels, test_size=0.2, random_state=7)

#Initiating vectorization (With stop word/common words with a max ratio of 0.7)
stidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Training and testing the data set with our program
stidf_train=stidf_vectorizer.fit_transform(x_train) 
stidf_test=stidf_vectorizer.transform(x_test)

#PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(stidf_train,y_train)

#Predicting data test set and getting the accuracy
y_pred=pac.predict(stidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Confusion matrix for false and true negatives, and positives
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

#Output
print(confusion_matrix(y_test, y_pred))