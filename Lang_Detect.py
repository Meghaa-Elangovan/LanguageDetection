import pandas as pd
import numpy as np

# Importing the dataset
df=pd.read_csv("D:\Dataset\Language Detection.csv")

# Splitting the feature and the target column
X=df['Text']
Y=df['Language']

# Preprocessing
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(Y)
df.dropna()

# Data cleaning
l=[]
import re
for i in X:
    i = re.sub('[!@#$(),"%^*?:;~`0-9]', ' ', i)
    i = re.sub ('[[]]',' ',i)
    i=i.lower()
    l.append(i)

# Using count vectorizer to create bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X = cv.fit_transform(l).toarray()

# Train-Test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20)

# Importing the model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
ac=accuracy_score(y_test, y_pred)
cm=confusion_matrix(y_test, y_pred)

def predict(text):
    text = re.sub('[!@#$(),"%^*?:;~`0-9]', ' ', text)
    text = re.sub ('[[]]',' ',text)
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The langauge is in",lang)
    
z=predict("എന്നിരുന്നാലും, കർണാടകയിലെയും തമിഴ്‌നാട്ടിലെയും" )

