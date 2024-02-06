#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


df=pd.read_csv("spam.csv",encoding='latin1')


# In[4]:


df.head()


# In[5]:


df=df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)


# In[6]:


df.rename({'v1':"Target",'v2':'mail'},inplace=True,axis=1)


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


import nltk
nltk.download("stopwords")
from nltk.stem import PorterStemmer
import string
tokill=string.punctuation


# In[10]:


from nltk.corpus import stopwords
sw = stopwords.words('english')


# In[11]:


ps=PorterStemmer()


# In[12]:


def data_cleaning(x):
    wordlist=[]
    for word in x.split():
        word=word.lower()
        if word not in sw:
            letterlist=[]
            for letter in word:
                if letter not in tokill:
                    letterlist.append(letter)
            stemword=ps.stem("".join(letterlist))
            wordlist.append(stemword)
    x=" ".join(wordlist)
    return x   


# In[13]:


df.head()


# In[14]:


df["Target"]=df["Target"].map({"ham":0,"spam":1})


# In[15]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(df["mail"],df["Target"],test_size=0.3,random_state=42)


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[17]:


# cv=CountVectorizer()
tf=TfidfVectorizer()


# In[18]:


xtrainout=tf.fit_transform(xtrain).toarray()


# In[19]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrainout,ytrain)


# In[20]:


xtestout=tf.transform(xtest)


# In[21]:


from sklearn.metrics import accuracy_score,classification_report
pred=lr.predict(xtestout)
print(accuracy_score(ytest,pred))
print(classification_report(ytest,pred))


# In[22]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(xtrainout,ytrain)
pred=rfc.predict(xtestout)
print(accuracy_score(ytest,pred))
print(classification_report(ytest,pred))


# In[23]:


cv=CountVectorizer()
xtrainout_2=cv.fit_transform(xtrain).toarray()
xtestout_2=cv.transform(xtest)
rfc=RandomForestClassifier()
rfc.fit(xtrainout_2,ytrain)
pred=rfc.predict(xtestout_2)
print(accuracy_score(ytest,pred))
print(classification_report(ytest,pred))


# In[24]:


cv=CountVectorizer()
lr=LogisticRegression()
xtrainout_2=cv.fit_transform(xtrain).toarray()
xtestout_2=cv.transform(xtest)
lr=LogisticRegression()
lr.fit(xtrainout_2,ytrain)
pred=lr.predict(xtestout_2)
print(accuracy_score(ytest,pred))
print(classification_report(ytest,pred))

