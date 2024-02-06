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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


sns.set_palette("husl", 8)
sns.set_style('darkgrid')


# In[5]:


df =  pd.read_csv('titanic1.csv')
df


# In[6]:


sns.countplot(x = 'Survived',data=df)


# In[7]:


sns.countplot(x = 'Survived',hue = 'Pclass',data=df)


# In[8]:


sns.countplot(x='Pclass',hue='Sex',data=df)


# In[9]:


sns.histplot(df['Age'],bins=40)


# In[10]:


sns.histplot(df['Fare'],bins=40)


# In[11]:


sns.pairplot(df,hue='Sex',palette='colorblind')


# In[12]:


sns.pairplot(df,hue='Pclass',palette='colorblind')


# In[13]:


df.isnull().sum()


# In[15]:


sns.heatmap(df.isnull(),cmap='plasma',cbar=True)


# In[16]:


df.drop(columns=['Cabin'],axis=1,inplace=True)


# In[17]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')


# In[18]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[19]:


df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)


# In[20]:


df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# In[21]:


df.isnull().sum()


# In[22]:


sns.heatmap(df.isnull(),yticklabels=False,cmap='plasma')


# In[23]:


df.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[24]:


df


# In[25]:


df_new = pd.get_dummies(df, columns=['Sex', 'Embarked'])


# In[26]:


df_new['Sex_female'] = df_new['Sex_female'].astype(int)
df_new['Sex_male'] = df_new['Sex_male'].astype(int)
df_new['Embarked_C'] = df_new['Embarked_C'].astype(int)
df_new['Embarked_Q'] = df_new['Embarked_Q'].astype(int)
df_new['Embarked_S'] = df_new['Embarked_S'].astype(int)


# In[27]:


df_new


# In[28]:


df_new.columns


# In[29]:


X = df_new[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female',
       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y = df_new['Survived']


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)


# In[32]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[33]:


from sklearn.neighbors import KNeighborsClassifier


# In[34]:


knn = KNeighborsClassifier(n_neighbors=10)


# In[35]:


knn.fit(X_train,y_train)


# In[36]:


pred_knn = knn.predict(X_test)


# In[37]:


print("Accuracy of Titanic Dataset using Knn = ",accuracy_score(y_test,pred_knn))


# In[38]:


print(classification_report(y_test,pred_knn))


# In[39]:


print(confusion_matrix(y_test,pred_knn))


# In[40]:


from sklearn.svm import SVC


# In[41]:


svc = SVC(kernel='rbf')


# In[42]:


svc.fit(X_train,y_train)


# In[43]:


pred_svc = svc.predict(X_test)


# In[44]:


print("Accuracy of Titanic Dataset using SVC = ",accuracy_score(y_test,pred_svc))


# In[45]:


print(classification_report(y_test,pred_svc))


# In[46]:


print(confusion_matrix(y_test,pred_svc))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




