#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT ALL NECCESSARY PACKAGES
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection

import nltk
nltk.download('stopwords')


# In[2]:


import glob, os
os.chdir("C:/Users/Raza Jutt/Desktop/txt_sentoken/pos/")
txt_pos = []
for file in glob.glob("*.txt"):
    txt_pos.append(file)


# In[3]:


dat = []
for i in txt_pos:
    tx = open("C:/Users/Raza Jutt/Desktop/txt_sentoken/pos/"+i, "r")
    dat.append([tx.read(),1])


# In[4]:


txt_neg = []
os.chdir("C:/Users/Raza Jutt/Desktop/txt_sentoken/neg/")
for file in glob.glob("*.txt"):
    txt_neg.append(file)
    
for i in txt_neg:
    tx = open("C:/Users/Raza Jutt/Desktop/txt_sentoken/neg/"+i, "r")
    dat.append([tx.read(),0])


# In[5]:


data = pd.DataFrame(dat, columns = ['review' , 'tag'])


# In[6]:


data = data.sample(frac = 1)


# In[7]:


data.columns


# In[8]:


data


# In[9]:


len(data)


# In[10]:


data.head()


# In[11]:


import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english')) 
sno = nltk.stem.SnowballStemmer('english') 

def cleanhtml(sentence): 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned


# In[12]:


i=0
str1=' '
final_string=[]
all_positive_words=[] 
all_negative_words=[] 
s=''
for sent in data['review'].values:
    filtered_sentence=[]
    sent=cleanhtml(sent) 
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (data['tag'].values)[i] == 1: 
                        all_positive_words.append(s) 
                    if(data['tag'].values)[i] == 0:
                        all_negative_words.append(s) 
                else:
                    continue
            else:
                continue 
    
    str1 = b" ".join(filtered_sentence) 
    
    final_string.append(str1)
    i+=1


# In[13]:


data['cleaned_review']=final_string


# In[14]:


def posneg(x):
    if x==0:
        return 0
    elif x==1:
        return 1
    return x

filtered_score = data["tag"].map(posneg)
data["score"] = filtered_score


# In[35]:


test_data = data[:300]
train_data = data[300:]


# In[36]:


X_train = train_data["cleaned_review"]
y_train = train_data["score"]

X_test = test_data["cleaned_review"]
y_test = test_data["score"]


# In[37]:


y_train=y_train.astype('int')
y_test=y_test.astype('int')


# In[38]:


#TF_IDF

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
tf_idf_test = tf_idf_vect.transform(X_test.values)


# In[39]:


alpha_range = list(np.arange(1,50,5))
len(alpha_range)


# In[40]:


from sklearn.naive_bayes import MultinomialNB

alpha_scores=[]

for a in alpha_range:
    clf = MultinomialNB(alpha=a)
    scores = cross_val_score(clf, tf_idf_train, y_train, cv=5, scoring='accuracy')
    alpha_scores.append(scores.mean())
    print(a,scores.mean())


# In[41]:


MSE = [1 - x for x in alpha_scores]


optimal_alpha_bnb = alpha_range[MSE.index(min(MSE))]

# plot misclassification error vs alpha
plt.plot(alpha_range, MSE)

plt.xlabel('hyperparameter alpha')
plt.ylabel('Misclassification Error')
plt.show()

# In[42]:


optimal_alpha_bnb


# In[43]:


clf = MultinomialNB(alpha=6)
clf.fit(tf_idf_train,y_train)


# In[44]:


y_pred_test = clf.predict(tf_idf_test)


# In[45]:


acc = accuracy_score(y_test, y_pred_test, normalize=True) * float(100)
print('\n****Test accuracy is',(acc))


# In[46]:


cm_test = confusion_matrix(y_test,y_pred_test)
cm_test


# In[47]:


import seaborn as sns

sns.heatmap(cm_test,annot=True,fmt='d')


# In[48]:


y_pred_train = clf.predict(tf_idf_train)


# In[49]:


acc = accuracy_score(y_train, y_pred_train, normalize=True) * float(100)
print('\n****Train accuracy is %d%%' % (acc))


# In[ ]:





# In[ ]:




