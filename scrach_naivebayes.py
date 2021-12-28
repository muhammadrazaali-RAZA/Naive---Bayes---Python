#!/usr/bin/env python
# coding: utf-8

# In[31]:


#IMPORT ALL NECCESSARY PACKAGES
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')


# In[32]:


import glob, os
os.chdir("C:/Users/Raza Jutt/Desktop/txt_sentoken/pos/")
txt_pos = []
for file in glob.glob("*.txt"):
    txt_pos.append(file)
    
dat = []
for i in txt_pos:
    tx = open("C:/Users/Raza Jutt/Desktop/txt_sentoken/pos/"+i, "r")
    dat.append([tx.read(),1])
    
txt_neg = []
os.chdir("C:/Users/Raza Jutt/Desktop/txt_sentoken/neg/")
for file in glob.glob("*.txt"):
    txt_neg.append(file)
    
for i in txt_neg:
    tx = open("C:/Users/Raza Jutt/Desktop/txt_sentoken/neg/"+i, "r")
    dat.append([tx.read(),0])
    
txt_neg = txt_pos = []


# In[33]:


data = pd.DataFrame(dat, columns = ['review' , 'tag'])
data = data.sample(frac = 1)
dat = txt_neg = txt_pos = []


# In[34]:


import re
from nltk.corpus import stopwords


# In[35]:


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


# In[36]:


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


# In[37]:


data['cleaned_review']=final_string


# In[38]:


def posneg(x):
    if x==0:
        return 0
    elif x==1:
        return 1
    return x

filtered_score = data["tag"].map(posneg)
data["score"] = filtered_score


# In[39]:


train_data = data[:]

X_train = train_data["cleaned_review"]
y_train = train_data["score"]

y_train=y_train.astype('int')


# In[40]:


#TF_IDF

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
tf_idf_train = tf_idf_vect.fit_transform(X_train.values)


# In[41]:


tf_data = pd.DataFrame(tf_idf_train.todense())
X = tf_data.to_numpy()
y = y_train.to_numpy()


# In[27]:


tf_data= tf_idf_train = X_train = train_data = data = []
#np.savetxt('data_numbers.txt', X, delimiter =', ')
#np.savetxt('label_number.txt', y)


# In[43]:


#tf_data.to_csv('data.csv', index=False)


# In[28]:


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# In[29]:


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




