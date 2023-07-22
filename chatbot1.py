#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


#Data set

df = pd.read_csv("jokes.csv")


# In[3]:


df.info


# In[4]:


#Show Data at head

df.head(10)


# 
# # Cleaning the dataset:

# In[5]:


#Drop ID colomn

df=df.drop(['ID'], axis=1)


# In[6]:


#Show Data at Random

df.sample(10)


# In[7]:



nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[8]:


#removing special characters, numbers, and stopwords

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    return text


# In[9]:


df['cleaned_Question'] = df['Question'].apply(clean_text)
df['cleaned_Answer'] = df['Answer'].apply(clean_text)


# In[11]:


df.sample(10)


# In[14]:


# Vectorize the text 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_Question'])
y = vectorizer.transform(df['cleaned_Answer'])


# In[19]:


#Training the chatbot
def generate_response(user_input):
    user_input = clean_text(user_input)
    user_input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vector, X).flatten()
    index = np.argmax(similarities)
    return df['cleaned_Answer'][index]


# In[ ]:


# Testing the chatbot
while True:
    user_input = input('You: ')
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input)
    print('Chatbot:', response)


# In[ ]:




