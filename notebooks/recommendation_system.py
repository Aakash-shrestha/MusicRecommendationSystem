#!/usr/bin/env python
# coding: utf-8

# In[128]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import nltk #natural language toolkit, for tokenization


# In[129]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# In[130]:


spotify_data = pd.read_csv('../data/spotify_millsongdata.csv')
# spotify_data.head()
small_spotify_data = spotify_data.iloc[:10000].copy()
small_spotify_data.head()


# In[131]:


nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(stop_words)


# In[132]:


#check for any null entries
small_spotify_data.isna().sum().sort_values(ascending=True)


# In[133]:


def clean_text(text):
    text = str(text).lower() #lower case
    text = re.sub(r'\r\n', ' ', text) #remove new lines
    text = re.sub(r'[^a-z\s]', '', text) #remove special chars/numbers
    tokens = text.split() #tokenize
    tokens = [t for t in tokens if t not in stop_words] #remove unwanted stopwords
    return ' '.join(tokens)


# In[134]:


#clean the spotify dataframe
small_spotify_data['cleaned_lyrics'] = spotify_data['text'].apply(clean_text) #apply clean_text function to clean the data
small_spotify_data[['song', 'cleaned_lyrics']].head(10)


# 
# TF-IDF
# Tf : Term Frequency - how often does the word appears in the document
# IDF : Inverse Document Frequency - how rare that word is across all songs
# <br>
# $TF-IDF = TF * IDF$ => It gives high weight to words that are important to one song but not common in all song

# In[135]:


#TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(small_spotify_data['cleaned_lyrics'])
X_tfidf.shape
print(X_tfidf.toarray())


# In[136]:


#PCA dimension reduction before kmeans clustering
X_pca = PCA(n_components=50).fit_transform(X_tfidf)


# In[137]:


#elbow to find inertia
ks = range(1,20)
inertias = []
for k in ks:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters k')
plt.ylabel('inertias')
plt.xticks(ks)
plt.show()
    
    


# In[138]:


#KMeans clustering
kmeans = KMeans(n_clusters=8, random_state=7)
kmeans.fit(X_pca)
small_spotify_data['cluster'] = kmeans.predict(X_pca)

#view the clusters
small_spotify_data[['artist', 'song', 'cluster']].head(10)


# In[139]:


#Cosine similarity-based recommendation
cos_sim = cosine_similarity(X_pca)

#define a function that returns the recommendation  : main algooo
def recommend(index, no_recommendations=5): #no_recommendations: how many songs to recommend
    sim_scores = list(enumerate(cos_sim[index])) #converts enumerate object (index, value) into list : index: song_index, value: cos_sim score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #key specifies on what basis the sorting should be done, here tuple's second index, ie. score
    top_indices = [i for i, score in sim_scores[1:no_recommendations+1]]
    return spotify_data.iloc[top_indices][['artist', 'song']]


# In[140]:


#dimensionality reduction using TSNE
n = small_spotify_data.shape[0] #no of rows
tsne = TSNE(learning_rate=n/12) #rule of thumb for learning_rate hyperparameter

X_tsne = tsne.fit_transform(X_pca)

#Add x_tsne to spotify_data dataframe
small_spotify_data['xs'] = X_tsne[:,0]
small_spotify_data['ys'] = X_tsne[:,1]


# In[141]:


#test of recommendation system hahah
small_spotify_data.iloc[200][['artist', 'song']]
# recommend(200)


# In[142]:


small_spotify_data.head()


# In[145]:


#plot the giraff
plt.figure(figsize=(5,3))
sns.scatterplot(data=small_spotify_data, x='xs', y='ys', hue='cluster', palette='colorblind')
plt.title('t-SNE Visualization of Lyrics Clusters')
plt.show()


# In[144]:


#silhouette_score for measuring cluster quality
from sklearn.metrics import silhouette_score
score = silhouette_score(X_pca, kmeans.labels_)
print("Silhouette Score:", score)


# very bad score!!! 😭 should be at least 0.2 gagagga
