# -*- coding: utf-8 -*-
"""
演示内容：文档的向量化
"""
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
'Jobs was the chairman of Apple Inc., and he was very famous',
'I like to use apple computer',
'And I also like to eat apple'
] 
 
#未经停用词过滤的文档向量化
vectorizer =CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())  #转化为完整特征矩阵
print(vectorizer.vocabulary_)
 
print(" ")
 
 
#经过停用词过滤后的文档向量化
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print (stopwords)
print(" ")
vectorizer =CountVectorizer(stop_words='english')
print("after stopwords removal:  ", vectorizer.fit_transform(corpus).todense())
print("after stopwords removal:  ", vectorizer.vocabulary_)
 
print(" ")
#采用ngram模式进行文档向量化
vectorizer =CountVectorizer(ngram_range=(1,2))#表示从1-2，既包括unigram，也包括bigram
print("N-gram mode:     ",vectorizer.fit_transform(corpus).todense())  #转化为完整特征矩阵
print(" ")
print("N-gram mode:         ",vectorizer.vocabulary_)