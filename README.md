# Tasty Tweets
Topic modeling and sentiment analysis of popular food and beverage chains on Twitter using Natural Language Processing and Machine Learning

## Project Introduction
* What are our customers saying about our brand on social media?  
* Is it positive, negative, or neutral?
* Are some aspects of our brand held in higher regard than others? (i.e. food, service, experience, etc.)
* Is the conversation a short blip or a persistent trend?

The focus of this project is to answer the questions above for three popular food and beverage brands: Chipotle Mexican Grill, Starbucks, and McDonalds.  The overall goal is to create a framework to answer these questions for any popular brand, product, or topic.

#### Methodologies
* Unsupervised learning to cluster tweets into latent topics
    * **Latent Semantic Analysis** - This method will use tf-idf document representation models.
    * **Latent Dirichlet Allocation** - This method will also use a term frequency vectorizer.
    * **Non-Negative Matrix Factorization** - This method will also use tf-idf.
    * **Deep Autoencoder Topic Model  [(DATM)](https://www.prhlt.upv.es/workshops/iwes15/pdf/iwes15-kumar-d'haro.pdf "DATM")** - This method will make use of word embeddings to relate words used in similar context with one another.
* Supervised and semi-supervised learning to classify sentiment of tweets in relation to brand
    * **Supervised** - Hand label a subset of tweets and use various machine learning algorithms to classify sentiment .
    * **Semi-supervised** - Read through the latent topics identified by the unsupervised models and determine if topics can be divided into generally positive, generally negative, and no sentiment.  Then use the supervised approach detailed above.

## The Process
My end to end process to extract insights from tweets was extensive.  In this section I'll explain the different steps I took.

#### Accessing Twitter's API
The first thing I did was begin pulling and saving tweets from Twitter's API.  I used a python Twitter API wrapper called Tweepy to make this easier.  Twitter limits the amount of calls you can make on their API to roughly 3,000 tweets every 15 minutes.  

#### Cleaning the Tweets
Tweets are notoriously messy text.  Computers don't like messy data, so cleaning the tweets was a big challenge.  Here are some of the issues I encountered:
* Misspellings, purposeful and not, are extremely common   
* Slang, acronyms and abbreviations are rampant
* Emojis are used widely and are represented in unicode in the raw tweets
* Links to webpages and images are common

To counter the issues above I took a number of steps:
* I created a slang translator to convert common slang terms and acronyms to plain english **ex: gonna -> going to, rn -> right now**
* I also utilized an emoji translator to convert emoji unicode to an english description of the emoji
* Lastly I removed links to webpages and pictures

#### Lemmatization and Removing Stop Words
* **Lemmatization** is the process of representing each word as the base dictionary form of that word *craving, craved -> crave*.
* **Stop Words** are words that make sentences grammatically correct, but don't usually change the meaning of the tweet. For instance *i need to get starbucks or i might die* has almost equivalent meaning to *need starbucks die*.  I removed the stop words in order to make my data less sparse, and thus easier to model.

#### Term Frequency - Inverse Document Frequency (tf-idf)

Tf-idf is way to represent documents, or tweets in our case, in a numerical vector form.  Each word is represented as **Term Frequency x Inverse Document Frequency**.  Once I created this matrix I then fed it into my decomposition algorithm.

* **Term Frequency = (Number of times term t appears in tweet) / (Total number of terms in tweet)**

* **Inverse Document Frequency = log_e(Total number of tweets / Number of tweets with term t in it)**

#### Word Embeddings

Word Embeddings are a way to represent each word in a vector space.  This is a recent breakthrough in Natural Language Processing which has led to huge gains in the field.  Each word in my vocabulary is assigned a vector based on how it is used in conjunction with other words.  I used the [Word2Vec model in gensim](https://radimrehurek.com/gensim/models/word2vec.html) to create my word embeddings.  My word embeddings were then fed into my autoencoder.

Word embeddings are great at representing the meaning of words in vector space.  For instance **"burrito"** and **"cheeseburger"** are close to each other in vector space, but they are both far from **"president"**.  Word embeddings also allow for exploration of word interactions by adding and subtracting embeddings from each other. For instance, **Cheeseburger + Chipotle - McDonalds = Burrito** and **Shake + Starbucks - McDonalds = Latte**.  
