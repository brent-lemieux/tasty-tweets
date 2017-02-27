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
    * **Non-Negative Factorization** - This method will also use tf-idf.
    * **Deep Autoencoder Topic Model  [(DATM)](https://www.prhlt.upv.es/workshops/iwes15/pdf/iwes15-kumar-d'haro.pdf "DATM")** - This method will make use of word embeddings to relate words used in similar context with one another.
* Supervised and semi-supervised learning to classify sentiment of tweets in relation to brand
    * **Supervised** - Hand label a subset of tweets and use various machine learning algorithms to classify sentiment .
    * **Semi-supervised** - Read through the latent topics identified by the unsupervised models and determine if topics can be divided into generally positive, generally negative, and no sentiment.  Then use the supervised approach detailed above.
