# Tasty Tweets
Topic modeling and sentiment analysis of popular food and beverage chains on Twitter using Natural Language Processing and Machine Learning

## Project Introduction
* What are our customers saying about our brand on social media?  
* Is it positive, negative, or neutral?
* Are some aspects of our brand held in higher regard than others? (i.e. food, service, experience, etc.)
* Is the conversation a short blip or a persistent trend?

The focus of this project is to answer the questions above for three popular food and beverage brands: Chipotle Mexican Grill, Starbucks, and McDonalds.  The overall goal is to create a framework to answer these questions for any popular brand, product, or topic.


## Final Methodology

The final models I settled on are:
* Term frequency - inverse document frequency (tf-idf) model to represent my tweets in numerical vectors
* Non-negative matrix factorization (NMF) to separate the tweets into latent topics

For a more extensive and technical discussion of my process and methodologies please click [here](https://github.com/brent-lemieux/tasty-tweets/methodologies/ "methodologies")

***I am also still exploring the combination of word embeddings and an autoencoder to model topics.  Early results are promising, but there is still quite a bit of tuning required to significantly outperform NMF.  Stay tuned for updates.***

## Insights
