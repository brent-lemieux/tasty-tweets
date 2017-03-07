# Tasty Tweets
Topic modeling and sentiment analysis of popular food and beverage chains on Twitter using Natural Language Processing and Machine Learning

## Project Introduction
Everyday, millions of people share their opinions on products and brands.  The goal of this project is to gain insight into what consumers are saying without spending countless hours reading through and cataloging millions of tweets.

* What are our customers saying about our brand on social media?  
* Is it positive, negative, or neutral?
* Are some aspects of our brand held in higher regard than others? (i.e. food, service, experience, etc.)
* Is the conversation a short blip or a persistent trend?

The focus of this project is to answer the questions above for three popular food and beverage brands: Chipotle Mexican Grill, Starbucks, and McDonalds.  


## Modeling Methodology

The final models I settled on are:
* **Term frequency - inverse document frequency (tf-idf)** model to represent my tweets in numerical vectors
* **Deep Autoencoder Topic Model** to shrink the feature space and prepare for a clustering algorithm
* **K-Means Clustering** to identify latent topics amongst the tweets.  ***(Note: a topic is a cluster of tweets that are similar to each other).***

***For a more extensive and technical discussion of my process and methodologies please click [here](https://github.com/brent-lemieux/tasty-tweets/methodologies/ "methodologies")***

## Insights

I collected tweets for over a month, Jan. 27 through Feb. 28 of 2017, with the key words "Chipotle", "McDonalds", and "Starbucks".  This section details some of my more interesting findings from modeling the tweets surrounding these brands.

### Starbucks
##### Refugee Hiring Announcement
Starbucks was in the news quite a bit during the period I was collecting data.  On January 31, 2017 they announced that their plan to hire 10,000 refugees over the next five years.  Of course, in this day in age, this was quickly politicized.  

Here is a word cloud of the most frequently used words for tweets that fall into the "refugee hiring" topic:
![Starbucks refugees](/final_plots/ae_starbucks4_cloud.png)

Here is the sentiment distribution for this topic:

![Starbucks refugees](/final_plots/sbux_refugee_sent.png)


And finally, a time-series of the topic prevalence:

![Starbucks refugees](/final_plots/sbux_refugee.png)

***I added day over day stock price change as a proxy for daily sales data.  Ideally, if I was working with the company, I would be able to show actual revenue numbers to infer how topics really affect the business.***

As you can see, the sentiment surrounding the topic is mostly negative (at least on Twitter).  However, the topic prevalence, while very significant at first, quickly fades from public discussion.  It does not appear to be an issue that Starbucks need to address.
