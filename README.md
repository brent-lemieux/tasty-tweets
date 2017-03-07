# Tasty Tweets
Topic modeling and sentiment analysis of popular food and beverage chains on Twitter using Natural Language Processing and Machine Learning

## Project Introduction
Everyday, millions of people share their opinions on products and brands.  The goal of this project is to gain insight into what consumers are saying without spending countless hours reading through and cataloging millions of tweets.

* What are our customers saying about our brand on social media?  
* Is it positive, negative, or neutral?
* Are some aspects of our brand held in higher regard than others? (i.e. food, service, experience, etc.)
* Is the conversation a short blip or a persistent trend?

The focus of this project is to answer the questions above for three popular food and beverage brands: Chipotle Mexican Grill, Starbucks, and McDonald's.  


## Modeling Methodology

The final models I settled on are:
* **Term frequency - inverse document frequency (tf-idf)** model to represent my tweets in numerical vectors
* **Deep Autoencoder Topic Model** to shrink the feature space and prepare for a clustering algorithm
* **K-Means Clustering** to identify latent topics amongst the tweets.  ***(Note: a topic is a cluster of tweets that are similar to each other).***

***For a more extensive and technical discussion of my process and methodologies please click [here](https://github.com/brent-lemieux/tasty-tweets/methodologies/ "methodologies")***

## Insights

I collected tweets for over a month, Jan. 27 through Feb. 28 of 2017, with the key words "Chipotle", "McDonald's", and "Starbucks".  This section details some of my more interesting findings from modeling the tweets surrounding these brands.

### Starbucks
#### Refugee Hiring Announcement
Starbucks was in the news quite a bit during the period I was collecting data.  On January 31, 2017 they announced that their plan to hire 10,000 refugees over the next five years.  Of course, in this day in age, this was quickly politicized.  

Here is a word cloud of the most frequently used words for tweets that fall into the "refugee hiring" topic:

![Starbucks refugees](/final_plots/ae_starbucks4_cloud.png)

Here is the sentiment distribution for this topic:

![Starbucks refugees](/final_plots/sbux_refugee_sent.png)


And finally, a time-series of the topic prevalence:

![Starbucks refugees](/final_plots/sbux_refugee.png)

***Note:  I added day over day stock price change as a proxy for daily sales data.  Ideally, if I was working with the company, I would be able to show actual revenue numbers to infer how topics really affect the business.***

As you can see, the sentiment surrounding the topic is mostly negative (at least on Twitter).  However, the topic prevalence, while very significant at first, quickly fades from public discussion.  It does not appear to be an issue that Starbucks need to address.

### Chipotle
#### Chipotle vs. the Competition

A small but constant portion of tweets about Chipotle, discuss Chipotle in reference to the competition (i.e. Qdoba, Moe's)

Here's a word cloud associated with this topic:

![Chipotle vs.](/final_plots/ae_chipotle13_cloud.png)

After "eat" and "qdoba", words like "better" and "gt" (greater than) stand out.  A lot of tweets in this topic are comparing two or more burrito restaurants.  

Here is how Chipotle stacks up:

![Chipotle sentiment](/final_plots/cmg_comp_sent.png)

This next plot shows the prevalence of this topic overtime.  It bounces around a little bit but tends to fall between 5 and 10 percent.

![Chipotle time-series](/final_plots/cmg_comp.png)

In this instance, my model provides a framework for companies to track their public opinion in relation to their competitors.

### McDonald's
#### McDonald's Shamrock Shake Release

Each year McDonald's releases the Shamrock Shake about a month before St. Patricks Day.  This year, along with the release of the shake, McDonald's also released a special edition "innovative" straw for maximum shake enjoyment.  

This word cloud shows words frequently used in tweeting about the Shamrock Shake / straw topic.

![McDonald's Shamrock](/final_plots/ae_mcdonalds6_cloud.png)

Next we see that the sentiment distribution is mostly positive with some negative.  Most of the negative tweets that fall into this topic are people making jabs at McDonald's for calling a straw innovative.

![McDonald's sentiment](/final_plots/mcd_shamrock_sent.png)

Lastly, we'll look at the topic prevalence over time.  Notice that hype begins to build a few days ahead of the release, and that it lasts for a little over a week afterwards.

![McDonald's time-series](/final_plots/mcd_shamrock.png)  

## Summary

The framework laid out here shows how companies can track topic prevalence and sentiment around events ranging from advertising campaigns and product releases to public relations issues.

If you would like to contact me about the project or data science in general, please email me at **blemieux4@gmail.com**
