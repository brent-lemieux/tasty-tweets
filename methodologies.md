## Explored Methodologies
* Unsupervised learning to cluster tweets into latent topics
    * **Latent Semantic Analysis** - This method utilizes tf-idf document representation models.
    * **Latent Dirichlet Allocation** - This method utilizes a term frequency vectorizer.
    * **Non-Negative Matrix Factorization** - This method utilizes tf-idf.
    * **Deep Autoencoder Topic Model  [(DATM)](https://www.prhlt.upv.es/workshops/iwes15/pdf/iwes15-kumar-d'haro.pdf "DATM")** - For this method I tried two ways of representing text in vector form:
        * Tf-idf
        * Word Embeddings
* Supervised learning to classify brand sentiment of tweets
    * **Supervised** - Hand labeled a subset of tweets and utilized various machine learning algorithms to classify sentiment.

## Final Methodologies and Process

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

***I ended up not using this method as it appears not to work as well for unsupervised learning tasks.  There is no mechanism for an autoencoder to learn word importance to overall tweet meaning in an unsupervised setting, therefore the tf-idf outperforms (at least in my experience).***


#### Deep Autoencoder Topic Model

The autoencoder takes each tweet as an input, compresses it into lower dimensional space, and attempts to reconstruct the original input.  It then scores itself based on a loss function (I am currently using binary cross-entropy) and uses backpropagation to update the weights between the fully connected layers.

To get the tweet representation in lower feature space, we take just the ***encode*** portion of the model and apply it to each tweet.  


![Autoencoder Chart](/final_plots/autoencoder.png)


#### K-Means Clustering

Once we have the compressed version of tweets we can use k-means clustering to separate the tweets into latent topics.  For k-means to work, a number of clusters must be specified.  The algorithm randomly places cluster centroids in the appropriate dimensional space.  It then updates by assigning the new centroids to the mean value of the clusters that were assigned with random initialization.

Here is how my first 5 clusters look when reduced to two-dimensions (using Principal Component Analysis):

![K-Means](/final_plots/pca_ae_plot.png)

***Note: I used a k-value of 15, not 5.  I reduced the number of topics to 5 here for illustration purposes***
