from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from decompose import print_top_words
from wordcloud import WordCloud, STOPWORDS
from collections import OrderedDict


data = pd.read_csv('/Users/blemieux/projects/tweets/csv/clean_master.csv')

# segment data by company

# Starbucks
sbux = data[data['tweets'].str.contains('starbucks')]
# load model
model = pickle.load(open('../models/sbux_nmf.pkl', 'rb'))
# define TfidfVectorizer parameters
n_features = 1000
ngrams = [1,3]
max_df = .5
min_df = .015
n_topics = 20
n_top_words = 10


#
# # Chipotle
# cmg = data[data['tweets'].str.contains('chipotle')]
# # load model
# model = pickle.load(open('../models/cmg_nmf.pkl', 'rb'))
# n_features = 1000
# ngrams = [1,3]
# max_df = .5
# min_df = .015
# n_topics = 20
# n_top_words = 10



# # McDonalds
# mcd = data[data['tweets'].str.contains('mcdonalds')]
# # load model
# model = pickle.load(open('../models/mcd_nmf.pkl', 'rb'))
# # define TfidfVectorizer parameters
# n_features = 1000
# ngrams = [1,3]
# max_df = .5
# min_df = .01
# n_topics = 18
# n_top_words = 10


# vectorize
tfidf = TfidfVectorizer(max_features=n_features, ngram_range=ngrams, max_df=max_df, min_df=min_df,stop_words='english')

vecs = tfidf.fit_transform(sbux['tweets']).todense()
feature_names = tfidf.get_feature_names()

sbux['topic'] = np.argmax(model.transform(vecs), axis=1)
# stock = pd.read_csv('../sbux_stock.csv')
# price_delta = stock['Price Change'].tolist()[::-1]

def plot_topic_trend(df,topic_index, topic_name, vday=None, stock=[], refugee=None):
    topic_share = []
    days = []
    dates = np.unique(df['date'])
    dts = [dt[-5:] for dt in dates]
    for i, date in enumerate(dates):
        df_date = df[df['date']==date]
        topic_tweets = df_date[df_date['topic'].isin(topic_index)]
        topic_share.append(len(topic_tweets)/len(df_date))
        days.append(i+1)
    plt.plot(days, topic_share, lw=2, label="Prevelance of Topic", color='navy')
    if len(stock) > 0:
        plt.plot(days, stock, color='teal', label="Stock Price Change", lw=2)
    plt.xticks(days, dts, rotation='vertical')
    plt.ylim((0,.45))
    if vday:
        plt.axvline(vday[0], color='red', lw=2, label=vday[1], alpha=.5)
    if refugee:
        plt.axvline(refugee[0], color='green', lw=2, label=refugee[1], alpha=.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.subplots_adjust(bottom=0.15)
    plt.title('{} Topic Overtime'.format(topic_name))
    plt.ylabel('Percentage of tweets about topic')
    plt.xlabel('Day')
    plt.savefig('../exploratory_plots/{}_topic_ts.png'.format(topic_name.replace(' ','_')))
    plt.close('all')



def create_cloud(df, topic_index, extra_stop, topic_name):
    for word in extra_stop:
        STOPWORDS.add(word)
    tweets = df[df['topic'].isin(topic_index)]['tweets'].tolist()
    tweets = ' '.join(tweets)
    wordcloud = WordCloud(stopwords=STOPWORDS, width=800,
     height=600).generate(tweets)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../exploratory_plots/{}_cloud.png'.format(topic_name.replace(' ','_')))
    plt.close('all')



def get_labeled_topics(model, tfidf, company):
    from load_and_process import load_xls
    df1 = load_xls('../../tweets/csv/test1.xls', slang=True, lemma=True, pos=False)
    df2 = load_xls('../../tweets/csv/test2.xls', slang=True, lemma=True, pos=False)
    df = pd.concat([df1, df2])
    df = df[df['tweets'].str.contains(company)]
    # df = pd.concat([df1, df2])
    X = tfidf.transform(df['tweets'])
    topics = np.argmax(model.transform(X), axis=1).tolist()
    df['topics'] = topics
    df['labels'] = df['labels'] - 2
    return df

def topic_distributions(df, topic, topic_index):
    topic_df = df[df['topics'].isin(topic_index)]
    topic = topic_df.groupby('labels')['tweets'].agg(['count'])
    topic['pct_of_tweets'] = topic['count'] / len(topic_df)
    print topic
    # plt.bar(topic.index, topic['pct_of_tweets'], align='center')
    colors = ['pale red', 'greyish', 'windows blue']
    current_palette_3 = sns.xkcd_palette(colors)
    sns.set_palette(current_palette_3)
    sns.set_style('darkgrid')
    sns.barplot(topic.index, topic['pct_of_tweets'])
    plt.xticks([0,1,2], ['Negative', 'Neutral', 'Positive'])
    plt.ylabel('Percent of Tweets in Topic')
    plt.xlabel('Brand Sentiment')
    plt.ylim((0.0,1.0))
    plt.title('{}_topic_{}_sentiment'.format(topic, topic_index[0]))
    plt.savefig('../exploratory_plots/nmf_{}_sentiment.png'.format(topic))
    plt.close('all')


if __name__ == '__main__':
    plt.close('all')
    plt.style.use('ggplot')
    co_string = 'starbucks'
    co = sbux
    # by_topic = co.groupby('topic')['tweets'].agg('count')
    # sns.barplot(by_topic.index, by_topic['count'])
    # plt.title('Number of Tweets in each Topic')
    # plt.savefig('starbucks_topics.png')
    for x in range(20):
        plt.close('all')
        topic_idx = [x]
        topic = 'nmf_Starbucks{}'.format(topic_idx[0])
        event1 = [2, 'Refugee Hiring Announcement']
        event2 = [12, "Valentine's Day"]
        plot_topic_trend(co, topic_idx, topic, vday=event2, refugee=event1)
        plt.close('all')
        create_cloud(co, topic_idx, [co_string], topic)
        plt.close('all')
        sent_df = get_labeled_topics(model, tfidf, co_string)
        topic_distributions(sent_df, topic, topic_idx)
