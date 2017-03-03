from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from wordcloud import WordCloud, STOPWORDS
from collections import OrderedDict


def find_k(encoded_tweets, k_cluster_range):
    scores = []
    for k in k_cluster_range:
        km = KMeans(n_clusters=k).fit(encoded_tweets)
        dists = km.transform(encoded_tweets)
        labels = km.labels_
        scores.append(silhouette_score(dists, labels))
    plt.plot(k_cluster_range, scores)
    plt.show()
    return zip(k_cluster_range, scores)


def kmeans(encoded_tweets, k, company):
    encoded_tweets = np.array(encoded_tweets)
    print encoded_tweets.shape
    km = KMeans(n_clusters=15).fit(encoded_tweets)
    labels = km.labels_
    print np.unique(labels)
    tweets = pickle.load(open('tweets_ae.pkl', 'rb'))
    dates = pickle.load(open('dates_ae.pkl', 'rb'))
    df = pd.DataFrame({'tweets':tweets,
                        'date':dates,
                        'topic':labels})
    return df


def plot_topic_trend(df, topic_index, topic_name, vday=None, stock=[], refugee=None):
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
    plt.ylim((0,.35))
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
    plt.show()

def create_cloud(df, company, k, topic_index):
    tweets = df
    stop = STOPWORDS.add(company)
    by_topic = tweets[tweets['topic']==topic_index]['tweets'].tolist()
    print '# of tweets in topic {} = {}'.format(topic_index, len(by_topic))
    string = ' '.join(by_topic)
    wordcloud = WordCloud(stopwords=stop, width=800, height=600).generate(string)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../exploratory_plots/ae_{}{}_cloud.png'.format(company, topic_index))


def get_labeled_topics(encoder, embedder, company, k_topics):
    from load_and_process import load_xls
    df1 = load_xls('../../tweets/csv/test1.xls', slang=True, lemma=True, pos=False)
    df2 = load_xls('../../tweets/csv/test2.xls', slang=True, lemma=True, pos=False)
    df = pd.concat([df1, df2])
    df = df[df['tweets'].str.contains(company)]
    tweets = [x.split(' ') for x in df['tweets'].tolist()]
    embedded_tweets = []
    master_tweets = []
    master_labels = []
    for i, tweet in enumerate(tweets):
        try:
            matrix = np.zeros((30, 100))
            for idx, word in enumerate(tweet):
                try:
                    matrix[idx,:] = embedder[word]
                except:
                    matrix[idx,:] = np.zeros((100))
            embedded_tweets.append(np.reshape(matrix,3000))
            master_tweets.append(' '.join(tweet))
            master_labels.append(df['labels'].tolist()[i])
        except:
            pass
    encoded = encoder.predict(np.array(embedded_tweets))
    km = KMeans(k_topics).fit(encoded)
    topics = km.labels_
    df = pd.DataFrame({'tweets':master_tweets, 'labels':master_labels, 'topics':topics})
    df['labels'] = df['labels'] - 2
    return df

def topic_distributions(df, topic_name, topic_index):
    topic_df = df[df['topics'].isin(topic_index)]
    topic = topic_df.groupby('labels')['tweets'].agg(['count'])
    topic['pct_of_tweets'] = topic['count'] / len(topic_df)
    colors = ['pale red', 'greyish', 'windows blue']
    current_palette_3 = sns.xkcd_palette(colors)
    sns.set_palette(current_palette_3)
    sns.set_style('darkgrid')
    sns.barplot(topic.index, topic['pct_of_tweets'])
    plt.xticks([0,1,2], ['Negative', 'Neutral', 'Positive'])
    plt.ylabel('Percent of Tweets in Topic')
    plt.xlabel('Brand Sentiment')
    plt.ylim((0.0,1.0))
    plt.title('{}_sentiment'.format(topic_name))
    plt.savefig('../exploratory_plots/ae_{}_sentiment.png'.format(topic_name))
    plt.show()


def create_plots(vecs, k, company, encoder, embedder):
    df = kmeans(vecs, k, company)
    for x in range(k):
        plt.close('all')
        topic = 'ae_Starbucks{}'.format(x)
        create_cloud(df, company, 15, x)
        plt.close('all')
        plot_topic_trend(df, [x], topic, vday=event2, refugee=event1)
        plt.close('all')
        sent_df = get_labeled_topics(encoder, embedder, company, k)
        topic_distributions(sent_df, topic, [x])

if __name__ == '__main__':
    plt.close('all')
    plt.style.use('ggplot')
    vecs = pickle.load(open('encoded_tweets.pkl', 'rb'))
    company = 'starbucks'
    event1 = [2, 'Refugee Hiring Announcement']
    event2 = [12, "Valentine's Day"]
    k = 15
    encoder = pickle.load(open('../models/encoder.pkl', 'rb'))
    embedder = pickle.load(open('../models/embed_model.pkl', 'rb'))
    create_plots(vecs, k, company, encoder, embedder)
