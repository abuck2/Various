#GEneral purpose packages
import numpy as np
import pandas as pd
import os, sys, time, gc, operator, pickle
from collections import defaultdict


#ML packages
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

#specific nlp packages
import string, re

#Local packages
import tokenization
from text_cleaning import text_cleaning, remove_bad_labels
from ClassificationReport import ClassificationReport
from DisasterDetector import DisasterDetector

#Exploration and viz
import missingno as msno
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
import plotly.offline as py
#py.init_notebook_mode(connected=True)
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns

class DisasterTweets:

    def __init__(self, filepath, explore = True, preparation = True):
        self.SEED = 42
        self.path = filepath
        self.explore = explore
        self.test = False #Pas assez de mémoire sur mon ordi
        self.preparation = preparation

    def open_data(self):
        self.df_train = pd.read_csv(self.path+"train.csv")
        self.df_test = pd.read_csv(self.path+"test.csv")

    def explore_data(self):
        print('Training Set Shape = {}'.format(self.df_train.shape))
        print('Training Set Memory Usage = {:.2f} MB'.format(self.df_train.memory_usage().sum() / 1024**2))
        print('Test Set Shape = {}'.format(self.df_test.shape))
        print('Test Set Memory Usage = {:.2f} MB'.format(self.df_test.memory_usage().sum() / 1024**2))
        print("Name of variables = {}".format(self.df_train.columns))    
        print("Example of data : ")
        print(self.df_train.iloc[1:10])
        print("Missing data?")
        print(self.df_train.isnull().sum())
        #print(self.df_train.info())
        msno.matrix(self.df_train)
        
        #tweets by country
        Loc = self.df_train['location'].value_counts()
        fig = px.choropleth(Loc.values, locations=Loc.index,
                            locationmode='country names',
                            color=Loc.values,
                            color_continuous_scale=px.colors.sequential.OrRd)
        fig.update_layout(title="Countrywise Distribution")
        py.iplot(fig, filename='test')

        #Tweets by categorie
        print("Checking if classes are balanced")
        cat = self.df_train['target'].value_counts()
        print(cat)
       
        #Wordcloud
        stopwords= set(STOPWORDS)
        wordcloud = WordCloud(width = 1000,
                height = 600,
                stopwords=stopwords,
                max_font_size = 200,
                max_words = 150,
                background_color='white').generate(" ".join(self.df_train.text))

        plt.figure(figsize=[10,10])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

        #Most frequent keyword
        plt.figure(figsize=(16,8))
        plt.title('Most frequent keywords',fontsize=16)
        plt.xlabel('keywords')
        sns.countplot(self.df_train.keyword,order=pd.value_counts(self.df_train.keyword).iloc[:15].index,
                palette=sns.color_palette("PuRd", 15))

        plt.xticks(size=16,rotation=60)
        plt.yticks(size=16)
        sns.despine(bottom=True, left=True)
        plt.show()
    
        #KW mostly present in true or false tweets
        self.df_train['target_mean'] = self.df_train.groupby('keyword')['target'].transform('mean')
        fig = plt.figure(figsize=(8, 72), dpi=100)
        sns.countplot(y=self.df_train.sort_values(by='target_mean', ascending=False)['keyword'],
                              hue=self.df_train.sort_values(by='target_mean', ascending=False)['target'])
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=12)
        plt.legend(loc=1)
        plt.title('Target Distribution in Keywords')
        plt.show()

        self.df_train.drop(columns=['target_mean'], inplace=True)

    def dataprep(self):
        print("Data preparation...")
        start_prep = time.time()

        self.feature_builder()
        if self.explore == True : 
            self.feature_explorer()
        self.data_prep_ngrams()
        
        if self.test == True:
            self.embeddings_testing() 
        print("Text cleaning...")
        self.clean_text()
        if self.test == True:
            self.embeddings_testing() 
        self.df_train = remove_bad_labels(self.df_train)
        end_prep = time.time()
        print("Dataprep done in {} minutes".format((end_prep - start_prep)/60))
        print("Persisting cleaned dataset on disk...")


    def feature_explorer(self):
        METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length',
                                'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']
        DISASTER_TWEETS = self.df_train['target'] == 1
        fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)
        for i, feature in enumerate(METAFEATURES):
            sns.distplot(self.df_train.loc[~DISASTER_TWEETS][feature],
                    label='Not Disaster', ax=axes[i][0], color='green')
            sns.distplot(self.df_train.loc[DISASTER_TWEETS][feature], label='Disaster', ax=axes[i][0], color='red')
            sns.distplot(self.df_train[feature], label='Training', ax=axes[i][1])
            sns.distplot(self.df_test[feature], label='Test', ax=axes[i][1])
            for j in range(2):
                axes[i][j].set_xlabel('')
                axes[i][j].tick_params(axis='x', labelsize=12)
                axes[i][j].tick_params(axis='y', labelsize=12)
                axes[i][j].legend()
            axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
            axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)
        plt.show()

    def feature_builder(self):
        """
        This function builds new features from the text
        """

        #missing data 
        for df in [self.df_train, self.df_test]:
            for col in ['keyword', 'location']:
                df[col] = df[col].fillna(f'no_{col}')

        # word_count
        self.df_train['word_count'] = self.df_train['text'].apply(lambda x: len(str(x).split()))
        self.df_test['word_count'] = self.df_test['text'].apply(lambda x: len(str(x).split()))

        # unique_word_count
        self.df_train['unique_word_count'] = self.df_train['text'].apply(lambda x: len(set(str(x).split())))
        self.df_test['unique_word_count'] = self.df_test['text'].apply(lambda x: len(set(str(x).split())))

        # stop_word_count
        self.df_train['stop_word_count'] = self.df_train['text']\
                .apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
        self.df_test['stop_word_count'] = self.df_test['text']\
                .apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

        # url_count
        self.df_train['url_count'] = self.df_train['text']\
                .apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
        self.df_test['url_count'] = self.df_test['text']\
                .apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

        # mean_word_length
        self.df_train['mean_word_length'] = self.df_train['text']\
                .apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        self.df_test['mean_word_length'] = self.df_test['text']\
                .apply(lambda x: np.mean([len(w) for w in str(x).split()]))

        # char_count
        self.df_train['char_count'] = self.df_train['text'].apply(lambda x: len(str(x)))
        self.df_test['char_count'] = self.df_test['text'].apply(lambda x: len(str(x)))

        # punctuation_count
        self.df_train['punctuation_count'] = self.df_train['text']\
                .apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
        self.df_test['punctuation_count'] = self.df_test['text']\
                .apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

        # hashtag_count
        self.df_train['hashtag_count'] = self.df_train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
        self.df_test['hashtag_count'] = self.df_test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

        # mention_count
        self.df_train['mention_count'] = self.df_train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
        self.df_test['mention_count'] = self.df_test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
        
        self.METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length',
                                'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']

        #idices of disasters tweets
        self.DISASTER_TWEETS = self.df_train['target'] == 1
    
    def generate_ngrams(self, text, n_gram=1):
        token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [' '.join(ngram) for ngram in ngrams]

    def data_prep_ngrams(self):
        N = 100
        # Unigrams
        disaster_unigrams = defaultdict(int)
        nondisaster_unigrams = defaultdict(int)

        for tweet in self.df_train[self.DISASTER_TWEETS]['text']:
            for word in self.generate_ngrams(tweet):
                disaster_unigrams[word] += 1

        for tweet in self.df_train[~self.DISASTER_TWEETS]['text']:
            for word in self.generate_ngrams(tweet):
                nondisaster_unigrams[word] += 1

        self.df_disaster_unigrams = pd.DataFrame(sorted(disaster_unigrams.items(), key=lambda x: x[1])[::-1])
        self.df_nondisaster_unigrams = pd.DataFrame(sorted(nondisaster_unigrams.items(), key=lambda x: x[1])[::-1])

        # Bigrams
        disaster_bigrams = defaultdict(int)
        nondisaster_bigrams = defaultdict(int)

        for tweet in self.df_train[self.DISASTER_TWEETS]['text']:
            for word in self.generate_ngrams(tweet, n_gram=2):
                disaster_bigrams[word] += 1

        for tweet in self.df_train[~self.DISASTER_TWEETS]['text']:
            for word in self.generate_ngrams(tweet, n_gram=2):
                nondisaster_bigrams[word] += 1

        self.df_disaster_bigrams = pd.DataFrame(sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1])
        self.df_nondisaster_bigrams = pd.DataFrame(sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1])
        
        # Trigrams
        disaster_trigrams = defaultdict(int)
        nondisaster_trigrams = defaultdict(int)

        for tweet in self.df_train[self.DISASTER_TWEETS]['text']:
            for word in self.generate_ngrams(tweet, n_gram=3):
                disaster_trigrams[word] += 1

        for tweet in self.df_train[~self.DISASTER_TWEETS]['text']:
            for word in self.generate_ngrams(tweet, n_gram=3):
                nondisaster_trigrams[word] += 1

        self.df_disaster_trigrams = pd.DataFrame(sorted(disaster_trigrams.items(), key=lambda x: x[1])[::-1])
        self.df_nondisaster_trigrams = pd.DataFrame(sorted(nondisaster_trigrams.items(), key=lambda x: x[1])[::-1])

    def embeddings_testing(self):
        glove_embeddings = np.load(self.path + 'glove.840B.300d.pkl', allow_pickle=True)
        fasttext_embeddings = np.load(self.path + 'crawl-300d-2M.pkl', allow_pickle=True)
        
        train_glove_oov, train_glove_vocab_coverage, train_glove_text_coverage = self.check_embeddings_coverage(
                self.df_train['text'], glove_embeddings)

        test_glove_oov, test_glove_vocab_coverage, test_glove_text_coverage = self.check_embeddings_coverage(
                self.df_test['text'], glove_embeddings)
    
        print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_glove_vocab_coverage, train_glove_text_coverage))

        print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_glove_vocab_coverage, test_glove_text_coverage))


        train_fasttext_oov, train_fasttext_vocab_coverage, train_fasttext_text_coverage = self.check_embeddings_coverage(
                self.df_train['text'], fasttext_embeddings)

        test_fasttext_oov, test_fasttext_vocab_coverage, test_fasttext_text_coverage = self.check_embeddings_coverage(
                self.df_test['text'], fasttext_embeddings)

        print('FastText Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_fasttext_vocab_coverage, train_fasttext_text_coverage))

        print('FastText Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_fasttext_vocab_coverage, test_fasttext_text_coverage))

        del glove_embeddings, fasttext_embeddings, train_glove_oov, test_glove_oov, train_fasttext_oov, test_fasttext_oov
        gc.collect()
    
    def build_vocab(self, X):
        """
        Function to buils vocab for cleaning
        """
        tweets = X.apply(lambda s: s.split()).values      
        vocab = {}
        for tweet in tweets:
            for word in tweet:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1                
        return vocab

    def check_embeddings_coverage(self, X, embeddings):
        """
        Check the coverage of the pretrained embeddings
        """

        vocab = build_vocab(X)    
        covered = {}
        oov = {}    
        n_covered = 0
        n_oov = 0
        for word in vocab:
            try:
                covered[word] = embeddings[word]
                n_covered += vocab[word]
            except:
                oov[word] = vocab[word]
                n_oov += vocab[word]
        vocab_coverage = len(covered) / len(vocab)
        text_coverage = (n_covered / (n_covered + n_oov))

        sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
        return sorted_oov, vocab_coverage, text_coverage

    def clean_text(self):
        self.df_train['text_cleaned'] = self.df_train['text'].apply(lambda s : text_cleaning(s))
        self.df_test['text_cleaned'] = self.df_test['text'].apply(lambda s : text_cleaning(s))
    

    def persist_cleaned_data(self):
        with open(self.path+"training_tweets.pkl", 'wb') as output:
            pickle.dump(self.df_train, output)
        with open(self.path+"test_tweets.pkl", 'wb') as output:
            pickle.dump(self.df_test, output)

    def open_prepared(self):
        with open(self.path+"training_tweets.pkl", 'rb') as data:
            self.df_train = pickle.load(data)
        with open(self.path+"test_tweets.pkl", 'rb') as data:
            self.df_test = pickle.load(data)

    def create_model(self):
        if self.preparation == False :
            self.open_prepared()
        self.prepare_crossval()
        self.modelization()

    def prepare_crossval(self):
        K = 2
        self.skf = StratifiedKFold(n_splits=K, random_state=self.SEED, shuffle=True)

        DISASTER = self.df_train['target'] == 1
        self.DISASTER = DISASTER
        print('Whole Training Set Shape = {}'.format(self.df_train.shape))
        print('Whole Training Set Unique keyword Count = {}'.format(self.df_train['keyword'].nunique()))
        print('Whole Training Set Target Rate (Disaster) {}/{} (Not Disaster)'.format(self.df_train[DISASTER]['target_relabeled'].count(), self.df_train[~DISASTER]['target_relabeled'].count()))

        for fold, (trn_idx, val_idx) in enumerate(self.skf.split(self.df_train['text_cleaned'], self.df_train['target']), 1):
            print('\nFold {} Training Set Shape = {} - Validation Set Shape = {}'.format(fold, self.df_train.loc[trn_idx, 'text_cleaned'].shape, self.df_train.loc[val_idx, 'text_cleaned'].shape))
            print('Fold {} Training Set Unique keyword Count = {} - Validation Set Unique keyword Count = {}'.format(fold, self.df_train.loc[trn_idx, 'keyword'].nunique(), self.df_train.loc[val_idx, 'keyword'].nunique()))    

    def modelization(self):
        #Load BERT layer
        bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=True)
        clf = DisasterDetector(bert_layer, max_seq_length=128, lr=0.0001, epochs=10, batch_size=32)

        clf.train(self.df_train, self.skf)
       
        y_pred = clf.predict(df_test)

        model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
        model_submission['target'] = np.round(y_pred).astype('int')
        model_submission.to_csv('model_submission.csv', index=False)
        model_submission.describe()

    def run(self):
        self.open_data()

        if self.explore == True :
            self.explore_data()

        if self.preparation == True :
            self.dataprep()
            self.persist_cleaned_data()
        
        self.create_model()        

if __name__ == "__main__":
    filepath = "/home/alexis/python_files/twitter/"
    predictor = DisasterTweets(filepath, explore = False, preparation = False)
    predictor.run()



