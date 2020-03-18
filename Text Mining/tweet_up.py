import pandas as pd 
import re
import numpy as np
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
##%matplotlib inline
import tweepy as twt
#Twitter API credentials
consumer_key = "Provide your Consumer Key"  #### Removed consumer_key as it violates security 
consumer_secret = "Provide your Consumer_secret"  #### Removed consumer_secret as it violates security 
access_key = "Provide your access_key"   #### Removed access_key as it violates security 
access_secret = "Provide your access_secret key"   #### Removed access_secret as it violates security 
alltweets = []	
def get_all_tweets(screen_name):
    auth = twt.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = twt.API(auth)
    new_tweets = api.user_timeline(screen_name = "@realDonaldTrump",count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1;
    while len(new_tweets)>0:
        new_tweets = api.user_timeline(screen_name = "@realDonaldTrump",count=200,max_id=oldest)
        #save most recent tweets
        alltweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))        
    # tweet.get('user', {}).get('location', {})
    outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
             tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
             tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
             tweet._json["user"]["utc_offset"]] for tweet in alltweets]
    tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                    "geo","id_str","lang","place","retweet_count","retweeted","source",
                                    "text","location","name","time_zone","utc_offset"])
    tweets_df["time"]  = pd.Series([str(i[0]) for i in outtweets])
    tweets_df["hashtags"] = pd.Series([str(i[1]) for i in outtweets])
    tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in outtweets])
    tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in outtweets])
    tweets_df["geo"] = pd.Series([str(i[4]) for i in outtweets])
    tweets_df["id_str"] = pd.Series([str(i[5]) for i in outtweets])
    tweets_df["lang"] = pd.Series([str(i[6]) for i in outtweets])
    tweets_df["place"] = pd.Series([str(i[7]) for i in outtweets])
    tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in outtweets])
    tweets_df["retweeted"] = pd.Series([str(i[9]) for i in outtweets])
    tweets_df["source"] = pd.Series([str(i[10]) for i in outtweets])
    tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
    tweets_df["location"] = pd.Series([str(i[12]) for i in outtweets])
    tweets_df["name"] = pd.Series([str(i[13]) for i in outtweets])
    tweets_df["time_zone"] = pd.Series([str(i[14]) for i in outtweets])
    tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in outtweets])
    tweets_df.to_csv("@realDonaldTrump"+"_tweets.csv")
    return tweets_df
tweets_df = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Text Mining\\@realDonaldTrump_tweets.csv")
##from sklearn.model_selection import train_test_split
##train,test  = train_test_split(tweets_df,test_size = 0.2) # 20% size
##train['text']
####user-defined function to remove unwanted text patterns from the tweets. It takes two 
##arguments, one is the original string of text and the other is the pattern of text that we
##want to remove from the string. The function returns the same input string but without the 
##given pattern. We will use this function to remove the pattern ‘@user’ from all the tweets in our data.
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt 
####let’s create a new column tidy_tweet, it will contain the cleaned and processed tweets. 
###Note that we have passed “@[\w]*” as the pattern to the remove_pattern function. It is actually 
###a regular expression which will pick any word starting with ‘@’.
# remove twitter handles (@user)
tweets_df['tidy_tweet'] = np.vectorize(remove_pattern)(tweets_df['text'], "@[\w]*") 
##Punctuations, numbers and special characters do not help much. It is better to remove 
##them from the text just as we removed the twitter handles. Here we will replace everything
##except characters and hashtags with spaces.   
# remove special characters, numbers, punctuations
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
#### Remove https
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("https", " ")
###### Now remove short words
#### Lets  remove all the words having length 3 or less. For example, terms like 
#### “hmm”, “oh” are of very little use. It is better to get rid of them.
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("[^\x00-\x7F]+"," ")
##### Removing stop words
with open("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Text Mining\\stop.txt","r") as sw:
    stopwords = sw.read()
stopwords = stopwords.split("\n")
##stopwords = pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Text Mining\\stop.xlsx")
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("stop"," ")
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("Tbgwkpv"," ")
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("YfUS"," ")
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("YqJrGDPAOj"," ")
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("this"," ")
#### Remove unwanted symbol in case if it exists
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("[^A-Za-z" "]+"," ")
tweets_df['tidy_tweet'] = tweets_df['tidy_tweet'].str.replace("[0-9" "]+"," ")
####Now we will tokenize all the cleaned tweets in our dataset. Tokens are individual terms or words, and 
####tokenization is the process of splitting a string of text into tokens.
tokenized_tweet = tweets_df['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()
#### Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) 
####from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” 
#### are the different variations of the word – “play”.
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()
### Now lets stitch these tokens back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
tweets_df['tidy_tweet'] = tokenized_tweet
##A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.
##Let’s visualize all the words our data using the wordcloud plot.
tweets_df['tidy_tweet'] = " ".join([text for text in tweets_df['tidy_tweet'] if not text in stopwords])
all_words = ' '.join([text for text in tweets_df['tidy_tweet']])
from wordcloud import WordCloud
import matplotlib.pyplot as plt
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])
normal_words =' '.join([text for text in tweets_df['tidy_tweet']])
wordcloud = WordCloud(width=800, height=500, max_font_size=110,background_color="white").generate(normal_words)
plt.figure(figsize=(15, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
##### Read positive and negative words from text files
with open("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Text Mining\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
with open("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Text Mining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")
######## Positive word cloud
# Choosing the only words which are present in positive words
donald_pos_in_pos = " ".join([text for text in tweets_df['tidy_tweet'] if text in poswords])
wordcloud_pos_in_pos = WordCloud(background_color='black',width=1800,height=1400,max_font_size=110).generate(donald_pos_in_pos)  
plt.figure(figsize=(15, 12))
plt.imshow(wordcloud_pos_in_pos, interpolation="bilinear")
plt.axis('off')
plt.show()
##### Choosing the only words which are present in negative words
donald_neg_in_neg = " ".join([text for text in tweets_df['tidy_tweet'] if text in negwords])
wordcloud_neg_in_neg = WordCloud(background_color='black',width=1800,height=1400,max_font_size=110).generate(donald_neg_in_neg)  
plt.figure(figsize=(15, 12))
plt.imshow(wordcloud_neg_in_neg, interpolation="bilinear")
plt.axis('off')
plt.show()
###### Hashtags in twitter are synonymous with the ongoing trends on twitter at any particular 
####point in time. We should try to check whether these hashtags add any value to our sentiment
#### analysis task, i.e., they help in distinguishing tweets into the different sentiments.
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags
HT_regular = hashtag_extract(tweets_df['tidy_tweet'])
# unnesting list
HT_regular = sum(HT_regular,[])
import nltk
import seaborn as sns
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,14))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
######## Extracting features from Cleaned Tweets
######To analyze a preprocessed data, it needs to be converted into features.
### Depending upon the usage, text features can be constructed using assorted techniques –
### Bag-of-Words, TF-IDF, and Word Embeddings
#### Bag-of-Words is a method to represent text into numerical features. Consider a corpus
## (a collection of texts) called C of D documents {d1,d2…..dD} and N unique tokens extracted
### out of the corpus C. The N tokens (words) will form a list, and the size of the bag-of-words 
###matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(tweets_df['tidy_tweet'])
###### This is another method which is based on the frequency method but it is different to the
### bag-of-words approach in the sense that it takes into account, not just the occurrence of a word 
##in a single document (or tweet) but in the entire corpus.
###TF = (Number of times term t appears in a document)/(Number of terms in the document)
##IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
##TF-IDF = TF*IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(tweets_df['tidy_tweet'])
tweets_df['label'] = np.random.choice([1,2], tweets_df.shape[0])
tweets_df = tweets_df.reset_index()
tweets_df['ID'] = tweets_df.index + 1
######### Building model using Bag-Of-Words features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
train=tweets_df.iloc[:2000,:]
test=tweets_df.iloc[2000:,:]
train_bow = bow[:2000,:]
test_bow = bow[2000:,:]
# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.38)
lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int,average=None)  #### 0.6266
f1_score(yvalid, prediction_int,average='weighted')  #### 0.349
f1_score(yvalid, prediction_int,average='micro') ##### 0.503
f1_score(yvalid, prediction_int,average='macro') ### 0.226
##### Now we will use the model to predict the test data
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['ID','label']]
submission.to_csv('sub_lreg_bow.csv', index=False)
####### Building model using TF-IDF features
train_tfidf = tfidf[:2000,:]
test_tfidf = tfidf[2000:,:]
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
lreg.fit(xtrain_tfidf, ytrain)
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int,average=None)  #### 0.6790
f1_score(yvalid, prediction_int,average='weighted') ### 0.349
f1_score(yvalid, prediction_int,average='macro')  #### 0.226
f1_score(yvalid, prediction_int,average='micro') ## 0.503