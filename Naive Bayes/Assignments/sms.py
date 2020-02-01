import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
email_data = pd.read_csv(r"C:\Users\hardi\OneDrive\Documents\Excelr_1\Python\Naive Bayes\Assignments\\sms_raw_NB.csv",encoding='latin-1')
# cleaning data 
import re
stop_words = []
with open("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Naive Bayes\\Assignments\\stop.txt") as f:
    stop_words = f.read()
# splitting the entire string by giving separator as "\n" to get list of 
# all stop words
stop_words = stop_words.split("\n")
"this is awsome 1231312 $#%$# a i he yu nwj"
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))
####Testing Purpose
cleaning_text("Words and hearts should be handled with care… for words when spoken and hearts when broken are the hardest things to repair.").split(" ")
# testing above function with sample text => removes punctuations, numbers
cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi how .. are you?")
email_data.text = email_data.text.apply(cleaning_text)
# removing empty rows 
email_data.shape
email_data = email_data.loc[email_data.text != " ",:]
# CountVectorizer
# Convert a collection of text documents to a matrix of token counts
# TfidfTransformer
# Transform a count matrix to a normalized tf or tf-idf representation
# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]
# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
email_train,email_test = train_test_split(email_data,test_size=0.3)
# Preparing email texts into word count matrix format 
emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.text)
# ["mailing","body","texting","good","awesome"]
# For all messages
all_emails_matrix = emails_bow.transform(email_data.text)
all_emails_matrix.shape # (5559,6661)
# For training messages
train_emails_matrix = emails_bow.transform(email_train.text)
train_emails_matrix.shape # (3891,6661)
# For testing messages
test_emails_matrix = emails_bow.transform(email_test.text)
test_emails_matrix.shape # (1668,6661)
####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.naive_bayes import ComplementNB as CB
from sklearn.naive_bayes import BernoulliNB as BB
from sklearn.naive_bayes import BaseDiscreteNB as BAB
# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_emails_matrix,email_train.type)
train_pred_m = classifier_mb.predict(train_emails_matrix)
accuracy_train_m = np.mean(train_pred_m==email_train.type) # 98%
test_pred_m = classifier_mb.predict(test_emails_matrix)
accuracy_test_m = np.mean(test_pred_m==email_test.type) # 97%
# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_emails_matrix.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_emails_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==email_train.type) # 90%
test_pred_g = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==email_test.type) # 83%
###Complement Naive Bayes
classifier_cb=CB()
classifier_cb.fit(train_emails_matrix.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_c = classifier_cb.predict(train_emails_matrix.toarray())
accuracy_train_c = np.mean(train_pred_c==email_train.type) # 96%
test_pred_c = classifier_cb.predict(test_emails_matrix.toarray())
accuracy_test_c = np.mean(test_pred_c==email_test.type) # 91%
###Bernoulli Naive Bayes
classifier_bb=BB()
classifier_bb.fit(train_emails_matrix.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_b = classifier_bb.predict(train_emails_matrix.toarray())
accuracy_train_b = np.mean(train_pred_b==email_train.type) # 97%
test_pred_b = classifier_bb.predict(test_emails_matrix.toarray())
accuracy_test_b = np.mean(test_pred_b==email_test.type) # 97%
#########################################################3
# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)
# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
train_tfidf.shape # (3891, 6661)
# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape #  (1668, 6661)
# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB
# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,email_train.type)
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m==email_train.type) # 96%
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==email_test.type) # 95%
# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_tfidf.toarray())
accuracy_train_g = np.mean(train_pred_g==email_train.type) # 91%
test_pred_g = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g = np.mean(test_pred_g==email_test.type) # 83%
###Complement Naive Bayes
classifier_cb=CB()
classifier_cb.fit(train_tfidf.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_c = classifier_cb.predict(train_tfidf.toarray())
accuracy_train_c = np.mean(train_pred_c==email_train.type) # 97%
test_pred_c = classifier_cb.predict(test_tfidf.toarray())
accuracy_test_c = np.mean(test_pred_c==email_test.type) # 92%
###Bernoulli Naive Bayes
classifier_bb=BB()
classifier_bb.fit(train_tfidf.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_b = classifier_bb.predict(train_tfidf.toarray())
accuracy_train_b = np.mean(train_pred_b==email_train.type) # 97%
test_pred_b = classifier_bb.predict(test_tfidf.toarray())
accuracy_test_b = np.mean(test_pred_b==email_test.type) # 97%
