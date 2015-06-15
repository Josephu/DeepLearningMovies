#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #

import os, pdb, csv
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn import cross_validation
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def main():
    start_time = datetime.now()

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    # test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
    #                quoting=3 )
    train, test, train_sentiment, test_sentiment = cross_validation.train_test_split(df['review'].values, df['sentiment'].values, test_size=0.4, random_state=0)

    print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...'
    #nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print "Cleaning and parsing the training set movie reviews...\n"
    for i in xrange(0, len(train)):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train[i], True)))


    # ****** Create a bag of words from the training set
    #
    print "Creating the bag of words...\n"


    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    # CountVectorizer transform each document into a word count vector, with each word as a feature.
    # Stop words are very frequent words in a language that may not have huge semantic impact, may dissolve the importance of other more meaning for words
    # N-gram provide word combination as a new feature.
    # stop_words: 'english' will use stop words from sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS, seem to make result more unstable.
    # max_feature: limit only the more commonly appeared words in document to be in array, None will allow all features, and increase vector size.
    # vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = 'english', max_features = None)

    # Tfidf: a normalization method to reduce weight of words that appear too frequent in dataset
    # TfidfVectorizer: CountVectorizer that run a tfidf normalization during transform
    vectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                 ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    print 'Train data feature shape: ' + str(train_data_features.shape)
    print 'Number of vocabularies/features: %d\n' %len(vectorizer.get_feature_names())

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    # ******* Train a model using the bag of words
    #
    print "Training the model (this may take a while)..."


    # Initialize a Random Forest classifier with 100 trees
    # clf = RandomForestClassifier(n_estimators=100)
    # clf = svm.LinearSVC(C=1)
    clf = LogisticRegressionCV(cv=3, scoring='roc_auc', solver='liblinear')

    # Cross validation, this takes a long time ...
    # print "4 Fold CV Score: ", np.mean(cross_validation.cross_val_score(clf, train_data_features, train_sentiment, cv=4, scoring='accuracy', n_jobs=4))

    # Fit the svc to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    model = clf.fit(train_data_features, train_sentiment)

    # Create an empty list and append the clean reviews one by one
    clean_test_reviews = []

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0, len(test)):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test[i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use svc to make sentiment label predictions
    print "Predicting test labels...\n"
    result = model.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame(data={"sentiment":test_sentiment, "predict_sentiment":result})
    output['succeed'] = output['sentiment'] == output['predict_sentiment']

    groupby = output.groupby('succeed')
    print 'Result Evaluation'
    print groupby['sentiment'].agg(['count'])

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=csv.QUOTE_MINIMAL)
    print "Wrote results to Bag_of_Words_model.csv"

    print datetime.now() - start_time
    print 'Cs_'
    print getattr(model, 'Cs_')
    print 'scores_'
    print getattr(model, 'scores_')
    print 'C_'
    print getattr(model,'C_')

if __name__ == "__main__":
    main()
