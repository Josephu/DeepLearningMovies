import os, pdb
from datetime import datetime
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import pandas as pd
import numpy as np

def main():
    start_time = datetime.now()

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    # test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)
    train, test, y, y_test = cross_validation.train_test_split(df['review'].values, df['sentiment'].values, test_size=0.4, random_state=0)

    print "Cleaning and parsing movie reviews...\n"      
    traindata = []
    for i in xrange(0, len(train)):
        traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train[i], False)))
    testdata = []
    for i in xrange(0, len(test)):
        testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test[i], False)))
    print 'vectorizing... ',
    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    X_all = traindata + testdata
    lentrain = len(traindata)

    print "fitting pipeline... ",
    tfv.fit(X_all)
    X_all = tfv.transform(X_all)

    X = X_all[:lentrain]
    X_test = X_all[lentrain:]

    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)
    print "10 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=10, scoring='roc_auc'))

    print "Retrain on all training data, predicting test labels...\n"
    model.fit(X, y)
    # result = model.predict_proba(X_test)[:,1] # predict as probability
    result = model.predict(X_test)
    # output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = pd.DataFrame(data={"sentiment":y_test, "predict_sentiment":result})
    output['succeed'] = output['sentiment'] == output['predict_sentiment']

    groupby = output.groupby('succeed')
    print 'Result Evaluation'
    print groupby['sentiment'].agg(['count'])

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model_linear.csv'), index=False, quoting=3)
    print "Wrote results to Bag_of_Words_model_linear.csv"

    print datetime.now() - start_time

if __name__ == "__main__":
    main()
