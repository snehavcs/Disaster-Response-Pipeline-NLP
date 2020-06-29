#import libraries
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import pandas as pd
import sys
import os
import re
from sqlalchemy import create_engine
import pickle
from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """
    Arguments:
        database_filepath - location to SQLite db
    Output:
        X - input features dataframe
        Y - output labels dataframe
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response_table',engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Arguments:
        text - messages in text format
    Output:
        tokens - cleaned tokens
    """
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    message_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return message_tokens


def build_model():
    """
    Output:
        tokens - Pipeline that tokenizes text messages & classifies them.
    """
    pipeline_model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))

        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline_model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Arguments:
        model - pipeline_model that tokenizes text messages & classifies them
        X_test: Test features dataframe
        Y_test: Test target label dataframe
        category_names: Categories classified
    Output:
        Display classification report & accuracy score for each category
    '''
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))
    pass


def save_model(model, model_filepath):
    """
    Arguments:
        model -> pipeline model
        model_filepath -> location to save .pkl file
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
