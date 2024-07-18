import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """Load the data from a sql database and store them in the variables needed for the ML-Pipeline

    Args:
    database_filepath: str. Path to the database

    Returns:
    X: Dataframe of messages 
    y: dataframe with the corresponding classification into different message categories
    category_names: list of the categories used for classification
    """
    engine_path = 'sqlite:///' + database_filepath
    engine = create_engine(engine_path)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """tokenize and lemmanize the message to use for ML algorithm"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """Setup of Pipeline and GridSearch to train the ML model"""
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2]
        }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=3)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evelatue the model performance on the Test Set"""
    y_pred = model.predict(X_test)
    for i in range(y_pred.shape[1]):
        print("=====================",Y_test.columns[i],"====================")
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))
    return


def save_model(model, model_filepath):
    """Save Model as a pickle file to later use in a webapp"""
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print(category_names)
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