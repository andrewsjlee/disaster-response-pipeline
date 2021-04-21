import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    INPUT
    database_filepath - path for database file generated in the 'process_data.py' script
    
    OUTPUT
    list of messages (X) and corresponding categories (y) and category names
    
    This function extracts data from the database and splits it out into messages and categories data 
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.DataFrame(engine.execute("SELECT * FROM DisasterResponseData").fetchall())
    df.columns = engine.execute("SELECT * FROM DisasterResponseData").keys()
    
    # Assign predictor variables (messages) to X, outcome variables (categories) to y
    X = df['message'].values
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = y.columns.values
    
    return X, y, category_names
    
def tokenize(text):
    '''
    INPUT
    text - text tokens
    
    OUTPUT
    clean_tokens - text tokens that are lemmatized, converted to lowercase, and stripped of whitespace
    
    This function, which tokenizes text strings, is an input to the CountVectorizer class 
    '''
    # Convert message strings to clean list of lemmatized words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT
    none
    
    OUTPUT
    model - model that integrates CountVectorizer, TfidfTransformer, and RandomForest using Pipeline and selects model parameters using GridSearch
    
    This function is used to instantiate the model which can then be fit to the text data
    '''
    # define pipeline that transforms tokenized words to a binary matrix
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
    'clf__estimator__min_samples_split': [25, 50],
    'clf__estimator__n_estimators': [100, 200]
    }
    
    # use Grid Search to find optimal parameters
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs= -1, cv=3)
    
    return model

def evaluate_model(model, X_test, y_test, category_names):
    '''
    INPUT
    model - output from build_model() function
    X_test - test predictors
    y_test - test outcome data
    category names - category names
    
    OUTPUT
    y_pred - data outputted from the model
    classification report showing precision, recall, and f-1 scores
    
    This function will generate the y_pred data and provide the performance of the fitted model by comparing y_pred to y_test
    '''
    # use fitted model to predict y values based on test predictors, then evaluate performance
    y_pred = pd.DataFrame(model.predict(X_test))
    y_pred.columns = category_names
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    # save model to pickle
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
