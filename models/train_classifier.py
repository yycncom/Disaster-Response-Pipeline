# import libraries
import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

ST_english = stopwords.words('english')

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table('merged', conn)

    X = df['message'].values
    # child_alone column only has one category record child_alone-0, so get ride of this column
    df_Y = df.loc[:, 'related':].drop(['child_alone'], axis=1)
    #There is 42 "2"s in related columns, so change it into "1"
    df_Y.loc[df_Y[df_Y['related'] == 2]['related'].index,'related'] =1
    category_names = list(df_Y.columns)
    Y = df_Y.values
    return X,Y,category_names

def tokenize(text):
    """
    Input text 
    Output tokenized text
    """
    #Clean data, remove all character except character and number,such as punctuation etc.
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in ST_english]
    return tokens


def build_model():
    pipeline = Pipeline([
           ("text_pipeline:", Pipeline([
               ("vect", CountVectorizer(tokenizer=tokenize)),
               ("tfidf", TfidfTransformer())
           ])),
           
           ("clf", MultiOutputClassifier(RandomForestClassifier()))
          ])
 
    parameters = {
        'clf__estimator__min_impurity_decrease':(0.0,0.01),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
   
    return cv

def evaluate_model(model, X_test, Y_test, category_names, df_Y=df_Y):
    Y_pred = model.predict(X_test)
    df_categories = df_Y
    for category_name in category_names: 
        print(
            classification_report(
            #Get the column from Y_test array, which column name is target_name.
            Y_test[:, category_names.index(category_name)], 
            #Get the column from Y_pred array, which column name is target_name.    
            Y_pred[:, category_names.index(category_name)], 
            category_names=[category_name +'-'+ str(cate) for cate in df_categories[category_name].unique()]
            )
        ) 
    
#     print("Labels:", labels)
#     print("Accuracy:", accuracy)
#     print("Precision:", Precision)
#     print("Recall:", Recall)
#     print("F1 score:", F1)
#     print("\nBest Parameters:", model.best_params_)

def save_model(model, model_filepath):
    with open(model_filepath,'wb') as file:
        pickle.dump(model, file)

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
