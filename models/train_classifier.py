#First, we import the relevant libraries for this model
import sys
import pickle
import re
import nltk
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#Next, we download necessary NLTK data
nltk.download(['punkt', 'wordnet'])

#Now, we define the regular expression to detect a url
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''This function loads the prepared data from a supplied database filepath.
    
    Then, it defines the independent variable (X) and the dependent variable (Y) respectively.   
    
    If the input is invalid or the data does not exist, this function will raise an error.
    
    INPUT:
    database_filepath --> the location of the SQLite file where the prepared data is stored
    
    OUTPUT:
    X --> This is defined as a set of the message column from the dataframe
    Y --> This is defined as a set of all the categorical columns from the dataframe
    Y.columns --> This is defined as a list of all categorical columns from the dataframe
    '''
    
    #First load the dataset from the database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('categorised_messages', engine)
    
    #Now define the X and Y variables and print both including the columns of Y to obtain a list of all categories
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    ''' This function breaks the texts into tokens, replaces URLs as well as lemmatizes the tokens. 
    
    If the input is invalid or the data does not exist, this function will raise an error.
    
    INPUT:
    text --> defined sentences
    
    OUTPUT:
    clean_tokens --> list of generated tokens
    '''
    
    urls_detected = re.findall(url_regex, text)
    for url in urls_detected:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens    


def build_model():
    ''' This function is the training model. It uses a pipeline object to group the transformers and predictors together. 
    
    These are the CountVectorizer, TfidfTransformer, and MultiOutputClassifier.
    
    As the dependent variable has multiple targets (i.e. many categorical columns),
    Using MultiOutputClassifier to run the RandomForestClassifier is a better fit for our ML model. 
    
    This function also makes use of the GridSearch method to improve the model.
    
    If the input is invalid or the data does not exist, this function will raise an error.
    
    INPUT:
    pipeline --> the ML model pipeline containing the transformers and predictor 
    parameters --> defined values for the GridSearch method
    
    OUTPUT:
    cv --> the created GridSearch object
    '''
    
    #First, we define the model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #Next, we specify parameters for the GridSearch method
    parameters = {
        'vect__max_features': [None, 100, 1000]
    }

    #Now, we create a GridSearch object to return
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
  
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    '''This function is used to evaluate the model built above on the test dataset. 
    
    Then, it iterates through the categorical columns and calls sklearn's classification_report on each.
    
    This gets us scores on:
        -precision
        -recall
        -f1-score
        
    If the input is invalid or the data does not exist, this function will raise an error.
    
    INPUT:
    Y_pred --> defined for the prediction on the dependent variable based on the independent variable from the test dataset
    
    OUTPUT:
    classification_report --> a report of scores listed above for each column
    '''
    #Predict the model on the test dependent variable
    Y_pred = model.predict(X_test)
    
    #Print a classification report for each column 
    for i, column in enumerate(category_names):
        print("Report for target column: " + column)
        print(classification_report(Y_test.values[:, i], Y_pred[:,i], target_names=["0", "1"]))
        


def save_model(model, model_filepath):
    '''This function saves the trained model as a pickle file.
    
    If the input is invalid or the data does not exist, this function will raise an error.
    
    OUTPUT:
    classifier.pkl --> the saved pickle file of the trained model
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))
    
    
    
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