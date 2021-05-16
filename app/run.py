import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorised_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

#Defining functions to create visualisations:
##New Visualisation 1: Distribution of Categories in Messages
def cat_dist(df):
    ''' This function calculates the total numbber of messages per category.
    
    If the input is invalid or the data does not exist, this function will raise an error.
    
    INPUT:
    df --> a Pandas DataFrame containing the prepared data
    
    OUTPUT:
    msg_count --> a Pandas DataFrame containing the message counts for each category
    categories --> the list of categories from the dataset
    '''
    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns
    count = []
    for category in categories:
        count.append(len(df[df[category] != 0]))
    cat_msg_counts_df = pd.DataFrame(count, columns=['Message Counts'], index=categories)
    msg_count = cat_msg_counts_df.sort_values(by=['Message Counts'], ascending=False)
    msg_count = msg_count['Message Counts']
    return msg_count, categories

def multiclass_dist(df):
    '''This function calculates the distribution of multiclassed messages.
    
    If the input is invalid or the data does not exist, this function will raise an error.
    
    INPUT:
    df --> A Pandas DataFrame containing the prepared data
    
    OUTPUT:
    multiclass_msgs --> A Pandas DataFrame containing the total number of messages 
    for each total relevant categories
    '''
    
    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories['total_cat'] = categories.sum(axis=1)
    multi_cat_msgs = categories.groupby('total_cat').count()[['related']]
    multi_cat_msgs.columns = ['total']
    
    return multi_cat_msgs

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    ##1. Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    ##2. Distribution of Message Categories
    cat_count, categories = cat_dist(df)
    
    ##3. Distribution of Multiclassed Messages
    multiclass_msgs = multiclass_dist(df)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=cat_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': 45
                },
                'margin': {
                    'b': 160
                }
            }
        },
        {
            'data': [
                Bar(
                    x=multiclass_msgs.index.tolist(),
                    y=multiclass_msgs['total'].tolist()
                )
            ],

            'layout': {
                'title': 'Distribution of Multiclassed Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Categories"
                },
                'margin': {
                    'b': 160
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()