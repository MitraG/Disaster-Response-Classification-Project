#First, we import the relevant libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''This function will load the messages and categories datasets.
    
    Then, this function will merge the datasets by left join using the common id and then return a pandas dataframe. 
    
    If the input is invalid or the data does not exist, this function will raise an error. 
    
    INPUT:
    messages_filepath --> location of messages data file from the project root
    categories_filepath --> location of the categories data file from the project root
    
    OUTPUT:
    df --> a DataFrame containing the merged dataset 
    '''
    
    #load the messages dataset
    messages = pd.read_csv(messages_filepath)
    
    #load the categories dataset
    categories = pd.read_csv(categories_filepath)
    
    #merge the two datasets
    df = pd.merge(messages, categories, on='id', how = 'left')
    
    return df


def clean_data(df):
    ''' This function will clean and prepare the merged data to make it more efficient to work with. 
    
    The steps this function will take to clean and prepare the data are:
        - Split the categories into separate category columns
        - Rename every column to its corresponding category
        - Convert category values to a boolean format (0 and 1)
        - Replace the original categories column in the merged dataframe with the new category columns
        - Drop any dulplicates in the newly merged dataset

    If the input is invalid or the data does not exist, this function will raise an error. 
    
    INPUT:
    df --> a Pandas DataFrame with the merged data
    
    OUTPUT:
    df --> a new Pandas Dataframe with each category as a column and its entries as 0/1 indicators.
    This is to flag if a message is classified under each category column.
    '''
    #Split the categories into 36 individual category columns and create a dataframe
    cat_cols = df["categories"].str.split(";", expand=True)
        
    #Rename every column to its corresponding category
    
    ##First, calling the first row of cat_cols to extract a new list of new column names
    ##Using a lambda function that takes everything 
    ##up to the second to last character of each string with slicing
    row = cat_cols.iloc[0]
    string_slicer = lambda x: x[:-2]
    cat_colnames = [string_slicer(i) for i in list(row)]
    cat_cols.columns = cat_colnames
    
    #Convert category values to a boolean format (0 and 1)
    
    #Iterating through the category columns in df to keep only the last character of each string (the 1 or 0) 
    ##Then convert the string into a numeric value 
    ##Using the slicing method once again
    int_slicer = lambda x: int(x[-1])
    for column in cat_cols:
        cat_cols[column] = [int_slicer(i) for i in list(cat_cols[column])]
        
    #Replace the original categories column in the merged dataframe with the new category columns
    df = df.drop(['categories'], axis=1)   
    df = pd.merge(df, cat_cols, left_index=True, right_index=True)
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    
    #Drop any dulplicates in the newly merged dataset
    df = df.drop_duplicates()
    
    return df
    


def save_data(df, database_filename):
    ''' This function will load the prepared data into a SQLite database file. 
    
    If the input is invalid or the data does not exist, this function will raise an error.
    
    INPUT:
    df --> a Pandas DataFrame containing the prepared data
    DisasterResponse.db --> database to store data for model ingestion 
    
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('categorised_messages', engine, index=False, if_exists='replace')


def main():
    ''' This is the mail ETL function that extracts, transforms and loads the data.
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()