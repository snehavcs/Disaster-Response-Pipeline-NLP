import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Input:
        messages_filepath - location to messages csv file
        categories_filepath - location to categories csv file
    Output:
        merged_df - output results as Pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merged_df = pd.merge(messages,categories,on='id')
    return merged_df


def clean_data(df):
    """
    Input:
        df - raw data
    Output:
        cleaned_df - cleaned dataframe
    """
    categories = df.categories.str.split(pat=';',expand=True)
    row = categories.iloc[0,:]
    category_cols = row.apply(lambda x:x[:-2])
    categories.columns = category_cols
    for col in categories:
        categories[col] = categories[col].str[-1]
        categories[col] = categories[col].astype(np.int)
    cleaned_df = df.drop('categories',axis=1)
    cleaned_df = pd.concat([cleaned_df,categories],axis=1)
    cleaned_df = cleaned_df.drop_duplicates()
    return cleaned_df


def save_data(df, database_filename):
    """
    Input:
        df - Cleaned dataframe
        database_filename - destination file path to db
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_response_table', engine, index=False, if_exists='replace')
    pass


def main():
    """
    Implement the ETL pipeline:
        1. Extract data from .csv
        2. Clean & process data
        3. Load data to SQLite database
    """
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
