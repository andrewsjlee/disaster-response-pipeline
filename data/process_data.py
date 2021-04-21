import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - path for 'disaster_messages.csv' file
    categories_filepath - path for 'disaster_categories.csv' file
    
    OUTPUT
    dataframe with merged messages data
    
    This function reads in the messages and categories data
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge messages and categories data into a single dataframe
    df = messages.merge(categories, how='left', on='id')
    
    return df

def clean_data(df):
    '''
    INPUT
    df - output from load_data function
    
    OUTPUT
    dataframe with cleaned category data
    
    This function cleans the messages data and prepares it for the database 
    '''
    # split category values into separate columns and save to new dataframe
    df_cat = df['categories'].str.split(';', expand=True)
    
    # select first row of the categories dataframe and remove the trailing digits
    row = df_cat.iloc[0]
    category_colnames = row.apply(lambda x: x[0:-2])
    
    # assign the clean column names to df_cat
    df_cat.columns = category_colnames

    for column in df_cat:
        # set each value to be the last character of the string
        df_cat[column] = df_cat[column].str[-1:]
        # convert column from string to numeric
        df_cat[column] = df_cat[column].astype(int)
        # replace '2' values with '1'
        df_cat[column] = df_cat[column].replace(2,1)

    # drop the earlier categories column from main dataframe
    df = df.drop(columns='categories')
    
    # concatenate the cleaned category data to the main dataframe
    df = pd.concat([df, df_cat], axis=1)

    # drop duplicate rows
    df = df.drop_duplicates() 
    
    return df
    
def save_data(df, database_filename):
    '''
    INPUT
    df - output from clean_data function
    database_filename - desired name of database
    
    OUTPUT
    dataframe with cleaned category data
    
    This function takes the cleaned data and outputs a .db file
    '''
    # save cleaned data to a database that can be read into the ML pipeline    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponseData', engine, index=False, if_exists='replace')  


def main():
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
