# Disaster Response Pipeline
Set of Python scripts and html files that implement an ML model that classifies tweets in a disaster area to help inform response efforts, outputting the model to an interactive web page.

Link to Github: https://github.com/andrewsjlee/disaster-response-pipeline

# Installation
The scripts were built on Python 3.8.5

The following libraries were used:
- pandas
- scikit-learn 
- numpy
- nltk
- sqlalchemy
- pickle

To run the program, download the files and run the following commands from the root directory:
- python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

Then, to run the ML model, run the following command:
- python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

To deploy the web app, run the following command from the app folder:
- python run.py

Then go to http://0.0.0.0:3001/

# Project Motivation and Summary
This code was written as part of Udacity's Data Science Nanodegree program. 

The objective was to demonstrate how to 1) implement an ETL workflow and store the data in a SQL database, 2) develop a machine learning classifier model that uses pipelines to streamline the data transformation process, and 3) output the model to an interactive web page.

Text transformation was performed using the tf-idf method, which was applied to the vectorized text data. The model was built using the random forest method. The model was deployed using Flask and Plotly.

# File Descriptions
* app
  * template
    * master.html  # main page of web app
    * go.html  # classification result page of web app
  * run.py  # Flask file that runs app

* data
  * disaster_categories.csv  # data to process 
  * disaster_messages.csv  # data to process
  * process_data.py

* models
  * train_classifier.py

# Contact Me
Feedback and suggestions are always welcome: andrewsungjaelee@gmail.com

# Licensing, Authors, Acknowledgements
Credit to the Udacity Data Science Course which provided the foundational materials needed to develop the code
