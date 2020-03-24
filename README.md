# Disaster Response Pipeline Project
   Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
Creating a machine learning pipeline to categorize these events so that sending the messages to an appropriate disaster relief agency.
This project including a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displaying visualizations of the data. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### What's included
  There are three components for this project.
```
.
|── app
│   |── run.py
│   |── templates
│       |── go.html
│       |── master.html
|── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   └── process_data.py
├── models
│   ├── classifier.pkl
│   └── train_classifier.py
└── README.md
```   
  1. ETL Pipeline
 
  In a Python script, process_data.py, data cleaning pipeline :
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database(DisasterResponse.db)    
  2. ML Pipeline
  
  In a Python script, train_classifier.py, a machine learning pipeline that:
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file(classifier.pkl)    
  3. Flask Web App
    - data visualizations using Plotly in the web app. 
    - classification results in several categories. 
    
  
