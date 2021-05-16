# Disaster Response Classification Project 
A repository containing a ETL pipeline, a ML pipeline and a Flask app that deploys a fully-functioning disaster message classification web app. 

### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
4. [Installation](#installation)
5. [Instructions](#instructions)
6. [ETL Pipeline Approach](#approach1)
7. [ML Pipeline Approach](#approach2)
8. [Results](#results)
9. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

The focus for my second project under the Data Science Nanodegree by Udacity apply data engineering and software engineering skills to analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. This project uses a data set containing real messages that were sent during disaster events. 

The goal of this project is to create a machine learning pipeline to categorize these events so that these messages can be sent to the appropriate disaster relief agency. This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


## File Descriptions <a name="files"></a>
Below is a summary of all files in this repository that is needed to achieve the goal of this project:
- app
  - templates
    - master.html (main page of the web app)
    - go.html  (classification page of the web app)
  - run.py  (Flask file that runs the web app)

- data
  - disaster_categories.csv  (dataset contaning the messages sent during disaster events) 
  - disaster_messages.csv  (dataset containing the categories to which each message belongs)
  - process_data.py (Python file that runs the ETL pipeline and exports a SQLite database)
  - InsertDatabaseName.db   (the exported Sqlite database containing the cleaned data)

- models
  - train_classifier.py (Python file that runs the ML pipeline and exports the model as a pickle file)
  - classifier.pkl  (the saved model)

## Installation <a name="installation"></a>

Other than the Anaconda distribution of Python versions 3 loaded into the code of this project, there's no need for any other libraries to successfully run the code.

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Now open anoother Terminal Window, and run env|grep WORK. You will see an output showing:
    - WORKSPACEDOMAIN=Udacity-student-workspaces.com
    - WORKSPACEID=view6914b2f4 
4. In a new browser type in the following: `https://WORKSPACEID-3001.WORKSPACEDOMAIN`. 

## ETL Pipeline Approach <a name="approach1"></a>
My approach to cleaning and preparing the data was to append each category as a unique column with its entries as 0/1 indicators. This is to flag if a message is classified under each category column using boolean data types.

I also dropped existing duplicates before koading the dataset into a SQLite database. 

My approach thoughout the pipeline is in more detail within the Python script `process_data.py`. 

## ML Pipeline Approach <a name="approach2"></a>
This dataset is imbalanced (ie some labels like water have few examples). To counter this imbalanced classification problem, I followed the ensemble methodology; this involves modifying existing classification algorithms to make them appropriate for imbalanced data sets.

The main objective of ensemble methodology is to improve the performance of single classifiers. The Random Forest Algorithm is one example of these technique. The MultiOutputClassifier function helps improve the performance of the model too, and hence has been included into the relevant code.

My approach throughout the pipeline is in more detail within the Python script `train_classifier.py`. 

## Results<a name="results"></a>
Screenshots of the produced web app are provided below:
![alt text](https://github.com/MitraG/Disaster-Response-Classification-Project/blob/main/images/home-page-screenshot-1.jpg)
![alt text](https://github.com/MitraG/Disaster-Response-Classification-Project/blob/main/images/home-page-screenshot-2.jpg)

The home page provides a space for one to classify a disaster message with three visualisations below it. The navigation bar on the top page provides links to Udacity, this Github page as well as my LinkedIn. 

![alt text](https://github.com/MitraG/Disaster-Response-Classification-Project/blob/main/images/classification-page-screenshot-1.jpg)
![alt text](https://github.com/MitraG/Disaster-Response-Classification-Project/blob/main/images/classification-page-screenshot-2.jpg)

When a Classify Message button is clicked, the app displays the predicted classification/category(ies) for the message. In the case, the predicted categories are `Related`, `Aid Related`, `Weather Related` and `Storm`.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I'd like to acknowledge Figure Eight for publicly providing an amazing dataset to perform this classification project. 

I'd also like to thank to Juno Lee and Andrew Paster for their detailed lessons on fundamental software engineering and data engineering skills. 
