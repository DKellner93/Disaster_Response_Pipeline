# Disaster Response Pipeline Project

This project is used to properply categorize messages sent during emergeny situations. This helps to forward these messages to the corresponding help organziation needed to solve the problem.

## Data structure:

- app:
   -  templates: contains templates to run the webapp
   -  run.py: App that runs the web apllication and includes the definition of the graphs used in the webapp
 
- data:
  - process_data.py: ETL pipeline that reads in and cleans the data used to train the ML model. Data is then saved into the DisasterResponse.db
  - disaster_categories.csv: Categories used to classify the messages
  - disaster_messages.csv: list of messages
  - DisasterResponse.db: Database where the cleaned data is stored
 
- models:
  - train_classifier.py: ML-pipeline that uses a database to setup and train a machine learning model to properly categorize new messages
  - classifier.pkl: trained ML-model saved to use in the webapp

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`
