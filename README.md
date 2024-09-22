# sentiment-analysis-app
A flask application which classifies the given text according to the sentiment.

## Creating virtaual env
`py -3 -m venv .venv`

## Activating virtual env
`.venv\Scripts\activate`

## Install flask
`pip install flask`

## Install other dependencies
`python -m spacy download en_core_web_trf`
`pip install vaderSentiment`
`pip install pandas`
`pip install transformers`

## Serving/Running application 
`flask --app app run`
or 
`flask run --debug`

## Server will run on the following address
`http://127.0.0.1:5000/`


## Use sample csv file which is in root folder or any csv following the sample csv structure.
