from newsapi import NewsApiClient
import random

import datetime
today = datetime.datetime.today()
startDate = today - datetime.timedelta(days=30)

# Create a Get News Method
# TODO: Move this to configuration
key = '4cff013574e34fb989b3641d3ef9cae0'
newsApiClient = NewsApiClient(api_key=key)

def GetNewsdata(sourceId):
    newses = newsApiClient.get_everything(
        sources=sourceId,
        domains='bbc.co.uk,techcrunch.com',
        from_param=startDate.strftime("%Y-%m-%d"),
        to=today.strftime("%Y-%m-%d"),
        language='en',
        sort_by='relevancy',
    )
    newsData = []
    for news in newses['articles']:
        list = (random.randint(0, 1000), news['title'],news['content'], 'REAL')
        newsData.append(list)
    return newsData

# Get News Sources
sourcesIdList = []
sources = newsApiClient.get_sources()
for source in sources['sources']:
    sourcesIdList.append(source['id'])

#Assuming id sources is always > 10
index = random.randint(0, len(sourcesIdList) - 11)

# Pick ramdom 10 sources
sourcesIdList = sourcesIdList[index: index+10]

print(f"Num of new sources: {len(sourcesIdList)}")

# Get News using Multiple Sources
newes = []
for sourceId in sourcesIdList:
    newes = newes + GetNewsdata(sourceId)


import pandas as pd
import numpy as np
# Task 5: Create a DataFrame of News
df = pd.DataFrame.from_records(newes)
df.columns = ['','title','text','label']
df.head()


# Load local news data and Concat the DataFrame
localDf = pd.read_csv("usercode/news.csv")
localDf.columns = ['','title','text','label']

df = pd.concat([df, localDf])
df.tail()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

#  Split the Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
X_train.head()

# Feature Selection
vectorizer = CountVectorizer(stop_words='english', max_df=0.7)
feature_train = vectorizer.fit_transform(X_train)
feature_test = vectorizer.transform(X_test)
feature_train_names = vectorizer.get_feature_names_out()
print(f"Found {len(feature_train_names)} features: {feature_train_names[:10]}")

# Initialise and Apply the Classifier
classifier = PassiveAggressiveClassifier(max_iter=500)
classifier.fit(feature_train, y_train)
print(f"Classifier Coefs: {classifier.coef_}")

# Test the classifier
y_pred = classifier.predict(feature_test)
acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy score on validation data: {acc_score*100:.2f}%")

# Load the Test Data
df_local = pd.read_csv("usercode/test_data.csv")
df_local.head()

# Select Features and Get Predictions
feature_local = vectorizer.transform(df_local['text'])
y_local_pred = classifier.predict(feature_local)

# Evaluate the Predictions
acc_local = accuracy_score(df_local['label'], y_local_pred)
print(f"Local data prediction accuracy {acc_local*100:.2f}%")