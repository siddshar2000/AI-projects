# Detecting Fake News
This project detects fake news!

## Installation
1.  Clone the repository: `git clone [repository URL]`
2.  Install dependencies: `pip install scikit-learn pandas newsapi numpy`
3.  Run the project: `python main.py`
## Usage
This project detects fake news by using natural language classifier. Train AI model using curated 30 days news from NewAPIClient from [link](https://newsapi.org/docs/client-libraries/python) and news dataset available on Kaggle


This is solving project mentioned on Educative.io site [link](https://www.educative.io/projects/fake-news-detection-using-scikit-learn)

## Model Scorecard
### Input data
. News dataset available on Kaggle.
. The second dataset we’ll create ourselves using the News API. We will use this API to load some data and then append that data to the other dataset.

### Model used
CountVectorizer to get tokens and passive aggressive classifier

### Evaluation 

```
Accuracy score on validation data: 91.32%
Local data prediction accuracy 100.00%
```
