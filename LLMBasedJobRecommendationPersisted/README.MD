# LLM based job recommendation
This project give recommendation for jobs for job seeker using Large Language model

## Installation
1.  Clone the repository: `git clone [repository URL]`
2.  Install dependencies: 
        `pip install scikit-learn pandas nltk transformers torch` 
        Install nltk dependencies:
            nltk.download('stopwords')
            nltk.download('punkt_tab')
            nltk.download('wordnet')
        Install hf_xet for Xert Storage
            pip install huggingface_hub[hf_xet]
3.  Run the project: `python main.py`

## Usage
Giving recommendation of Jobs based on query you ask and uses LLM based model to find semantically matching top 3 jobs.

Only first 100 jobs from Kaggle's dataset are considered

## Model Scorecard
### Input data
[Kaggle Job and description](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)

### Model used
Google's `bert-base-uncased`

### Sample output 
```
Enter text for job you want or enter exit
java with web development experience
\nSimilar jobs are:




job num 61:

JobTitle: JavaScript Developer
JobDescription: Javascript developers

As a javascript developer you’ll be responsible for creating frontend application using angular and other similar frameworks.

You’ll be working with an amazing team of frontend developers to package JS applications to be deployed on iOS and android devices.     

You will need to have experience in integrating different third party APIs using JS, to provide functionality to our websites and apps. 

Experience with frontend development frameworks, jQuery and Git will be a bonus.

Email your CV



job num 162:

JobTitle: Node js developer
JobDescription: Skills
Nodejs, Express JS, Javascript, ES6, jQuery, GIT
Roles And Responsibilities
Assisting in the integration of technical and application components.
Coding and testing program modules; assisting in design
Maintaining, tuning and repairing applications; providing user support
Participating in execution and documentation of tests
Executing conversion plans and documenting possible improvements; identifying disruptions in the processes



job num 1:

JobTitle: Django Developer
JobDescription: PYTHON/DJANGO (Developer/Lead) - Job Code(PDJ - 04)
Strong Python experience in API development (REST/RPC).
Experience working with API Frameworks (Django/flask).
Experience evaluating and improving the efficiency of programs in a Linux environment.
Ability to effectively handle multiple tasks with a high level of accuracy and attention to detail.
Good verbal and written communication skills.
Working knowledge of SQL.
JSON experience preferred.
Good knowledge in automated unit testing using PyUnit.
```