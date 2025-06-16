import logging.config
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from transformers import BertTokenizer, BertModel
import torch
import logging

MAX_NUM_JOBS = 500
JOB_DESC_COLUMN_NAME = 'Job Description'

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

df = pd.read_csv('data/job_title_des.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
logger.debug(f"Total jobs found: {df.count()}")

engStopWord = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')
max_tokens = model.config.max_position_embeddings

def PreProcessJobDesc(jobDesc):
    tokens = word_tokenize(jobDesc.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in engStopWord and token not in string.punctuation]

    tokens = tokens[:max_tokens]

    logger.debug(f"\nPreprocess text tokens with size: {len(tokens)}, is: \n\t {tokens}")

    return ' '.join(tokens)

def ConvertJobDescToBertInputs(jobDesc):
    logger.debug(f"Starting creating BERT tokens for job desc: \n\t{jobDesc}")

    text = PreProcessJobDesc(jobDesc)

    # Get Bert inputs. Trim len to maximum context size of model
    inputs = tokenizer(text, return_tensors='pt', max_length=max_tokens)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    logger.debug(f"Created BERT tokens with size {len(tokens)} is: \n\t{tokens}")
    return inputs

# job_desc_embedding in shape (batch_size, seq_leg, last_hidden_layer_state)
def ConvertToSingleJobDescEmbedding(job_desc_embedding):
    return job_desc_embedding[:, 0, :]
    # Difference solution to take mean on all token embedding
    #return torch.mean(job_desc_embedding, dim=1)

def GetJobDescBertEmbedding(jobDesc):
    inputs = ConvertJobDescToBertInputs(jobDesc)

    with torch.no_grad():
        outputs = model(**inputs)
        # Last hidden state has all text embedding in shape(batch_size, seq_length, hidden_layer)
        job_desc_embedding = outputs.last_hidden_state

        # [CLS] token contain embedding for whole job description
        # as we have batch of 1
        embedding = ConvertToSingleJobDescEmbedding(job_desc_embedding)

        logger.debug(f"Job desc embedding with length {embedding.size()} is:\n\t{embedding}")

        return embedding

df_job_desc_embedding = pd.DataFrame(columns=['Embedding'])

def BuildAllJobDescriptionsEmbedding():
    # Build all jobs embedding
    for i in range(MAX_NUM_JOBS):
        jobDesc = df[JOB_DESC_COLUMN_NAME][i]
        jobDescEmbedding = GetJobDescBertEmbedding(jobDesc)

        logger.info(f"JOB num: {i}\nJOB Description:\n\t{jobDesc}\nEmbedding created with size {jobDescEmbedding.size()}:\n\t{jobDescEmbedding[:10]}")

        df_job_desc_embedding.loc[len(df_job_desc_embedding)] = {'Embedding': jobDescEmbedding}

def FinJobsWithSematicSimilar(query, topN = 3):
    query_embedding = GetJobDescBertEmbedding(query)

    similarities = []

    for i, row in df_job_desc_embedding.iterrows():
        logger.debug(f"For query {query}\nMatching Job num {i}")
        similarity = torch.cosine_similarity(query_embedding, row['Embedding'], dim=1)
        similarities.append((i, similarity))

    similarities.sort(key=lambda x:x[1], reverse = True)

    return similarities[:topN]


BuildAllJobDescriptionsEmbedding()

while (True):
    print("\n\nEnter text for job you want or enter exit")
    text = str(input())

    if (text.lower() == 'exit'):
        exit(0)

    similar_jobs = FinJobsWithSematicSimilar(text)

    print("\nSimilar jobs are:\n")
    for i, s in similar_jobs:
        print(f"\n\n\njob num {i}: \n\t")
        print(f"JobTitle: {df.loc[i]['Job Title']}")
        print(f"JobDescription: {df.loc[i]['Job Description']}")


