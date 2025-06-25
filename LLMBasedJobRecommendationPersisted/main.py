import pandas as pd
import logging
from VectorDB import VectorDBForJobs

MAX_NUM_JOBS = 500
JOB_DESC_COLUMN_NAME = 'Job Description'

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

jobsDB = VectorDBForJobs(logger)

df = pd.read_csv('data/job_title_des.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
logger.debug(f"Total jobs found: {df.count()}")

# df_job_desc_embedding = pd.DataFrame(columns=['Embedding'])

def BuildAllJobDescriptionsEmbedding():
    # Build all jobs embedding
    for i in range(MAX_NUM_JOBS):
        jobDesc = df[JOB_DESC_COLUMN_NAME][i]
        jobsDB.upsertToCollection([jobDesc], [str(i)])

        logger.info(f"Added to vector db JOB num: {i}\nJOB Description:\n\t{jobDesc}")

        # df_job_desc_embedding.loc[len(df_job_desc_embedding)] = {'Embedding': jobDescEmbedding}

def FinJobsWithSematicSimilar(query: str, topN = 3) -> tuple[list[str], list[str]]:
    result_job_descs, results_ids = jobsDB.queryFromCollection([query])

    return (result_job_descs, results_ids)


# TODO: Make this conditional and make sure ids are matching whats in DB and what is cvs file
BuildAllJobDescriptionsEmbedding()

while (True):
    print("\n\nEnter text for job you want or enter exit")
    text = str(input())

    if (text.lower() == 'exit'):
        exit(0)

    similar_jobs = FinJobsWithSematicSimilar(text)

    print("\nSimilar jobs are:\n")
    for id, job_desc in zip(similar_jobs):
        print(f"\n\n\njob num {id}: \n\t")
        # TODO: Get from metadata
        print(f"JobTitle: {df.loc[int(id)]['Job Title']}")
        print(f"JobDescription: {job_desc}")


