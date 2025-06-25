from chromadb import Document, Documents, EmbeddingFunction, Embedding, Embeddings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from transformers import BertTokenizer, BertModel
import torch
import logging

class CustomEmbeddingFunction(EmbeddingFunction):
    
    def __init(self, logger: logging.Logger):
        self.__self.__logger = logger
        self.__engStopWord = set(stopwords.words('english'))
        self.__lemmatizer = WordNetLemmatizer()

        self.__tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.__model = BertModel.from_pretrained('bert-base-uncased')
        self.__max_tokens = self.__model.config.max_position_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []

        # TODO: Can be done in batches
        for document in input:
            embedding = self.GetJobDescBertEmbedding(document)
            embeddings.append(embedding)

        return embeddings

    def __PreProcessJobDesc(self, jobDesc: Document) -> Document:
        tokens = word_tokenize(jobDesc.lower())
        tokens = [self.__lemmatizer.lemmatize(token) for token in tokens if token not in engStopWord and token not in string.punctuation]

        tokens = tokens[:self.__max_tokens]

        self.__logger.debug(f"\nPreprocess text tokens with size: {len(tokens)}, is: \n\t {tokens}")

        return ' '.join(tokens)

    def __ConvertJobDescToBertInputs(self, jobDesc: Document):
        self.__logger.debug(f"Starting creating BERT tokens for job desc: \n\t{jobDesc}")

        text = self.__PreProcessJobDesc(jobDesc)

        # Get Bert inputs. Trim len to maximum context size of model
        inputs = self.__tokenizer(text, return_tensors='pt', max_length=self.__max_tokens)

        tokens = self.__tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        self.__logger.debug(f"Created BERT tokens with size {len(tokens)} is: \n\t{tokens}")
        return inputs

    # job_desc_embedding in shape (batch_size, seq_leg, last_hidden_layer_state)
    def __ConvertToSingleJobDescEmbedding(self, job_desc_embedding):
        return job_desc_embedding[:, 0, :]
        # Difference solution to take mean on all token embedding
        #return torch.mean(job_desc_embedding, dim=1)

    def GetJobDescBertEmbedding(self, jobDesc: Document) -> Embedding:
        inputs = self.__ConvertJobDescToBertInputs(jobDesc)

        with torch.no_grad():
            outputs = self.__model(**inputs)
            # Last hidden state has all text embedding in shape(batch_size, seq_length, hidden_layer)
            job_desc_embedding = outputs.last_hidden_state

            # [CLS] token contain embedding for whole job description
            # as we have batch of 1
            embedding = self.__ConvertToSingleJobDescEmbedding(job_desc_embedding)

            self.__logger.debug(f"Job desc embedding with length {embedding.size()} is:\n\t{embedding}")

            return embedding
