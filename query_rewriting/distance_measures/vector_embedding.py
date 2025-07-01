"""
Embed vectors using transformers
"""
import os
import time
from typing import Literal

import numpy as np
import spacy
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from spacy import Language
from spacy.tokens import Doc
from torch import Tensor
from numpy import dot
from numpy.linalg import norm

from query_rewriting.utilities.statistics import add_tokens

# Global variable for setting the model to use for the embedding
model: SentenceTransformer
# tokens used in the methods of this file
prompt_tokens_embedding: int = 0


def set_up_model(wanted_model: str):
    """
    Set up the model that we want for the embedding, e.g. 'sentence-transformers/all-mpnet-base-v2'.
    Needs to be called before the other functions, if a model from sentence-transformers is to be used.
    The model will use its default distance function for vector distances.

    :param str wanted_model: The model that we want to use
    """
    global model
    # Load a pretrained Sentence Transformer
    model = SentenceTransformer(wanted_model)


# not tested, be careful!
def set_up_external_model(wanted_model: str):
    """
    Set up the model that we want for the embedding, e.g. 'dunzhang/stella_en_400M_v5'.
    Needs to be called before the other functions, if a model from sentence-transformers is to be used.
    The model will use its default distance function for vector distances.
    The model load is for custom (not from sentence transformers) models with trusted code.
    Only do this after checking the code of the model.

    :param str wanted_model: The model that we want to use
    """
    global model
    # Load a pretrained Sentence Transformer
    model = SentenceTransformer(wanted_model, trust_remote_code=True)


def model_embedding(input_sentence: str) -> Tensor:
    """
    Use the set model to embed a single sentence.

    :param str input_sentence: The sentence we want to embed
    :return: The embedding of the sentence as a Tensor (multidimensional array)
    :rtype: Tensor
    """
    # Calculate embedding of the input
    global model
    embedding: Tensor = model.encode(input_sentence)
    # print(embedding)
    return embedding


def model_embedding_list(input_list: list[str]) -> np.ndarray:
    """
    Use the set model to embed a list of sentences.

    :param list[str] input_list: The sentences we want to embed
    :return: The embedded sentences in a matrix, such that the shape is (num_sentences, embedding_dim)
    :rtype: np.ndarray
    """
    # Calculate embedding of the input list
    global model
    embedding: np.ndarray = model.encode(input_list)
    # print(embedding)
    return embedding


def model_similarity(input_tensor1: Tensor, input_tensor2: Tensor) -> float:
    """
    Calculate the similarity between two tensors, where each tensor only embeds one sentence/word.
    The higher the similarity, the better.

    :param Tensor input_tensor1: The tensor of the first sentence/word
    :param Tensor input_tensor2: The tensor of the second sentence/word
    :return: The similarity between the two tensors using the set distance function (default ist cosine)
    :rtype: float
    """
    # Calculate the embedding similarities (default function: cosine) as a matrix
    global model
    similarity: Tensor = model.similarity(input_tensor1, input_tensor2)
    # Extract the only existent similarity from the matrix
    sim_value: float = similarity.numpy().item()
    return sim_value


def model_similarity_for_lists(input_tensor1: Tensor | np.ndarray, input_tensor2: Tensor | np.ndarray) -> Tensor:
    """
    Calculate the similarity between two tensors of variable length.

    :param Tensor input_tensor1: The tensor of the first sentences
    :param Tensor input_tensor2: The tensor of the second sentences
    :return: The similarity between the two tensors (as a matrix of size numEmbeddings1 x numEmbeddings2)
             using the set distance function (default ist cosine)
    :rtype: Tensor
    """
    # Calculate the embedding similarities (default function: cosine) as a matrix
    global model
    similarity: Tensor = model.similarity(input_tensor1, input_tensor2)
    return similarity


# Model embedding for models that need prompts (for testing different models)
# not tested
def model_similarity_with_prompts(input_queries: list[str], input_docs: list[str], prompt: str) -> list[float]:
    """
    Calculate the similarity between query and document for sentence transformers with prompts.

    :param list[str] input_queries: The given queries
    :param list[str] input_docs: The docs we want to compare
    :param str prompt: The prompt for the model
    :return: The similarities as a matrix
    :rtype: list[float]
    """
    query_embed = model.encode(input_queries, prompt_name=prompt)
    doc_embed = model.encode(input_docs)
    return list(query_embed @ doc_embed.T)


# not tested in test_query_rewriting.py, as it makes LLM call
def embedding_gpt(input_string: str, wanted_model: str = 'text-embedding-3-small') -> list[float]:
    """
    Embed the string using an embedder model from the GPT API

    :param str input_string: The sentence we want to embed
    :param str wanted_model: The embedding model used for the call
           (text-embedding-3-large, text-embedding-3-small or text-embedding-ada-002)
    :return: The vector returned from the API call
    :rtype: A list with floats representing the vector
    """
    # Set the GPT API key
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    # Make the request
    embedding = client.embeddings.create(input=[input_string], model=wanted_model)
    global prompt_tokens_embedding
    prompt_tokens_embedding = embedding.usage.prompt_tokens
    print(f"Tokens used in the embedding input: {prompt_tokens_embedding}\n")
    add_tokens(embedding.usage.prompt_tokens, 0, embedding.usage.prompt_tokens)
    return embedding.data[0].embedding


# Done already in config to save time, only called if really needed
def load_en_core_web_lg() -> Language:
    """
    Load the en_core_web_lg model from spacy. This needs to be executed before using it in any method.

    :return: The en_core_web_lg model language
    """
    # Load the model
    # start_time: float = time.time()
    nlp: Language = spacy.load('en_core_web_lg')
    # print(f"Spacy en_core_web_lg is in folder: {nlp.path}")
    # end_time: float = time.time()
    # print(f"Loading en_core_web_lg took {end_time-start_time} seconds.")
    return nlp


def similarity_spacy_en_core_web_lg(input_str1: str, input_str2: str, nlp: Language) -> float:
    """
    Calculate the similarity of two words/sentences using spacy en_core_web_lg
    (similarity function is cosine similarity).

    :param str input_str1: The first string to compare
    :param str input_str2: The second string to compare
    :param Language nlp: The preloaded spacy model
    :return: The cosine similarity between the spacy embeddings two strings
    :rtype: float
    """
    # Spacy function:
    # nlp(string): Split input into words and annotate (no stopword removal!)
    # Doc has:
    #   tokens (for token in doc) with the attributes:
    #       token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #       token.shape_, token.is_alpha, token.is_stop,
    #       token.has_vector, token.vector_norm, token.is_oov
    #   entities (for ent in doc.ents) with the attributes:
    #       ent.text, ent.start_char, ent.end_char, ent.label_
    doc1 = nlp(input_str1)
    doc2 = nlp(input_str2)
    # Return cosine similarity
    return doc1.similarity(doc2)


def precalculate_docs_for_spacy_similarity(input_list: list[str], nlp: Language) -> list[Doc]:
    """
    Calculate the tokenization etc. from spacy for all elements in the input list.

    :param list[str] input_list: The input elements
    :param Language nlp: The preloaded spacy model
    :return: All documents prepared for spacy similarity
    :rtype: list[Doc]
    """
    return_list: list[Doc] = []
    for input_str in input_list:
        return_list.append(nlp(input_str))
    return return_list


def calculate_cosine_sim(input_vector1: list[float], input_vector2: list[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    :param list[float] input_vector1: The first vector as a list of floats
    :param list[float] input_vector2: The second vector as a list of floats
    :return: The cosine similarity between the two vectors
    :rtype: float
    """
    return dot(input_vector1, input_vector2) / (norm(input_vector1) * norm(input_vector2))


options = Literal["cos", "dot", "euc", "man"]


# Does not need testing, only uses library functions
def calculate_tensor_sim_via_sentence_transformers(input1: Tensor, input2: Tensor, sim_func: options = "cos") -> float:
    """
    Calculate the cosine similarity, dot product, Euclidean similarity or manhattan similarity
    between two vectors using the sentence transformers library.
    Warning: Not all options have the same value range!

    :param list[float] input1: The first vector as a Tensor
    :param list[float] input2: The second vector as a Tensor
    :param options sim_func: The similarity function used
    :return: The cosine similarity between the two Tensors
    :rtype: float
    """
    if sim_func == "cos":
        # Extract the only existent similarity from the matrix
        return util.cos_sim(input1, input2).numpy().item()
    elif sim_func == "dot":
        return util.dot_score(input1, input2).numpy().item()
    elif sim_func == "euc":
        return util.euclidean_sim(input1, input2).numpy().item()
    elif sim_func == "man":
        return util.manhattan_sim(input1, input2).numpy().item()
