"""
A file to combine all methods of ranking algorithms.
"""

import numpy as np

from collections.abc import Callable
# from openai.types.chat import ChatCompletion
from collections.abc import Iterable
from sklearn.cluster import KMeans, DBSCAN  #, HDBSCAN
#from itertools import zip_longest
from inspect import signature

from query_rewriting.config import RankingNotPossible
from query_rewriting.distance_measures.sql_queries_comparison import difflib_simple_comparison
from query_rewriting.distance_measures.vector_embedding import model_embedding
from query_rewriting.utilities.gpt_functions import gpt_api_call, strip_code_block_output, \
    strip_starting_and_ending_characters
import query_rewriting.config as config
from query_rewriting.utilities.statistics import add_tokens

prompt_tokens_ranking_algorithms: int = 0
completion_tokens_ranking_algorithms: int = 0
total_tokens_ranking_algorithms: int = 0


# Rankers that use similarity to original query and similarity to other queries get 2 distance functions:
#  1. distance to original query using intent, 2. distance to other queries using string similarity


def maximal_marginal_relevance(input_request: str, alternative_queries: list[str],
                               similarity_measure: Callable[[str, str], float],
                               intent_similarity_measure: Callable[[str, str], float] | Callable[
                                   [str, list[str]], list[float]], lambda_var: float,
                               output_length: int) -> list[str]:
    """
    Rank the queries that are produced as alternatives for the original input using
    the maximal marginal relevance algorithm.
    It is given two distances, one for similarity calculation between queries
    and one for similarity calculation to the original query.

    :param str input_request: The input request that the system got
    :param list[str] alternative_queries: The produced alternative queries that should be ranked
    :param Callable[[str,str],float] similarity_measure: The similarity measure used
           to calculate the similarity between queries
    :param Callable[[str,str],float] | Callable[[str, list[str]], list[float]] intent_similarity_measure:
           The similarity measure used to calculate the similarity to the original query
    :param float lambda_var: Value between 0 and 1 to weigh the similarities
    :param int output_length: The length of the output list (only top-k queries returned)
    :return: The produced alternative queries reranked in a new list (ranking is given by the order in the list)
    :rtype: list[str]
    """
    # define the similarity matrix and vector for the queries
    dimension: int = len(alternative_queries)
    similarity_matrix: list[list[float]] = np.zeros((dimension, dimension), dtype=float).tolist()
    similarity_vector: list[float] = np.zeros(dimension, dtype=float).tolist()
    # calculate the matrix for the query similarities
    # and the vector for the similarity to the original query
    sim_func_signature = signature(intent_similarity_measure)
    # print(str(sim_func_signature))
    for i, elem1 in enumerate(alternative_queries):
        if str(sim_func_signature).endswith('float'):
            # print('entered case 1')
            similarity_vector[i] = intent_similarity_measure(input_request, elem1)
        for j, elem2 in enumerate(alternative_queries):
            similarity_matrix[i][j] = similarity_measure(elem1, elem2)
    if str(sim_func_signature).endswith('list[float]'):
        # print('entered case 2')
        similarity_vector = intent_similarity_measure(input_request, alternative_queries)
    ranked_queries: list[str] = []
    ranked_queries_indices: list[int] = []
    # rank all existent queries
    while len(ranked_queries) < output_length:
        max_sim: float = -1.1
        max_index: int = dimension  # would produce an error if array access called on this
        # search for the query with the highest MMR value
        for i, elem1 in enumerate(alternative_queries):
            if i in ranked_queries_indices:
                # query is already in ranked output
                continue
            max_sim_to_doc: float = -1.1
            # calculate the maximum similarity to an already returned document
            for index in ranked_queries_indices:
                current_sim: float = similarity_matrix[i][index]
                if current_sim > max_sim_to_doc:
                    max_sim_to_doc = current_sim
            # calculate the MMR formula
            sim: float = lambda_var * similarity_vector[i] - (1 - lambda_var) * max_sim_to_doc
            if sim > max_sim:
                # the current query is the new maximum
                max_sim = sim
                max_index = i
        # add the maximum to the ranked queries
        ranked_queries.append(alternative_queries[max_index])
        ranked_queries_indices.append(max_index)
    return ranked_queries


def simple_ranker(input_request: str, alternative_queries: list[str],
                  similarity_measure: Callable[[str, str], float] | Callable[[str, list[str]], list[float]],
                  output_length: int) -> list[str]:
    """
    Rank the queries that are produced as alternatives for the original input using a simple sorting
    based on the similarity to the original query.

    :param str input_request: The input request that the system got
    :param list[str] alternative_queries: The produced alternative queries that should be ranked
    :param Callable[[str,str],float] | Callable[[str, list[str]], list[float]] similarity_measure:
           The similarity measure used to calculate the similarity to another query
    :param int output_length: The length of the output list (only top-k queries returned)
    :return: The produced alternative queries reranked in a new list (ranking is given by the order in the list)
    :rtype: list[str]
    """
    # Return element if there is just 1 elem in list
    # Otherwise: Calculate the similarities to the original query
    if len(alternative_queries) == 1 and output_length == 1:
        return alternative_queries
    similarities: list[(str, float)] = []
    sim_func_signature = signature(similarity_measure)
    if str(sim_func_signature).endswith('float'):
        for query in alternative_queries:
            similarities.append((query, similarity_measure(input_request, query)))
    elif str(sim_func_signature).endswith('list[float]'):
        sims: list[float] = similarity_measure(input_request, alternative_queries)
        for i in range(0, len(alternative_queries)):
            similarities.append((alternative_queries[i], sims[i]))
    # Get the top alternatives by sorting after the similarity (descending)
    top_alternatives: list[str] = []
    sorted_similarities: list[(str, float)] = sorted(similarities, key=lambda x: x[1], reverse=True)
    for i in range(0, output_length):
        top_alternatives.append(sorted_similarities[i][0])
    return top_alternatives


def rank_via_clustering_dbscan(input_request: str, alternative_queries: list[str],
                               similarity_measure: Callable[[str, str], float],
                               intent_similarity_measure: Callable[[str, str], float] | Callable[
                                   [str, list[str]], list[float]],
                               output_length: int) -> list[str]:
    """
    Rank the queries that are produced as alternatives for the original input using a clustering algorithm (DBSCAN).
    The top queries are selected from the different clusters and sorted based on similarity to the original query.
    Outliers are eliminated.

    :param str input_request: The input request that the system got
    :param list[str] alternative_queries: The produced alternative queries that should be ranked
    :param Callable[[str,str],float] similarity_measure: The similarity measure used
           to calculate the similarity between queries (assumed to be in the interval [0,1])
    :param Callable[[str,str],float] | Callable[[str, list[str]], list[float]] intent_similarity_measure:
           The similarity measure used to calculate the similarity to the original query
    :param int output_length: The length of the output list (only top-k queries returned)
    :return: The produced alternative queries reranked in a new list (ranking is given by the order in the list)
    :rtype: list[str]
    """
    # implemented via scikit-learn
    # Make the distance matrix
    dimension: int = len(alternative_queries)
    distance_matrix: list[list[float]] = np.zeros((dimension, dimension), dtype=float).tolist()
    for i, query1 in enumerate(alternative_queries):
        for j, query2 in enumerate(alternative_queries):
            if query1 != query2:
                distance_matrix[i][j] = 1 - similarity_measure(query1, query2)
            else:
                distance_matrix[i][j] = 0
    # print(distance_matrix)
    # Cluster the elements
    clustering: DBSCAN = (DBSCAN(min_samples=int(min(dimension / output_length, 5)), metric='precomputed')
                          .fit(distance_matrix))
    # print(int(min(dimension / output_length, 5)))
    cluster_indices: list[float] = list(clustering.labels_)  # Noise samples have value -1
    # print(cluster_indices)
    # Get the number of clusters and the existing indices
    distinct_cluster_indices: list[float] = list(set(cluster_indices))
    num_clusters: int = len(distinct_cluster_indices) - (1 if -1 in cluster_indices else 0)
    print(f"Clusters formed by DBScan: {num_clusters}")
    # print(distinct_cluster_indices)
    if -1 in distinct_cluster_indices:
        distinct_cluster_indices.remove(-1)
    cluster_ranked_lists: list[list[str]] = []
    # Sort the elements per cluster into ranked lists
    for cluster_index in distinct_cluster_indices:
        cluster_queries: list[str] = [alternative_queries[i] for i in range(0, len(cluster_indices))
                                      if cluster_indices[i] == cluster_index]
        ranked_cluster_queries: list[str] = simple_ranker(input_request, cluster_queries,
                                                          intent_similarity_measure, len(cluster_queries))
        cluster_ranked_lists.append(ranked_cluster_queries)
    # Go through the first, second,... elements in each cluster and sort those and add them to the result
    result_list: list[str] = []
    while cluster_ranked_lists != [] and len(result_list) < output_length:
        current_queries: list[str] = [cluster_list.pop(0) for cluster_list in cluster_ranked_lists]
        cluster_ranked_lists = [cluster_list for cluster_list in cluster_ranked_lists if cluster_list != []]
        # Needed length is either all elements or
        # only those to fully fill the result length to the required output length
        needed_length: int = min(len(current_queries), output_length - len(result_list))
        ranked_current_queries: list[str] = simple_ranker(input_request, current_queries,
                                                          intent_similarity_measure, needed_length)
        result_list.extend(ranked_current_queries)
    # Add outliers if the result list is too short
    if len(result_list) < output_length:
        outlier_queries: list[str] = [alternative_queries[i] for i in range(0, len(cluster_indices))
                                      if cluster_indices[i] == -1]
        needed_length: int = min(len(outlier_queries), output_length - len(result_list))
        ranked_outliers: list[str] = simple_ranker(input_request, outlier_queries, intent_similarity_measure,
                                                   needed_length)
        result_list.extend(ranked_outliers)
    return result_list


def rank_via_clustering_kmeans(input_request: str, alternative_queries: list[str],
                               intent_similarity_measure: Callable[[str, str], float] | Callable[
                                   [str, list[str]], list[float]],
                               output_length: int) -> list[str]:
    """
    Rank the queries that are produced as alternatives for the original input using a clustering algorithm (k-means).
    The clustering is done based on the embeddings of the different alternative queries.
    The top queries are selected from the different clusters and sorted based on similarity to the original query.
    One element from each cluster is taken for the result.

    :param str input_request: The input request that the system got
    :param list[str] alternative_queries: The produced alternative queries that should be ranked
    :param Callable[[str,str],float] | Callable[[str, list[str]], list[float]] intent_similarity_measure:
           The similarity measure used to calculate the similarity to the original query
    :param int output_length: The length of the output list (only top-k queries returned)
    :return: The produced alternative queries reranked in a new list (ranking is given by the order in the list)
    :rtype: list[str]
    """
    # Calculate the embeddings
    query_tensors: list[np.array] = []
    for query in alternative_queries:
        query_tensors.append(model_embedding(query))
    # Cluster using k-means with number of clusters = length of output
    clustering: KMeans = KMeans(n_clusters=output_length).fit(query_tensors)
    cluster_indices: list[int] = list(clustering.labels_)
    # print(cluster_indices)
    # Sort the elements per cluster into ranked lists
    top_cluster_elements: list[str] = []
    # numbering of clusters from 0 to output_length - 1
    for cluster_index in range(0, output_length):
        cluster_queries: list[str] = [alternative_queries[i] for i in range(0, len(cluster_indices))
                                      if cluster_indices[i] == cluster_index]
        ranked_cluster_queries: list[str] = simple_ranker(input_request, cluster_queries,
                                                          intent_similarity_measure, len(cluster_queries))
        # Take the best element of the cluster
        top_cluster_elements.append(ranked_cluster_queries[0])
    # Rank the top elements of all the clusters, length of them should be output_length here
    top_cluster_elements_ranked: list[str] = simple_ranker(input_request, top_cluster_elements,
                                                           intent_similarity_measure, len(top_cluster_elements))
    return top_cluster_elements_ranked


# tested in workflow
def rank_via_llm(input_request: str, alternative_queries: list[str],
                 output_length: int) -> list[str]:
    """
    Rank the queries that are produced as alternatives for the original input using an LLM.
    The LLM is asked to sort the queries based on their similarity to the original query
    while also considering heterogeneity in the result.

    :param str input_request: The input request that the system got
    :param list[str] alternative_queries: The produced alternative queries that should be ranked
    :param int output_length: The length of the output list (only top-k queries returned)
    :return: The produced alternative queries reranked in a new list (ranking is given by the order in the list)
    :rtype: list[str]
    """
    # Prompt for LLM
    alt_queries_formatted: list[str] = [q.strip() if q.strip().endswith(";")
                                        else q.strip() + ";" for q in alternative_queries]
    alt_queries_str: str = "\n".join(alt_queries_formatted)
    # Prompt for LLM
    content_for_gpt = (f"I will give you a single SQL query called original query.\n"
                       f"I will also give you multiple other SQL queries called alternative queries.\n"
                       f"I want you to rank the alternative queries according to their similarity to the original "
                       f"query (the highest similarity should be at the top).\n"
                       f"The similarity of two SQL queries in this context is defined as follows:\n"
                       f"Two queries are similar if the insight they give to a human being when executed "
                       f"on a database is similar.\n"
                       f"I.e. the intents of the two queries are similar from a human perspective.\n"
                       f"So to get the similarity think step-by-step:\n"
                       f"1. What is the intent from the alternative query when abstracting from the exact SQL? "
                       f"Think about the intent the user has in mind.\n"
                       f"2. Which similarity value would this alternative query get to the original one? "
                       f"Take into consideration that queries that have results that somehow correlate with the "
                       f"original query should have a high similarity.\n"
                       f"3. Sort the alternative queries based on this rank. "
                       f"The ranking should also be heterogeneous, "
                       f"so if multiple queries are really similar based on their strings, "
                       f"you might reorder them, such that they are not directly after each other.\n"
                       f"Give back nothing else but all the alternative queries "
                       f"sorted by the ranking and separated with new lines.\n"
                       f"Here is the original query:\n"
                       f"{input_request}\n"
                       f"Here are the alternative queries (separated by semicolons):\n"
                       f"{alt_queries_str}")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in SQL."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # print(f"Prompt:\n{content_for_gpt}")
    # print()
    # GPT call and result extraction
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    suggested_ranking: str = completion["choices"][0]["message"]["content"]
    # print(f"GPT output:\n{suggested_ranking}")
    suggested_ranking = strip_code_block_output(suggested_ranking)
    separated_ranked_queries: list[str] = strip_starting_and_ending_characters(suggested_ranking).splitlines()
    separated_ranked_queries = [query.strip() for query in separated_ranked_queries if query.strip() != ""]
    # print(separated_ranked_queries)
    # Postprocessing to check LLM results
    resulting_rank: list[str] = []
    alt_queries_copy: list[str] = alt_queries_formatted.copy()
    for ranked_query in separated_ranked_queries:
        # print(ranked_query)
        if ranked_query in alt_queries_copy:
            # print("Found")
            resulting_rank.append(ranked_query)
            alt_queries_copy.remove(ranked_query)
            # print(resulting_rank)
        else:
            # print("Not found")
            similarities: list[float] = (
                list(map(lambda x: difflib_simple_comparison(input_request, x), alt_queries_copy)))
            max_index: int = similarities.index(max(similarities))
            max_query: str = alt_queries_copy[max_index]
            resulting_rank.append(max_query)
            alt_queries_copy.remove(max_query)
            # print(resulting_rank)
    if llm_used:
        global prompt_tokens_ranking_algorithms
        prompt_tokens_ranking_algorithms = completion["usage"]["prompt_tokens"]
        global completion_tokens_ranking_algorithms
        completion_tokens_ranking_algorithms = completion["usage"]["completion_tokens"]
        global total_tokens_ranking_algorithms
        total_tokens_ranking_algorithms = completion["usage"]["total_tokens"]
        print(
            f"Tokens used after the ranking: \n\tPrompt Tokens: {prompt_tokens_ranking_algorithms}, "
            f"Completion Tokens: {completion_tokens_ranking_algorithms}, Total Tokens: "
            f"{total_tokens_ranking_algorithms}\n")
        add_tokens(completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"],
                   completion["usage"]["total_tokens"])
    else:
        print("No tokens used for ranking due to reproducibility DB.")
    # print(resulting_rank)
    if len(resulting_rank) < output_length:
        raise RankingNotPossible("Not enough queries in the ranking of the LLM.")
    return resulting_rank[:output_length]




