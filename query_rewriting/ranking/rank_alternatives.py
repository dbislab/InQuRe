"""
Ranking algorithms for the produced alternative queries
"""
from query_rewriting.config import NotYetSupportedException, RankingNotPossible
from collections.abc import Callable
from query_rewriting.distance_measures.sql_queries_comparison import difflib_simple_comparison, \
    similarity_using_clauses, similarity_using_tables, llm_intent_similarity_measure, \
    llm_intent_and_embedding_similarity
from query_rewriting.ranking.prune_alternatives import prune_alternatives
from query_rewriting.ranking.ranking_algorithms import maximal_marginal_relevance, simple_ranker, \
    rank_via_clustering_dbscan, rank_via_clustering_kmeans, rank_via_llm

import query_rewriting.config as config

# Ranking algorithm and distance function configurable separately to combine in multiple ways
# tested in the workflow
def rank_alternative_queries(input_request: str, alternative_queries: list[str], ranker_kind: int,
                             output_length: int) -> list[str]:
    """
    Rank the queries that are produced as alternatives for the original input.

    :param str input_request: The input request that the system got
    :param list[str] alternative_queries: The produced alternative queries that should be ranked
    :param int ranker_kind: Defines what ranking algorithm to perform (1 for a simple ranker)
    :param int output_length: The length of the output list (only top-k queries returned)
    :return: The produced alternative queries reranked in a new list (ranking is given by the order in the list)
    :rtype: list[str]
    """
    alternative_queries_pruned: list[str] = prune_alternatives(input_request, alternative_queries)
    # Check if there are enough queries to start the ranking process
    if len(alternative_queries_pruned) < output_length:
        raise RankingNotPossible("Too many queries were pruned to have enough queries for the output.")
    ranked_queries: list[str] = []
    # Set the wanted string similarity function
    if config.sim_measure_string == 1:
        string_sim_func: Callable[[str, str], float] = difflib_simple_comparison
    elif config.sim_measure_string == 2:
        string_sim_func: Callable[[str, str], float] = similarity_using_clauses
    else:
        raise NotYetSupportedException(f"Ranking queries using string measure {config.sim_measure_string} "
                                       f"is not yet supported")
    # Set the wanted intent similarity function
    if config.sim_measure_intent == 1:
        intent_sim_func: Callable[[str, str], float] = similarity_using_tables
    elif config.sim_measure_intent == 2:
        intent_sim_func: Callable[[str, list[str]], list[float]] = llm_intent_similarity_measure
    elif config.sim_measure_intent == 3:
        intent_sim_func: Callable[[str, list[str]], list[float]] = llm_intent_and_embedding_similarity
    else:
        raise NotYetSupportedException(f"Ranking queries using intent measure {config.sim_measure_intent} "
                                       f"is not yet supported")
    if ranker_kind == 1:
        # use MMR (lambda=0.5) with the inbuilt difflib string comparison
        ranked_queries = simple_ranking_algorithm(input_request, alternative_queries_pruned,
                                                  difflib_simple_comparison, 0.5, output_length)
    elif ranker_kind == 2:
        ranked_queries = simple_ranker(input_request, alternative_queries_pruned, intent_sim_func, output_length)
    elif ranker_kind == 3:
        ranked_queries = maximal_marginal_relevance(input_request, alternative_queries_pruned, string_sim_func,
                                                    intent_sim_func, 0.7, output_length)
    elif ranker_kind == 4:
        ranked_queries = rank_via_clustering_dbscan(input_request, alternative_queries_pruned, string_sim_func,
                                                    intent_sim_func, output_length)
    elif ranker_kind == 5:
        ranked_queries = rank_via_clustering_kmeans(input_request, alternative_queries_pruned, intent_sim_func,
                                                    output_length)
    elif ranker_kind == 6:
        ranked_queries = rank_via_llm(input_request, alternative_queries_pruned, output_length)
    else:
        raise NotYetSupportedException(f"Ranking queries using kind {ranker_kind} is not yet supported")
    # Check if there are still enough queries after the ranking
    if len(ranked_queries) < output_length:
        raise RankingNotPossible("Too few queries were returned after the ranking.")
    return ranked_queries


# no test needed (simple function call)
def simple_ranking_algorithm(input_request: str, alternative_queries: list[str],
                             similarity_measure: Callable[[str, str], float], lambda_var: float, output_length: int) \
        -> list[str]:
    """
    Rank the queries that are produced as alternatives for the original input using a basic ranking algorithm.
    The algorithm compares the different SQL clauses using a simple inbuilt
    string similarity and ranks them based on MMR.

    :param str input_request: The input request that the system got
    :param list[str] alternative_queries: The produced alternative queries that should be ranked
    :param Callable[[str,str],float] similarity_measure: The similarity measure used to
           calculate the similarity between queries for the ranking
    :param float lambda_var: Value between 0 and 1 to weigh the similarities in the MMR algorithm
    :param int output_length: The length of the output list (only top-k queries returned)
    :return: The produced alternative queries reranked in a new list (ranking is given by the order in the list)
    :rtype: list[str]
    """
    # calculate MMR
    return maximal_marginal_relevance(input_request, alternative_queries, similarity_measure, similarity_measure,
                                      lambda_var, output_length)






