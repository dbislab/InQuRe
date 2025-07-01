"""
Prune the bad alternative queries before even going to the ranking.
"""
import os.path

import query_rewriting.config as config
from query_rewriting.utilities.sql_parsing import extract_tables
from query_rewriting.utilities.statistics import add_num_pruned_queries


# no tests currently needed
def prune_alternatives(input_request: str, alternative_queries: list[str]) -> list[str]:
    """
    Prune the alternative queries that are too far off from the original query.

    :param str input_request: The original query from the input
    :param list[str] alternative_queries: The queries produced as alternatives
    :return: All queries that were not pruned
    :rtype: list[str]
    """
    pruned_queries: list[str] = alternative_queries.copy()
    if config.db_prefixes:
        pruned_queries = simple_pruning_via_prefix_of_db(alternative_queries)
    return pruned_queries


def simple_pruning_via_prefix_of_db(alternative_queries: list[str]) -> list[str]:
    """
    Prune those queries that contain tables from different databases
    (only usable if tables from the same DB have the same prefix).

    :param list[str] alternative_queries: The queries produced as alternatives
    :return: Only those queries using tables from one DB
    :rtype: list[str]
    """
    pruned_queries: list[str] = []
    for query in alternative_queries:
        tables: list[str] = extract_tables(query)
        common_prefix: str = os.path.commonprefix(tables)
        if common_prefix.endswith("_") or len(common_prefix) > 5:
            pruned_queries.append(query)
    add_num_pruned_queries(len(alternative_queries) - len(pruned_queries))
    return pruned_queries