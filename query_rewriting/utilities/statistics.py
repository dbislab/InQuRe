"""
Use variables and functions to save some statistics of the execution (for printout purposes).
"""

from typing import Tuple

# For counting purposes
total_prompt_tokens: int = 0
total_completion_tokens: int = 0
total_tokens: int = 0

# For timing purposes
table_filtering_time: list[float] = []
rewriting_time: list[float] = []

# For prefilter 2
num_suggested_tables_llm: int = 0
num_correctly_suggested_tables_llm: int = 0
num_slightly_wrong_suggested_tables_llm: int = 0
num_wrong_suggested_tables_llm: int = 0

# For pruning
list_num_pruned_queries: list[int] = []


def add_tokens(prompt: int, completion: int, tokens: int):
    """
    Add all kinds of tokens to the overall tokens

    :param int prompt: Number of prompt tokens to add
    :param int completion: Number of completion tokens to add
    :param int tokens: Number of total tokens to add
    """
    global total_prompt_tokens
    global total_completion_tokens
    global total_tokens
    total_prompt_tokens += prompt
    total_completion_tokens += completion
    total_tokens += tokens


def get_tokens() -> Tuple[int, int, int]:
    """
    Return the total number of tokens used

    :return: Total prompt tokens, total completion tokens, total tokens
    :rtype: Tuple[int, int, int]
    """
    return total_prompt_tokens, total_completion_tokens, total_tokens


def change_prefilter_2_statistics(suggested: int, correctly: int, slightly_wrong: int, wrong: int):
    """
    Add the statistics of the current query to the overall statistics

    :param int suggested: Number of suggested tables from the LLM
    :param int correctly: Number of correctly suggested tables from the LLM
    :param int slightly_wrong: Number of nearly correctly suggested tables from the LLM
    :param int wrong: Number of wrongly suggested tables from the LLM
    """
    global num_suggested_tables_llm
    global num_correctly_suggested_tables_llm
    global num_slightly_wrong_suggested_tables_llm
    global num_wrong_suggested_tables_llm
    num_suggested_tables_llm += suggested
    num_correctly_suggested_tables_llm += correctly
    num_slightly_wrong_suggested_tables_llm += slightly_wrong
    num_wrong_suggested_tables_llm += wrong


def get_prefilter_2_statistics() -> Tuple[int, int, int, int]:
    """
    Return the complete statistics for the prefilter 2

    :return: The suggested tables and from them: correct, nearly correct, wrong tables
    :rtype: Tuple[int, int, int, int]
    """
    return (num_suggested_tables_llm, num_correctly_suggested_tables_llm,
            num_slightly_wrong_suggested_tables_llm, num_wrong_suggested_tables_llm)


def add_rewriting_timings(time_prefilter: float, time_rewrite: float):
    """
    Set the correct timings of the prefilter and the rewrite part in the rewrite method

    :param float time_prefilter: Time the pre-filtering took
    :param float time_rewrite: Time the rewriting took
    """
    global table_filtering_time
    global rewriting_time
    table_filtering_time.append(time_prefilter)
    rewriting_time.append(time_rewrite)


def get_rewriting_time() -> Tuple[list[float], list[float]]:
    """
    Get the timings from the rewrite method

    :return: The table filtering time and the rewrite time
    :rtype: Tuple[list[float], list[float]]
    """
    return table_filtering_time, rewriting_time


def add_num_pruned_queries(pr: int):
    """
    Add the number of pruned queries for one query.

    :param int pr: Number of pruned queries for this query
    """
    global list_num_pruned_queries
    list_num_pruned_queries.append(pr)


def get_num_pruned_queries() -> list[int]:
    """
    Get the number of pruned alternatives per query

    :return: The number of pruned alternatives per query
    :rtype: list[int]
    """
    return list_num_pruned_queries