"""
Functions for getting rewrites of the original query
"""
import time
from typing import Tuple

# from openai.types.chat import ChatCompletion

from query_rewriting.config import NotYetSupportedException, NoRewritesFoundException
from collections.abc import Iterable

import query_rewriting.config as config
from query_rewriting.utilities.duckdb_functions import get_usable_constraints_from_db
from query_rewriting.utilities.find_metadata import llm_find_intent_sql
from query_rewriting.utilities.gpt_functions import strip_sql_output, gpt_api_call, \
    prepare_db_schema_for_prompt_including_fk
from query_rewriting.rewrite_generator.prefilter_tables import prefilter_tables
from query_rewriting.utilities.statistics import add_rewriting_timings, add_tokens

# tokens used in the methods of this file
prompt_tokens_query_rewriting: int = 0
completion_tokens_query_rewriting: int = 0
total_tokens_query_rewriting: int = 0


# tested in main execution method only
def rewrite_query(query: str, number_of_alternatives: int, rewrite_kind: int, prefilter_kind: int) \
        -> Tuple[list[str], dict]:
    """
    Produce multiple rewrites for a single query using different parameters for tuning.

    :param str query: The query to rewrite (currently a SQL query)
    :param int number_of_alternatives: The number of alternative queries to generate
    :param int rewrite_kind: Define what rewrite method to use (1 for zero-shot rewriting on the query)
    :param int prefilter_kind: Define what table filter method to use (1 for simple filter)
    :return: A list of the rewrites of the query and the proposed tables used for the rewrite
    :rtype: Tuple[list[str], dict]
    """
    # Get all tables from the database that could be used
    start_time_filter: float = time.time()
    proposed_tables: dict = prefilter_tables(query, prefilter_kind)
    end_time_filter: float = time.time()
    if len(proposed_tables) == 0:
        # No available tables for a rewrite
        add_rewriting_timings(end_time_filter - start_time_filter, 0)
        raise NoRewritesFoundException("No tables in the database can be used to rewrite the query.")
    start_time_rewrite: float = time.time()
    if rewrite_kind == 1:
        # Simple zero-shot prompting should be used
        # used_function = simple_gpt_rewriting
        result: list[str] = simple_gpt_rewriting(query, number_of_alternatives, proposed_tables)
    elif rewrite_kind == 2:
        # First get the intent and then the rewrites
        result: list[str] = simple_gpt_rewrite_using_nl(query, number_of_alternatives, proposed_tables)
    else:
        # More complex prompting should be used
        raise NotYetSupportedException(
            f"Configured rewrite mode not implemented (method: {rewrite_kind}). Skipping query '{query}'.")
    end_time_rewrite: float = time.time()
    add_rewriting_timings(end_time_filter - start_time_filter, end_time_rewrite - start_time_rewrite)
    if len(result) == 0:
        # No rewrites were found
        raise NoRewritesFoundException(
            "No rewrites were found for the given query and the fitting tables in the database.")
    elif len(result) < number_of_alternatives:
        # Too few rewrites found
        raise NoRewritesFoundException("Not enough rewrites were found for the given query.")
    else:
        return result, proposed_tables


# tested in main execution method only
def simple_gpt_rewriting(query: str, number_of_alternatives: int, proposed_tables: dict) -> list[str]:
    """
    Use a zero-shot approach to rewrite the query using GPT.

    :param str query: The query we want to rewrite
    :param int number_of_alternatives: The number of alternative queries to generate
    :param dict proposed_tables: Tables from the DB identified as usable for rewriting
    :return: Alternative queries produced by GPT
    :rtype: list[str]
    """
    # get the proposed tables as string for prompt
    # Check for Foreign Key Constraints
    foreign_keys: dict = get_usable_constraints_from_db(list(proposed_tables.keys()), False)
    proposed_table_str: str = prepare_db_schema_for_prompt_including_fk(proposed_tables, foreign_keys)
    content_for_gpt: str = (f"I have the following SQL query:\n"
                            f"{query}\n"
                            f"I do not have access to the tables needed in the query.\n"
                            f"I do have the following tables in my database "
                            f"(written in the format: "
                            f"table: column1 type1, column2 type2 \\n Foreign keys (if existent)):\n"
                            f"{proposed_table_str}"
                            f"Keep the foreign keys in mind if you join any tables.\n"
                            f"I want to have queries that keep the same human intent and satisfy "
                            f"the information need as the given query.\n"
                            f"These new queries should use the provided tables and columns from my database.\n"
                            f"Give me {number_of_alternatives} alternative queries. "
                            f"They should be as diverse as possible.\n"
                            f"Only give the SQL queries and nothing more. Use a semicolon to separate the queries.")
    # print(f"\nPrompt for GPT:\n{content_for_gpt}\n")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in SQL."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    # Split the alternative queries via the token given to GPT
    sql_queries_gpt: str = completion["choices"][0]["message"]["content"]
    # print(f"Answer from GPT:\n{sql_queries_gpt}")
    alt_queries: list[str] = sql_queries_gpt.split(";")  #("!NEXT QUERY!")
    # Extract only the SQL string from the response (for each query in the result list), remove empty queries
    alt_queries_formatted = [strip_sql_output(query) for query in alt_queries if strip_sql_output(query) != '']
    # Count how many tokens were used
    if llm_used:
        global prompt_tokens_query_rewriting
        prompt_tokens_query_rewriting = completion["usage"]["prompt_tokens"]
        global completion_tokens_query_rewriting
        completion_tokens_query_rewriting = completion["usage"]["completion_tokens"]
        global total_tokens_query_rewriting
        total_tokens_query_rewriting = completion["usage"]["total_tokens"]
        print(
            f"Tokens used after the query rewriting: \n\tPrompt Tokens: {prompt_tokens_query_rewriting}, "
            f"Completion Tokens: {completion_tokens_query_rewriting}, Total Tokens: {total_tokens_query_rewriting}\n")
        add_tokens(completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"],
                   completion["usage"]["total_tokens"])
    else:
        print("No tokens used for query rewriting due to reproducibility DB.")
    return alt_queries_formatted


# tested in workflow
def simple_gpt_rewrite_using_nl(query: str, number_of_alternatives: int, proposed_tables: dict) -> list[str]:
    """
    Use a zero-shot approach to rewrite the query using GPT.
    This time we first ask the LLM for the intent and then ask for queries fitting the intent.

    :param str query: The query we want to rewrite
    :param int number_of_alternatives: The number of alternative queries to generate
    :param dict proposed_tables: Tables from the DB identified as usable for rewriting
    :return: Alternative queries produced by GPT
    :rtype: list[str]
    """
    # Get the intent of the SQL query
    intent_of_query: str = llm_find_intent_sql(query)
    # Get the foreign keys from the DB and make everything into a string
    foreign_keys: dict = get_usable_constraints_from_db(list(proposed_tables.keys()), False)
    proposed_table_str: str = prepare_db_schema_for_prompt_including_fk(proposed_tables, foreign_keys)
    # Make the LLM call with the right prompt
    content_for_gpt: str = (f"I have the following request deducted from a SQL query:\n"
                            f"'{intent_of_query}'\n"
                            f"I have the following tables in my database "
                            f"(written in the format: "
                            f"table: column1 type1, column2 type2 \\n Foreign keys (if existent)):\n"
                            f"{proposed_table_str}"
                            f"I want to have SQL queries that have the same human intent and satisfy "
                            f"the same information need as the given request.\n"
                            f"These new queries should use the provided tables and columns from my database.\n"
                            f"Keep the foreign keys in mind if you join any tables.\n"
                            f"Give me {number_of_alternatives} alternative queries. "
                            f"They should be as diverse as possible.\n"
                            f"Only give the SQL queries and nothing more. Use a semicolon to separate the queries.")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in SQL."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # Get an answer from the LLM and process it (just as in simple_gpt_rewriting)
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    sql_queries_gpt: str = completion["choices"][0]["message"]["content"]
    alt_queries: list[str] = sql_queries_gpt.split(";")
    alt_queries_formatted = [strip_sql_output(query) for query in alt_queries if strip_sql_output(query) != '']
    # Count how many tokens were used
    if llm_used:
        global prompt_tokens_query_rewriting
        prompt_tokens_query_rewriting = completion["usage"]["prompt_tokens"]
        global completion_tokens_query_rewriting
        completion_tokens_query_rewriting = completion["usage"]["completion_tokens"]
        global total_tokens_query_rewriting
        total_tokens_query_rewriting = completion["usage"]["total_tokens"]
        print(
            f"Tokens used after the query rewriting: \n\tPrompt Tokens: {prompt_tokens_query_rewriting}, "
            f"Completion Tokens: {completion_tokens_query_rewriting}, Total Tokens: {total_tokens_query_rewriting}\n")
        add_tokens(completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"],
                   completion["usage"]["total_tokens"])
    else:
        print("No tokens used for query rewriting due to reproducibility DB.")
    return alt_queries_formatted

