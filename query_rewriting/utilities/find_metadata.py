"""
Methods to find different metadata from a query (e.g. intent, topics,...)
for better generation of alternative queries and ranking/evaluation methods.
Methods that use an LLM are prefixed with "llm_".
"""
from collections.abc import Iterable
from itertools import islice

# from openai.types.chat import ChatCompletion

from query_rewriting.utilities.gpt_functions import gpt_api_call, strip_preceding_keywords, strip_whitespaces, \
    strip_starting_and_ending_characters, strip_code_block_output
import query_rewriting.config as config
from query_rewriting.utilities.statistics import add_tokens

# Tokens used in the methods of this file
prompt_tokens_find_metadata: int = 0
completion_tokens_find_metadata: int = 0
total_tokens_find_metadata: int = 0


# not tested in test_query_rewriting.py, as it makes LLM call
def llm_find_intent_nl(input_request: str) -> str:
    """
    Find the intent of the given natural language input query using an LLM.

    :param str input_request: The input query to find intent from
    :return: The intent of the query in natural language
    :rtype: str
    """
    # Define the message
    content_for_gpt: str = (f"I have the following query in natural language:\n"
                            f"{input_request}\n "
                            f"Give me the information need of the query, i.e., "
                            f"what a humans intent is when writing the query.\n "
                            f"Do not just repeat the query itself as an intent, but abstract from it. Be concise.\n "
                            f"Only return the intent preceded by the keyword 'Intent:'.")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in natural language."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # Make the call to GPT
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    # Count the tokens
    if llm_used:
        add_tokens_metadata(completion)
    else:
        print("No tokens used to find the intent due to reproducibility DB.")
    # Get the intent and strip it
    intent_gpt: str = completion["choices"][0]["message"]["content"]
    intent: str = strip_whitespaces(strip_preceding_keywords(intent_gpt, "Intent:"))
    return intent


# not tested in test_query_rewriting.py, as it makes LLM call
def llm_find_intent_sql(input_request: str) -> str:
    """
    Find the intent of the given SQL input query using an LLM.

    :param str input_request: The input query (in SQL) to find intent from
    :return: The intent of the query in natural language
    :rtype: str
    """
    # Define the message
    content_for_gpt: str = (f"I have the following query in SQL:\n"
                            f"{input_request}\n "
                            f"Give me the information need of the query, i.e., "
                            f"what a humans intent is when writing the query.\n "
                            f"Do not just repeat the query itself as an intent, but abstract from it. Be concise.\n "
                            f"Only return the intent preceded by the keyword 'Intent:'.")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in SQL."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # Make the call to GPT
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    # Count the tokens
    if llm_used:
        add_tokens_metadata(completion)
    else:
        print("No tokens used to find intent due to reproducibility DB.")
    # Get the intent and strip it
    intent_gpt: str = completion["choices"][0]["message"]["content"]
    intent: str = strip_whitespaces(strip_preceding_keywords(intent_gpt, "Intent:"))
    return intent


# not tested in test_query_rewriting.py, as it makes LLM call
# Prompt tested via ChatGPT
def llm_find_intent_sql_list(input_requests: list[str]) -> list[str]:
    """
    Find the intents of the given SQL input queries using an LLM.
    Warning: There is no check if all queries get an intent.

    :param list[str] input_requests: The input queries (in SQL) to find intent from
    :return: The intents of the query in natural language, listed in the same order as the queries
    :rtype: list[str]
    """
    queries_formatted: list[str] = [q.strip() if q.strip().endswith(";")
                                    else q.strip() + ";" for q in input_requests]
    amount_of_queries_per_request = 10
    queries_split_for_request: list[list[str]] = \
        [list(islice(queries_formatted, i, i + amount_of_queries_per_request)) for i in
         range(0, len(queries_formatted), amount_of_queries_per_request)]
    all_intents: list[str] = []
    for queries in queries_split_for_request:
        queries_formatted_t = "\n".join(queries)
        content_for_gpt: str = (f"I have the following queries in SQL:\n"
                                f"{queries_formatted_t}\n"
                                f"For each query give me the information need of the query, i.e., "
                                f"what a humans intent is when writing the query.\n "
                                f"Do not just repeat the query itself as an intent, but abstract from it. Be concise.\n "
                                f"For each query only return the intent.\n"
                                f"Give the intents in the same order as the input queries.\n"
                                f"Separate the intents of the single queries with a new line each.")
        message: Iterable = [
            {"role": "system", "content": "We will work with databases and queries in SQL."},
            {
                "role": "user",
                "content": content_for_gpt
            }
        ]
        # Make the call to GPT
        completion, llm_used = gpt_api_call(config.gpt_model, message)
        # Count the tokens
        if llm_used:
            add_tokens_metadata(completion)
        else:
            print("No tokens used to find intent list due to reproducibility DB.")
        # Get the intents and strip them
        llm_response: str = completion["choices"][0]["message"]["content"]
        suggested_intents: list[str] = (strip_starting_and_ending_characters(strip_code_block_output(llm_response))
                                        .splitlines())
        suggested_intents_cleaned: list[str] = [query.strip() for query in suggested_intents if query.strip() != ""]
        amount_added = 0
        while len(suggested_intents_cleaned) < len(queries):
            amount_added += 1
            suggested_intents_cleaned.append("")
        if amount_added > 0:
            print(f"Added {amount_added} empty strings as intents.")
        all_intents.extend(suggested_intents_cleaned)
    if len(all_intents) != len(input_requests):
        raise config.RankingNotPossible("The LLM gave not the exact amount of intents for all alternative queries.")
    # Return all intents
    return all_intents


# not tested in test_query_rewriting.py, as it makes LLM call
def llm_find_intent_keywords_nl(input_request: str, num_keywords: int) -> list[str]:
    """
    Find the intent (in keywords) of the given natural language input query using an LLM.

    :param str input_request: The input query to find intent keywords from
    :param int num_keywords: The number of keywords the LLM should generate
    :return: A list of all relevant keywords for the input query, the length is given by the num_keywords parameter
    :rtype: list[str]
    """
    # Define the message
    content_for_gpt: str = (f"I have the following query in natural language:\n"
                            f"{input_request}\n "
                            f"Give me {num_keywords} keywords that fit the information need of the query, i.e., "
                            f"what a humans intent is when writing the query.\n "
                            f"Do not just repeat the query words itself as keywords, but abstract from them. "
                            f"Be concise.\n "
                            f"Only return the keywords separated with commas.")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in natural language."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # Make the call to GPT
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    # Count the tokens
    if llm_used:
        add_tokens_metadata(completion)
    else:
        print("No tokens used to find intent keywords due to reproducibility DB.")
    # Get the intent keywords
    intent_keywords_gpt: str = completion["choices"][0]["message"]["content"]
    # Strip points etc.
    intent_keywords_gpt = strip_whitespaces(strip_starting_and_ending_characters(intent_keywords_gpt))
    # Split the keywords into a list and clean up the whitespaces for each one
    intent_keyword_list: list[str] = intent_keywords_gpt.split(",")
    keywords: list[str] = [strip_whitespaces(word) for word in intent_keyword_list]
    return keywords


# not tested in test_query_rewriting.py, as it makes LLM call
def llm_find_intent_keywords_sql(input_request: str, num_keywords: int) -> list[str]:
    """
    Find the intent (in keywords) of the given SQL input query using an LLM.

    :param str input_request: The input query (in SQL) to find intent keywords from
    :param int num_keywords: The number of keywords the LLM should generate
    :return: A list of all relevant keywords for the input query, the length is given by the num_keywords parameter
    :rtype: list[str]
    """
    # Define the message
    content_for_gpt: str = (f"I have the following query in SQL:\n"
                            f"{input_request}\n "
                            f"Give me {num_keywords} keywords that fit the information need of the query, i.e., "
                            f"what a humans intent is when writing the query.\n "
                            f"Do not just repeat the query tables and columns themselves as keywords, "
                            f"but abstract from them. Be concise.\n "
                            f"Only return the keywords separated with commas.")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in SQL."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # Make the call to GPT
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    # Count the tokens
    if llm_used:
        add_tokens_metadata(completion)
    else:
        print("No tokens used to find intent keywords due to reproducibility DB.")
    # Get the intent keywords
    intent_keywords_gpt: str = completion["choices"][0]["message"]["content"]
    # Strip points etc.
    intent_keywords_gpt = strip_whitespaces(strip_starting_and_ending_characters(intent_keywords_gpt))
    # Split the keywords into a list and clean up the whitespaces for each one
    intent_keyword_list: list[str] = intent_keywords_gpt.split(",")
    keywords: list[str] = [strip_whitespaces(word) for word in intent_keyword_list]
    return keywords


# not tested in test_query_rewriting.py, as it makes LLM call
def llm_find_topics_sql(input_request: str, num_topics: int) -> list[str]:
    """
    Find the topics covered by the given SQL input query using an LLM.

    :param str input_request: The input query (in SQL) to find topics from
    :param int num_topics: The number of topics the LLM should generate
    :return: A list of all relevant topics for the input query, the length is given by the num_topics parameter
    :rtype: list[str]
    """
    # Define the message
    content_for_gpt: str = (f"I have the following query in SQL:\n"
                            f"{input_request}\n "
                            f"Give me {num_topics} topics that fit the information need of the query, i.e., "
                            f"what a humans intent is when writing the query.\n "
                            f"Do not just repeat the query tables and columns themselves as topics, "
                            f"but abstract from them. Be concise.\n "
                            f"Only return the topics separated with commas.")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in SQL."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # Make the call to GPT
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    # Count the tokens
    if llm_used:
        add_tokens_metadata(completion)
    else:
        print("No tokens used to find topics due to reproducibility DB.")
    # Get the intent keywords
    topics_gpt: str = completion["choices"][0]["message"]["content"]
    # Strip points etc.
    topics_gpt = strip_whitespaces(strip_starting_and_ending_characters(topics_gpt))
    # Split the keywords into a list and clean up the whitespaces for each one
    topics_list: list[str] = topics_gpt.split(",")
    topics: list[str] = [strip_whitespaces(word) for word in topics_list]
    return topics


# does not need testing
def add_tokens_metadata(completion: dict):
    """
    Add the tokens used in the GPT call to the overall tokens used by the methods in this file.

    :param dict completion: The return of the GPT API call as a dictionary.
    """
    # Count how many tokens were used
    global prompt_tokens_find_metadata
    prompt_tokens_find_metadata = completion["usage"]["prompt_tokens"]
    global completion_tokens_find_metadata
    completion_tokens_find_metadata = completion["usage"]["completion_tokens"]
    global total_tokens_find_metadata
    total_tokens_find_metadata = completion["usage"]["total_tokens"]
    print(
        f"Tokens used to find metadata for the query: \n Prompt Tokens: {prompt_tokens_find_metadata}, "
        f"Completion Tokens: {completion_tokens_find_metadata}, Total Tokens: {total_tokens_find_metadata}\n")
    add_tokens(completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"],
               completion["usage"]["total_tokens"])