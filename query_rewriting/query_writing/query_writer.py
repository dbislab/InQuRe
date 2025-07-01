"""
Write a query from NL and proposed tables
"""
from collections.abc import Iterable

# from openai.types.chat import ChatCompletion

import query_rewriting.config as config
from query_rewriting.utilities.gpt_functions import strip_sql_output, gpt_api_call
from query_rewriting.utilities.statistics import add_tokens

# tokens used in the methods of this file
prompt_tokens_query_writing: int = 0
completion_tokens_query_writing: int = 0
total_tokens_query_writing: int = 0


# no tests currently needed, as not integrated in workflow
def write_query_from_tables(schema: dict, request: str) -> str:
    """
    Make a request to a LLM (default GPT) to suggest a query for the given natural language request and proposed tables.

    :param dict schema: The proposed tables that could be used to answer the query
    :param str request: The natural language request
    :return: The proposed SQL query as a string
    :rtype: str
    """
    # Only really simple implementation, not too reliable
    # Turn the schema into readable text
    schema_text: str = ''
    for key, value in schema.items():
        schema_text += f'{key}:'
        schema_text += ','.join(value) + '\n'
    # print(f"Given Schema: {schema_text}")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in natural language."},
        {
            "role": "user",
            "content": (f"I have the following natural language request:\n"
                        f"{request}\n"
                        f"You gave me the following tables to use for an SQL query:\n"
                        f"{schema_text}"
                        f"Give me the SQL query that answers the request using the tables. Only output the SQL query. ")
        }
    ]
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    # Extract only the SQL string from the response
    sql: str = completion["choices"][0]["message"]["content"]
    sql = strip_sql_output(sql)
    # Count how many tokens were used
    if llm_used:
        global prompt_tokens_query_writing
        prompt_tokens_query_writing = completion["usage"]["prompt_tokens"]
        global completion_tokens_query_writing
        completion_tokens_query_writing = completion["usage"]["completion_tokens"]
        global total_tokens_query_writing
        total_tokens_query_writing = completion["usage"]["total_tokens"]
        print(
            f"Tokens used in the query writing process: \n\tPrompt Tokens: {prompt_tokens_query_writing}, "
            f"Completion Tokens: {completion_tokens_query_writing}, Total Tokens: {total_tokens_query_writing}\n")
        add_tokens(completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"],
                   completion["usage"]["total_tokens"])
    else:
        print("No tokens used for query writing due to reproducibility DB.")
    return sql

