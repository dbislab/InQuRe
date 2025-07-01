"""
Functions to handle GPT in- and outputs
"""
import os
import time
from collections.abc import Iterable
# from http.client import responses
from typing import Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletion

import query_rewriting.config as config
from query_rewriting.utilities.reproducibility_functions import check_for_entry, write_new_entry


# no direct testing in test method needed
def gpt_api_call(used_model: str, input_message: Iterable, temperature: float = 1.0,
                 top_p: float = 1.0, seed: int = -1) -> Tuple[dict, bool]:
    """
    Function to call the GPT API.

    :param str used_model: The GPT model we want to use
    :param Iterable input_message: The message we want to send
            (example message:
            [{"role": "system", "content": "We will work with SQL."},
            {"role": "user", "content": f"I have the following SQL query:\n {query}\n"}]
            )
    :param float temperature: The temperature we want to use (in the range of 0: deterministic to 2:random, default: 1)
    :param float top_p: The top x percent of the probability mass to consider (alter with temperature)
    :param int seed: Sample seed used by the LLM (if this is set the answers are mostly deterministic),
           if set to -1 it will not be used
    :return: The completion object from the GPT API as a dictionary and true if the LLM was used, false otherwise
    :rtype: Tuple[dict, bool]
    """
    # Set the bool for return
    llm_used: bool = True
    # checked reproducibility part in workflow
    if config.reproducibility:
        response: dict = check_for_entry(used_model, input_message, False)
        if response != dict():
            # Response was found in the DB
            # token count should not go up if entry found in DB and no LLM used -> careful with this
            print("Response was already in DB. LLM not used.")
            llm_used = False
            return response, llm_used
    # Set the GPT API key
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    # Make the request (either with or without a seed)
    time_start = time.time()
    if seed == -1:
        completion: ChatCompletion = client.chat.completions.create(
            model=used_model,
            messages=input_message,
            # Sets the sampling temperature between 0 and 2 (the higher, the more random; the lower, the more focused)
            temperature=temperature,
            top_p=top_p
        )
    else:
        completion: ChatCompletion = client.chat.completions.create(
            model=used_model,
            messages=input_message,
            # Sets the sampling temperature between 0 and 2 (the higher, the more random; the lower, the more focused)
            temperature=temperature,
            top_p=top_p,
            seed=seed
        )
    time_end = time.time()
    result: dict = completion.to_dict()
    # print(result)  # checked if this is ok in workflow (valid python dict)
    print(f"Time for API Call: {time_end - time_start}")
    if config.reproducibility:
        # Save the response in the DB
        print("Response was not in DB. Saving it now.")
        write_new_entry(completion.id, used_model, input_message, result, completion.system_fingerprint, False)
    return result, llm_used


def strip_whitespaces(input_str: str) -> str:
    """
    Strip linebreaks, replace double whitespaces with single ones and remove tabs within a string.
    Also strip the start and end of the string of such whitespaces.

    :param str input_str: The string to strip
    :return: The stripped string
    :rtype: str
    """
    # Remove line breaks
    cleared_str: str = " ".join(input_str.splitlines())
    # Replace double whitespaces and tabs
    # (two times for security if space before and after newline or tabs)
    cleared_str = cleared_str.replace("\t", " ")
    cleared_str = cleared_str.replace("  ", " ")
    cleared_str = cleared_str.replace("  ", " ")
    cleared_str = cleared_str.strip()
    return cleared_str


def strip_sql_output(gpt_response: str) -> str:
    """
    Strip the Markdown annotations from a GPT response that contains only a SQL query.

    :param str gpt_response: The GPT response containing only SQL in Markdown format
    :return: The SQL query without line breaks
    :rtype: str
    """
    # print(f"String before formatting:\n{gpt_response}")
    # Remove newlines (default done via strip)
    sql: str = gpt_response.strip()
    # Strip starting and ending formatting
    sql = sql.strip("`")
    # Strip the sql tag at the start
    sql = sql.removeprefix("sql")
    sql = sql.strip()
    # Remove all whitespaces etc.
    sql = strip_whitespaces(sql)
    # print(f"String after formatting:\n{sql}")
    return sql


def strip_code_block_output(gpt_response: str) -> str:
    """
    Strip the Markdown annotations from a GPT response that has a code block.

    :param str gpt_response: The GPT response containing only a code block in Markdown format
           (but with line breaks etc.)
    :return: The content of the code block
    :rtype: str
    """
    # print(f"String before formatting:\n{gpt_response}")
    # Remove newlines (default done via strip)
    code: str = gpt_response.strip()
    # Strip starting and ending formatting
    code = code.strip("`")
    # Strip the sql tag at the start
    code = code.removeprefix("sql")
    code = code.removeprefix("css")
    code = code.removeprefix("plaintext")
    code = code.strip()
    # print(f"String after formatting:\n{sql}")
    return code


def strip_preceding_keywords(gpt_response: str, keyword: str) -> str:
    """
    Strip a keyword that is used at the start of the output of GPT.

    :param str gpt_response: The GPT response
    :param str keyword: The keyword (preceding the answer) we want to strip
    :return: The answer without the keyword
    :rtype: str
    """
    # Remove newlines (default done via strip)
    stripped: str = gpt_response.strip()
    # Remove the keyword
    stripped = stripped.removeprefix(keyword)
    # Remove all whitespaces etc.
    stripped = strip_whitespaces(stripped)
    return stripped


def strip_starting_and_ending_characters(gpt_response: str) -> str:
    """
    Strip some common symbols that GPT uses to start or end messages with.

    :param str gpt_response: The GPT response
    :return: The response stripped from certain characters
    :rtype: str
    """
    stripped: str = gpt_response.strip()
    # Remove points from the end
    stripped = stripped.removesuffix(".")
    return stripped


def prepare_db_schema_for_prompt(db_schema: dict) -> str:
    """
    Get a string representation of the DB schema for the LLM prompt.

    :param dict db_schema: The DB schema in the form {table1:[column1 type1,column2 type2]}
    :return: The string representation of the DB schema
    :rtype: str
    """
    result_str: str = ""
    for table, columns in db_schema.items():
        result_str += f"{table}: {', '.join(columns)}\n"
    return result_str


def prepare_db_schema_for_prompt_including_fk(db_schema: dict, foreign_keys: dict) -> str:
    """
    Get a string representation of the DB schema and the foreign key constraints for the LLM prompt.

    :param dict db_schema: The DB schema in the form {table1:[column1 type1,column2 type2]}
    :param dict foreign_keys: The foreign key constraints for the LLM prompt
    :return: The string representation of the DB schema and the constraints
    :rtype: str
    """
    result_str: str = ""
    for table, columns in db_schema.items():
        result_str += f"{table}: {', '.join(columns)}\n"
        possible_references: list[str] = foreign_keys.get(table, [])
        if possible_references:
            reference_str: str = f"Foreign Keys: {', '.join(possible_references)}\n"
            result_str += reference_str
        else:
            result_str += "No Foreign Keys\n"
    return result_str
