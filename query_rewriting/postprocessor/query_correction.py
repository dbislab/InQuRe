"""
Correct the queries produced by the LLM to work on the database.
The self-correction prompts for the LLM are from the Din-SQL Paper.
"""
from typing import Tuple

import duckdb
from collections.abc import Iterable

# from openai.types.chat import ChatCompletion

import query_rewriting.config as config
from query_rewriting.utilities.duckdb_functions import get_result_with_column_names, get_usable_constraints_from_db
from query_rewriting.utilities.gpt_functions import gpt_api_call, strip_sql_output, \
    prepare_db_schema_for_prompt_including_fk
from query_rewriting.utilities.statistics import add_tokens

# tokens used in the methods of this file
prompt_tokens_query_correction: int = 0
completion_tokens_query_correction: int = 0
total_tokens_query_correction: int = 0

# tested in workflow for queries that need correction (not for non-correctable queries)
def query_correction_and_execution(input_queries: list[str], usable_tables: dict, num_iterations: int = 3) \
        -> Tuple[list[str], list[str], list[list], int]:
    """
    Correct a list of queries. First check if each query is executable.
    Correct the ones that are not (worst-case in multiple turns for the LLM).
    If a query cannot be corrected in multiple turns, it is flagged as non-executable.
    If a query is executable (before or after correction),
    the results are also returned to avoid having to execute the query multiple times.

    :param list[str] input_queries: The queries that we want to correct
    :param dict usable_tables: The usable tables that were selected to rewrite the query
           in the form {table:[column1 type1, column2 type2]}
    :param int num_iterations: The number of iterations done to try and correct the query (default: 3)
    :return: 1. A list of queries (corrected if it was correctable, original otherwise)
             2. A list of error messages (empty string if the query is executable now, the error message otherwise)
             3. A list of the results of the query (if the query was executable, otherwise the result [] is appended)
             4. The number of queries that were corrected in this run
    :rtype: Tuple[list[str], list[str], list[list]]
    """
    corrected_queries: list[str] = []
    error_messages: list[str] = []
    results: list[list] = []
    num_corrections: int = 0
    for input_query in input_queries:
        # Try for each query if it is executable
        executable, result, correctable, error_msg = get_error_message_or_result(input_query, False)
        if executable:
            # The query is executable, we just add everything
            print(f"The query '{input_query}' did not need correction.")
            corrected_queries.append(input_query)
            error_messages.append(error_msg)
            results.append(result)
        else:
            print(f"The query '{input_query}' needed correction.")
            num_corrections += 1
            # The query is not directly executable
            if correctable:
                # The query can be corrected
                corrected_version_found: bool = False
                corrected_query: str = input_query
                error_msg2: str = error_msg
                for i in range(num_iterations):
                    # Try num_iterations times to correct the query (iteratively) and check if it was corrected
                    corrected_query = gentle_self_correction(corrected_query, usable_tables, error_msg2)
                    executable2, result2, correctable2, error_msg2 = get_error_message_or_result(corrected_query, False)
                    if executable2:
                        # The query is now executable, so append the corrected query,
                        # the new error message and the new result and break the loop
                        corrected_queries.append(corrected_query)
                        error_messages.append(error_msg2)
                        results.append(result2)
                        corrected_version_found = True
                        break
                if not corrected_version_found:
                    # No corrected version was found, so insert the original query, error message and (empty) result
                    corrected_queries.append(input_query)
                    error_messages.append(error_msg)
                    results.append(result)
            else:
                # The query cannot be corrected, so we add everything including the error message
                corrected_queries.append(input_query)
                error_messages.append(error_msg)
                results.append(result)
    return corrected_queries, error_messages, results, num_corrections


# is tested in workflow for queries that need correction (not for non-correctable queries)
def gentle_self_correction(input_query: str, usable_tables: dict, error_message: str) -> str:
    """
    Uses the LLM itself (zero-shot setting) to correct one non-executable query.
    It is assumed (unlike in DinSQL) that the input query is not executable.
    For similarity/correct results regarding the rewriting we do not use correction,
        but filter the best rewrites via ranking (unlike DinSQL) before the correction.
    Therefore, only syntactic correction is needed, as the top-k queries are already assumed to be good rewrites.
    This mainly corrects syntactic errors.

    :param str input_query: The query we want to correct
    :param dict usable_tables: Dictionary of usable tables that were selected to rewrite the query in the form
           {table:[column1 type1, column2 type2]}
    :param str error_message: The error message that was produced during the execution
    :return: A corrected version of the query from the LLM (no guarantee to be executable)
    :rtype: str
    """
    # Get the proposed tables as string for prompt
    # Checked if Foreign Key Constraints are there
    foreign_keys: dict = get_usable_constraints_from_db(list(usable_tables.keys()), False)
    proposed_table_str: str = prepare_db_schema_for_prompt_including_fk(usable_tables, foreign_keys)
    # Prompt the LLM
    content_for_gpt: str = (
        f"For the given SQL query, use the provided tables and columns to fix it for the problems "
        f"arising during its execution in DuckDB.\n"
        f"Only change parts of the query that need correction and keep the intent of the query when correcting it.\n"
        f"Use the following error message as a hint to identify the issue with the query:\n"
        f"{error_message}\n"
        f"I have the following query in SQL:\n"
        f"{input_query}\n"
        f"I have the following tables in my database "
        f"(format: table1: column1 type1, column2 type2 \\n Foreign keys (if existent)):\n"
        f"{proposed_table_str}"
        f"Keep the foreign keys in mind if the query joins any tables.\n"
        f"You should provide only a corrected version of the query that is executable on the database, nothing else.")
    print(f"\nCorrection prompt for GPT:\n{content_for_gpt[:1500]}\n")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in SQL."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # Get the response from the LLM
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    stripped_query: str = strip_sql_output(completion["choices"][0]["message"]["content"])
    # Count the tokens
    if llm_used:
        global prompt_tokens_query_correction
        prompt_tokens_query_correction = completion["usage"]["prompt_tokens"]
        global completion_tokens_query_correction
        completion_tokens_query_correction = completion["usage"]["completion_tokens"]
        global total_tokens_query_correction
        total_tokens_query_correction = completion["usage"]["total_tokens"]
        print(f"Tokens used for query correction: \n\tPrompt Tokens: {prompt_tokens_query_correction}, Completion "
              f"Tokens: {completion_tokens_query_correction}, Total Tokens: {total_tokens_query_correction}")
        add_tokens(completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"],
                   completion["usage"]["total_tokens"])
    else:
        print("No tokens used for query correction due to reproducibility DB.")
    return stripped_query


def get_error_message_or_result(input_query: str, test: bool) -> Tuple[bool, list, bool, str]:
    """
    Check if the query is executable on our database and give precise errors (or results if it is executable).
    This is done in one method to avoid having to execute correct queries two times.

    :param str input_query: The query to check for execution
    :param bool test: indicates if tests are run currently
    :return: A tuple including:
             1. An indicator if the query is executable
             2. The result (from .fetchall() including the columns) of the query if it was executable
                (otherwise the result is the empty list)
             3. An indicator if the query can be corrected
                (for some DuckDB errors, the query is assumed to be not correctable)
             4. The error message if the query is correctable (if it is executable the error message is empty)
    :rtype: Tuple[bool, list, bool, str]
    """
    # Establish the DB connection
    if test:
        path = config.test_db_file
    else:
        path = config.db_file
    con = duckdb.connect(path)
    executable: bool = True
    res: list = []
    correctable: bool = True
    error_msg: str = ""
    try:
        # Try to execute the query
        con.execute(input_query)
        res = get_result_with_column_names(con)
    except duckdb.ProgrammingError as e:
        # Catches:
        # BinderException, ParserException, CatalogException,
        # InvalidInputException, InvalidTypeException, SyntaxException
        executable = False
        error_msg = f"{str(e)}"
    except duckdb.DataError as e:
        # Catches:
        # ConversionException, TypeMismatchException, OutOfRangeException
        executable = False
        error_msg = f"{str(e)}"
    except Exception as e:
        # The query is not correctable with all other exceptions
        executable = False
        correctable = False
        error_msg = f"Other Exception:\n{str(e)}"
    finally:
        con.close()
    return executable, res, correctable, error_msg
