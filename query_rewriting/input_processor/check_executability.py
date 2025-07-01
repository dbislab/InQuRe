"""
Get proposed tables to check executability of NL/ Check executability in DB for SQL
"""
from collections.abc import Iterable
from typing import Tuple

import duckdb
# from openai.types.chat import ChatCompletion

import query_rewriting.config as config
from query_rewriting.utilities.duckdb_functions import get_result_with_column_names
from query_rewriting.utilities.gpt_functions import gpt_api_call
from query_rewriting.utilities.statistics import add_tokens

# Tokens used in the methods of this file
prompt_tokens_input_proc: int = 0
completion_tokens_input_proc: int = 0
total_tokens_input_proc: int = 0


# not tested in test_query_rewriting.py, as it makes LLM call
def proposed_tables(nl_input: str) -> dict:
    """
    Make a request to a LLM (default: GPT) to suggest tables for the given natural language request.

    :param str nl_input: The given natural language request
    :return: The suggested tables in the form {table1: [column1, column2]}
    :rtype: dict
    """
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in natural language."},
        {
            "role": "user",
            "content": (f"I have the following natural language request:\n"
                        f"{nl_input}\n"
                        f"Give me tables from a database that can be used in an SQL statement to answer the request.\n"
                        f"Give me the tables in the following format:\n"
                        f"table1:column1,column2,column3\n"
                        f"table2:column1,column2\n"
                        f"Give only the tables and no explanation.  ")
        }
    ]
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    print(completion["choices"][0]["message"]["content"])
    # Save the tables and columns suggested by the LLM to a dictionary
    schema: dict = dict()
    table_lines: list = completion["choices"][0]["message"]["content"].splitlines()
    for line in table_lines:
        table: str = line.split(":")[0].strip()
        columns: list = line.split(":")[1].strip().split(",")
        current_columns: list = schema.get(table, [])
        current_columns.extend(columns)
        schema[table] = current_columns
    # Count how many tokens were used
    if llm_used:
        global prompt_tokens_input_proc
        prompt_tokens_input_proc = completion["usage"]["prompt_tokens"]
        global completion_tokens_input_proc
        completion_tokens_input_proc = completion["usage"]["completion_tokens"]
        global total_tokens_input_proc
        total_tokens_input_proc = completion["usage"]["total_tokens"]
        print(
            f"Tokens used after tables are proposed: \n Prompt Tokens: {prompt_tokens_input_proc}, "
            f"Completion Tokens: {completion_tokens_input_proc}, Total Tokens: {total_tokens_input_proc}\n")
        add_tokens(completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"],
                   completion["usage"]["total_tokens"])
    else:
        print("No tokens used for table proposition due to reproducibility DB.")
    # print(f"Schema: {schema}")
    return schema



def check_proposed_tables_exact(schema: dict, test: bool) -> Tuple[bool, dict]:
    """
    Try to find the tables suggested in the input in the database schema.

    :param dict schema: A dictionary of the proposed schema
    :param bool test: indicates if tests are run currently
    :return: True if all the tables are found, false otherwise.
             As a second return all elements from the input
             that were found in the DB schema are written in a dictionary.
    :rtype: Tuple[bool, dict]
    """
    # Get all existent tables from the database
    if test:
        path = config.test_db_file
    else:
        path = config.db_file
    con = duckdb.connect(path)
    res: list = con.execute("SELECT table_name FROM duckdb_tables()").fetchall()
    existent_tables: dict = dict()
    existent_tables_count: int = 0
    # Fetch all table names into a single tuple
    # (throws error if no tables in DB, but this is checked at the execution start)
    db_tables: tuple = list(zip(*res))[0]
    # Iterate over given tables to try and find them in the DB
    for table, columns in schema.items():
        # First: Check if tables exist
        if table in db_tables:
            # Table exists, so get the columns of the table in the DB
            res2: list = con.execute("SELECT column_name FROM duckdb_columns() WHERE table_name=?",
                                     [table]).fetchall()
            db_columns: tuple = list(zip(*res2))[0]
            columns_set: set = set(columns)
            db_columns_set: set = set(db_columns)
            # Calculate the intersection of the DB and given schema columns
            intersection: set = columns_set.intersection(db_columns_set)
            if columns_set.issubset(db_columns_set):
                # All columns exist in the DB as well
                existent_tables_count += 1
                existent_tables[table] = list(intersection)
            elif len(intersection) > 0:
                # There are columns in the given schema that exist in the DB
                existent_tables[table] = list(intersection)
    con.close()
    if existent_tables_count == len(schema):
        # All tables were found
        return True, existent_tables
    else:
        # Not all tables were found
        return False, existent_tables


def check_query_execution(query_input: str, test: bool) -> Tuple[bool, list]:
    """
    Check if the input SQL query is executable.

    :param str query_input: The input SQL query
    :param bool test: indicates if tests are run currently
    :return: True if the input SQL query is executable, false otherwise.
             If the input was executable, the result of the query is returned as well (including column names)
    :rtype: Tuple[bool, list]
    """
    # Establish the DB connection
    if test:
        path = config.test_db_file
    else:
        path = config.db_file
    con = duckdb.connect(path)
    executable: bool = True
    res: list = []  # con.sql("SELECT false").fetchall()
    try:
        # Try to execute the query
        # res = con.execute(query_input).fetchall()
        con.execute(query_input)
        res = get_result_with_column_names(con)
    # Catch the possible exceptions that can be produced from a non-executable query
    # (set executable to false if an exception is thrown)
    except duckdb.ParserException as e:
        executable = False
        print(f"Query not executable on database. Parser error thrown with the message:\n{e}\n")
    except duckdb.ProgrammingError as e:
        executable = False
        print(f"Query not executable on database. Programming error thrown with the message:\n{e}\n")
    except duckdb.Error as e:
        executable = False
        print(f"Query not executable on database. General error thrown with the message:\n{e}\n")
    finally:
        # Close the connection
        con.close()
        # Return the boolean showing if the query executed without errors
        return executable, res
