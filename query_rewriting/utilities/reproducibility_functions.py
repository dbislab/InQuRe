"""
All functions that make reproducible results possible are written here.
"""
import hashlib
import json
from collections.abc import Iterable
from typing import Iterator

import duckdb

import query_rewriting.config as config


# used in main method at start and before tests
def create_reproducibility_database(test: bool):
    """
    Creates a reproducibility database to save already made LLM requests.

    :param bool test: Indicates whether tests are currently run
    """
    # Schema: ID, Model, Request, Request Hash, Result, System Fingerprint
    # Create a DB using the path in the config plus an addition
    db_path: str = calculate_reproducibility_db_path(test)
    con = duckdb.connect(db_path)
    create_statement: str = (f"CREATE TABLE IF NOT EXISTS {config.db_reproducibility_table_name} "
                             f"(ID VARCHAR PRIMARY KEY, model VARCHAR, request VARCHAR, requestHash VARCHAR, "
                             f"response VARCHAR, fingerprint VARCHAR);")
    con.execute(create_statement)
    check_statement: str = (f"SELECT table_name FROM duckdb_tables() "
                            f"WHERE table_name = '{config.db_reproducibility_table_name}'")
    db_tables: list = con.execute(check_statement).fetchall()
    con.close()
    assert len(db_tables) == 1


def check_for_entry(model: str, request: Iterable, test: bool) -> dict:
    """
    Check if an entry exists in the database using the request string.

    :param str model: The model used as the LLM
    :param str request: The request posted to the LLM
    :param bool test: Indicates whether tests are currently run
    :return: An empty dict if the entry does not exist, the response taken from the DB otherwise
    :rtype: dict
    """
    found_entry: dict = dict()
    db_path: str = calculate_reproducibility_db_path(test)
    con = duckdb.connect(db_path)
    # Check if hash exists
    request_str: str = get_request_string_from_iterable(request)
    request_hashed: str = get_hash_from_request_string(request_str)
    resulting_entries: list = (con.execute(f"SELECT * FROM {config.db_reproducibility_table_name} "
                                           f"WHERE model = ? AND requestHash = ?;", [model, request_hashed])
                               .fetchall())
    if len(resulting_entries) == 0:
        return found_entry
    # Check if exact string exists
    rechecked_entries: list = (con.execute(f"SELECT response FROM {config.db_reproducibility_table_name} "
                                           f"WHERE model = ? AND requestHash = ? AND request = ?;",
                                           [model, request_hashed, request_str])
                               .fetchall())
    if len(rechecked_entries) == 0:
        return found_entry
    # Get response from DB
    # print(rechecked_entries)
    entry: str = rechecked_entries[0][0]
    found_entry = string_to_dictionary(entry)
    con.close()
    return found_entry


def write_new_entry(entry_id: str, model: str, request: Iterable, response: dict, fingerprint: str, test: bool):
    """
    Write a new entry into the database using the request and response string from the LLM.

    :param str entry_id: The ID of the LLM response
    :param str model: The model used as the LLM
    :param str request: The request posted to the LLM
    :param dict response: The response received from the LLM as a dictionary
    :param str fingerprint: The fingerprint of the LLM response
    :param bool test: Indicates whether tests are currently run
    """
    db_path: str = calculate_reproducibility_db_path(test)
    con = duckdb.connect(db_path)
    # Get needed values
    request_str: str = get_request_string_from_iterable(request)
    hash_val: str = get_hash_from_request_string(request_str)
    response_str: str = dictionary_to_string(response)
    # Make insert into DB
    # Escape the JSON string of the response using a prepared statement
    con.execute(f"INSERT INTO {config.db_reproducibility_table_name} VALUES (?,?,?,?,?,?);",
                [entry_id, model, request_str, hash_val, response_str, fingerprint])
    con.close()


def get_request_string_from_iterable(request: Iterable) -> str:
    """
    Make the request given as an iterable into a string.

    :param Iterable request: The request for the LLM
    :return: The string representation of the iterable
    :rtype: str
    """
    objects: Iterator = iter(request)
    result_str: str = "["
    curr_object = next(objects, None)
    while not curr_object is None:
        result_str += str(curr_object)
        result_str += ","
        curr_object = next(objects, None)
    result_str = result_str[:-1] + "]"
    return result_str


def get_hash_from_request_string(request: str) -> str:
    """
    Calculate a hash value for a given string for quicker comparison.
    It uses the in-built hash function of python.

    :param str request: The request given as a string
    :return: The hash value of the request
    :rtype: str
    """
    return hashlib.sha256(request.encode()).hexdigest()


def dictionary_to_string(d: dict) -> str:
    """
    Make a string from a dictionary (to save the string in the DB).

    :param dict d: The dictionary to make a string from
    :return: The dictionary converted to a string via JSON dumps
    :rtype: str
    """
    output: str = json.dumps(d)
    return output


def string_to_dictionary(s: str) -> dict:
    """
    Make a string into a dictionary (to convert the string into the DB back to a dictionary).

    :param str s: The string from the DB to convert to a dictionary
    :return: The dictionary converted from the string
    :rtype: dict
    """
    output: dict = json.loads(s)
    return output


def calculate_reproducibility_db_path(test: bool) -> str:
    """
    Calculate the path to the reproducibility database where all info from LLM calls is saved.

    :param bool test: Indicates whether tests are currently run.
    :return: The path to the reproducibility database
    :rtype: str
    """
    if test:
        reproducibility_db_path: str = config.test_db_file
    else:
        reproducibility_db_path: str = config.db_file
    # add LLM model to path and reproducibility flag
    reproducibility_db_path = (reproducibility_db_path[:reproducibility_db_path.rindex(".db")]
                               + config.db_reproducibility_addition + '.db')
    print(f"Found the following reproducibility path: {reproducibility_db_path}")
    return reproducibility_db_path


# Helper functions done by hand, no testing
def find_entry_with_certain_request(request: str, test: bool):
    """
    Find and print all entries where the request contains the given string

    :param str request: The string contained in the request
    :param bool test: Indicates whether tests are currently run.
    """
    db_path: str = calculate_reproducibility_db_path(test)
    con = duckdb.connect(db_path)
    resulting_entries: list = (con.execute(f"SELECT * FROM {config.db_reproducibility_table_name} "
                                           f"WHERE contains(request, ?);", [request]).fetchall())
    if len(resulting_entries) == 0:
        print("Nothing found in DB")
    else:
        print(f"{len(resulting_entries)} entries found:")
        print(*resulting_entries, sep="\n")
    con.close()


# Helper functions done by hand, no testing
def delete_entry_with_certain_id(request_id: str, test: bool):
    """
    Delete an entry with a certain id.

    :param str request_id: The id of the entry to delete.
    :param bool test: Indicates whether tests are currently run.
    """
    db_path: str = calculate_reproducibility_db_path(test)
    con = duckdb.connect(db_path)
    resulting_entries: list = (con.execute(f"SELECT * FROM {config.db_reproducibility_table_name} "
                                           f"WHERE ID = ?;", [request_id]).fetchall())
    print(f"Before deletion:\nEntries with the ID {request_id}: {len(resulting_entries)}")
    con.execute(f"DELETE FROM {config.db_reproducibility_table_name} WHERE ID = ?;",
                [request_id])
    resulting_entries_after: list = (con.execute(f"SELECT * FROM {config.db_reproducibility_table_name} "
                                                 f"WHERE ID = ?;", [request_id]).fetchall())
    print(f"After deletion:\nEntries with the ID {request_id}: {len(resulting_entries_after)}")
    con.close()



