"""
Utilities used for queries on DuckDB.
"""
from duckdb.duckdb import DuckDBPyConnection
import duckdb

import query_rewriting.config as config


def get_result_with_column_names(connection: DuckDBPyConnection) -> list[list]:
    """
    Get the results saved in the connection after execute,
    such that the first list element describes the returned columns of the query.

    :param DuckDBPyConnection connection: The connection on which the query was executed
    :return: A list with all results, where the first tuple describes the returned columns
    :rtype: list[list]
    """
    description: list = connection.description
    results: list[list] = connection.fetchall()
    cleaned_columns: list = []
    for column in description:
        cleaned_columns.append(column[0])
    results.insert(0, cleaned_columns)
    return results


def get_tables_from_db(test: bool) -> dict:
    """
    Get all the tables and columns existent in the database

    :param bool test: Indicates if tests are currently run
    :return: A dictionary containing the tables in the form {table:[column1 type1, column2 type2]}
    :rtype: dict
    """
    db_schema: dict = dict()
    # Get all existent tables from the database
    if test:
        path = config.test_db_file
    else:
        path = config.db_file
    con = duckdb.connect(path)
    db_tables_unprocessed: list = con.execute("SELECT table_name FROM duckdb_tables()").fetchall()
    # Fetch all table names into a single tuple
    # (throws error if no tables in DB, but this is checked at the execution start)
    db_tables: list = list(list(zip(*db_tables_unprocessed))[0])
    for table in db_tables:
        # Get the columns of the table in the DB and put entry into dictionary
        db_columns_unprocessed: list = con.execute(
            "SELECT column_name, data_type FROM duckdb_columns() WHERE table_name=?", [table]).fetchall()
        db_columns: list = list(list(zip(*db_columns_unprocessed))[0])
        db_column_types: list = list(list(zip(*db_columns_unprocessed))[1])
        db_columns_and_type: list = [i + " " + j for i, j in zip(db_columns, db_column_types)]
        db_schema[table] = db_columns_and_type
    con.close()
    # print(f"DB Schema: {db_schema}")
    return db_schema


# Gets constraints like key relations to make it easier for LLM to make less errors
#   (duckdb_constraints() function)
def get_usable_constraints_from_db(rewrite_tables: list[str], test: bool) -> dict:
    """
    Find constraints in the database that exist between two tables used for rewriting a query.

    :param list[str] rewrite_tables: All the tables that are used for the rewrite of the query
    :param bool test: Indicates if tests are currently run
    :return: A dictionary mapping from the table name to its usable foreign keys (as a list of constraint strings)
    :rtype: dict
    """
    # Set the path
    if test:
        path = config.test_db_file
    else:
        path = config.db_file
    # Get all constraints from the database with foreign keys
    con = duckdb.connect(path)
    constraints: list = con.execute("SELECT table_name, constraint_text "
                                    "FROM duckdb_constraints() WHERE constraint_type = 'FOREIGN KEY'").fetchall()
    con.close()
    # Return empty dict if no constraints are there
    if len(constraints) == 0:
        return dict()
    tables_names = list(zip(*constraints))[0]
    constraint_texts = list(zip(*constraints))[1]  # e.g. FOREIGN KEY (h) REFERENCES tf_3(h)
    # Get the references for each table
    found_mappings: dict = dict()
    for table_name, constraint_text in zip(tables_names, constraint_texts):
        if table_name in rewrite_tables:
            # Get the referenced table
            find_string: str = ' REFERENCES '
            index_start: int = constraint_text.find(find_string)
            found_table_reference = constraint_text[index_start + len(find_string):]
            index_end: int = found_table_reference.find('(')
            found_table_reference = found_table_reference[:index_end]
            found_table_reference = found_table_reference.strip().strip('"').strip("'")
            # Check if the referenced table is also in the tables used for rewriting
            if index_start != -1 and index_end != -1 and found_table_reference in rewrite_tables:
                already_found_constraints: list[str] = found_mappings.get(table_name, [])
                already_found_constraints.append(constraint_text)
                found_mappings[table_name] = already_found_constraints
    return found_mappings


def check_existence_of_tables(test: bool) -> bool:
    """
    Check if there are tables in the database.

    :param bool test: Indicates if tests are currently run
    :return: True if there are tables, false otherwise
    :rtype: bool
    """
    # Set the path
    if test:
        path = config.test_db_file
    else:
        path = config.db_file
    # Get the tables
    connection = duckdb.connect(path)
    db_tables: list = connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
    connection.close()
    if len(db_tables) == 0:
        return False
    else:
        return True
