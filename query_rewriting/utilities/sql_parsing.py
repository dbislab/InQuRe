"""
Methods to parse SQL in different ways.
"""

from sql_metadata import Parser


# https://pypi.org/project/sql_metadata/


# no tests needed for such simple function
def extract_tables(input_query: str) -> list[str]:
    """
    Extract the used tables from an input SQL query.

    :param str input_query: The SQL query to get the tables from
    :return: The tables used in the query as a list of strings
    :rtype: list[str]
    """
    parser: Parser = Parser(input_query)
    return parser.tables


# no tests needed for such simple function
def extract_columns(input_query: str) -> list[str]:
    """
    Extract the used columns from an input SQL query.

    :param str input_query: The SQL query to get the columns from
    :return: The columns used in the query as a list of strings
    :rtype: list[str]
    """
    parser: Parser = Parser(input_query)
    return parser.columns


# no tests needed for such simple function
def extract_columns_with_place(input_query: str) -> dict[str, list[str]]:
    """
    Extract the used columns from an input SQL query.
    Save for each column the place where it was used in the query
    (select, where, order_by, group_by, join, insert and update).

    :param str input_query: The SQL query to get the columns from
    :return: The columns in a dictionary with possible keys: select, where, order_by, group_by, join, insert and update
    :rtype: dict[str, list[str]]
    """
    parser: Parser = Parser(input_query)
    return parser.columns_dict
