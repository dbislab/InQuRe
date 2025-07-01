"""
Methods to prefilter the tables in the database, s.t. the prompt can get existent tables to use for alternative queries.
"""
import duckdb
import query_rewriting.config as config

from query_rewriting.config import NotYetSupportedException
from query_rewriting.distance_measures.vector_embedding import similarity_spacy_en_core_web_lg
from query_rewriting.rewrite_generator.prefilter_tables_llm import simple_prefilter_via_llm, complex_prefilter_via_llm
from query_rewriting.rewrite_generator.prefilter_tables_summaries import prefilter_tables_via_summaries
from query_rewriting.utilities.duckdb_functions import get_tables_from_db
from query_rewriting.utilities.sql_parsing import extract_tables


# tested in the main workflow
def prefilter_tables(input_query: str, prefilter_kind: int) -> dict:
    """
    Pre-filter the tables s.t. they can be given as usable tables for generating alternative queries.

    :param str input_query: The query which should be rewritten on the tables of the database
    :param int prefilter_kind: The kind of algorithm to use for pre-filtering
    :return: A dictionary containing the usable tables in the form {table:[column1 type1, column2 type2]}
    :rtype: dict
    """
    db_tables: dict = get_tables_from_db(False)
    if prefilter_kind == 1:
        return simple_prefilter_via_scipy_embedding(input_query, db_tables)
    elif prefilter_kind == 2:
        return simple_prefilter_via_llm(input_query, db_tables)
    elif prefilter_kind == 3:
        return complex_prefilter_via_llm(input_query, db_tables)
    elif prefilter_kind == 4:
        return prefilter_tables_via_summaries(input_query, db_tables)
    else:
        raise NotYetSupportedException(f"Pre-filtering tables using kind {prefilter_kind} is not yet supported")


def simple_prefilter_via_scipy_embedding(input_query: str, db_tables: dict) -> dict:
    """
    Prefilter the tables via finding tables that are similar to the ones in the query.
    Similarity is measured using the cosine similarity of the word embeddings via scipy.

    :param str input_query: The query that is used to prefilter the tables of the database
    :param dict db_tables: The tables in the database in the form {table:[column1 type1, column2 type2]}
    :return: A dictionary containing the usable tables in the form {table:[column1 type1, column2 type2]}
    :rtype: dict
    """
    # Preparation of dict and word embedding
    proposed_tables: dict = dict()
    # nlp: Language = load_en_core_web_lg()
    sql_tables: list[str] = extract_tables(input_query)
    for sql_table in sql_tables:
        for db_table, db_columns in db_tables.items():
            if similarity_spacy_en_core_web_lg(sql_table, db_table, config.nlp_language_model) > 0.4:
                proposed_tables[db_table] = db_columns
                # print(f"Table {db_table} has been added for SQL query table {sql_table}.")
    print(f"The following {len(proposed_tables)} tables were found in the database: ")
    print(f"{', '.join(proposed_tables.keys())}")
    return proposed_tables


# Idea for function that does postprocessing:
#   Get all the tables found by the table filtering algorithm
#   Look at those tables pairwise
#   If two tables have a join using only one other table: add this table to the result
def add_joining_tables(prefiltered_tables: dict, db_tables: dict, test: bool) -> dict:
    """
    For the found tables from the pre-filtering, add tables that are needed to join two of these tables together.
    It is checked pairwise if both tables have a reference to the same table,
    i.e. also tables that are referenced by both tables are added (even if the two tables are already joinable).

    :param dict prefiltered_tables: The tables that were found in the pre-filtering
           in the form {table:[column1 type1, column2 type2]}
    :param dict db_tables: The tables in the database in the form {table:[column1 type1, column2 type2]}
    :param bool test: Indicates if tests are currently run
    :return: The dictionary of prefiltered tables enhanced with the joining tables
    :rtype: dict
    """
    # Configure DB path
    if test:
        path = config.test_db_file
    else:
        path = config.db_file
    # Get all constraints from the database with foreign keys
    con = duckdb.connect(path)
    constraints: list = con.execute("SELECT table_name, constraint_text "
                                    "FROM duckdb_constraints() WHERE constraint_type = 'FOREIGN KEY'").fetchall()
    con.close()
    if len(constraints) == 0:
        return prefiltered_tables
    tables_names = list(zip(*constraints))[0]
    constraint_texts = list(zip(*constraints))[1]  # e.g. FOREIGN KEY (h) REFERENCES tf_3(h)
    # Save all references regarding the tables
    references: dict = dict()
    for table_name, constraint_text in zip(tables_names, constraint_texts):
        find_string: str = ' REFERENCES '
        index_start: int = constraint_text.find(find_string)
        found_table_reference = constraint_text[index_start + len(find_string):]
        index_end: int = found_table_reference.find('(')
        found_table_reference = found_table_reference[:index_end]
        found_table_reference = found_table_reference.strip().strip('"').strip("'")
        if index_end != -1 and index_start != -1:
            existent_references1 = references.get(table_name, [])
            existent_references2 = references.get(found_table_reference, [])
            existent_references1.append(found_table_reference)
            existent_references2.append(table_name)
            references[table_name] = existent_references1
            references[found_table_reference] = existent_references2
    # Find all tables that could be added
    possible_additions: set[str] = set()
    found_tables: dict = dict()
    for table1 in prefiltered_tables:
        for table2 in prefiltered_tables:
            if table1 == table2:
                continue
            references1: list[str] = references.get(table1, [])
            references2: list[str] = references.get(table2, [])
            intersection: set[str] = set(references1).intersection(set(references2))
            possible_additions.update(intersection)
    for table in possible_additions:
        columns: list[str] = db_tables.get(table, [])
        if len(columns) != 0:
            found_tables[table] = columns
    # Add the tables to the found tables from the prefilter
    prefiltered_tables.update(found_tables)
    return prefiltered_tables


