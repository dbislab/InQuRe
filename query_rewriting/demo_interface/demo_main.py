"""
This file contains the function that can later be called by the UI for the Demo
"""
import os

import query_rewriting.config as config
from query_rewriting.demo_interface.demo_callable import DemoCallable
from query_rewriting.distance_measures.vector_embedding import set_up_model
from query_rewriting.main import execute_query_rewriting
from query_rewriting.utilities.duckdb_functions import check_existence_of_tables
from query_rewriting.utilities.reproducibility_functions import create_reproducibility_database


def run_rewriter_for_ui(original_query: str, db_file_path: str, gpt_model: str, num_alternatives_returned: int,
                        additional_num_queries_produced: int, prefilter_kind: int, rewrite_kind: int,
                        ranker_kind: int, ranker_kind_string_sim: int, ranker_kind_intent_sim: int,
                        database_prefix: bool, embedding_threshold: float, sllm_percent_returned_tables: float,
                        cllm_threshold: float, mmr_lambda: float, llms_package_size: int, num_correction_tries: int,
                        demo_object: DemoCallable):
    """
    Run the rewriter with the given input from the UI.Return intermediate results via the demo_callable object.

    :param original_query: The query that one wants to be rewritten as a string.
    :param db_file_path: The path to the database file (either relative or absolute) as a string.
    :param gpt_model: The model from GPT one wants to use. Currently supported models are gpt-4o, gpt-4o-mini, o1-preview, and o1-mini.
    :param num_alternatives_returned: The number of alternative queries to be returned to the user in the end.
    :param additional_num_queries_produced: This number is added to num_alternatives_returned. The total is then the number of rewrites produced, to account for pruned queries.
    :param prefilter_kind: The kind of table filter used by the system. 1 is for the embedding filter (E), 2 for the simple LLM filter (SLLM), and 3 for the complex LLM filter (CLLM). Set this to -1 for no filter.
    :param rewrite_kind: The kind of rewriter to use. 1 is for the simple rewriting (S) and 2 for the NL rewriting (NL).
    :param ranker_kind: The kind of ranker to use in the system. 2 is for the simple ranker (I) and 3 for MMR. Set this to -1 for no ranking.
    :param ranker_kind_string_sim: The kind of string similarity to use. There is only one available string similarity (number 1).
    :param ranker_kind_intent_sim: The kind of similarity to use to measure the intent similarity between two queries. 1 is the embedding similarity via tables (ES) and 2 is the LLM similarity (LLMS).
    :param database_prefix: True if the database has prefixes in the table names for tables from different sources, false otherwise. This should be set to True for Spider and False for IMDB.
    :param embedding_threshold: The threshold for table similarity in the embedding table filter. A table is considered if it has a higher embedding similarity to a query table than this. The value range is 0 to 1 and a sensible value is 0.4.
    :param sllm_percent_returned_tables: This describes how many percent of the tables of our database we think will be usable for the rewrites. The values range is 0 to 1 and a sensible value is, e.g., 0.1.
    :param cllm_threshold: The similarity threshold above which a database table is considered relevant. If a table has a higher similarity than this to a suggested table from the LLM it will be used for the rewriting phase. The value range is 0 to 1, a sensible value is 0.7.
    :param mmr_lambda: The lambda parameter for the MMR algorithm. It determines the importance of the similarity of a rewrite to the original query. The value range is 0 to 1, a sensible value is 0.7.
    :param llms_package_size: This value determines how many rewrites are given to the LLM in bulk when asking the LLM for similarity values to the original query. The minimum is 1, a sensible value is 10. It is recommended to not make this value higher than 20, since LLMs struggle with giving  back a specified number of similarities.
    :param num_correction_tries: The maximal amount of iterations for correcting a query via the LLM. The minimum is 0 (no correction), a sensible value is 3. For more than that it can happen that the query deviates too much from the original rewrite.
    :param demo_object: The callable object whose functions are used to return values to the UI.
    """
    # TODO implement no filter, no ranker for -1; no ranker also no pruner?
    # TODO callable method if an error occurs: method 4 not needed?
    # Set the config parameters
    config.db_file = db_file_path
    config.gpt_model = gpt_model
    config.num_alternatives = num_alternatives_returned + additional_num_queries_produced
    config.num_results = num_alternatives_returned
    config.rewrite_kind = rewrite_kind
    config.ranker_kind = ranker_kind
    config.prefilter_kind = prefilter_kind
    config.db_prefixes = database_prefix
    config.sim_measure_string = ranker_kind_string_sim
    config.sim_measure_intent = ranker_kind_intent_sim
    config.demo_callback = demo_object
    # Params set to defaults
    config.reproducibility = False
    config.check_executability = False
    # additional params
    config.embedding_threshold = embedding_threshold
    config.sllm_percent_returned_tables = sllm_percent_returned_tables
    config.cllm_threshold = cllm_threshold
    config.mmr_lambda = mmr_lambda
    config.llms_package_size = llms_package_size
    config.num_correction_tries = num_correction_tries
    # Check if we have a query
    if original_query.strip == "" or original_query is None:
        raise config.RewritingNotPossible("Input Query is empty") # TODO is this checked via interface?
    # Check if the database exists
    if not (os.path.isfile(config.db_file)):
        # Results in maybe an empty DB:
        # An empty DB does not make sense for rewriting queries
        print("\nWarning: The specified DB file does not exist. It will be created on the first access.")
        raise config.RewritingNotPossible("Database file not found")
    # Check if there are tables in the database, if not the rewriting on existent tables does not make sense
    if not check_existence_of_tables(False):
        raise config.RewritingNotPossible("There are no tables in the database.\nNo executable rewrite can be produced.")
    # Check that the number of result queries is not bigger than the number of produced alternatives
    if config.num_alternatives < config.num_results:
        raise config.RewritingNotPossible(f"Cannot output more queries ({config.num_results}) "
                                   f"than the number of alternatives produced ({config.num_alternatives}).")
    # Set up of all needed elements
    set_up_model(config.sentence_embedder)
    # Set up the reproducibility DB
    if config.reproducibility:
        create_reproducibility_database(False)
    # Execute the workflow
    execute_query_rewriting([['SQL',original_query]], config.num_alternatives, config.rewrite_kind,
                            config.ranker_kind, config.num_results, config.prefilter_kind)  # TODO after each phase call demo object function with right statistics