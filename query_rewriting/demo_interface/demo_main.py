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
    # TODO write which param should have which values for which version of stuff (documentation)
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
    # Reproducibility always set to false currently
    # additional params
    config.embedding_threshold = embedding_threshold
    config.sllm_percent_returned_tables = sllm_percent_returned_tables
    config.cllm_threshold = cllm_threshold
    config.mmr_lambda = mmr_lambda
    config.llms_package_size = llms_package_size
    config.num_correction_tries = num_correction_tries
    # Check if we have a query
    if original_query.strip == "" or original_query is None:
        raise config.RewritingNotPossible("Input Query is empty")
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