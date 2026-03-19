"""
This file contains the function that can later be called by the UI for the Demo
"""
import os
import random
import time

import duckdb

import query_rewriting.config as config
from query_rewriting.demo_interface.demo_callable import DemoCallable
from query_rewriting.demo_interface.phase1_statistics import Phase1Statistics
from query_rewriting.demo_interface.phase2_statistics import Phase2Statistics
from query_rewriting.demo_interface.phase3_statistics import Phase3Statistics
from query_rewriting.demo_interface.phase4_statistics import Phase4Statistics
from query_rewriting.distance_measures.vector_embedding import set_up_model
from query_rewriting.main import execute_query_rewriting
from query_rewriting.utilities.duckdb_functions import check_existence_of_tables
from query_rewriting.utilities.reproducibility_functions import create_reproducibility_database


def run_rewriter_for_ui(original_query: str, db_file_connection: duckdb.DuckDBPyConnection,
                        gpt_model: str, num_alternatives_returned: int,
                        additional_num_queries_produced: int, prefilter_kind: int, rewrite_kind: int,
                        ranker_kind: int, ranker_kind_string_sim: int, ranker_kind_intent_sim: int,
                        database_prefix: bool, embedding_threshold: float, sllm_percent_returned_tables: float,
                        cllm_threshold: float, mmr_lambda: float, llms_package_size: int, num_correction_tries: int,
                        demo_object: DemoCallable):
    """
    Run the rewriter with the given input from the UI.Return intermediate results via the demo_callable object.

    :param original_query: The query that one wants to be rewritten as a string.
    :param db_file_connection: The database connection as an object from DuckDB.
    :param gpt_model: The model from GPT one wants to use. Currently supported models are gpt-4o, gpt-4o-mini, o1-preview, and o1-mini as older models. Now, gpt-5, gpt-5.1, and gpt-5.2 also work.
    :param num_alternatives_returned: The number of alternative queries to be returned to the user in the end.
    :param additional_num_queries_produced: This number is added to num_alternatives_returned. The total is then the number of rewrites produced, to account for pruned queries.
    :param prefilter_kind: The kind of table filter used by the system. 1 is for the embedding filter (E), 2 for the simple LLM filter (SLLM), and 3 for the complex LLM filter (CLLM). Set this to -1 for no filter.
    :param rewrite_kind: The kind of rewriter to use. 1 is for the simple rewriting (S) and 2 for the NL rewriting (NL).
    :param ranker_kind: The kind of ranker to use in the system. 2 is for the simple ranker (I) and 3 for MMR. Set this to -1 for no ranking (pruner still active then).
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
    # TODO LATER check for DDL Statements (either in Input or rewrites or both) and do not execute them (or make transactions and roll it back if tables change from that)
    # TODO ONLY IF NEEDED check for global variables/other variables that they are reset after each run for performance reasons? -> reset method if we run into problems only
    # TODO LATER implement loading from cache?
    # TODO OPTIONAL prune queries using non-existent table before ranking?
    # TODO LATER if multiple UI calls come at once: problem for different configuration with clash in config file...fix!
    # TODO NOW also give statistics that are already available if phase throws an error, check when statistics are set!!
    # TODO LATER also continue if too few rewrites produced/other errors that do not kill process?
    # Set the config parameters
    config.db_file = ""
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
    # Additional params
    config.embedding_threshold = embedding_threshold
    config.sllm_percent_returned_tables = sllm_percent_returned_tables
    config.cllm_threshold = cllm_threshold
    config.mmr_lambda = mmr_lambda
    config.llms_package_size = llms_package_size
    config.num_correction_tries = num_correction_tries
    # Reset statistics
    config.statistics1 = Phase1Statistics()
    config.statistics2 = Phase2Statistics()
    config.statistics3 = Phase3Statistics()
    config.statistics4 = Phase4Statistics()
    # Check if we have a query and notify UI if not
    if original_query.strip == "" or original_query is None:
        demo_object.parameter_check_failed("Input Query is empty")
    # Check if there are tables in the database, if not the rewriting on existent tables does not make sense (then UI is notified)
    if not check_existence_of_tables(False,db_file_connection):
        demo_object.parameter_check_failed("There are no tables in the database. No executable rewrite can be produced.")
    # Check that the number of result queries is not bigger than the number of produced alternatives (if not UI is notified)
    if config.num_alternatives < config.num_results:
        demo_object.parameter_check_failed(f"Cannot output more queries ({config.num_results}) "
                                           f"than the number of alternatives produced ({config.num_alternatives}).")
    # Notify the callback that check was successful
    demo_object.parameter_check_successful()
    # Set up of all needed elements
    set_up_model(config.sentence_embedder)
    # Set up the reproducibility DB
    if config.reproducibility:
        create_reproducibility_database(False)
    # Execute the workflow (after each phase it calls demo object function with right statistics)
    execute_query_rewriting([['SQL',original_query]], config.num_alternatives, config.rewrite_kind,
                            config.ranker_kind, config.num_results, config.prefilter_kind, db_file_connection)



def run_rewriter_for_ui_test(original_query: str, db_file_connection: duckdb.DuckDBPyConnection,
                             gpt_model: str, num_alternatives_returned: int,
                            additional_num_queries_produced: int, prefilter_kind: int, rewrite_kind: int,
                            ranker_kind: int, ranker_kind_string_sim: int, ranker_kind_intent_sim: int,
                            database_prefix: bool, embedding_threshold: float, sllm_percent_returned_tables: float,
                            cllm_threshold: float, mmr_lambda: float, llms_package_size: int, num_correction_tries: int,
                            demo_object: DemoCallable):
    """
    This is a function for testing the UI. It has the same parameters as the real one, but does not have AI API calls.
    It also returns results via the statistics objects (with wait time in between) and sometimes throws an error via randomization.
    """
    # Set the config parameters
    config.db_file = ""
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
    # Additional params
    config.embedding_threshold = embedding_threshold
    config.sllm_percent_returned_tables = sllm_percent_returned_tables
    config.cllm_threshold = cllm_threshold
    config.mmr_lambda = mmr_lambda
    config.llms_package_size = llms_package_size
    config.num_correction_tries = num_correction_tries
    # Reset statistics
    config.statistics1 = Phase1Statistics()
    config.statistics2 = Phase2Statistics()
    config.statistics3 = Phase3Statistics()
    config.statistics4 = Phase4Statistics()
    # Check if we have a query and notify UI if not
    if original_query.strip == "" or original_query is None:
        demo_object.parameter_check_failed("Input Query is empty")
    # Check if there are tables in the database, if not the rewriting on existent tables does not make sense (then UI is notified)
    if not check_existence_of_tables(False, db_file_connection):
        demo_object.parameter_check_failed(
            "There are no tables in the database. No executable rewrite can be produced.")
    # Check that the number of result queries is not bigger than the number of produced alternatives (if not UI is notified)
    if config.num_alternatives < config.num_results:
        demo_object.parameter_check_failed(f"Cannot output more queries ({config.num_results}) "
                                           f"than the number of alternatives produced ({config.num_alternatives}).")
    # Notify the callback that check was successful
    demo_object.parameter_check_successful()
    # Set up of all needed elements
    set_up_model(config.sentence_embedder)
    # Set up the reproducibility DB
    if config.reproducibility:
        create_reproducibility_database(False)
    # Test calls for the callback object
    config.statistics1.input_tokens = 1000
    config.statistics1.output_tokens = 2000
    config.statistics1.llm_prompts = ["Example prompt 1", "Example prompt 2"]
    config.statistics1.llm_answers = ["Example answer 1", "Example answer 2"]
    config.statistics1.num_selected_tables = 3
    config.statistics1.selected_tables = {"table1": ["column1 type1", "column2 type2"], "table2": ["column3 type3"], "table3": ["column4 type4", "column5 type5"]}
    if random.uniform(0,1) < 0.1:
        demo_object.first_phase_error("No tables found", config.statistics1)
        return
    demo_object.first_phase_done(config.statistics1)
    time.sleep(2)
    config.statistics2.input_tokens = 1000
    config.statistics2.output_tokens = 2000
    config.statistics2.llm_prompts = ["Example prompt 1", "Example prompt 2"]
    config.statistics2.llm_answers = ["Example answer 1", "Example answer 2"]
    config.statistics2.num_produced_rewrites = 2
    config.statistics2.current_rewrites = ["Example rewrite 1", "Example rewrite 2"]
    if random.uniform(0,1) < 0.1:
        demo_object.second_phase_error("No rewrites found", config.statistics2)
        return
    demo_object.second_phase_done(config.statistics2)
    time.sleep(2)
    config.statistics3.input_tokens = 1000
    config.statistics3.output_tokens = 2000
    config.statistics3.llm_prompts = ["Example prompt 1", "Example prompt 2"]
    config.statistics3.llm_answers = ["Example answer 1", "Example answer 2"]
    config.statistics3.num_pruned_queries = 1
    config.statistics3.ranked_queries = ["Example rewrite 1"]
    if random.uniform(0,1) < 0.1:
        demo_object.third_phase_error("TOo many queries pruned", config.statistics3)
        return
    demo_object.third_phase_done(config.statistics3)
    time.sleep(2)
    config.statistics4.input_tokens = 1000
    config.statistics4.output_tokens = 2000
    config.statistics4.llm_prompts = ["Example prompt 1", "Example prompt 2"]
    config.statistics4.llm_answers = ["Example answer 1", "Example answer 2"]
    config.statistics4.error_messages = ["", "unexpected * in WHERE", ""]
    config.statistics4.final_rewrites = ["Example rewrite 1", "Broken Rewrite", "Example rewrite 3"]
    config.statistics4.results_of_final_rewrites = [[["column1", "column2"], ["value11", "value12"], ["value21", "value22"]], [], [["column1", "column2"]]]
    demo_object.fourth_phase_done(config.statistics4)