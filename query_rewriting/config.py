"""
Global variables and classes for the project
"""
import os.path

import spacy
from spacy import Language

from query_rewriting.demo_interface.demo_callable import DemoCallable
from query_rewriting.demo_interface.phase1_statistics import Phase1Statistics
from query_rewriting.demo_interface.phase2_statistics import Phase2Statistics
from query_rewriting.demo_interface.phase3_statistics import Phase3Statistics
from query_rewriting.demo_interface.phase4_statistics import Phase4Statistics

# The following variables are accessed by different packages
# and can be set here or via the command line
# The default absolute path to the input file with the SQL queries
file_input_string: str = os.path.join('resources','spider_with_prefixes_input.txt')
# The default absolute path to the DuckDB database file
db_file: str = os.path.join('resources','spider_with_prefixes.db')
# Indicator if the Database has prefixes for tables from different schemas using underlines
db_prefixes: bool = True # Should be left as is
# The used GPT model in every API call (options: gpt-4o, gpt-4o-mini, o1-preview, o1-mini)
gpt_model: str = "gpt-4o-mini"
# A test DB file for testing
test_db_file: str = os.path.join('..','resources','test.db')
# The distance function used for string similarity
sim_measure_string: int = 1
# The distance function used for intent similarity
sim_measure_intent: int = 1
# Whether we want reproducible results (LLM requests that happened before will be taken from a DB)
reproducibility: bool = False # Should be left as is
# The following variables are given to methods as parameters,
# but are still here for configuration without the command line
# The number of alternative queries produced during the rewrite
num_alternatives: int = 5
# The number of returned (and ranked) result queries
num_results: int = 5
# The algorithm used for pre-filtering the tables before the prompt (1 for simple Word2Vec)
prefilter_kind: int = 1
# The algorithm used for rewriting (1 for simple rewrite prompt)
rewrite_kind: int = 1
# The algorithm used for ranking (1 for simple string distance and MMR)
ranker_kind: int = 1
# The flag to decide if executability of the input query is checked (if the query is executable, no rewriting will be done)
# True if it should be checked, False otherwise
check_executability: bool = False

# Additional params, only changeable here or via UI method
# Similarity threshold for the embedding similarity filter
embedding_threshold: float = 0.4  # Used in prefilter_tables
# Percentage of tables expected to be usable
sllm_percent_returned_tables: float = 0.1 # Used in prefilter_tables_llm
# Threshold for a table to be considered similar in the complex LLM filter
cllm_threshold: float = 0.7 # Used in prefilter_tables_llm
# Lambda parameter for MMR algorithm (ranker 3)
mmr_lambda: float = 0.7 # Used in rank_alternatives
# Number of rewrites per LLM call asking for similarity to original query
llms_package_size: int = 10 # Used in sql_queries_comparison
# Maximum number of iterations to try and correct a query
num_correction_tries: int = 3 # Used in main as param for function from query_correction

# Callback object and statistics to show progress in the demo UI and save params here
# UI: Callback Object
demo_callback: DemoCallable|None = None
# UI: Phase1 Statistics
statistics1: Phase1Statistics = Phase1Statistics()
# UI: Phase2 Statistics
statistics2: Phase2Statistics = Phase2Statistics()
# UI: Phase3 Statistics
statistics3: Phase3Statistics = Phase3Statistics()
# UI: Phase4 Statistics
statistics4: Phase4Statistics = Phase4Statistics()

# The following variables are here to be consistently used in the whole project
# The output length available for gpt-4o and gpt-4o-mini (in tokens)
output_length_gpt_4o: int = 16384
# The output length available for o1-preview and o1-mini (in tokens)
output_length_gpt_o1: int = 32768
# The output length available for newer models (gpt-5.1, gpt-5.4) (in tokens)
output_length_gpt_5: int = 128000
# The input length available for all older models (context window)
input_length_gpt: int = 128000
# The input length for newer models (gpt-5 to 5.2)
input_length_gpt_new: int = 400000
# Suffix added to the DB file name, defining the DB where metadata for this table is stored
db_metadata_addition: str = '_metadata'
# Name of the table for the metadata
db_metadata_table_name: str = 'Metadata'
# Suffix added to the DB file name, defining the DB where responses from the LLM are stored
db_reproducibility_addition: str = '_reproducibility'
# Name of the table for reproducibility of LLM outputs
db_reproducibility_table_name: str = 'LLMOutputs'
# Names and types of the columns in the metadata table (topics and keywords are comma separated strings)
# First column has to be the table name, order is relevant for pre-filtering via table summaries
db_metadata_column_names_and_types: list[str] = ['table_name VARCHAR PRIMARY KEY', 'nl_description VARCHAR',
                                                 'topics VARCHAR', 'keywords VARCHAR']
# Names of the columns in the metadata table (topics and keywords are comma separated strings)
db_metadata_column_names: list[str] = ['table_name', 'nl_description', 'topics', 'keywords']
# Name of the used sentence-embedder
sentence_embedder: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
# Limit of the maximum amount of summaries created by the LLM at once (-1 to deactivate)
max_num_summaries: int = 30

# The following variables are set once,
# as doing it more than once would take a lot of time in the program
nlp_language_model: Language = spacy.load('en_core_web_lg')


class NotYetSupportedException(Exception):
    """
    This exception is raised when a part of workflow that would be needed in this certain case is not yet implemented.
    """


class NoRewritesFoundException(Exception):
    """
    This exception is raised when no rewrites could be found for the query using the available data.
    """


class RewritingNotPossible(Exception):
    """
    This exception is raised when a rewrite is not possible from the start.
    """


class RankingNotPossible(Exception):
    """
    This exception is raised if the ranking algorithm fails somehow.
    """
