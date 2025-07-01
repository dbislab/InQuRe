"""
Global variables and classes for the project
"""
import os.path

import spacy
from spacy import Language

# The following variables are accessed by different packages
# and can be set here or via the command line
# The default absolute path to the input file with the SQL queries
file_input_string: str = os.path.join('resources','spider_with_prefixes_input.txt')
# The default absolute path to the DuckDB database file
db_file: str = os.path.join('resources','spider_with_prefixes.db')
# Indicator if the Database has prefixes for tables from different schemas using underlines
db_prefixes: bool = False # Should be left as is
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
num_alternatives: int = 1
# The number of returned (and ranked) result queries
num_results: int = 1
# The algorithm used for pre-filtering the tables before the prompt (1 for simple Word2Vec)
prefilter_kind: int = 1
# The algorithm used for rewriting (1 for simple rewrite prompt)
rewrite_kind: int = 1
# The algorithm used for ranking (1 for simple string distance and MMR)
ranker_kind: int = 2

# The following variables are here to be consistently used in the whole project
# The output length available for gpt-4o and gpt-4o-mini (in tokens)
output_length_gpt_4o: int = 16384
# The output length available for o1-preview and o1-mini (in tokens)
output_length_gpt_o1: int = 32768
# The input length available for all models (context window)
input_length_gpt: int = 128000
# Suffix added to the DB file name, defining the DB where metadata for this table is stored
db_metadata_addition: str = '_metadata'
# Name of the table for the metadata
db_metadata_table_name: str = 'Metadata'
# Suffix added to the DB file name, defining the DB where responses from the LLM are stored
db_reproducibility_addition: str = '_reproducibility'
# Name of the table for reproducibility of LLM outputs
db_reproducibility_table_name: str = 'LLMOutputs'
# Names and types of the columns in the metadata table (topics and keywords are comma seperated strings)
# First column has to be the table name, order is relevant for pre-filtering via table summaries
db_metadata_column_names_and_types: list[str] = ['table_name VARCHAR PRIMARY KEY', 'nl_description VARCHAR',
                                                 'topics VARCHAR', 'keywords VARCHAR']
# Names of the columns in the metadata table (topics and keywords are comma seperated strings)
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
