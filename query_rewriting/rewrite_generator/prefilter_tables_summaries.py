"""
Methods to prefilter the tables based on table and query summaries.
"""
import math
from itertools import islice
from typing import Tuple
from collections.abc import Iterable

import duckdb
from spacy.tokens import Doc
# from openai.types.chat import ChatCompletion

# from query_rewriting.config import NoRewritesFoundException
import query_rewriting.config as config
from query_rewriting.distance_measures.vector_embedding import calculate_tensor_sim_via_sentence_transformers, \
    model_embedding
from query_rewriting.utilities.find_metadata import llm_find_intent_sql, llm_find_intent_keywords_sql, \
    llm_find_topics_sql
from query_rewriting.utilities.gpt_functions import gpt_api_call, strip_starting_and_ending_characters, \
    strip_whitespaces, strip_code_block_output
from query_rewriting.utilities.statistics import add_tokens

prompt_tokens_prefilter_summary: int = 0
completion_tokens_prefilter_summary: int = 0
total_tokens_prefilter_summary: int = 0


# checked created database tuples -> look ok
# tested only in workflow
def prefilter_tables_via_summaries(input_query: str, db_tables: dict) -> dict:
    """
    Prefilter the tables via summarizing the tables and the query in natural language.
    Similar tables are chosen using similarity metrics for the natural language descriptions.

    :param str input_query: The query that is used to prefilter the tables of the database
    :param dict db_tables: The tables in the database in the form {table:[column1 type1, column2 type2]}
    :return: A dictionary containing the usable tables in the form {table:[column1 type1, column2 type2]}
    :rtype: dict
    """
    # Get the metadata path
    metadata_db_path: str = calculate_metadata_db_path(False)
    # Check if summaries are existent
    possible_tables_without_info: list[str] = check_for_summaries(db_tables, metadata_db_path)
    if len(possible_tables_without_info) != 0:
        # There are tables without info in the metadata table
        # Compute and save info for those
        summaries, topics, keywords = create_summaries(possible_tables_without_info, config.max_num_summaries)
        successful_write: bool = save_tables_info(possible_tables_without_info, summaries, topics, keywords,
                                                  metadata_db_path)
        if not successful_write:
            print("Warning: The metadata for the tables could not be created completely.")
    # Get the usable tables
    resulting_tables: dict = compare_tables_and_query(input_query, db_tables, metadata_db_path)
    return resulting_tables


def check_for_summaries(db_tables: dict, metadata_db_path: str) -> list[str]:
    """
    Check if all tables in the database already have a summary.

    :param dict db_tables: The tables in the database in the form {table:[column1 type1, column2 type2]}
    :param str metadata_db_path: The path to the metadata db file
    :return: A list of the tables in the database that have no metadata information
    :rtype: list[str]
    """
    con = duckdb.connect(metadata_db_path)
    # Create the needed metadata table
    create_statement: str = \
        (f"CREATE TABLE IF NOT EXISTS {config.db_metadata_table_name} "
         f"({", ".join(config.db_metadata_column_names_and_types)});")
    # print(create_statement)
    con.execute(create_statement)
    tables: list[str] = con.execute(f"SELECT * FROM {config.db_metadata_table_name}").fetchall()
    con.close()
    # Maybe also check if there is something actually written in the columns
    #  -> always written together, so should be ok
    # print(list(zip(tables)))
    # Tables could be empty
    if len(tables) == 0:
        tables_with_info: tuple = tuple()
    else:
        tables_with_info: tuple = list(zip(*tables))[0]
    # print(tables_with_info)
    num_table_infos: int = len(tables_with_info)
    new_info_needed: bool = (num_table_infos < len(db_tables.keys()))
    tables_needing_info: list[str] = []
    if new_info_needed:
        print(f"New metadata calculation needed.")
        if len(tables_with_info) == 0:
            print(f"No tables with metadata existent.")
        else:
            print(f"Tables with metadata: {', '.join(tables_with_info)}")
        # Calculate tables that need new metadata
        tables_needing_info = list(set(db_tables.keys()).difference(set(tables_with_info)))
    else:
        print(f"All needed metadata is available.")
    return tables_needing_info


# tested only in workflow, as it makes LLM call
def create_summaries(tables_needing_summary: list[str], max_tables_per_prompt: int = -1) -> Tuple[dict, dict, dict]:
    """
    Create the summaries (and more) for all available tables in the database.

    :param dict tables_needing_summary: The tables in the database that need a summary etc.
    :param int max_tables_per_prompt: The maximum amount of tables asked per prompt
    :return: 3 dictionaries: table_summaries: summaries produced for the tables of the form {table: table_summary}
                             table_topics: topics produced for the tables in the form {table: [topic1, topic2, ...]}
                             table_keywords: keywords produced for the tables in the form {table: [keyword1, ...]}
    :rtype: Tuple[dict, dict, dict]
    """
    # Returned element
    found_summaries: dict = dict()
    found_topics: dict = dict()
    found_keywords: dict = dict()
    # Current metadata is ok -> should be checked before
    # Get the CREATE TABLE statements for all the tables from the database
    #  (database is not empty, checked at start of main method)
    con = duckdb.connect(config.db_file)
    create_statements_list: list = con.execute("SELECT table_name, sql FROM duckdb_tables()").fetchall()
    con.close()
    create_statements_dict: dict = dict(create_statements_list)
    # Slice the list into parts of the right size for the LLM
    max_output_tokens: int = 130
    tables_sliced, max_output_tokens_per_table = (
        slice_list_of_tables_for_summaries(tables_needing_summary, create_statements_dict,
                                           max_output_tokens, max_tables_per_prompt))
    # Set the wanted parameters
    num_topics_per_table: int = 5
    num_keywords_per_table: int = 5
    # Length of summary: max_tokens - 4*(num_topics+num_keywords) (assuming 4 tokens per topic/keyword)
    max_tokens_summary_per_table: int = (
            max_output_tokens_per_table - 4 * (num_topics_per_table + num_keywords_per_table))
    # Count tokens
    local_prompt_tokens: int = 0
    local_completion_tokens: int = 0
    local_total_tokens: int = 0
    total_llm_use: bool = False
    for tables in tables_sliced:
        # Count statistics
        tables_not_found: int = 0
        incorrect_format: int = 0
        # LLM prompt
        # For topics: maybe tell LLM to stay consistent -> does not help for multiple api calls if topics not given
        #   -> not too bad as they are compared via embedding
        create_statements_tables: list[str] = [create_statements_dict.get(t) for t in tables]
        content_for_gpt: str = (f"Given the following CREATE TABLE statements in SQL:\n"
                                f"{'\n'.join(create_statements_tables)}\n"
                                f"I want you to do three things:\n"
                                f"1. Summarize\n"
                                f"Generate a summary to describe the table with maximal "
                                f"{max_tokens_summary_per_table} tokens regarding the information need it satisfies.\n"
                                f"Think about how you would describe an entity in this table and use this as a "
                                f"guideline for the table description.\n"
                                f"Only give the description in natural language sentences, "
                                f"mimicking the human understanding of the table.\n"
                                f"Do not describe the table column by column. Be concise.\n"
                                f"2. Topics\n"
                                f"Give me {num_topics_per_table} topics that fit the information need of the table, "
                                f"i.e., what a human would see as topics in the table.\n"
                                f"Do not just repeat the query tables and columns themselves as topics, "
                                f"but abstract from them. Be concise.\n"
                                f"3. Keywords\n"
                                f"Give me {num_keywords_per_table} keywords that fit the information need of the "
                                f"table, i.e., what a human would see as keywords in the table.\n"
                                f"Do not just repeat the query tables and columns themselves as keywords, "
                                f"but abstract from them. Be concise.\n"
                                f"\n"
                                f"I want you to give me these three things in the following output format "
                                f"(adapting the number of topics and keywords to the above given numbers):\n"
                                f"table_name[summary;topic1,topic2;keyword1,keyword2]\n"
                                f"Return exactly this format for each given table of the CREATE statements "
                                f"in the same order as the CREATE statements given to you.\n"
                                f"Separate the outputs for the tables by using a new line for each output.")
        message: Iterable = [
            {"role": "system", "content": "We will work with databases and queries in SQL."},
            {
                "role": "user",
                "content": content_for_gpt
            }
        ]
        completion, llm_used = gpt_api_call(config.gpt_model, message)
        gpt_answer: str = completion["choices"][0]["message"]["content"]
        # print(completion.choices[0].finish_reason)
        # print(gpt_answer)
        # Process the answer by splitting the lines and then the elements
        gpt_answer = strip_code_block_output(gpt_answer)
        separated_metadata: list[str] = strip_starting_and_ending_characters(gpt_answer).splitlines()
        separated_metadata = [line for line in separated_metadata if line.strip() != ""]
        # print('\n'.join(separated_metadata))
        print(f"Found metadata: {len(separated_metadata)}, Tables in DB: {len(tables)}")
        if not len(separated_metadata) == len(tables):
            print("Warning: Metadata missing for some tables in the current prompt iteration.")
        for m in separated_metadata:
            m_split_table: list[str] = m.split("[")
            if len(m_split_table) != 2:  # Control of LLM output
                incorrect_format += 1
                continue
            m_table: str = strip_whitespaces(m_split_table[0])
            m_formatted: str = strip_whitespaces(m_split_table[1]).removesuffix("]").removeprefix("[")
            m_split: list[str] = m_formatted.split(";")
            if len(m_split) != 3:  # Control of LLM output
                incorrect_format += 1
                continue
            if m_table in tables:  # Table from LLM exists in the database
                found_summaries[m_table] = strip_whitespaces(m_split[0]).replace("'", "''")
                found_topics[m_table] = [strip_whitespaces(k).replace("'", "''") for k in m_split[1].split(",")]
                found_keywords[m_table] = [strip_whitespaces(k).replace("'", "''") for k in m_split[2].split(",")]
            else:
                tables_not_found += 1
        # Add the used tokens
        if llm_used:
            local_prompt_tokens += completion["usage"]["prompt_tokens"]
            local_completion_tokens += completion["usage"]["completion_tokens"]
            local_total_tokens += completion["usage"]["total_tokens"]
            total_llm_use = True
        print(f"Tables not found in the DB but given by LLM (in this iteration): {tables_not_found}")
        print(f"Tables that were incorrectly formatted (in this iteration): {incorrect_format}")
    # Check if all tables got answers
    if not len(found_summaries) == len(found_topics) == len(found_keywords) == len(tables_needing_summary):
        print(f"Warning: {len(tables_needing_summary) - len(found_summaries)} tables did not get correct metadata. "
              f"They will not be considered during rewriting.")
    # Write statistics from the execution
    if total_llm_use:
        global prompt_tokens_prefilter_summary
        prompt_tokens_prefilter_summary = local_prompt_tokens
        global completion_tokens_prefilter_summary
        completion_tokens_prefilter_summary = local_completion_tokens
        global total_tokens_prefilter_summary
        total_tokens_prefilter_summary = local_total_tokens
        print(
            f"Tokens used after the table filtering with summaries: \n\t"
            f"Prompt Tokens: {prompt_tokens_prefilter_summary}, "
            f"Completion Tokens: {completion_tokens_prefilter_summary}, "
            f"Total Tokens: {total_tokens_prefilter_summary}\n")
        add_tokens(local_prompt_tokens, local_completion_tokens, local_total_tokens)
    else:
        print("No tokens used for table filtering with summaries due to reproducibility DB.")
    return found_summaries, found_topics, found_keywords


def slice_list_of_tables_for_summaries(table_list: list[str], create_statements: dict,
                                       max_tokens: int, max_tables_per_prompt: int = -1) -> Tuple[list[list[str]], int]:
    """
    Slice the list of the tables into parts, s.t. the in- and output tokens of the LLM are enough to process them.

    :param list[str] table_list: The list of the tables need to be in the prompt
    :param dict create_statements: The CREATE statements for the tables in the database in the form {table:statement}
    :param int max_tokens: The maximum number of tokens for each table in the output prompt
    :param int max_tables_per_prompt: The maximum amount of tables asked per prompt
    :return: The list sliced into parts, each containing a list of table names for one prompt;
             as a second element the tokens per table possible in the prompt output
    :rtype: Tuple[list[list[str]], int]
    """
    # Input tables: Input tokens/(avg length in characters of CREATE statements/4) as ca. 4 chars per token
    length_create_statements: list[int] = [len(item) for item in create_statements.values()]
    avg_table_create_length: float = sum(length_create_statements) / len(length_create_statements)
    avg_table_create_tokens: float = avg_table_create_length / 4
    # Subtract length of prompt from input length before division (roughly 300)
    max_input_tables: int = math.floor((config.input_length_gpt - 300) / avg_table_create_tokens)
    # get output length of models (in tokens)
    if config.gpt_model == "gpt-4o" or config.gpt_model == "gpt-4o-mini":
        output_length: int = config.output_length_gpt_4o
    else:
        output_length: int = config.output_length_gpt_o1
    # input_tables = output_tokens/max_tokens
    max_output_tables: int = math.floor(output_length / max_tokens)
    # Minimum of tables for in- and output is taken
    num_tables_per_slice: int = min(max_input_tables, max_output_tables)
    if max_tables_per_prompt != -1:
        num_tables_per_slice = min(num_tables_per_slice, max_tables_per_prompt)
    num_output_tokens_per_table: int = math.floor(output_length / num_tables_per_slice)
    # Split the tables into slices
    tables_split_for_requests: list[list[str]] = \
        [list(islice(table_list, i, i + num_tables_per_slice)) for i in range(0, len(table_list), num_tables_per_slice)]
    return tables_split_for_requests, num_output_tokens_per_table


def save_tables_info(tables: list[str], table_summaries: dict, table_topics: dict,
                     table_keywords: dict, metadata_db_path: str) -> bool:
    """
    Save the summaries and other info produced for the tables in a database.
    It is assumed that the tables given in the input do not have metadata saved yet.

    :param list[str] tables: the tables that have info to save
    :param dict table_summaries: The summaries produced for the tables in the form {table: table_summary}
    :param dict table_topics: The topics produced for the tables in the form {table: [topic1, topic2, ...]}
    :param dict table_keywords: The keywords produced for the tables in the form {table: [keyword1, keyword2, ...]}
    :param str metadata_db_path: The path to the metadata database where all info is saved
    :return: Whether writing back was successful
    :rtype: bool
    """
    successful_write: bool = True
    con = duckdb.connect(metadata_db_path)
    # Create the needed metadata table
    create_statement: str = \
        (f"CREATE TABLE IF NOT EXISTS {config.db_metadata_table_name} "
         f"({", ".join(config.db_metadata_column_names_and_types)});")
    # print(create_statement)
    con.execute(create_statement)
    con.close()
    for table in tables:
        con = duckdb.connect(metadata_db_path)
        summary: str = table_summaries.get(table, "")
        topics: list[str] = table_topics.get(table, [])
        keywords: list[str] = table_keywords.get(table, [])
        topic_string = ','.join(topics)
        keyword_string = ','.join(keywords)
        column_names: str = ','.join(config.db_metadata_column_names)
        if summary == "" or topics == [] or keywords == []:
            print(f"Metadata for table {table} not saved as creation of metadata not complete.")
            successful_write = False
            continue
        query: str = (f"INSERT INTO {config.db_metadata_table_name} ({column_names}) "
                      f"VALUES ('{table}','{summary}','{topic_string}','{keyword_string}')")
        try:
            con.execute(query)
        except duckdb.Error as e:
            successful_write = False
            print(f"Metadata for table {table} not saved due to the following error:\n{e}")
        finally:
            con.close()
    return successful_write


# tested in workflow (uses LLM)
def get_query_info(input_query: str) -> Tuple[str, list[str], list[str]]:
    """
    Get some information from the query.

    :param str input_query: The query we want to get some more information on.
    :return: The summary of the query in natural language,
             the topics existent in the query and keywords describing the query.
    :rtype: Tuple[str, list[str], list[str]]
    """
    # Ask the LLM for all the info on the query
    # Prompts were ok when being tested with ChatGPT
    summary: str = llm_find_intent_sql(input_query)
    topics: list[str] = llm_find_topics_sql(input_query, 5)
    keywords: list[str] = llm_find_intent_keywords_sql(input_query, 5)
    return summary, topics, keywords


# tested with examples in workflow
def compare_tables_and_query(input_query: str, db_tables: dict, metadata_db_path: str) -> dict:
    """
    Get all the usable tables from the database that could be used to answer the given query.

    :param str input_query: The given query for rewriting
    :param dict db_tables: The tables in the database in the form {table:[column1 type1, column2 type2]}
    :param str metadata_db_path: The path to the metadata database where all info is saved
    :return: All tables that qualify for being used in the rewrite in the form {table:[column1 type1, column2 type2]}
    :rtype: dict
    """
    feasible_tables = dict()
    # Get the query infos
    query_summary, query_topics, query_keywords = get_query_info(input_query)
    # Get the tables infos (all tables should have metadata at this point)
    con = duckdb.connect(metadata_db_path)
    tables: list[str] = con.execute(f"SELECT * FROM {config.db_metadata_table_name}").fetchall()
    con.close()
    all_tables_in_db: list[str] = list(db_tables.keys())
    tables_found_via_summary: int = 0
    tables_found_via_topics: int = 0
    tables_found_via_keywords: int = 0
    tables_found_via_summary_list: list[str] = []
    tables_found_via_topics_list: list[str] = []
    tables_found_via_keywords_list: list[str] = []
    #Precalculate stuff
    summary_query_tensor = model_embedding(query_summary)
    query_topics_string: str = ' '.join(query_topics)
    query_keywords_string: str = ' '.join(query_keywords)
    query_topics_nlp: Doc = config.nlp_language_model(query_topics_string)
    query_keywords_nlp: Doc = config.nlp_language_model(query_keywords_string)
    for table in tables:
        table_name: str = table[0]
        table_summary: str = table[1]
        table_topics: str = table[2]
        table_keywords: str = table[3]
        try:
            all_tables_in_db.remove(table_name)
        except ValueError:
            print("Warning: Table that has metadata is not in the database.")
            continue
        # Compare the summaries
        summary_table_tensor = model_embedding(table_summary)
        summary_sim: float = (
            calculate_tensor_sim_via_sentence_transformers(summary_query_tensor, summary_table_tensor))
        if summary_sim > 0.5:
            feasible_tables[table_name] = db_tables.get(table_name)
            tables_found_via_summary += 1
            tables_found_via_summary_list.append(table_name)
            continue
        # Compare the topics
        table_topics_string: str = ' '.join(table_topics.split(","))
        topic_sim: float = (query_topics_nlp
                            .similarity(config.nlp_language_model(table_topics_string)))
        if topic_sim > 0.8:
            feasible_tables[table_name] = db_tables.get(table_name)
            tables_found_via_topics += 1
            tables_found_via_topics_list.append(table_name)
            continue
        # Compare the keywords
        table_keywords_string: str = ' '.join(table_keywords.split(","))
        keyword_sim: float = (query_keywords_nlp
                              .similarity(config.nlp_language_model(table_keywords_string)))
        if keyword_sim > 0.8:
            feasible_tables[table_name] = db_tables.get(table_name)
            tables_found_via_keywords += 1
            tables_found_via_keywords_list.append(table_name)
            continue
    if len(all_tables_in_db) > 0:
        print(f"Warning: {len(all_tables_in_db)} tables were not considered in the filtering due to missing metadata.")
    if len(feasible_tables) > 0:
        print(f"The following {len(feasible_tables)} tables were found in the database:")
        print(f"{', '.join(feasible_tables.keys())}")
        print(f"{tables_found_via_summary} tables were found via the summary, {tables_found_via_topics} were found via "
              f"the topics, {tables_found_via_keywords} were found via the keywords.")
        print(f"Summary: {', '.join(tables_found_via_summary_list)}\n"
              f"Topics: {', '.join(tables_found_via_topics_list)}\n"
              f"Keywords: {', '.join(tables_found_via_keywords_list)}")
    else:
        print("No tables were found in the database.")
    return feasible_tables


def calculate_metadata_db_path(test: bool) -> str:
    """
    Calculate the path to the metadata database where all info is saved.

    :param bool test: Indicates whether tests are currently run.
    :return: The path to the metadata database
    :rtype: str
    """
    if test:
        metadata_db_path: str = config.test_db_file
    else:
        metadata_db_path: str = config.db_file
    # add LLM model to path and metadata flag
    metadata_db_path = (metadata_db_path[:metadata_db_path.rindex(".db")] + config.db_metadata_addition
                        + '_' + config.gpt_model.replace("-", "_") + '.db')
    print(f"Found the following metadata path: {metadata_db_path}")
    return metadata_db_path
