"""
All the prefilter method which directly rely on prompting the LLM for the tables.
"""
from difflib import SequenceMatcher
from itertools import islice
import re
import math

# from openai.types.chat import ChatCompletion
from collections.abc import Iterable
from nltk.stem import PorterStemmer
from spacy.tokens import Doc

from query_rewriting.distance_measures.vector_embedding import precalculate_docs_for_spacy_similarity
from query_rewriting.utilities.gpt_functions import gpt_api_call, strip_whitespaces, \
    strip_starting_and_ending_characters
import query_rewriting.config as config
from query_rewriting.utilities.sql_parsing import extract_tables
from query_rewriting.utilities.statistics import change_prefilter_2_statistics, add_tokens

prompt_tokens_prefilter: int = 0
completion_tokens_prefilter: int = 0
total_tokens_prefilter: int = 0


# not tested in test_query_rewriting.py, as it makes LLM call; only called in workflow
def simple_prefilter_via_llm(input_query: str, db_tables: dict) -> dict:
    """
    Prefilter the tables via finding tables that are similar to the ones in the query.
    Similar tables are chosen by the LLM from the ones in the database.

    :param str input_query: The query that is used to prefilter the tables of the database
    :param dict db_tables: The tables in the database in the form {table:[column1 type1, column2 type2]}
    :return: A dictionary containing the usable tables in the form {table:[column1 type1, column2 type2]}
    :rtype: dict
    """
    # return object
    found_tables: dict = dict()
    tables_sliced_for_prompts: list[list[str]] = get_tables_in_slices_for_llm_call(db_tables, 0.1)
    # Count tokens
    local_prompt_tokens: int = 0
    local_completion_tokens: int = 0
    local_total_tokens: int = 0
    total_llm_use: bool = False
    # Count if LLM gives existent tables back
    correct_tables_returned: int = 0
    slightly_wrong_tables_returned: int = 0
    wrong_tables_returned: int = 0
    for table_slice in tables_sliced_for_prompts:
        # Prompt for GPT
        # LLM sometimes hallucinates that original table from query is in DB -> postprocessing is enough to catch this
        content_for_gpt_1: str = \
            (f"I want you to decide which from the given tables are useful to answer a given query.\n"
             f"It is important that the chosen tables can help to answer the query either directly or indirectly.\n"
             f"If a table cannot be used directly, think about if it can be used due to it being correlated "
             f"regarding human intuition.\n"
             f"E.g. tables on park locations and crime rates could help to know areas with high rent.\n"
             f"So in conclusion, the tables should be usable to answer the query "
             f"such that the information gain stays the same for a human.\n"
             f"So synonyms, hyponyms, correlations and similar topics should be important "
             f"when choosing the usable tables.\n")
        content_for_gpt_2: str = \
            (f"I have the following SQL query:\n"
             f"{input_query}\n"
             f"Here are the tables from my database (just the names):\n"
             f"{', '.join(table_slice)}\n"
             f"If none of these tables are usable, then only respond with 'No tables usable'.\n"
             f"If some tables are usable, only respond with the names, separated by semicolons.")
        # Prompt part to take prefixes of databases into account
        possible_content_for_table_prefixes: str = \
            (f"When choosing tables, please also take into account that tables from the same "
             f"schema are prefixed with the same string (the name of the schema).\n"
             f"Such tables could be joined and used together to answer the query.\n")
        if config.db_prefixes:
            # We want the part mentioning the prefixes
            content_for_gpt = f"{content_for_gpt_1}{possible_content_for_table_prefixes}{content_for_gpt_2}"
        else:
            content_for_gpt = f"{content_for_gpt_1}{content_for_gpt_2}"
        message: Iterable = [
            {"role": "system", "content": "We will work with databases and queries in SQL."},
            {
                "role": "user",
                "content": content_for_gpt
            }
        ]
        # print(content_for_gpt)
        # print(f"GPT message content:\n{content_for_gpt}")
        completion, llm_used = gpt_api_call(config.gpt_model, message)
        possible_tables: str = completion["choices"][0]["message"]["content"]
        if re.search('No tables usable', strip_whitespaces(possible_tables), flags=re.IGNORECASE) is not None:
            # No tables of this batch are usable
            print(f"No tables were deemed usable in this table batch by the LLM.")
        else:
            # There are usable tables
            prepared_possible_tables: str = strip_starting_and_ending_characters(strip_whitespaces(possible_tables))
            # Separate the single tables and check if they are really in the DB -> avoid hallucination
            for table in prepared_possible_tables.split(";"):
                # Find the columns for the tables in the db_tables and add them as return
                searched_table: str = table.strip()
                if searched_table == '':
                    continue
                columns: list[str] = db_tables.get(searched_table, [])
                if len(columns) > 0:
                    found_tables[searched_table] = columns
                    correct_tables_returned += 1
                else:
                    # Search for element with the smallest distance (.ratio returns the similarity)
                    min_dist_table: str = max(db_tables.keys(),
                                              key=lambda key: SequenceMatcher(None, table, key).ratio())
                    if SequenceMatcher(None, table, min_dist_table).ratio() > 0.85:
                        # Distance of found item small (similarity is high)
                        # count+1 on small error and take table (real found name, not from LLM)
                        columns: list[str] = db_tables.get(min_dist_table, [])
                        found_tables[min_dist_table] = columns
                        slightly_wrong_tables_returned += 1
                    else:
                        # No table with small distance found
                        # count error and do not add anything to result
                        wrong_tables_returned += 1
            print(f"The following tables were suggested by the LLM:")
            print(f"{prepared_possible_tables}")
        # Add the used tokens
        if llm_used:
            local_prompt_tokens += completion["usage"]["prompt_tokens"]
            local_completion_tokens += completion["usage"]["completion_tokens"]
            local_total_tokens += completion["usage"]["total_tokens"]
            total_llm_use = True
    if len(found_tables) > 0:
        print(f"The following {len(found_tables)} tables were found in the database:")
        print(f"{', '.join(found_tables.keys())}")
    else:
        print("No tables were found in the database.")
    # Write statistics from the execution
    print(f"Tables from the LLM:\nCorrect tables: {correct_tables_returned}, "
          f"Slight errors: {slightly_wrong_tables_returned}, Wrong tables: {wrong_tables_returned}")
    print(f"Total number of tables suggested by the LLM: "
          f"{correct_tables_returned + slightly_wrong_tables_returned + wrong_tables_returned}")
    total_suggested_tables: int = correct_tables_returned + slightly_wrong_tables_returned + wrong_tables_returned
    change_prefilter_2_statistics(total_suggested_tables, correct_tables_returned,
                                  slightly_wrong_tables_returned, wrong_tables_returned)
    if total_llm_use:
        global prompt_tokens_prefilter
        prompt_tokens_prefilter = local_prompt_tokens
        global completion_tokens_prefilter
        completion_tokens_prefilter = local_completion_tokens
        global total_tokens_prefilter
        total_tokens_prefilter = local_total_tokens
        print(
            f"Tokens used after the table filtering: \n\tPrompt Tokens: {prompt_tokens_prefilter}, "
            f"Completion Tokens: {completion_tokens_prefilter}, Total Tokens: {total_tokens_prefilter}\n")
        add_tokens(local_prompt_tokens, local_completion_tokens, local_total_tokens)
    else:
        print("No tokens used for table filtering due to reproducibility DB.")
    return found_tables


def get_tables_in_slices_for_llm_call(input_tables: dict, percentage_for_output: float) -> list[list[str]]:
    """
    Slice the table names into lists, such that each of the lists can be used in one LLM prompt.
    It is sliced such that the output is big enough to be able
    to give the specified percentage of input tables as an output.

    :param dict input_tables: The tables in the database in the form {table:[column1 type1, column2 type2]}
    :param float percentage_for_output: The percentage of tables that the LLM should be able to output in each call
           (e.g. 0.5 for 50%)
    :return: A list such that each element of the list contains the tables usable in one prompt
    :rtype: list[list[str]]
    """
    # Calculate how many tokens there are in average for a table name
    num_tables: int = len(input_tables)
    num_underlines: int = 0
    num_words: int = 0
    num_chars: int = 0
    # Assume that words in tables names are usually separated by underlines
    for table in input_tables:
        words: list[str] = table.split("_")
        num_underlines += len(words) - 1
        num_words += len(words)
        num_chars += sum(len(w) for w in words)
    # Every underline starts a new token:
    # Average is average number of underlines + 1 (avg number of words) times average tokens for average length of words
    # Number of tokens: references OpenAI: roughly 4 chars are one token
    # This calculates the average table length in tokens
    avg_table_length: float = ((num_underlines / num_tables) + 1) * ((num_chars / num_words) / 4)
    # get output length of models (in tokens)
    if config.gpt_model == "gpt-4o" or config.gpt_model == "gpt-4o-mini":
        output_length: int = config.output_length_gpt_4o
    else:
        output_length: int = config.output_length_gpt_o1
    # Assumption: maximum tables that are returned: percentage of output
    tables_per_request: int = math.floor((output_length / avg_table_length) * (1 / percentage_for_output))
    table_keys: list[str] = list(input_tables.keys())
    # Split the tables into slices
    tables_split_for_requests: list[list[str]] = \
        [list(islice(table_keys, i, i + tables_per_request)) for i in range(0, len(table_keys), tables_per_request)]
    return tables_split_for_requests


# not tested in test_query_rewriting.py, just in workflow
def complex_prefilter_via_llm(input_query: str, db_tables: dict) -> dict:
    """
    Prefilter the tables via asking LLM for usable tables.
    Then check the database for those tables and find best matches.

    :param str input_query: The query that is used to prefilter the tables of the database
    :param dict db_tables: The tables in the database in the form {table:[column1 type1, column2 type2]}
    :return: A dictionary containing the usable tables in the form {table:[column1 type1, column2 type2]}
    :rtype: dict
    """
    # Prompt for LLM
    content_for_gpt = (f"I want you to give me tables that are usable to answer the SQL query I give you.\n"
                       f"These tables should include synonyms and hyponyms of the tables in the query,"
                       f" but also tables that could have correlated values to those needed in the query.\n"
                       f"Such tables could maybe not directly answer the query, "
                       f"but keep the same information gain for a human.\n"
                       f"An example for this is that e.g. school and park locations influence housing prices.\n"
                       f"Other important tables could be ones that can be joined "
                       f"to produce the needed answer together.\n"
                       f"So incorporate all of these options in your answer "
                       f"and give me all the tables you can think of.\n" 
                       f"I have the following SQL query:\n"
                       f"{input_query}\n"
                       f"Only respond with possible tables separated with semicolons. "
                       f"If you think there exist no tables to answer the query only respond with 'No tables usable'.")
    message: Iterable = [
        {"role": "system", "content": "We will work with databases and queries in SQL."},
        {
            "role": "user",
            "content": content_for_gpt
        }
    ]
    # print(content_for_gpt)
    completion, llm_used = gpt_api_call(config.gpt_model, message)
    suggested_tables: str = completion["choices"][0]["message"]["content"]
    matching_tables: dict = dict()
    if re.search('No tables usable', strip_whitespaces(suggested_tables), flags=re.IGNORECASE) is not None:
        # No tables of this batch are usable
        print(f"No usable tables were found by the LLM.")
    else:
        # Tables are usable, get the single tables from the LLM response
        prepared_suggested_tables: list[str] = (
            strip_starting_and_ending_characters(strip_whitespaces(suggested_tables)).split(";"))
        prepared_suggested_tables = [table.strip() for table in prepared_suggested_tables if table.strip() != '']
        print(f"The following tables were suggested by the LLM:")
        print(f"{suggested_tables}")
        # Add the tables from the query (possible that similar ones are already in the database)
        tables_from_query: list[str] = extract_tables(input_query)
        prepared_suggested_tables = [*prepared_suggested_tables, *tables_from_query]
        print(f"The following tables from the query were added: ")
        print(f"{', '.join(tables_from_query)}")
        # Find similar tables
        matching_tables = find_similar_tables(prepared_suggested_tables, db_tables)
    if len(matching_tables) > 0:
        print(f"The following {len(matching_tables)} tables were found in the database:")
        print(f"{', '.join(matching_tables.keys())}")
    else:
        print("No tables were found in the database.")
    # Add tokens to count
    if llm_used:
        global prompt_tokens_prefilter
        prompt_tokens_prefilter = completion["usage"]["prompt_tokens"]
        global completion_tokens_prefilter
        completion_tokens_prefilter = completion["usage"]["completion_tokens"]
        global total_tokens_prefilter
        total_tokens_prefilter = completion["usage"]["total_tokens"]
        print(
            f"Tokens used after the table filtering: \n\tPrompt Tokens: {prompt_tokens_prefilter}, "
            f"Completion Tokens: {completion_tokens_prefilter}, Total Tokens: {total_tokens_prefilter}\n")
        add_tokens(completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"],
                   completion["usage"]["total_tokens"])
    else:
        print("No tokens used for table filtering due to reproducibility DB.")
    return matching_tables


def find_similar_tables(suggested_table_list: list[str], db_tables: dict) -> dict:
    """
    Find tables in the database that are similar to the suggested tables.
    Use different measures to define similarity.

    :param list[str] suggested_table_list: Tables that were suggested
    :param dict db_tables: The real tables in the database
    :return: The tables from the database that are the most similar to the suggested ones
    :rtype: dict
    """
    # Original tables with NLP calculation
    found_tables: dict = dict()
    suggested_tables_nlp: list[Doc] = (
        precalculate_docs_for_spacy_similarity(suggested_table_list, config.nlp_language_model))
    # Split the table names into words and calculate NLP calculation
    suggested_tables_tokenized_tokens: list[list[str]] = tokenize(suggested_table_list)
    suggested_tables_toc_and_stem: list[list[str]] = tokenize_and_stem(suggested_table_list)
    suggested_tables_tokenized: list[str] = [' '.join(t) for t in suggested_tables_tokenized_tokens]
    suggested_tables_tokenized_nlp: list[Doc] = (
        precalculate_docs_for_spacy_similarity(suggested_tables_tokenized, config.nlp_language_model))
    # Database table NLP calculation
    db_tables_as_list: list[str] = [x for x in db_tables]
    db_tables_nlp: list[Doc] = (
        precalculate_docs_for_spacy_similarity(db_tables_as_list, config.nlp_language_model))
    # Calculate stuff for the prefixed part
    db_tables_tokenized_tokens: list[list[str]] = tokenize(db_tables_as_list)
    db_tables_tok_and_stem: list[list[str]] = tokenize_and_stem(db_tables_as_list)
    db_tables_tokenized: list[str] = [' '.join(t) for t in db_tables_tokenized_tokens]
    db_tables_tokenized_nlp: list[Doc] = (
        precalculate_docs_for_spacy_similarity(db_tables_tokenized, config.nlp_language_model))
    for i, suggested_table in enumerate(suggested_table_list):
        for j, db_table in enumerate(db_tables):
            if config.db_prefixes:
                # Tokenize and stem both DB tables and suggested ones and then do set intersection >= 1
                if len(set(suggested_tables_toc_and_stem[i]).intersection(set(db_tables_tok_and_stem[j]))) > 0:
                    found_tables[db_table] = db_tables.get(db_table, [])
            else:
                # Improved via tokenization, by adding tables with whitespaces and then using embedding
                if suggested_tables_nlp[i].similarity(db_tables_nlp[j]) > 0.7:
                    found_tables[db_table] = db_tables.get(db_table, [])
                elif suggested_tables_tokenized_nlp[i].similarity(db_tables_nlp[j]) > 0.7:
                    found_tables[db_table] = db_tables.get(db_table, [])
                elif suggested_tables_nlp[i].similarity(db_tables_tokenized_nlp[j]) > 0.7:
                    found_tables[db_table] = db_tables.get(db_table, [])
                elif suggested_tables_tokenized_nlp[i].similarity(db_tables_tokenized_nlp[j]) > 0.7:
                    found_tables[db_table] = db_tables.get(db_table, [])
    return found_tables


def tokenize_and_stem(names: list[str]) -> list[list[str]]:
    """
    Tokenize and stem the words in the list.

    :param list[str] names: the words to process
    :return: tokenized and stemmed list for each word
    :rtype: list[list[str]]
    """
    result: list[list[str]] = []
    porter = PorterStemmer()
    tokenized_list: list[list[str]] = tokenize(names)
    for words in tokenized_list:
        name_list = [porter.stem(w) for w in words]
        result.append(name_list)
    return result


# does not need testing, done in tokenize_and_stem
def tokenize(names: list[str]) -> list[list[str]]:
    """
    Tokenize the words in the list.

    :param list[str] names: the words to process
    :return: tokenized list for each word
    :rtype: list[list[str]]
    """
    result: list[list[str]] = []
    expr = re.compile('([A-Z]*[a-z]+)([A-Z]?)')
    for name in names:
        if name.find("_") == -1:
            name = expr.sub(lambda x: (x.group(1) + '_' + x.group(2)), name).strip('_')
        name = name.lower()
        name_list: list[str] = name.split('_')
        result.append(name_list)
    return result
