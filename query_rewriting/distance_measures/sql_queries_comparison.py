"""
Similarity/Comparison functions for two SQL queries.
All functions give a similarity score between 0 and 1.
"""
from difflib import SequenceMatcher
from itertools import islice

from spacy.tokens import Doc
# from openai.types.chat import ChatCompletion
from collections.abc import Iterable

from torch import Tensor

from query_rewriting.config import RankingNotPossible
from query_rewriting.distance_measures.vector_embedding import precalculate_docs_for_spacy_similarity, \
    model_embedding, calculate_tensor_sim_via_sentence_transformers
from query_rewriting.rewrite_generator.prefilter_tables_llm import tokenize
from query_rewriting.utilities.find_metadata import llm_find_intent_sql_list, llm_find_intent_sql
from query_rewriting.utilities.gpt_functions import gpt_api_call, strip_code_block_output, \
    strip_starting_and_ending_characters
from query_rewriting.utilities.sql_parsing import extract_tables, extract_columns, extract_columns_with_place
import query_rewriting.config as config
from query_rewriting.utilities.statistics import add_tokens

prompt_tokens_query_comparison: int = 0
completion_tokens_query_comparison: int = 0
total_tokens_query_comparison: int = 0


def similarity_using_clauses(input_query1: str, input_query2: str) -> float:
    """
    Calculates the similarity of two SQL queries via comparing the single clauses.
    (Similar ideas in:
      - 'Similarity Metrics for SQL Query Clustering'
      - 'Search-by-example over SQL repositories using structural and intent-driven similarity')
    Intent is not really captured with this comparison, as it just calculates Jaccard values and combines them.

    :param str input_query1: The first SQL query
    :param str input_query2: The second SQL query
    :return: The similarity of the two queries (as a float)
    :rtype: float
    """
    tables1: list[str] = extract_tables(input_query1)
    tables2: list[str] = extract_tables(input_query2)
    columns1: list[str] = extract_columns(input_query1)
    columns2: list[str] = extract_columns(input_query2)
    columns_dict1: dict[str, list[str]] = extract_columns_with_place(input_query1)
    columns_dict2: dict[str, list[str]] = extract_columns_with_place(input_query2)
    if columns_dict1 is None:
        columns_dict1 = dict()
    if columns_dict2 is None:
        columns_dict2 = dict()
    tables1_set = set(tables1)
    tables2_set = set(tables2)
    columns1_set = set(columns1)
    columns2_set = set(columns2)
    union1: int = len(tables1_set.union(tables2_set))
    union2: int = len(columns1_set.union(columns2_set))
    table_jaccard: float = (len(tables1_set.intersection(tables2_set)) / union1) if union1 > 0 else 0
    column_jaccard: float = (len(columns1_set.intersection(columns2_set)) / union2) if union2 > 0 else 0
    temp_sum: float = 0
    temp_parts: float = 0
    copy: dict[str, list[str]] = columns_dict1.copy()
    for part in copy:
        part_cols1: set[str] = set(columns_dict1.pop(part, []))
        part_cols2: set[str] = set(columns_dict2.pop(part, []))
        union_len: int = len(part_cols1.union(part_cols2))
        temp_sum += (len(part_cols1.intersection(part_cols2)) / union_len) if union_len > 0 else 0
        temp_parts += 1
    for part in columns_dict2:
        # Only contains columns not in columns_dict1 at this point
        # columns_dict2.pop(part, [])
        temp_parts += 1
    column_parts_jaccard_mean: float = temp_sum / temp_parts if temp_parts > 0 else 0
    return (table_jaccard + column_jaccard + column_parts_jaccard_mean) / 3


def similarity_using_tables(input_query1: str, input_query2: str) -> float:
    """
    Calculates the similarity of two SQL queries via comparing the tables (also using synonyms).
    This similarity partly includes intent due to embeddings.

    :param str input_query1: The first SQL query
    :param str input_query2: The second SQL query
    :return: The similarity of the two queries (as a float)
    :rtype: float
    """
    tables_q1: list[str] = extract_tables(input_query1)
    tables_q2: list[str] = extract_tables(input_query2)
    tokenized_tables1: list[str] = [' '.join(t) for t in tokenize(tables_q1)]
    tokenized_tables2: list[str] = [' '.join(t) for t in tokenize(tables_q2)]
    docs1: list[Doc] = precalculate_docs_for_spacy_similarity(tokenized_tables1, config.nlp_language_model)
    docs2: list[Doc] = precalculate_docs_for_spacy_similarity(tokenized_tables2, config.nlp_language_model)
    similarity_sum: float = 0
    num_comparisons: int = 0
    # print(docs1)
    # print(docs2)
    for i, doc1 in enumerate(docs1):
        temp_sum: float = 0
        temp_comparisons: float = 0
        if tokenized_tables1[i] in tokenized_tables2:
            temp_sum += 1
            temp_comparisons += 1
            # continue
        else:
            for doc2 in docs2:
                temp_sum += doc1.similarity(doc2)
                temp_comparisons += 1
        similarity_sum += temp_sum / temp_comparisons
        num_comparisons += 1
    return similarity_sum / num_comparisons



def difflib_simple_comparison(input_query1: str, input_query2: str) -> float:
    """
    Calculates the similarity of two SQL queries via the inbuilt library difflib.
    The return value is in the range [0,1] and can be dependent on the input order.
    This is a comparison based on simple string similarity.

    :param str input_query1: The first SQL query
    :param str input_query2: The second SQL query
    :return: The similarity of the two queries (as a float between 0 and 1)
    :rtype: float
    """
    return SequenceMatcher(None, input_query1, input_query2).ratio()


# Not tested in test_query_rewriting.py (uses LLM)
# prompts tested
def llm_intent_similarity_measure(input_query: str, alternative_queries: list[str]) -> list[float]:
    """
    Calculate the similarity of a SQl query to a bunch of alternative queries using an LLM.
    The LLM is asked to give similarity values between 0 and 1 based on the intent of the queries.

    :param str input_query: The query to compare to all others
    :param list[str] alternative_queries: The queries of which we want to know similarities to the original query
    :return: The similarities (based on intent) of all the alternative queries to the given query
             (in the same order as the input list)
    :rtype: list[float]
    """
    # Prepare prompt for LLM
    prompt_tokens_query_comparison_t = 0
    completion_tokens_query_comparison_t = 0
    total_tokens_query_comparison_t = 0
    # Assumption: All queries fit into one prompt -> no
    alt_queries_formatted: list[str] = [q.strip() if q.strip().endswith(";")
                                        else q.strip() + ";" for q in alternative_queries]
    amount_of_queries_per_run: int = 10
    queries_split_for_request: list[list[str]] = \
        [list(islice(alt_queries_formatted, i, i + amount_of_queries_per_run)) for i in
         range(0, len(alt_queries_formatted), amount_of_queries_per_run)]
    all_results_together: list[float] = []
    for alt_queries in queries_split_for_request:
        alt_queries_prompt_str: str = "\n".join(alt_queries)
        content_for_gpt: str = (f"I will give you a single SQL query called original query.\n"
                                f"I will also give you multiple other SQL queries called alternative queries.\n"
                                f"For each of the alternative queries, I want to calculate its similarity to the "
                                f"original query.\n"
                                f"These similarities should be floating point values between 0 and 1, and you should use "
                                f"the following guidelines to appoint them:\n"
                                f"The similarity between two queries is solely based on their intent.\n"
                                f"If the expected result of the queries gives the same insight to a human being, "
                                f"their similarity should be 1.\n"
                                f"This can include results that are correlated (e.g. rent and crime rate in a city) or "
                                f"that are virtually the same.\n"
                                f"If the given insight is similar, but not the same, the value should be a "
                                f"floating point number between 0 and 1, depending on how high the similarity is.\n"
                                f"If the query intents do not have anything to do with each other, "
                                f"the similarity should be 0.\n"
                                f"For each given alternative query only return the assigned similarity value "
                                f"between 0 and 1.\n"
                                f"Separate the similarity values using semicolons.\n"
                                f"Here is the original query:\n"
                                f"{input_query}\n"
                                f"Here are the alternative queries:\n"
                                f"{alt_queries_prompt_str}\n"
                                f"Give me only the similarity values separated with semicolons.")
        message: Iterable = [
            {"role": "system", "content": "We will work with databases and queries in SQL."},
            {
                "role": "user",
                "content": content_for_gpt
            }
        ]
        # Get the values from the prompt
        completion, llm_used = gpt_api_call(config.gpt_model, message)
        similarity_values_str: str = completion["choices"][0]["message"]["content"]
        similarity_values_str = strip_starting_and_ending_characters(strip_code_block_output(similarity_values_str))
        similarity_values: list[float] = [float(num.strip()) for num in
                                          similarity_values_str.split(";") if num.strip() != '']
        if llm_used:
            prompt_tokens_query_comparison_t += completion["usage"]["prompt_tokens"]
            completion_tokens_query_comparison_t += completion["usage"]["completion_tokens"]
            total_tokens_query_comparison_t += completion["usage"]["total_tokens"]
        else:
            print("No tokens used for similarity measure calculation due to reproducibility DB.")
        amount_added = 0
        if len(similarity_values) > len(alt_queries):
            print(similarity_values_str)
            raise RankingNotPossible("The LLM gave too many similarity values for all alternative queries.")
        while len(similarity_values) < len(alt_queries):
            amount_added += 1
            similarity_values.append(0)
        if amount_added > 0:
            print(f"Added {amount_added} zeros to similarities.")
        all_results_together.extend(similarity_values)
    global prompt_tokens_query_comparison, completion_tokens_query_comparison, total_tokens_query_comparison
    prompt_tokens_query_comparison = prompt_tokens_query_comparison_t
    completion_tokens_query_comparison = completion_tokens_query_comparison_t
    total_tokens_query_comparison = total_tokens_query_comparison_t
    print(f"Tokens used for the similarity measure calculation: \n"
          f"\tPrompt Tokens: {prompt_tokens_query_comparison}, "
          f"Completion Tokens: {completion_tokens_query_comparison}, "
          f"Total Tokens: {total_tokens_query_comparison}\n")
    add_tokens(prompt_tokens_query_comparison, completion_tokens_query_comparison,
               total_tokens_query_comparison)
    if len(all_results_together) != len(alternative_queries):
        raise RankingNotPossible("The LLM did not give a similarity value for all alternative queries.")
    else:
        return all_results_together


# Not tested in test_query_rewriting.py (uses LLM)
# prompts tested
def llm_intent_and_embedding_similarity(input_query: str, alternative_queries: list[str]) -> list[float]:
    """
    Calculate the similarity of a SQl query to a bunch of alternative queries using an LLM.
    The LLM is asked for intents of the queries, which are then compared using embeddings.

    :param str input_query: The query to compare to all others
    :param list[str] alternative_queries: The queries of which we want to know similarities to the original query
    :return: The similarities (based on intent) of all the alternative queries to the given query
             (in the same order as the input list)
    :rtype: list[float]
    """
    intents: list[str] = llm_find_intent_sql_list(alternative_queries)
    original_intent: str = llm_find_intent_sql(input_query)
    if len(intents) != len(alternative_queries):
        raise RankingNotPossible("The LLM did not give an intent for all alternative queries.")
    resulting_similarity_values: list[float] = []
    original_query_vector: Tensor = model_embedding(original_intent)
    alt_queries_vectors: list[Tensor] = [model_embedding(i) for i in intents]
    for alt_query_vector in alt_queries_vectors:
        resulting_similarity_values.append(
            calculate_tensor_sim_via_sentence_transformers(original_query_vector, alt_query_vector))
    return resulting_similarity_values

