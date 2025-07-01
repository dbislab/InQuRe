"""
Main execution file for query rewriting
"""
import argparse
import os
import time
from argparse import Namespace

# import sys
import config as config
from query_rewriting.config import NotYetSupportedException, NoRewritesFoundException, RewritingNotPossible, \
    RankingNotPossible
from query_rewriting.distance_measures.vector_embedding import set_up_model
from query_rewriting.input_processor.check_executability import check_query_execution
from query_rewriting.postprocessor.query_correction import query_correction_and_execution
from query_rewriting.ranking.rank_alternatives import rank_alternative_queries
from query_rewriting.rewrite_generator.generate_rewrites import rewrite_query
from query_rewriting.utilities.duckdb_functions import check_existence_of_tables
from query_rewriting.utilities.reproducibility_functions import create_reproducibility_database
from query_rewriting.utilities.statistics import get_rewriting_time, get_prefilter_2_statistics, get_tokens, \
    get_num_pruned_queries


natural_language_string = 'Natural Language'
sql_string = 'SQL'


def main():
    """
    The method executed in the command line. It reads the arguments from the execution and then starts the workflow.
    """
    # input_requests: list = []
    '''if len(sys.argv)>1 :
        # a path to the input was given with the program call
        input_requests: list = read_file(sys.argv[1])
    else:
        # fallback to the default input path
        input_requests: list = read_file(config.file_input_string)
    if len(sys.argv)>2:
        # both a path to the input and the db file was provided
        config.db_file = sys.argv[2]
    print(input_requests)'''
    # Add the possible arguments for the command line (default type of argument is string)
    # Parameter for test file is in test main
    parser = argparse.ArgumentParser(prog='Query Rewriting',
                                     description='Read requests from the input and execute them or a rewrite')
    parser.add_argument('-i', '--input-file', default=config.file_input_string, help='Absolute path to the input file')
    parser.add_argument('-d', '--database-file', default=config.db_file, help='Absolute path to the DuckDB file')
    parser.add_argument('-m', '--gpt-model', default=config.gpt_model,
                        choices=['gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini'], help='API GPT model to use')
    parser.add_argument('-n', '--alternative-queries', default=config.num_alternatives, type=int,
                        help='Number of alternative queries produced per input query')
    parser.add_argument('-k', '--result-quantity', default=config.num_results, type=int,
                        help='Number of alternative queries returned per input query')
    parser.add_argument('-re', '--rewrite-kind', default=config.rewrite_kind, type=int,
                        help='Kind of rewriter used (1 for simple rewriter)')
    parser.add_argument('-ra', '--ranker-kind', default=config.ranker_kind, type=int,
                        help='Kind of ranker used (1 for simple ranker)')
    parser.add_argument('-p', '--prefilter-kind', default=config.prefilter_kind, type=int,
                        help='Kind of prefilter used to select usable tables (1 for simple prefilter)')
    parser.add_argument('-dbp', '--database-prefix', default=config.db_prefixes, type=bool,
                        help='True if the database has table name prefixes for tables from the same schema')
    parser.add_argument('-ras', '--ranker-kind-string-sim', default=config.sim_measure_string, type=int,
                        help='Kind of string similarity function used for the ranker')
    parser.add_argument('-rai', '--ranker-kind-intent-sim', default=config.sim_measure_intent, type=int,
                        help='Kind of intent similarity function used for the ranker')
    parser.add_argument('-rep', '--reproducibility', default=config.reproducibility, type=bool,
                        help='True if we want to use saved LLM responses from a DB')
    # Parse the arguments from the command line
    # (arguments names use first long flag without the initial "--", replacing "-" in the name with "_")
    args: Namespace = parser.parse_args()
    config.file_input_string = args.input_file
    config.db_file = args.database_file
    config.gpt_model = args.gpt_model
    config.num_alternatives = args.alternative_queries
    config.num_results = args.result_quantity
    config.rewrite_kind = args.rewrite_kind
    config.ranker_kind = args.ranker_kind
    config.prefilter_kind = args.prefilter_kind
    config.db_prefixes = args.database_prefix
    config.sim_measure_string = args.ranker_kind_string_sim
    config.sim_measure_intent = args.ranker_kind_intent_sim
    config.reproducibility = args.reproducibility
    # Check the used parameters:
    print("Executing the query rewriting:  ")
    print(f"The used parameters are:  ")
    print(f"Input file: {config.file_input_string}  ")
    print(f"DB file: {config.db_file}  ")
    print(f"GPT model: {config.gpt_model}  ")
    print(f"Number of alternatives: {config.num_alternatives}, Number of results: {config.num_results}  ")
    print(
        f"Prefilter kind: {config.prefilter_kind}, Rewrite kind: {config.rewrite_kind}, "
        f"Ranker kind: {config.ranker_kind} (String similarity function: {config.sim_measure_string}, "
        f"Intent similarity function: {config.sim_measure_intent})  ")
    if config.prefilter_kind == 4:
        if config.max_num_summaries == -1:
            print("No maximum number for the amount of table summaries per LLM request in the prefilter was set.  ")
        else:
            print(f"The maximum number of table summaries per LLM request in the prefilter "
                  f"was set to {config.max_num_summaries}.  ")
    if config.db_prefixes:
        print(f"Tables from the same schema in the database have the same prefix.  ")
    else:
        print(f"No prefixes exist for the tables in the database.  ")
    if config.reproducibility:
        print(f"Reproducibility: ON (DB used)\n")
    else:
        print(f"Reproducibility: OFF (LLM used)\n")
    # Check if the input files exist
    if not (os.path.isfile(config.file_input_string)):
        print("\nWarning: The specified input file does not exist.")  # would throw error in reading
        raise RewritingNotPossible("Input file not found")
    # Read the input file provided via the input file path
    input_requests: list[list[str]] = read_file(config.file_input_string)
    print("The input queries are:  ")
    print(*input_requests, sep="\n")
    # Check if we have queries, if not no execution is needed
    if len(input_requests) == 0:
        raise RewritingNotPossible("There are no queries in the input.\nNo rewrites can be produced.")
    # Check if the database exists
    if not (os.path.isfile(config.db_file)):
        # Results in maybe an empty DB:
        # An empty DB does not make sense for rewriting queries
        print("\nWarning: The specified DB file does not exist. It will be created on the first access.")
        raise RewritingNotPossible("Database file not found")
    # Check if there are tables in the database, if not the rewriting on existent tables does not make sense
    if not check_existence_of_tables(False):
        raise RewritingNotPossible("There are no tables in the database.\nNo executable rewrite can be produced.")
    # Check that the number of result queries is not bigger than the number of produced alternatives
    if config.num_alternatives < config.num_results:
        raise RewritingNotPossible(f"Cannot output more queries ({config.num_results}) "
                                   f"than the number of alternatives produced ({config.num_alternatives}).")
    # Set up of all needed elements
    set_up_model(config.sentence_embedder)
    # Set up the reproducibility DB
    if config.reproducibility:
        create_reproducibility_database(False)
    # Execute the workflow
    execute_query_rewriting(input_requests, config.num_alternatives, config.rewrite_kind,
                            config.ranker_kind, config.num_results, config.prefilter_kind)


def read_file(file_dir: str = config.file_input_string) -> list[list[str]]:
    """
    Read the input file including SQL and natural language requests.
    Every request is preceded by either a line containing the keyword 'SQL' or 'Natural Language'.
    Requests are separated by an empty line.

    :param str file_dir: Absolute path to the input file
    :return: A list of lists, where each inner list contains the type of request (SQL or Natural Language)
             and the request itself
    :rtype: list[list[str]]
    """
    print('Reading file: ' + fr'{file_dir}')
    requests = []
    # Open the input file with a reader
    with open(fr'{file_dir}', 'r', encoding='utf-8') as file:
        input_type: bool = True
        # Set the current request as a list with two empty elements
        current_request: list[str] = ['', '']
        for line in file:
            # Parse all lines in the file
            if input_type:
                # The last request is handled, now a new one needs to be parsed, including its type
                # Save the type (if given correctly) in the first element of the current request list
                if natural_language_string in line:
                    current_request[0] = natural_language_string
                    input_type = False
                elif sql_string in line:
                    current_request[0] = sql_string
                    input_type = False
                elif line.strip() == '':
                    # Line is just empty -> pass
                    pass
                else:
                    # No valid input type found
                    raise Exception('Need either a statement of Natural Language or SQL')
            else:
                # We parse the request lines
                if line.strip() != '':
                    # The request goes on (no empty line yet)
                    current_request[1] = current_request[1] + ' ' + line.strip()
                else:
                    # The request is finished (empty line), so save it to the already found requests
                    current_request[1] = current_request[1].strip()
                    requests.append(current_request)
                    # Reset the current request
                    current_request = ['', '']
                    input_type = True
        else:
            # We reached the end of the file
            if current_request[0] != '' and current_request[1] != '':
                # There is a parsed request that needs to be added to the current requests
                current_request[1] = current_request[1].strip()
                requests.append(current_request)
    return requests


def execute_query_rewriting(input_queries: list[list[str]], number_of_alternatives: int, rewrite_kind: int,
                            ranker_kind: int, number_of_results: int, prefilter_kind: int):
    """
    Executes the whole workflow of the system for all input queries.
    SQL and NL inputs are treated separately using their corresponding methods.

    :param list[list[str]] input_queries: The input queries from the user
    :param int number_of_alternatives: Number of alternative queries to produce
    :param bool rewrite_kind: Defines the rewrite method to be used (1 for simple rewrite)
    :param bool ranker_kind: Defines the ranker method to be used (1 for simple ranker)
    :param int number_of_results: Number of wanted result queries (top-k queries after the ranking)
    :param int prefilter_kind: Defines the table pre-filter method to be used (1 for simple filter)
    """
    # Count the number of rewrites that are corrected and returned (top-k)
    num_no_rewrites_found_queries: int = 0
    num_output_rewrites: float = 0
    num_corrections_total: int = 0
    list_num_corrections_total: list[int] = []
    list_num_non_correctable_queries: list[int] = []
    # Count the number of queries that needed rewrites
    num_queries_sql: int = 0
    num_queries_needing_rewrite_sql: int = 0
    # Time executions
    time_list_sql_check_execution: list[float] = []
    time_list_sql_rewrite: list[float] = []
    time_list_sql_rank: list[float] = []
    time_list_sql_correction: list[float] = []
    # start_time_sql = time.time()
    # end_time_sql = time.time()
    # Rewriting for all found input queries
    for request_tuple in input_queries:
        print(f"\n\nCurrently processing:\n{request_tuple[1]}\nQuery type: {request_tuple[0]}")
        print("")
        if request_tuple[0] == natural_language_string:
            # The next request is in natural language
            # NL not implemented yet
            raise NotYetSupportedException(
                f"Natural Language Input not yet supported, request '{request_tuple[1]}' will be skipped.")
        elif request_tuple[0] == sql_string:
            # The next request is in SQL
            num_queries_sql += 1
            query: str = request_tuple[1]
            # Check if the query is executable
            start_time_sql = time.time()
            query_execution_possible, possible_result = check_query_execution(query, False)
            end_time_sql = time.time()
            time_list_sql_check_execution.append(end_time_sql - start_time_sql)
            if query_execution_possible:
                # Execute the query (possibly altered with other tables)
                print("Query was executed with the following result:")
                print(*possible_result, sep='\n')
            else:
                # Rewrites are needed
                num_queries_needing_rewrite_sql += 1
                start_time = time.time()
                try:
                    alternative_queries, proposed_tables = rewrite_query(query, number_of_alternatives, rewrite_kind,
                                                                         prefilter_kind)
                    end_time = time.time()
                    time_list_sql_rewrite.append(end_time - start_time)
                except NoRewritesFoundException as e:
                    # No rewrites were found (either no tables are usable or the intent cannot be kept using the tables)
                    # Skip this query
                    print(f"No rewrites found for the query:\n"
                          f"'{query}'\n"
                          f"The error was:\n"
                          f"{e}\n"
                          f"Continuing with the next query.")
                    num_no_rewrites_found_queries += 1
                    end_time = time.time()
                    time_list_sql_rewrite.append(end_time - start_time)
                    continue
                # Rewrites were found
                print(f"Alternative queries ({len(alternative_queries)}):")
                print(*alternative_queries, sep="\n")
                print("")
                # Rank the rewrites
                start_time = time.time()
                try:
                    ranked_alternative_queries: list[str] = rank_alternative_queries(query, alternative_queries,
                                                                                     ranker_kind, number_of_results)
                    end_time = time.time()
                    time_list_sql_rank.append(end_time - start_time)
                except RankingNotPossible as e:
                    # No ranking was found (the ranking did not work correctly)
                    # Skip this query
                    print(f"No ranking found for the query:\n"
                          f"'{query}'\n"
                          f"The error was:\n"
                          f"{e}\n"
                          f"Continuing with the next query.")
                    num_no_rewrites_found_queries += 1
                    end_time = time.time()
                    time_list_sql_rank.append(end_time - start_time)
                    continue
                # Correct the top-k rewrites and annotate uncorrected ones
                start_time = time.time()
                corrected_queries, error_messages, query_results, num_corrections_one_query = (
                    query_correction_and_execution(ranked_alternative_queries, proposed_tables))
                end_time = time.time()
                time_list_sql_correction.append(end_time - start_time)
                num_output_rewrites += number_of_results
                num_corrections_total += num_corrections_one_query
                list_num_corrections_total.append(num_corrections_one_query)
                num_non_correctable_queries: int = 0
                # Output the corrected, reranked alternative queries and the result (if possible)
                print(f"Produced top-{number_of_results} rewrites:")
                for rank, alternative_query in enumerate(corrected_queries):
                    if rank < number_of_results:
                        print(f"{rank + 1}. {alternative_query}")
                        if error_messages[rank] == "":
                            print("\tResults:")
                            print("\t\t", end="")
                            print(*query_results[rank], sep='\n\t\t')
                        else:
                            print("\tQuery was not executable after correction. The following error occurred:")
                            print(f"\t\t{error_messages[rank]}")
                            num_non_correctable_queries += 1
                    else:
                        list_num_non_correctable_queries.append(num_non_correctable_queries)
                        break
                list_num_non_correctable_queries.append(num_non_correctable_queries)
        else:
            # Input type of the request is not supported by the system (neither SQL nor NL)
            raise Exception(f"Unsupported Input Type: {request_tuple[0]} for request '{request_tuple[1]}'")
    # Print a bunch of statistics
    print(f"\n\nStatistics:")
    print(f"\nNumber of queries in input: {len(input_queries)}  ")
    print(f"Number of queries in SQL: {num_queries_sql}  ")
    print(f"Number of queries in SQL that needed rewrites: {num_queries_needing_rewrite_sql}  ")
    print(f"Number of queries where no rewrite was found: {num_no_rewrites_found_queries}  ")
    if config.prefilter_kind == 2:
        print(f"For prefilter kind 2 the following number of tables were suggested by the LLM (sorted by kinds):  ")
        num1, num2, num3, num4 = get_prefilter_2_statistics()
        print(f"Total: {num1}, Correctly: {num2}, Slightly off: {num3}, Not in the DB: {num4}  ")
    print(f"Number of queries that were output as a top-k rewrite: {num_output_rewrites}  ")
    list_pruned_queries: list[int] = get_num_pruned_queries()
    print(f"Number of alternative queries pruned per query: {list_pruned_queries}")
    print(f"Number of correction iterations: {num_corrections_total}  ")  # Queries needing correction
    print(f"Broken down per query: {list_num_corrections_total}  ")
    if num_output_rewrites > 0:
        percentage_of_rewrites: float = (num_corrections_total / num_output_rewrites) * 100
        print(f"Percentage of queries that needed correction: {percentage_of_rewrites} %")
    print(f"Alternative queries per query that could not be corrected: {list_num_non_correctable_queries}  ")
    if len(time_list_sql_check_execution) > 0:
        print(f"\nThe following times were recorded (in seconds):  ")
        print(f"Average time for checking query execution: "
              f"{sum(time_list_sql_check_execution) / len(time_list_sql_check_execution)}  ")
        print(f"Exact times: {time_list_sql_check_execution}  ")
    if len(time_list_sql_rewrite) > 0:
        print(f"Average time for producing rewrites (including table filtering): "
              f"{sum(time_list_sql_rewrite) / len(time_list_sql_rewrite)}  ")
        print(f"Exact times: {time_list_sql_rewrite}  ")
    filter_time, rewrite_time = get_rewriting_time()
    if len(filter_time) > 0:
        print(f"\tFrom those:  ")
        print(f"\tAverage time for table filtering: {sum(filter_time) / len(filter_time)}  ")
        print(f"\tExact times: {filter_time}  ")
        if config.prefilter_kind == 4:
            print(f"\t\tThe first filter time (possibly including metadata creation) was: {filter_time[0]}  ")
    if len(rewrite_time) > 0:
        print(f"\tAverage time for rewrites: {sum(rewrite_time) / len(rewrite_time)}  ")
        print(f"\tExact times: {rewrite_time}  ")
    if len(time_list_sql_rank) > 0:
        print(f"Average time for ranking the rewrites: {sum(time_list_sql_rank) / len(time_list_sql_rank)}  ")
        print(f"Exact times: {time_list_sql_rank}  ")
    if len(time_list_sql_correction) > 0:
        print(f"Average time for correcting queries: {sum(time_list_sql_correction) / len(time_list_sql_correction)}")
        print(f"Exact times: {time_list_sql_correction}  ")
    print(f"\nTokens for the LLM:  ")
    prom, comp, tot = get_tokens()
    print(f"Prompt tokens: {prom}  ")
    print(f"Completion tokens: {comp}  ")
    print(f"Total tokens: {tot}")


if __name__ == "__main__":
    """
    Main method to execute the intent-keeping query rewriter.
    """
    main()
