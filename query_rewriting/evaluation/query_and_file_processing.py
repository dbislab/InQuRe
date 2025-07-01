"""
Methods to read the alternative queries from all runs and deduplicate them.
"""
import os
import random
# from sys import executable
from typing import Tuple
from string import whitespace

import duckdb

from query_rewriting.distance_measures.sql_queries_comparison import difflib_simple_comparison


# Nothing is tested in test files, just in some execution tests

def read_file_eval(path: str) -> list[str]:
    """
    Read a file and return all queries within the file.

    :param str path: Path to the file with the queries
    :return: The SQL queries as strings
    :rtype: list[str]
    """
    print('Reading file: ' + fr'{path}')
    queries: list[str] = []
    with open(fr'{path}', 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("-") or line.strip() == '':
                continue
            else:
                q: str = line.strip()
                if q.endswith(";"):
                    queries.append(q)
                else:
                    q_new: str = q + ";"
                    queries.append(q_new)
    return queries


def deduplicate_queries(queries: list[str]) -> list[str]:
    """
    Remove duplicates from the queries.

    :param list[str] queries: The given queries
    :return: The queries without duplicates
    :rtype: list[str]
    """
    return list(set(queries))


def execute_deduplication(path: str, query_num: int):
    """
    Execute the process of deduplication and print some sizes

    :param str path: The path to the file
    :param int query_num: Currently evaluated query
    """
    q: list[str] = read_file_eval(path)
    # print(f"Original:")
    # print(*q, sep="\n")
    q_set: list[str] = deduplicate_queries(q)
    # print(f"Deduplicated:")
    # print(*q_set, sep="\n")
    print(f"Q{query_num}")
    print(f"Size Original: {len(q)}")
    print(f"Size Deduplicated: {len(q_set)}")


def read_and_write_back_shuffled(path_input: str, path_output: str, query_num: int):
    """
    Read and then write back the deduplicated queries to a file for the user study

    :param str path_input: The path to the input file with the alternative queries
    :param int query_num: Currently evaluated query
    :param str path_output: The path to the output file with the deduplicated, shuffled queries
    """
    queries: list[str] = read_file_eval(path_input)
    print(f"Original Size: {len(queries)}")
    queries_new: list[str] = deduplicate_queries(queries)
    print(f"New Size: {len(queries_new)}")
    random.shuffle(queries_new)
    query_str: str = '\n'.join(queries_new)
    if query_num < 10:
        file_name: str = rf'{path_output}\Query_0{query_num}.txt'
    else:
        file_name: str = rf'{path_output}\Query_{query_num}.txt'
    with open(file_name, "w", encoding='utf-8') as file:
        file.write(query_str)


def read_user_study_file(path_input: str) -> dict[int, list[Tuple[str, int, int]]]:
    """
    Read all the files from all users and save for each original query the alternatives with their relevance.

    :param str path_input: The path to the folder with the results from the users
    :return: A dictionary with the query number as the key and a list of ratings as the values.
             Each list element is a tuple with the query, the user ID and the rating (0: irrelevant, 1: relevant).
    :rtype: dict[int, list[Tuple[str, int, int]]]
    """
    result: dict[int, list[Tuple[str, int, int]]] = dict()
    for root, dirs, files in os.walk(path_input):
        for name in files:
            file_path = os.path.join(root, name)
            file_name_parts: list[str] = name.split(".")[0].split("_")
            query_id: int = int(file_name_parts[1].removeprefix("0"))
            user_id: int = int(file_name_parts[3].removeprefix("0"))
            with open(fr'{file_path}', 'r', encoding='utf-8') as file:
                for line in file:
                    l: str = line.strip()
                    if l.startswith("x") or l.startswith("X"):
                        query: str = (",".join(l.split(",")[1:])).strip()
                        relevance_tuple: Tuple[str, int, int] = (query, user_id, 1)
                    elif l.startswith("#"):
                        continue
                    elif l == '':
                        continue
                    elif line.startswith(tuple(w for w in whitespace)):
                        query: str = l
                        relevance_tuple: Tuple[str, int, int] = (query, user_id, 1)
                    else:
                        relevance_tuple: Tuple[str, int, int] = (l, user_id, 0)
                    current_list: list[Tuple[str, int, int]] = result.get(query_id, [])
                    current_list.append(relevance_tuple)
                    result[query_id] = current_list
    return result


def write_user_study_results_to_db(path_to_db: str, user_results: dict[int, list[Tuple[str, int, int]]]):
    """
    Write the extracted results from the user study into the database

    :param str path_to_db: The path to the database
    :param dict[int, list[Tuple[str, int, int]]] user_results: The extracted results from the users
    """
    con = duckdb.connect(path_to_db)
    for query_id in user_results:
        results: list[tuple[str, int, int]] = user_results[query_id]
        for result in results:
            con.execute("INSERT INTO user_reviews VALUES (?,?,?,?)", [query_id, result[0], result[1], result[2]])
    con.close()


def calculate_majority_vote(path_to_db: str):
    """
    Fill the table containing the majority vote from the user study

    :param str path_to_db: The path to the database
    """
    con = duckdb.connect(path_to_db)
    statement1: str = ("INSERT INTO combined_reviews "
                       "(SELECT queryID, query, 1 FROM user_reviews "
                       "GROUP BY queryID, query HAVING sum(rating)/count(userID) > 0.5)")
    statement2: str = ("INSERT INTO combined_reviews "
                       "(SELECT queryID, query, 0 FROM user_reviews "
                       "GROUP BY queryID, query HAVING sum(rating)/count(userID) <= 0.5)")
    con.execute(statement1)
    con.execute(statement2)
    con.close()


def write_queries_from_approaches_to_db(path_to_db: str, path_to_user_study_db: str, file_paths: list[str],
                                        query_ids: list[int], pruned_path: str):
    """
    Read the queries from the different approaches and write them to a table in the DB

    :param str path_to_db: The path to the database where the queries can be executed
    :param str path_to_user_study_db: The path to the user study database
    :param list[str] file_paths: The paths to the files to read
    :param list[str] query_ids: The ids of the queries (one for each file)
    :param str pruned_path: The path to the file with pruned queries
    """
    # Read the pruned queries
    pruned_queries: list[str] = read_file_eval(pruned_path)
    # print(len(pruned_queries))
    # print(*pruned_queries, sep="\n")
    # Read the alternative queries
    con = duckdb.connect(path_to_db)
    con_users = duckdb.connect(path_to_user_study_db)
    for path, query_id in zip(file_paths, query_ids):
        approach: int = 0
        with open(fr'{path}', 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith("-"):
                    # Next approach found
                    approach += 1
                    continue
                elif line.strip() == '':
                    continue
                else:
                    q: str = line.strip()
                    if q.endswith(";"):
                        query: str = q
                    else:
                        q_new: str = q + ";"
                        query: str = q_new
                    # Check if query is pruned and if it is executable
                    pruned: int = 0
                    if query in pruned_queries:
                        pruned = 1
                    executability: int = 1
                    try:
                        con.execute(query)
                    except duckdb.Error:
                        executability = 0
                    # Insert query into DB
                    con_users.execute("INSERT INTO query_alternatives VALUES (?,?,?,?,?)",
                                      [query_id, query, approach, executability, pruned])
    con.close()
    con_users.close()


def precision_of_approaches(path_to_user_study_db: str):
    """
    Calculate the precision of each approach
    :param str path_to_user_study_db: The path to the user study database
    """
    con = duckdb.connect(path_to_user_study_db)
    # Request to get general precision ignoring pruning and executability
    request: str = ("SELECT q.queryID, q.approach, sum(c.rating)/count(c.rating) as precision, "
                    "sum(c.rating) AS tp, count(c.rating) as t "
                    "FROM combined_reviews c, query_alternatives q "
                    "WHERE c.queryID = q.queryID AND c.query = q.query "
                    "GROUP BY q.queryID, q.approach "
                    "ORDER BY q.approach, q.queryID")
    request2: str = ("SELECT q.approach, sum(c.rating)/count(c.rating) as precision, "
                     "sum(c.rating) AS tp, count(c.rating) as t "
                     "FROM combined_reviews c, query_alternatives q "
                     "WHERE c.queryID = q.queryID AND c.query = q.query "
                     "GROUP BY q.approach "
                     "ORDER BY q.approach")
    request2_2: str = ("SELECT q.queryID, sum(c.rating)/count(c.rating) as precision, "
                       "sum(c.rating) AS tp, count(c.rating) as t "
                       "FROM combined_reviews c, query_alternatives q "
                       "WHERE c.queryID = q.queryID AND c.query = q.query "
                       "GROUP BY q.queryID "
                       "ORDER BY q.queryID")
    # Requests not ignoring the pruning
    request3: str = ("SELECT q.queryID, q.approach, "
                     "(sum(c.rating) FILTER (q.pruned != 1))/count(c.rating) as precision, "
                     "(sum(c.rating) FILTER (q.pruned != 1)) as tp, count(c.rating) AS t "
                     "FROM combined_reviews c, query_alternatives q "
                     "WHERE c.queryID = q.queryID AND c.query = q.query "
                     "GROUP BY q.queryID, q.approach "
                     "ORDER BY q.approach, q.queryID")
    request4: str = ("SELECT q.approach, (sum(c.rating) FILTER (q.pruned != 1))/count(c.rating) as precision, "
                     "(sum(c.rating) FILTER (q.pruned != 1)) as tp, count(c.rating) AS t "
                     "FROM combined_reviews c, query_alternatives q "
                     "WHERE c.queryID = q.queryID AND c.query = q.query "
                     "GROUP BY q.approach "
                     "ORDER BY q.approach")
    request4_2: str = ("SELECT q.queryID, (sum(c.rating) FILTER (q.pruned != 1))/count(c.rating) as precision, "
                       "(sum(c.rating) FILTER (q.pruned != 1)) as tp, count(c.rating) AS t "
                       "FROM combined_reviews c, query_alternatives q "
                       "WHERE c.queryID = q.queryID AND c.query = q.query "
                       "GROUP BY q.queryID "
                       "ORDER BY q.queryID")
    # Requests not ignoring execution
    request5: str = ("SELECT q.queryID, q.approach, "
                     "(sum(c.rating) FILTER (q.executable != 0))/count(c.rating) as precision, "
                     "(sum(c.rating) FILTER (q.executable != 0)) AS tp, count(c.rating) AS t "
                     "FROM combined_reviews c, query_alternatives q "
                     "WHERE c.queryID = q.queryID AND c.query = q.query "
                     "GROUP BY q.queryID, q.approach "
                     "ORDER BY q.approach, q.queryID")
    request6: str = ("SELECT q.approach, (sum(c.rating) FILTER (q.executable != 0))/count(c.rating) as precision, "
                     "(sum(c.rating) FILTER (q.executable != 0)) AS tp, count(c.rating) AS t "
                     "FROM combined_reviews c, query_alternatives q "
                     "WHERE c.queryID = q.queryID AND c.query = q.query "
                     "GROUP BY q.approach "
                     "ORDER BY q.approach")
    request6_2: str = ("SELECT q.queryID, (sum(c.rating) FILTER (q.executable != 0))/count(c.rating) as precision, "
                       "(sum(c.rating) FILTER (q.executable != 0)) AS tp, count(c.rating) AS t "
                       "FROM combined_reviews c, query_alternatives q "
                       "WHERE c.queryID = q.queryID AND c.query = q.query "
                       "GROUP BY q.queryID "
                       "ORDER BY q.queryID")
    # Requests neither ignoring pruning nor execution
    request7: str = ("SELECT q.queryID, q.approach, "
                     "(sum(c.rating) FILTER (q.pruned != 1 AND q.executable != 0))/count(c.rating) as precision, "
                     "(sum(c.rating) FILTER (q.pruned != 1 AND q.executable != 0)) AS tp, count(c.rating) AS t "
                     "FROM combined_reviews c, query_alternatives q "
                     "WHERE c.queryID = q.queryID AND c.query = q.query "
                     "GROUP BY q.queryID, q.approach "
                     "ORDER BY q.approach, q.queryID")
    request8: str = ("SELECT q.approach, "
                     "(sum(c.rating) FILTER (q.pruned != 1 AND q.executable != 0))/count(c.rating) as precision, "
                     "(sum(c.rating) FILTER (q.pruned != 1 AND q.executable != 0)) AS tp, count(c.rating) AS t "
                     "FROM combined_reviews c, query_alternatives q "
                     "WHERE c.queryID = q.queryID AND c.query = q.query "
                     "GROUP BY q.approach "
                     "ORDER BY q.approach")
    request8_2: str = ("SELECT q.queryID, "
                       "(sum(c.rating) FILTER (q.pruned != 1 AND q.executable != 0))/count(c.rating) as precision, "
                       "(sum(c.rating) FILTER (q.pruned != 1 AND q.executable != 0)) AS tp, count(c.rating) AS t "
                       "FROM combined_reviews c, query_alternatives q "
                       "WHERE c.queryID = q.queryID AND c.query = q.query "
                       "GROUP BY q.queryID "
                       "ORDER BY q.queryID")
    res1: list = con.execute(request).fetchall()
    res2: list = con.execute(request2).fetchall()
    res2_2: list = con.execute(request2_2).fetchall()
    res3: list = con.execute(request3).fetchall()
    res4: list = con.execute(request4).fetchall()
    res4_2: list = con.execute(request4_2).fetchall()
    res5: list = con.execute(request5).fetchall()
    res6: list = con.execute(request6).fetchall()
    res6_2: list = con.execute(request6_2).fetchall()
    res7: list = con.execute(request7).fetchall()
    res8: list = con.execute(request8).fetchall()
    res8_2: list = con.execute(request8_2).fetchall()
    print(f"General Precision per Query and Approach:")
    print(*res1, sep="\n")
    print(f"General Precision per Approach:")
    print(*res2, sep="\n")
    print(f"General Precision per Query:")
    print(*res2_2, sep="\n")
    print(f"Precision per Query and Approach considering Pruning:")
    print(*res3, sep="\n")
    print(f"Precision per Approach considering Pruning:")
    print(*res4, sep="\n")
    print(f"Precision per Query considering Pruning:")
    print(*res4_2, sep="\n")
    print(f"Precision per Query and Approach considering Executability:")
    print(*res5, sep="\n")
    print(f"Precision per Approach considering Executability:")
    print(*res6, sep="\n")
    print(f"Precision per Query considering Executability:")
    print(*res6_2, sep="\n")
    print(f"Precision per Query and Approach ignoring nothing:")
    print(*res7, sep="\n")
    print(f"Precision per Approach ignoring nothing:")
    print(*res8, sep="\n")
    print(f"Precision per Query ignoring nothing:")
    print(*res8_2, sep="\n")
    con.close()


def homogeneity_of_lists(ranked_elements: list[str], top_k: int) -> float:
    """
    Calculate the average string similarity between the top k elements in the list.

    :param list[str] ranked_elements: the ranked list
    :param int top_k: the elements we want to compare
    :return: the average similarity of the top k elements in the list
    :rtype float
    """
    num_comps: float = 0
    sum_sims: float = 0
    for i in range(0, top_k):
        for j in range(i + 1, top_k):
            # print(f"Comparing:")
            # print(ranked_elements[i])
            # print(ranked_elements[j])
            num_comps += 1
            sum_sims += difflib_simple_comparison(ranked_elements[i], ranked_elements[j])
    # print(num_comps, sum_sims)
    avg_sim: float = sum_sims / num_comps
    return avg_sim
