"""
Test the ranking algorithms for queries
"""
import unittest

from query_rewriting.distance_measures.sql_queries_comparison import difflib_simple_comparison
from query_rewriting.distance_measures.vector_embedding import set_up_model
from query_rewriting.ranking.ranking_algorithms import maximal_marginal_relevance, simple_ranker, \
    rank_via_clustering_dbscan, rank_via_clustering_kmeans

test_words: list[str]
input_word: str

test_sentences: list[str]
input_sentence: str
test_sql: list[str]
input_sql: str


def sim_func_between_queries(q1: str, q2: str) -> float:
    """
    Return predefined similarities for 2 queries from test_sql.
    """
    if q1 == q2:
        return 1
    elif q1 == test_sql[0] and q2 == test_sql[1] or q1 == test_sql[1] and q2 == test_sql[0]:
        return 1
    elif q1 == test_sql[0] and q2 == test_sql[2] or q1 == test_sql[2] and q2 == test_sql[0]:
        return 0
    elif q1 == test_sql[1] and q2 == test_sql[2] or q1 == test_sql[2] and q2 == test_sql[1]:
        return 0.4
    else:
        return 0


def sim_func_between_queries2(q1: str, q2: str) -> float:
    """
    Return predefined similarities for 2 queries from test_sql.
    """
    if q1 == q2:
        return 1
    elif q1 == test_sql[0] and q2 == test_sql[1] or q1 == test_sql[1] and q2 == test_sql[0]:
        return 1
    elif q1 == test_sql[0] and q2 == test_sql[2] or q1 == test_sql[2] and q2 == test_sql[0]:
        return 0
    elif q1 == test_sql[1] and q2 == test_sql[2] or q1 == test_sql[2] and q2 == test_sql[1]:
        return 1
    else:
        return 0


def sim_func_to_input(input_q: str, q: str) -> float:
    """
     Return predefined similarities for the input query (input_sql) and another query from test_sql.
    """
    if input_q == q:
        return 1
    elif q == test_sql[0]:
        return 1
    elif q == test_sql[1]:
        return 0.8
    elif q == test_sql[2]:
        return 0.4
    else:
        return 0


def sim_func_to_input_list(input_q: str, alt_q: list[str]) -> list[float]:
    """
    Return predefined similarities for the input query (input_sql) and all other queries from test_sql.
    """
    results: list[float] = []
    for q in alt_q:
        results.append(sim_func_to_input(input_q, q))
    return results


class TestRankQueries(unittest.TestCase):

    def setUp(self):
        """
        Set up the example data to test the ranking algorithms.
        """
        global test_words, input_word, test_sentences, input_sentence, test_sql, input_sql
        test_words = ["house", "hat", "hos"]
        input_word = "huse"
        test_sentences = ["Willkommen zur Ubung Informationssysteme an der TU Kaiserslautern.",
                          "Sich in einer Stadt wie Kaiserslautern zu orientieren erfordert etwas Ubung, "
                          "wenn man vom Land kommt.",
                          "Mit etwas Ubung kann jeder den Umgang mit Datenbanken erlernen"]
        input_sentence = "Willkommen zu Informationssysteme."
        test_sql = ["SELECT avg(income) AS aIncome FROM income GROUP BY district ORDER BY aIncome DESC;",
                    "SELECT percentage FROM crimeRate ORDER BY percentage ASC;",
                    "SELECT avg(monthlyExpense) AS mExpense FROM income GROUP BY district ORDER BY mExpense DESC;"]
        input_sql = "SELECT avg(rent) AS aRent FROM apartments GROUP BY district ORDER BY aRent DESC;"

    def test_mmr_difflib_similarity(self):
        """
        Test the MMR function using the inbuilt difflib similarity.
        """
        # print(test_words, input_word, test_sentences, input_sentence, test_sql, input_sql)
        ranked_list: list[str] = maximal_marginal_relevance(input_word, test_words, difflib_simple_comparison,
                                                            difflib_simple_comparison, 0.5, 3)
        # print(ranked_list)
        self.assertEqual(ranked_list, test_words)

    def test_mmr_difflib_similarity_sentence(self):
        """
        Test the MMR function for sentences using the inbuilt difflib similarity.
        """
        ranked_list: list[str] = maximal_marginal_relevance(input_sentence, test_sentences,
                                                            difflib_simple_comparison, difflib_simple_comparison,
                                                            0.5, 3)
        # print(ranked_list)
        self.assertEqual(ranked_list, test_sentences)

    def test_mmr_difflib_similarity_sql(self):
        """
        Test the MMR function for SQL using the inbuilt difflib similarity.
        """
        ranked_list: list[str] = maximal_marginal_relevance(input_sql, test_sql, difflib_simple_comparison,
                                                            difflib_simple_comparison, 0.5, 3)
        # print(ranked_list)
        self.assertEqual(ranked_list, test_sql)

    def test_mmr_with_2_dist_functions(self):
        """
        Test the MMR function with 2 different similarity methods.
        """
        # print(test_words, input_word, test_sentences, input_sentence, test_sql, input_sql)
        ranked_list: list[str] = maximal_marginal_relevance(input_sql, test_sql,
                                                            sim_func_between_queries, sim_func_to_input,
                                                            0.5, 2)
        check_list: list[str] = [test_sql[0], test_sql[2]]
        self.assertEqual(ranked_list, check_list)
        ranked_list2: list[str] = maximal_marginal_relevance(input_sql, test_sql,
                                                             sim_func_between_queries, sim_func_to_input_list,
                                                             0.5, 2)
        self.assertEqual(ranked_list2, check_list)

    def test_simple_ranker(self):
        """
        Test the simple ranking algorithm.
        """
        # print(test_words, input_word, test_sentences, input_sentence, test_sql, input_sql)
        # print(type(sim_func_to_input))
        # print(type(sim_func_to_input_list))
        ranked_list1: list[str] = simple_ranker(input_sql, test_sql, sim_func_to_input, 1)
        ranked_list2: list[str] = simple_ranker(input_sql, test_sql, sim_func_to_input, 2)
        ranked_list3: list[str] = simple_ranker(input_sql, test_sql, sim_func_to_input, 3)
        self.assertEqual(ranked_list1, [test_sql[0]])
        self.assertEqual(ranked_list2, [test_sql[0], test_sql[1]])
        self.assertEqual(ranked_list3, [test_sql[0], test_sql[1], test_sql[2]])
        # Test with other similarity measure type
        ranked_list4: list[str] = simple_ranker(input_sql, test_sql, sim_func_to_input_list, 1)
        ranked_list5: list[str] = simple_ranker(input_sql, test_sql, sim_func_to_input_list, 2)
        ranked_list6: list[str] = simple_ranker(input_sql, test_sql, sim_func_to_input_list, 3)
        self.assertEqual(ranked_list4, [test_sql[0]])
        self.assertEqual(ranked_list5, [test_sql[0], test_sql[1]])
        self.assertEqual(ranked_list6, [test_sql[0], test_sql[1], test_sql[2]])

    def test_rank_via_clustering_dbscan(self):
        """
        Test the clustering and ranking with DBScan.
        """
        # Distance metric: [[0, 0, 1], [0, 0, 0.6], [1, 0.6, 0]]
        ranked_list: list[str] = rank_via_clustering_dbscan(input_sql, test_sql,
                                                            sim_func_between_queries, sim_func_to_input, 2)
        # print(ranked_list)
        self.assertEqual(ranked_list, [test_sql[0], test_sql[2]])
        ranked_list2: list[str] = rank_via_clustering_dbscan(input_sql, test_sql,
                                                             sim_func_between_queries2, sim_func_to_input, 1)
        # print(ranked_list2)
        self.assertEqual(ranked_list2, [test_sql[0]])

    def test_rank_via_clustering_kmeans(self):
        """
        Test the clustering and ranking with k-Means.
        """
        set_up_model('sentence-transformers/all-mpnet-base-v2')
        ranked_list: list[str] = rank_via_clustering_kmeans(input_sql, test_sql,
                                                            sim_func_to_input, 2)
        # print(ranked_list)
        # Sometimes wrong due to random start nature of k-means
        self.assertEqual(ranked_list, [test_sql[0], test_sql[1]])
        ranked_list2: list[str] = rank_via_clustering_kmeans(input_sql, test_sql,
                                                             sim_func_to_input, 1)
        self.assertEqual(ranked_list2, [test_sql[0]])
        ranked_list3: list[str] = rank_via_clustering_kmeans(input_sql, test_sql,
                                                             sim_func_to_input, 3)
        self.assertEqual(ranked_list3, [test_sql[0], test_sql[1], test_sql[2]])