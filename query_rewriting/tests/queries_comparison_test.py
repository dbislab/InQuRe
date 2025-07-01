"""
Test the distance functions for SQL
"""
import unittest

from query_rewriting.distance_measures.sql_queries_comparison import difflib_simple_comparison, \
    similarity_using_tables, similarity_using_clauses


class TestComparison(unittest.TestCase):

    def test_difflib_comparison_equal(self):
        """
        Assert that two same words get a similarity of 1.
        """
        self.assertEqual(difflib_simple_comparison("house", "house"), 1)

    def test_difflib_comparison_not_equal(self):
        """
        Assert that two completely different words get a similarity of 0.
        """
        self.assertEqual(difflib_simple_comparison("house", "park"), 0)

    def test_difflib_comparison_similar(self):
        """
        Assert that two similar words get a high similarity.
        """
        sim: float = difflib_simple_comparison("hose", "house")
        # print(sim)
        self.assertGreater(sim, 0.8)

    def test_difflib_comparison_not_similar(self):
        """
        Assert that two similar words get a high similarity.
        """
        sim: float = difflib_simple_comparison("hat", "house")
        # print(sim)
        self.assertGreater(0.5, sim)

    def test_similarity_using_tables(self):
        """
        Test how the method comparing tables of two SQL queries behaves.
        """
        q1: str = "SELECT * FROM zoo"
        q2: str = "SELECT * FROM employees"
        q3: str = "SELECT * FROM employee"
        q4: str = "SELECT * FROM zoo, staff"
        q5: str = "SELECT * FROM employee, staff"
        sim1: float = similarity_using_tables(q1, q2)
        # print(sim1)
        self.assertGreater(0.16, sim1)
        self.assertGreater(sim1, 0.15)
        sim2: float = similarity_using_tables(q2, q3)
        # print(sim2)
        self.assertGreater(0.83, sim2)
        self.assertGreater(sim2, 0.82)
        sim3: float = similarity_using_tables(q1, q1)
        # print(sim3)
        self.assertGreaterEqual(1.0, sim3)
        self.assertGreaterEqual(sim3, 1.0)
        sim4: float = similarity_using_tables(q4, q5)
        # print(sim4)
        self.assertGreater(0.61, sim4)
        self.assertGreater(sim4, 0.6)

    def test_similarity_using_clauses(self):
        """
        Test how the method comparing clauses of two SQL queries behaves.
        """
        q1: str = "SELECT zoo.animal FROM zoo JOIN staff ON zoo.id = staff.employer WHERE zoo.name = 'KL'"
        q2: str = "SELECT zoo.city FROM zoo JOIN staff ON zoo.id = staff.employer WHERE zoo.name = 'KL'"
        q3: str = "SELECT zoo.city FROM zoo JOIN staff ON zoo.id = staff.employer"
        sim1: float = similarity_using_clauses(q1, q1)
        sim2: float = similarity_using_clauses(q2, q1)
        sim3: float = similarity_using_clauses(q1, q2)
        sim4: float = similarity_using_clauses(q2, q3)
        sim5: float = similarity_using_clauses(q3, q2)
        self.assertEqual(sim1, 1)
        self.assertEqual(sim2, sim3)
        self.assertGreater(sim2, 0.75)
        self.assertGreater(0.76, sim2)
        self.assertEqual(sim4, sim5)
        self.assertGreater(sim4, 0.80)
        self.assertGreater(0.81, sim4)


