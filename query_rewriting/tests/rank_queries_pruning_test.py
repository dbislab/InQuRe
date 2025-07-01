"""
Test the pruning part of the ranking.
"""
import unittest

from query_rewriting.ranking.prune_alternatives import simple_pruning_via_prefix_of_db


class TestRankQueriesPruning(unittest.TestCase):

    def test_simple_prefix_pruning(self):
        query1: str = "SELECT * FROM pre_table, pre_anotherTable"
        query2: str = "SELECT * FROM table, otherTable;"
        query3: str = "SELECT * FROM prefiTable, prefiOtherTable"
        query4: str = "SELECT * FROM prefixTable, prefixOtherTable"
        list_queries: list[str] = [query1, query2, query3, query4]
        pruned_queries: list[str] = simple_pruning_via_prefix_of_db(list_queries)
        control: list[str] = [query1, query4]
        self.assertCountEqual(pruned_queries, control)

