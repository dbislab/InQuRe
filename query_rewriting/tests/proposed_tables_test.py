"""
Check if the proposed tables for NL that are found are returned correctly
"""
import unittest

import duckdb

import query_rewriting.config as config
from query_rewriting.input_processor.check_executability import check_proposed_tables_exact


class TestProposedTables(unittest.TestCase):

    def setUp(self):
        """
        Set up the DB connection and create a test table.
        """
        con = duckdb.connect(config.test_db_file)
        con.execute("CREATE TABLE IF NOT EXISTS tbl (i INTEGER, j VARCHAR);")
        con.close()

    def test_checking_db_for_tables(self):
        """
        Check if the return is true and the exact schema if they are the same.
        """
        schema: dict = {"tbl": ["i", "j"]}
        check_bool, check_schema = check_proposed_tables_exact(schema, True)
        self.assertTrue(check_bool)
        list1: list[str] = schema.pop("tbl", [])
        list2: list[str] = check_schema.pop("tbl", [])
        self.assertNotEqual(len(list1), 0)
        self.assertNotEqual(len(list2), 0)
        self.assertCountEqual(list1, list2)
        self.assertEqual(schema, check_schema)  # both should be empty now
        # self.assertEqual(check_schema, schema)

    def test_checking_db_for_not_existent_relation(self):
        """
        Check that all tables need to be found to return true.
        """
        schema: dict = {"tbl": ["i", "j"], "table": ["i", "j"]}
        check_bool, check_schema = check_proposed_tables_exact(schema, True)
        wanted_schema: dict = {"tbl": ["i", "j"]}
        self.assertFalse(check_bool)
        list1: list[str] = wanted_schema.pop("tbl", [])
        list2: list[str] = check_schema.pop("tbl", [])
        self.assertNotEqual(len(list1), 0)
        self.assertNotEqual(len(list2), 0)
        self.assertCountEqual(list1, list2)
        self.assertEqual(wanted_schema, check_schema)  # both should be empty now
        # self.assertEqual(check_schema, {"tbl": ["i", "j"]})

    def test_checking_db_for_too_few_existent_relations(self):
        """
        Check that the table name is relevant for all the column names to be found.
        """
        schema = {"table": ["i", "j"]}
        check_bool, check_schema = check_proposed_tables_exact(schema, True)
        self.assertFalse(check_bool)
        self.assertEqual(check_schema, dict())

    def test_checking_db_for_not_existent_columns(self):
        """
        Check that the table name is relevant for the column name to be found.
        """
        schema = {"table": ["i"]}
        check_bool, check_schema = check_proposed_tables_exact(schema, True)
        self.assertFalse(check_bool)
        self.assertEqual(check_schema, dict())

    def test_checking_db_for_one_existent_column(self):
        """
        Check that all needed columns are present for the function to return true.
        """
        schema = {"tbl": ["i"]}
        check_bool, check_schema = check_proposed_tables_exact(schema, True)
        self.assertTrue(check_bool)
        self.assertEqual(check_schema, schema)

    def test_checking_db_for_too_few_existent_column(self):
        """
        Check that the table is found, but not all columns (but the found ones are returned).
        """
        schema = {"tbl": ["i", "j", "k"]}
        check_bool, check_schema = check_proposed_tables_exact(schema, True)
        self.assertFalse(check_bool)
        wanted_schema = {"tbl": ["i", "j"]}
        list1: list[str] = wanted_schema.pop("tbl", [])
        list2: list[str] = check_schema.pop("tbl", [])
        self.assertNotEqual(len(list1), 0)
        self.assertNotEqual(len(list2), 0)
        self.assertCountEqual(list1, list2)
        self.assertEqual(wanted_schema, dict())
        self.assertEqual(wanted_schema, check_schema)  # both should be empty now
        # self.assertEqual(check_schema, {"tbl": ["i", "j"]})

    def tearDown(self):
        """
        Drop the test table and close the DB connection.
        """
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE tbl;")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()
