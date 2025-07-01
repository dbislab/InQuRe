"""
Test the database connection to DuckDB
"""
import unittest

import duckdb

import query_rewriting.config as config


class TestDatabaseConnection(unittest.TestCase):

    def test_method_duck_db(self):
        """
        Test if the given DuckDB database works.
        """
        con = duckdb.connect(config.test_db_file)
        con.execute("CREATE TABLE IF NOT EXISTS tbl (i INTEGER, j VARCHAR);")
        con.sql("DESCRIBE tbl;").show()
        con.execute("INSERT INTO tbl VALUES(1,'one');")
        con.execute("SELECT * FROM tbl;")
        res = con.fetchone()
        self.assertEqual(res, (1, 'one'))
        assert con.fetchone() is None
        # con.sql("SHOW ALL TABLES;").show()
        con.close()
        # fetchall/fetchone on con after execute to get items

    def test_method_duck_db_delete(self):
        """
        Test if the deletion of the database works.
        """
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE tbl;")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        # print("Table successfully deleted")
        con.close()
