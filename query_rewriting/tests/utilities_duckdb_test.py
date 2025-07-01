"""
Test the functions that are written in the utilities package for DuckDB
"""
import unittest

import duckdb

import query_rewriting.config as config
from query_rewriting.utilities.duckdb_functions import get_result_with_column_names, get_tables_from_db, \
    check_existence_of_tables, get_usable_constraints_from_db


class TestUtilitiesDuckDB(unittest.TestCase):

    def test_duck_db_utilities(self):
        """
        Test the utility methods for DuckDB.
        """
        con = duckdb.connect(config.test_db_file)
        con.execute("CREATE TABLE IF NOT EXISTS tbl (i INTEGER, j VARCHAR);")
        con.execute("INSERT INTO tbl VALUES(1,'one');")
        con.execute("INSERT INTO tbl VALUES(2,'two');")
        con.execute("SELECT * FROM tbl;")
        res: list[list] = get_result_with_column_names(con)
        self.assertEqual(res[0], ["i", "j"])
        self.assertCountEqual(res, [["i", "j"], (1, 'one'), (2, 'two')])
        con.sql("DROP TABLE tbl;")
        con.close()

    def test_duck_db_utilities_empty_result(self):
        con = duckdb.connect(config.test_db_file)
        con.execute("CREATE TABLE IF NOT EXISTS tbl (i INTEGER, j VARCHAR);")
        con.execute("SELECT * FROM tbl;")
        res: list[list] = get_result_with_column_names(con)
        self.assertEqual(res, [["i", "j"]])
        con.sql("DROP TABLE tbl;")
        con.close()

    def test_get_tables_from_db(self):
        """
        Test if the right tables are returned from the database.
        """
        # Set up
        con = duckdb.connect(config.test_db_file)
        con.execute("CREATE TABLE IF NOT EXISTS tbl (i INTEGER, j VARCHAR);")
        con.execute("CREATE TABLE IF NOT EXISTS tbl2 (k INTEGER, l VARCHAR, m DATE);")
        con.close()
        # Test
        db_tables: dict = get_tables_from_db(True)
        wanted_schema = {"tbl": ["i INTEGER", "j VARCHAR"], "tbl2": ["k INTEGER", "l VARCHAR", "m DATE"]}
        list1: list[str] = wanted_schema.pop("tbl", [])
        list2: list[str] = db_tables.pop("tbl", [])
        self.assertNotEqual(len(list1), 0)
        self.assertNotEqual(len(list2), 0)
        self.assertCountEqual(list1, list2)
        list3: list[str] = wanted_schema.pop("tbl2", [])
        list4: list[str] = db_tables.pop("tbl2", [])
        self.assertNotEqual(len(list3), 0)
        self.assertNotEqual(len(list4), 0)
        self.assertCountEqual(list3, list4)
        self.assertEqual(db_tables, dict())
        self.assertEqual(db_tables, wanted_schema)
        # Tear down
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE tbl;")
        con.sql("DROP TABLE tbl2;")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()

    def test_check_existence_of_tables(self):
        # Set up
        con = duckdb.connect(config.test_db_file)
        assert not check_existence_of_tables(True)
        con.execute("CREATE TABLE IF NOT EXISTS tbl (i INTEGER, j VARCHAR);")
        assert check_existence_of_tables(True)
        con.execute("CREATE TABLE IF NOT EXISTS tbl2 (k INTEGER, l VARCHAR, m DATE);")
        assert check_existence_of_tables(True)
        con.close()
        # Tear down
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE tbl;")
        con.sql("DROP TABLE tbl2;")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()

    def test_check_result_to_dict(self):
        """
        Check if turning the result of a DuckDB query into a dictionary works.
        """
        # Set up
        con = duckdb.connect(config.test_db_file)
        create1: str = "CREATE TABLE IF NOT EXISTS tbl (i INTEGER, j VARCHAR);"
        create2: str = "CREATE TABLE IF NOT EXISTS tbl2 (k INTEGER, l VARCHAR, m DATE);"
        create_in_db1: str = "CREATE TABLE tbl(i INTEGER, j VARCHAR);"
        create_in_db2: str = "CREATE TABLE tbl2(k INTEGER, l VARCHAR, m DATE);"
        con.execute(create1)
        con.execute(create2)
        create_statements_list: list = con.execute("SELECT table_name, sql FROM duckdb_tables()").fetchall()
        create_statements_dict: dict = dict(create_statements_list)
        con.close()
        self.assertEqual(create_statements_dict.pop("tbl"), create_in_db1)
        self.assertEqual(create_statements_dict.pop("tbl2"), create_in_db2)
        self.assertEqual(create_statements_dict, dict())
        # Tear down
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE tbl;")
        con.sql("DROP TABLE tbl2;")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()

    def test_get_usable_constraints_from_db(self):
        """
        Test if the right foreign keys are returned for different constellations.
        """
        # First test
        self.assertEqual(dict(), get_usable_constraints_from_db([], True))
        # Set up
        con = duckdb.connect(config.test_db_file)
        con.execute("CREATE TABLE t1 (i INTEGER PRIMARY KEY, j VARCHAR);")
        con.execute("CREATE TABLE t2 (k INTEGER, l VARCHAR PRIMARY KEY, m DATE, FOREIGN KEY (k) REFERENCES t1(i));")
        con.execute("CREATE TABLE t3 (n INTEGER, o VARCHAR, FOREIGN KEY (n) REFERENCES t1(i), "
                    "FOREIGN KEY (o) REFERENCES t2(l));")
        con.close()
        # Check the method
        rewrite_tables1: dict = get_usable_constraints_from_db(['t1'], True)
        rewrite_tables2: dict = get_usable_constraints_from_db(['t1', 't2'], True)
        rewrite_tables3: dict = get_usable_constraints_from_db(['t1', 't2', 't3'], True)
        rewrite_tables4: dict = get_usable_constraints_from_db(['t1', 't3'], True)
        ground_truth1: dict = dict()
        ground_truth2: dict = {'t2': ['FOREIGN KEY (k) REFERENCES t1(i)']}
        ground_truth3: dict = {'t2': ['FOREIGN KEY (k) REFERENCES t1(i)'],
                               't3': ['FOREIGN KEY (n) REFERENCES t1(i)', 'FOREIGN KEY (o) REFERENCES t2(l)']}
        ground_truth4: dict = {'t3': ['FOREIGN KEY (n) REFERENCES t1(i)']}
        self.assertEqual(rewrite_tables1, ground_truth1)
        self.assertEqual(rewrite_tables2, ground_truth2)
        self.assertEqual(rewrite_tables3.pop('t2'), ground_truth3.pop('t2'))
        self.assertCountEqual(rewrite_tables3.pop('t3'), ground_truth3.pop('t3'))
        self.assertEqual(rewrite_tables3, ground_truth3)
        self.assertEqual(rewrite_tables4, ground_truth4)
        # Tear down
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE t3;")
        con.sql("DROP TABLE t2;")
        con.sql("DROP TABLE t1;")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()
