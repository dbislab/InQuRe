"""
Test different methods that check query execution from input- and postprocessing.
"""
import unittest

import duckdb

import query_rewriting.config as config
from query_rewriting.input_processor.check_executability import check_query_execution
from query_rewriting.postprocessor.query_correction import get_error_message_or_result


class TestExecutionMethods(unittest.TestCase):

    def setUp(self):
        """
        Set up the database.
        """
        con = duckdb.connect(config.test_db_file)
        con.execute("CREATE TABLE IF NOT EXISTS tbl (i INTEGER, j VARCHAR);")
        con.execute("INSERT INTO tbl VALUES(1,'one');")
        con.close()

    def test_check_query_execution(self):
        """
        Check the query execution method from the input processor.
        """
        executable, result = check_query_execution("SELECT * FROM table;", True)
        self.assertFalse(executable)
        self.assertEqual(result, [])
        executable, result = check_query_execution("SELECT k FROM tbl;", True)
        self.assertFalse(executable)
        self.assertEqual(result, [])
        executable, res_list = check_query_execution("SELECT i FROM tbl;", True)
        self.assertTrue(executable)
        self.assertEqual(res_list, [['i'], (1,)])

    def test_check_query_execution_correction(self):
        """
        Check the query execution method from the correction module.
        """
        executable, result, correctable, error_msg = get_error_message_or_result("SELECT * FROM table;", True)
        self.assertFalse(executable)
        self.assertEqual(result, [])
        self.assertTrue(correctable)
        # print(f"Found another error message:\n{error_msg}")
        self.assertTrue(error_msg.startswith("Parser Error"))
        executable, result, correctable, error_msg = get_error_message_or_result("SELECT k FROM tbl;", True)
        self.assertFalse(executable)
        self.assertEqual(result, [])
        self.assertTrue(correctable)
        # print(f"Found another error message:\n{error_msg}")
        self.assertTrue(error_msg.startswith("Binder Error"))
        executable, result, correctable, error_msg = get_error_message_or_result("SELECT k FROM ?;", True)
        self.assertFalse(executable)
        self.assertEqual(result, [])
        self.assertTrue(correctable)
        # print(f"Found another error message:\n{error_msg}")
        self.assertTrue(error_msg.startswith("Parser Error"))
        executable, result, correctable, error_msg = get_error_message_or_result("SELECT i FROM tbl;", True)
        self.assertTrue(executable)
        self.assertEqual(result, [['i'], (1,)])
        self.assertTrue(correctable)
        self.assertTrue(error_msg == "")

    def tearDown(self):
        """
        Delete the stuff from the database.
        """
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE tbl;")
        con.close()
