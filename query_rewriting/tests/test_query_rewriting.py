"""
Main test file
"""
import argparse
import unittest

import query_rewriting.config as config

if __name__ == '__main__':
    """
    A second main method to execute all the tests.
    """
    # Add the arguments to make the test DB file configurable
    parser = argparse.ArgumentParser(prog='Query Rewriting',
                                     description='Read requests from the input and execute them or a rewrite')
    parser.add_argument('-d', '--database-file', default=config.test_db_file, help='DuckDB file')
    args = parser.parse_args()
    config.test_db_file = args.database_file
    # print(input_requests)
    # Load the unittest files that end with _test.py
    loader = unittest.TestLoader()
    start_directory = '.'
    suite = loader.discover(start_directory, pattern='*_test.py')
    # Run all tests
    runner = unittest.TextTestRunner()
    runner.run(suite)
    # unittest.main()
