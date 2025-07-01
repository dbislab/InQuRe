"""
Test all the helper methods to create summaries of the tables for pre-filtering.
"""
import math
import unittest
import duckdb

from random import choices
from string import ascii_uppercase, digits

import query_rewriting.config as config
from query_rewriting.rewrite_generator.prefilter_tables_summaries import calculate_metadata_db_path, save_tables_info, \
    check_for_summaries, slice_list_of_tables_for_summaries


class TestRewriteMethodsSummaries(unittest.TestCase):

    def test_calculate_db_metadata_path(self):
        """
        Test if the path to the metadata database is calculated correctly.
        """
        control_path: str = config.test_db_file
        control_path = (control_path[:-3] + config.db_metadata_addition + '_'
                        + config.gpt_model.replace("-", "_") + '.db')
        calculated_path: str = calculate_metadata_db_path(True)
        # print(f"Control: {control_path}\nActual:  {calculated_path}")
        self.assertEqual(calculated_path, control_path)

    def test_check_for_summaries(self):
        """
        Test if it is correctly determined if a summary is needed.
        """
        metadata_db_path: str = calculate_metadata_db_path(True)
        res: list[str] = check_for_summaries({'t1': ['c1'], 't2': ['c2']}, metadata_db_path)
        self.assertCountEqual(res, ['t1', 't2'])
        tables = ['t1', 't2']
        table_summaries: dict = {'t1': 'Some random summary.', 't2': 'A second random summary.'}
        table_topics: dict = {'t1': ['topic1', 'topic2'], 't2': ['topic3', 'topic4']}
        table_keywords: dict = {'t1': ['keyword1', 'keyword2'], 't2': ['keyword3', 'keyword4']}
        working = save_tables_info(tables, table_summaries, table_topics, table_keywords, metadata_db_path)
        assert working
        res = check_for_summaries({'t1': ['c1'], 't2': ['c2']}, metadata_db_path)
        self.assertCountEqual(res, [])
        res = check_for_summaries({'t1': ['c1'], 't2': ['c2'], 't3': ['c3']}, metadata_db_path)
        self.assertCountEqual(res, ['t3'])
        con = duckdb.connect(metadata_db_path)
        con.sql(f"DROP TABLE {config.db_metadata_table_name};")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()

    def test_save_tables_info(self):
        """
        Test if the tables info is saved correctly.
        """
        tables = ['t1', 't2']
        table_summaries: dict = {'t1': 'Some random summary.', 't2': 'A second random summary.'}
        table_topics: dict = {'t1': ['topic1', 'topic2'], 't2': ['topic3', 'topic4']}
        table_keywords: dict = {'t1': ['keyword1', 'keyword2'], 't2': ['keyword3', 'keyword4']}
        wrong_summary: dict = {'t2': 'A second random summary.'}
        wrong_topics: dict = {'t2': ['topic3', 'topic4']}
        wrong_keywords: dict = {'t2': ['keyword3', 'keyword4']}
        metadata_db_path: str = calculate_metadata_db_path(True)
        con = duckdb.connect(metadata_db_path)
        working: bool = save_tables_info(tables, wrong_summary, table_topics, table_keywords, metadata_db_path)
        assert not working
        self.assertEqual(con.execute(f"SELECT * FROM {config.db_metadata_table_name}").fetchall(),
                         [('t2', 'A second random summary.', 'topic3,topic4', 'keyword3,keyword4')])
        con.execute(f"DELETE FROM {config.db_metadata_table_name} WHERE {config.db_metadata_column_names[0]} = 't2'")
        working = save_tables_info(tables, table_summaries, wrong_topics, table_keywords, metadata_db_path)
        assert not working
        self.assertEqual(con.execute(f"SELECT * FROM {config.db_metadata_table_name}").fetchall(),
                         [('t2', 'A second random summary.', 'topic3,topic4', 'keyword3,keyword4')])
        con.execute(f"DELETE FROM {config.db_metadata_table_name} WHERE {config.db_metadata_column_names[0]} = 't2'")
        working = save_tables_info(tables, table_summaries, table_topics, wrong_keywords, metadata_db_path)
        assert not working
        self.assertEqual(con.execute(f"SELECT * FROM {config.db_metadata_table_name}").fetchall(),
                         [('t2', 'A second random summary.', 'topic3,topic4', 'keyword3,keyword4')])
        con.execute(f"DELETE FROM {config.db_metadata_table_name} WHERE {config.db_metadata_column_names[0]} = 't2'")
        working = save_tables_info(tables, table_summaries, table_topics, table_keywords, metadata_db_path)
        assert working
        res: list = con.execute(f"SELECT * FROM {config.db_metadata_table_name}").fetchall()
        control: list = [('t1', 'Some random summary.', 'topic1,topic2', 'keyword1,keyword2'),
                         ('t2', 'A second random summary.', 'topic3,topic4', 'keyword3,keyword4')]
        self.assertCountEqual(res, control)
        con.sql(f"DROP TABLE {config.db_metadata_table_name};")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()

    def test_slice_list_of_tables_for_summaries(self):
        """
        Test the case when only one slice of the tables is created.
        """
        create_statements: dict = dict()
        tables: list[str] = []
        for i in range(0, 1000):
            key: str = ''.join(choices(ascii_uppercase + digits, k=20))
            create_statements[str(i)] = key
            tables.append(str(i))
        for i in range(1000, 2000):
            key: str = ''.join(choices(ascii_uppercase + digits, k=16))
            create_statements[str(i)] = key
            tables.append(str(i))
        for i in range(2000, 3000):
            key: str = ''.join(choices(ascii_uppercase + digits, k=24))
            create_statements[str(i)] = key
            tables.append(str(i))
        # avg_create_table_tokens: float = ((20 + 16 + 24) / 3) / 4  # 5
        # output_length: int = 16384
        # input_length = 128000
        max_tokens: int = 5
        # tables per request: tables_per_slice: min(>20000, ca. 3000) -> one slice
        # Use the function to slice the tables
        tables_sliced_from_function, tokens_per_table = (
            slice_list_of_tables_for_summaries(tables, create_statements, max_tokens))
        last_tables: list[str] = tables_sliced_from_function[-1]
        self.assertEqual(3000, len(last_tables))
        self.assertGreater((16384 - 300) / 5, len(last_tables))
        for table in last_tables:
            create_statements.pop(table)
        self.assertEqual(create_statements, dict())
        self.assertEqual(tokens_per_table, math.floor(16384 / 3000))

    def test_slice_list_of_tables_for_summaries2(self):
        """
        Test the case when multiple slices of the tables is created.
        """
        create_statements: dict = dict()
        tables: list[str] = []
        for i in range(0, 1000):
            key: str = ''.join(choices(ascii_uppercase + digits, k=20))
            create_statements[str(i)] = key
            tables.append(str(i))
        for i in range(1000, 2000):
            key: str = ''.join(choices(ascii_uppercase + digits, k=16))
            create_statements[str(i)] = key
            tables.append(str(i))
        for i in range(2000, 3000):
            key: str = ''.join(choices(ascii_uppercase + digits, k=24))
            create_statements[str(i)] = key
            tables.append(str(i))
        create_statements2: dict = create_statements.copy()
        # avg_create_table_tokens: float = ((20 + 16 + 24) / 3) / 4  # 5
        output_length: int = 16384
        # input_length = 128000
        max_tokens: int = 100
        # tables per request: tables_per_slice: min(>20000, ca. 163) -> multiple slices
        tables_per_request: int = math.floor(output_length / max_tokens)
        # print(tables_per_request)
        # Use the function to slice the tables
        tables_sliced_from_function, tokens_per_table = (
            slice_list_of_tables_for_summaries(tables, create_statements, max_tokens))
        self.assertEqual(19, len(tables_sliced_from_function))
        for tables_sliced in tables_sliced_from_function[:-1]:
            self.assertEqual(len(tables_sliced), tables_per_request)
            for table in tables_sliced:
                create_statements.pop(table)
        last_tables: list[str] = tables_sliced_from_function[-1]
        self.assertEqual(66, len(last_tables))
        self.assertGreater(tables_per_request, len(last_tables))
        for table in last_tables:
            create_statements.pop(table)
        self.assertEqual(create_statements, dict())
        # Try the max tables
        tables_per_request = 50
        tables_sliced_from_function2, tokens_per_table2 = (
            slice_list_of_tables_for_summaries(tables, create_statements2, max_tokens, tables_per_request))
        self.assertEqual(60, len(tables_sliced_from_function2))
        for tables_sliced in tables_sliced_from_function2[:-1]:
            self.assertEqual(len(tables_sliced), tables_per_request)
            for table in tables_sliced:
                create_statements2.pop(table)
        last_tables: list[str] = tables_sliced_from_function2[-1]
        self.assertEqual(tables_per_request, len(last_tables))
        self.assertGreaterEqual(tables_per_request, len(last_tables))
        for table in last_tables:
            create_statements2.pop(table)
        self.assertEqual(create_statements2, dict())

