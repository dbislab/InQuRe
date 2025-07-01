"""
Test prefilter methods from the rewriting package (but only methods used for the pre-filtering by the LLM).
"""
import math
import unittest
from random import choices
from string import ascii_uppercase, digits

from query_rewriting.rewrite_generator.prefilter_tables_llm import get_tables_in_slices_for_llm_call, \
    find_similar_tables, tokenize_and_stem


class TestRewriteMethodsLLM(unittest.TestCase):

    def test_get_slices_for_llm_call(self):
        """
        Test the case when only one slice of the tables is created.
        """
        schema: dict = dict()
        for i in range(0, 1000):
            key: str = ''.join(choices(ascii_uppercase + digits, k=20))
            schema[key] = i
        for i in range(1000, 2000):
            key: str = (''.join(choices(ascii_uppercase + digits, k=16)) + '_' +
                        ''.join(choices(ascii_uppercase + digits, k=16)))
            schema[key] = i
        for i in range(2000, 3000):
            key: str = (''.join(choices(ascii_uppercase + digits, k=24)) + '_' +
                        ''.join(choices(ascii_uppercase + digits, k=24)) + '_' +
                        ''.join(choices(ascii_uppercase + digits, k=24)))
            schema[key] = i
        avg_table_len: float = ((3000 / 3000) + 1) * (((20 + 16 + 16 + 24 + 24 + 24) / 6) / 4)  # 10.3333
        output_length: int = 16384
        percentage_for_output: float = 0.5
        # tables per request: 3172 -> one slice
        tables_per_request: int = math.ceil((output_length / avg_table_len) * (1 / percentage_for_output))
        # Use the function to slice the tables
        tables_sliced_from_function: list[list[str]] = get_tables_in_slices_for_llm_call(schema, percentage_for_output)
        last_tables: list[str] = tables_sliced_from_function[-1]
        self.assertEqual(3000, len(last_tables))
        self.assertGreater(tables_per_request, len(last_tables))
        for table in last_tables:
            schema.pop(table)
        self.assertEqual(schema, dict())

    def test_get_slices_for_llm_call_2(self):
        """
        Test the case when multiple slices of the tables is created.
        """
        schema: dict = dict()
        for i in range(0, 1000):
            key: str = ''.join(choices(ascii_uppercase + digits, k=48))
            schema[key] = i
        for i in range(1000, 2000):
            key: str = (''.join(choices(ascii_uppercase + digits, k=36)) + '_' +
                        ''.join(choices(ascii_uppercase + digits, k=36)))
            schema[key] = i
        for i in range(2000, 3000):
            key: str = (''.join(choices(ascii_uppercase, k=24)) + '_' +
                        ''.join(choices(ascii_uppercase, k=24)) + '_' +
                        ''.join(choices(ascii_uppercase, k=24)))
            schema[key] = i
        avg_table_len: float = ((3000 / 3000) + 1) * (((48 + 36 + 36 + 24 + 24 + 24) / 6) / 4)  # 16
        output_length: int = 16384
        percentage_for_output: float = 0.5
        # tables per request: 2048 -> two slices
        tables_per_request: int = math.ceil((output_length / avg_table_len) * (1 / percentage_for_output))
        # Use the function to slice the tables
        tables_sliced_from_function: list[list[str]] = get_tables_in_slices_for_llm_call(schema, percentage_for_output)
        for tables in tables_sliced_from_function[:-1]:
            self.assertEqual(len(tables), tables_per_request)
            for table in tables:
                schema.pop(table)
        last_tables: list[str] = tables_sliced_from_function[-1]
        self.assertEqual(952, len(last_tables))
        self.assertGreater(tables_per_request, len(last_tables))
        for table in last_tables:
            schema.pop(table)
        self.assertEqual(schema, dict())

    def test_find_similar_tables(self):
        """
        Tests if finding similar tables does the right thing.
        """
        suggested_tables: list[str] = ['student', 'lecture']
        db_tables: dict = {'lecture': ['c1 type1'], 'zoo': ['c1 t1']}
        control: dict = {'lecture': ['c1 type1']}
        result: dict = find_similar_tables(suggested_tables, db_tables)
        self.assertEqual(len(control), len(result))
        self.assertEqual(control.pop('lecture'), result.pop('lecture'))
        self.assertEqual(len(result), 0)

    def test_tokenize_and_stem(self):
        """
        Test tokenization and stemming.
        """
        res: list[list[str]] = tokenize_and_stem(['word', 'Words', 'dogHouse', 'snake_case', 'Other_Snake_Case',
                                                  'HTTPResponse', 'ClassName'])
        control: list[list[str]] = [['word'], ['word'], ['dog', 'hous'], ['snake', 'case'], ['other', 'snake', 'case'],
                                    ['httprespons'], ['class', 'name']]
        for r, c in zip(res, control):
            self.assertCountEqual(r, c)