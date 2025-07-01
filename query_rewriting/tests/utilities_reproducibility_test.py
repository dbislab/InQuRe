"""
Test all the utility functions for reproducibility.
"""
import unittest
from collections.abc import Iterable

import duckdb

import query_rewriting.config as config
from query_rewriting.utilities.reproducibility_functions import calculate_reproducibility_db_path, \
    dictionary_to_string, string_to_dictionary, get_hash_from_request_string, get_request_string_from_iterable, \
    create_reproducibility_database, check_for_entry, write_new_entry


class TestUtilitiesReproducibility(unittest.TestCase):

    def setUp(self):
        """
        Set Up the DB used in the tests.
        """
        create_reproducibility_database(True)

    def test_calculate_db_path(self):
        control_path: str = config.test_db_file[:-3] + config.db_reproducibility_addition + ".db"
        calculated_path: str = calculate_reproducibility_db_path(True)
        self.assertEqual(calculated_path, control_path)

    def test_string_dict_conversions(self):
        additional_company_schema: dict = {
            "projects": ["pID INTEGER PRIMARY KEY", "company INTEGER", "cost FLOAT", "gain FLOAT", "startDate DATE",
                         "endDate DATE", "numWorkers INTEGER"],
            "employees": ["eID INTEGER PRIMARY KEY", "name VARCHAR", "dateOfBirth DATE", "specialization VARCHAR"],
            "workedIn": ["project INTEGER", "employee INTEGER"]
        }
        # print(dictionary_to_string(additional_company_schema))
        self.assertEqual(additional_company_schema,
                         string_to_dictionary(dictionary_to_string(additional_company_schema)))

    def test_hash_consistency(self):
        request: str = "This is a certain string that could be given for a request."
        hash_val: str = get_hash_from_request_string(request)
        self.assertEqual(hash_val, "e0545a1fe0dfecb309c7fd58c0eb3d3b762f90709171015f6694687f20d5b3d0")
        for i in range(0, 100):
            self.assertEqual(get_hash_from_request_string(request),
                             "e0545a1fe0dfecb309c7fd58c0eb3d3b762f90709171015f6694687f20d5b3d0")

    def test_request_str_from_iterable(self):
        content_for_gpt: str = "This is a test request."
        message: Iterable = [
            {"role": "system", "content": "We will work with databases and queries in SQL."},
            {
                "role": "user",
                "content": content_for_gpt
            }
        ]
        calculated: str = get_request_string_from_iterable(message)
        control: str = ("[{'role': 'system', 'content': 'We will work with databases and queries in SQL.'},"
                        "{'role': 'user', 'content': 'This is a test request.'}]")
        # print(calculated)
        self.assertEqual(calculated, control)

    def test_create_reproducibility_database(self):
        create_reproducibility_database(True)

    def test_check_for_entry(self):
        content_for_gpt: str = "This is a test request."
        message: Iterable = [
            {"role": "system", "content": "We will work with databases and queries in SQL."},
            {
                "role": "user",
                "content": content_for_gpt
            }
        ]
        message_str: str = get_request_string_from_iterable(message)
        message_hash: str = get_hash_from_request_string(message_str)
        res1: dict = check_for_entry("Model", message, True)
        self.assertEqual(res1, dict())
        con = duckdb.connect(calculate_reproducibility_db_path(True))
        json_str: str = '{"message": "Hello"}'
        con.execute(f"INSERT INTO {config.db_reproducibility_table_name} VALUES "
                    f"('1','Model',?,?,?,'Some fingerprint')", [message_str, message_hash, json_str])
        res2: dict = check_for_entry("Model", message, True)
        self.assertEqual(res2.get("message"), "Hello")
        self.assertEqual(len(res2.keys()), 1)
        con.execute(f"DELETE FROM {config.db_reproducibility_table_name} WHERE ID = '1'")
        check_res: list = con.execute(f"SELECT * FROM {config.db_reproducibility_table_name}").fetchall()
        con.close()
        self.assertEqual(len(check_res), 0)

    def test_write_new_entry(self):
        content_for_gpt: str = "This is a test request."
        message: Iterable = [
            {"role": "system", "content": "We will work with databases and queries in SQL."},
            {
                "role": "user",
                "content": content_for_gpt
            }
        ]
        response_chat: dict = {
            "id": "chatcmpl-123456",
            "object": "chat.completion",
            "created": 100,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hi there! How can I assist you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 19,
                "completion_tokens": 10,
                "total_tokens": 29,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            },
            "system_fingerprint": "fp"
        }  # Taken as an example from https://platform.openai.com/docs/api-reference/chat/object
        write_new_entry('1', 'Model', message, response_chat, "Fingerprint", True)
        con = duckdb.connect(calculate_reproducibility_db_path(True))
        entries: list = con.execute(f"SELECT * FROM {config.db_reproducibility_table_name}").fetchall()
        # print(entries)
        control = [('1', 'Model', "[{'role': 'system', 'content': 'We will work with databases and queries in SQL.'},"
                                  "{'role': 'user', 'content': 'This is a test request.'}]",
                    '705c426ae702ab874e18ebb95d57fcbf89eae255ba910fd1d5f8ef32b960ca45',
                    '{"id": "chatcmpl-123456", "object": "chat.completion", "created": 100, '
                    '"model": "gpt-4o-2024-08-06", "choices": [{"index": 0, "message": {"role": "assistant", '
                    '"content": "Hi there! How can I assist you today?"}, "finish_reason": "stop"}], '
                    '"usage": {"prompt_tokens": 19, "completion_tokens": 10, "total_tokens": 29, '
                    '"prompt_tokens_details": {"cached_tokens": 0}, '
                    '"completion_tokens_details": {"reasoning_tokens": 0, "accepted_prediction_tokens": 0, '
                    '"rejected_prediction_tokens": 0}}, "system_fingerprint": "fp"}', 'Fingerprint')]
        self.assertEqual(entries, control)
        con.execute(f"DELETE FROM {config.db_reproducibility_table_name} WHERE ID = '1'")
        check_res: list = con.execute(f"SELECT * FROM {config.db_reproducibility_table_name}").fetchall()
        self.assertEqual(len(check_res), 0)
        con.close()

    def test_combination_of_write_and_find(self):
        content_for_gpt: str = "This is a test request."
        message: Iterable = [
            {"role": "system", "content": "We will work with databases and queries in SQL."},
            {
                "role": "user",
                "content": content_for_gpt
            }
        ]
        response_chat: dict = {
            "id": "chatcmpl-123456",
            "object": "chat.completion",
            "created": 100,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hi there! How can I assist you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 19,
                "completion_tokens": 10,
                "total_tokens": 29,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            },
            "system_fingerprint": "fp"
        }  # Taken as an example from https://platform.openai.com/docs/api-reference/chat/object
        write_new_entry('1', 'Model', message, response_chat, "Fingerprint", True)
        res_dict: dict = check_for_entry('Model', message, True)
        self.assertEqual(res_dict, response_chat)
        con = duckdb.connect(calculate_reproducibility_db_path(True))
        con.execute(f"DELETE FROM {config.db_reproducibility_table_name} WHERE ID = '1'")
        check_res: list = con.execute(f"SELECT * FROM {config.db_reproducibility_table_name}").fetchall()
        self.assertEqual(len(check_res), 0)
        con.close()


