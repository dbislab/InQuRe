"""
Test the GPT API key and some API calls, remove skip to execute the tests
"""
import unittest

from query_rewriting.input_processor.check_executability import proposed_tables
from query_rewriting.query_writing.query_writer import write_query_from_tables
from query_rewriting.utilities.find_metadata import llm_find_intent_nl, llm_find_intent_sql, \
    llm_find_intent_keywords_nl, llm_find_intent_keywords_sql


class TestGPT(unittest.TestCase):

    # WARNING: Each execution of one of these tests will cost something as they use the GPT API

    def setUp(self):
        """
        No setup done here
        """
        pass

    @unittest.skip("Uses GPT API, every test costs")
    def test_method_gpt_prompt(self):
        """
        Test the proposal of tables for a NL request.
        """
        pass
        schema: dict = proposed_tables('Which European country would one want to live in?')
        print(f"Schema for european countries:\n {schema}")

    @unittest.skip("Uses GPT API, every test costs")
    def test_method_gpt_rewrite(self):
        """
        Test the writing of a query from an NL request and proposed tables.
        """
        request: str = "Which Thrillers would other people recommend to read? "
        schema: dict = {'books': ['book_title', 'author', 'genre'],
                        'reviews': ['book_id', 'user_rating', 'user_review'],
                        'recommendations': ['book_id', 'recommended_by']}
        sql: str = write_query_from_tables(schema, request)
        print(f"SQL for Thriller recommendations:\n {sql}")

    @unittest.skip("Uses GPT API, every test costs")
    def test_find_intent_nl(self):
        """
        Test what intent is found for a NL request and if the format is right.
        """
        request: str = "Which Thrillers would other people recommend to read? "
        print(f"\nIntent of: {request} ")
        print(llm_find_intent_nl(request))
        print("")

    @unittest.skip("Uses GPT API, every test costs")
    def test_find_intent_sql(self):
        """
        Test what intent is found for a SQL request and if the format is right.
        """
        request: str = "SELECT avg(rent) FROM city GROUP BY district; "
        print(f"\nIntent of: {request} ")
        print(llm_find_intent_sql(request))
        print("")

    @unittest.skip("Uses GPT API, every test costs")
    def test_find_intent_keywords_nl(self):
        """
        Test what intent keywords are found for a NL request and if the format is right.
        """
        request: str = "Which Thrillers would other people recommend to read? "
        print(f"\nIntent keywords of: {request} ")
        print(llm_find_intent_keywords_nl(request, 5))
        print("")

    @unittest.skip("Uses GPT API, every test costs")
    def test_find_intent_keywords_sql(self):
        """
        Test what intent keywords are found for a SQL request and if the format is right.
        """
        request: str = "SELECT avg(rent) FROM city GROUP BY district; "
        print(f"\nIntent keywords of: {request} ")
        print(llm_find_intent_keywords_sql(request, 5))
        print("")

    def tearDown(self):
        """
        No teardown done here
        """
        pass
