"""
Test the functions that are written in the utilities package for GPT (except the GPT API call)
"""
import unittest

from query_rewriting.utilities.gpt_functions import strip_sql_output, strip_preceding_keywords, \
    strip_starting_and_ending_characters, prepare_db_schema_for_prompt, strip_code_block_output, \
    prepare_db_schema_for_prompt_including_fk


class TestUtilitiesGPT(unittest.TestCase):

    def test_sql_stripping(self):
        """
        Test if stripping the Markdown annotations from SQL works
        """
        markdown: str = ("\n```sql SELECT b.book_title, b.author \n FROM books b\n "
                         "JOIN recommendations r ON b.book_id = r.book_id WHERE b.genre = 'Thriller';```\n")
        stripped_md: str = strip_sql_output(markdown)
        sql: str = ("SELECT b.book_title, b.author FROM books b JOIN recommendations r ON b.book_id = r.book_id "
                    "WHERE b.genre = 'Thriller';")
        self.assertEqual(stripped_md, sql)

    def test_sql_stripping_without_sql_annotator(self):
        """
        Test if stripping the Markdown annotations from SQL works, even if the SQL tag is not there
        """
        markdown: str = ("``` SELECT b.book_title, b.author FROM books b JOIN recommendations r "
                         "ON b.book_id = r.book_id WHERE b.genre = 'Thriller';```")
        stripped_md: str = strip_sql_output(markdown)
        sql: str = ("SELECT b.book_title, b.author FROM books b JOIN recommendations r ON b.book_id = r.book_id "
                    "WHERE b.genre = 'Thriller';")
        self.assertEqual(stripped_md, sql)

    def test_strip_preceding_keywords(self):
        """
        Test if stripping certain keywords at the start of the string works
        """
        markdown: str = "Intent: I want to test this."
        stripped_md: str = strip_preceding_keywords(markdown, "Intent:")
        correct_md: str = "I want to test this."
        self.assertEqual(stripped_md, correct_md)

    def test_strip_preceding_keywords_keyword_not_there(self):
        """
        Test if stripping certain keywords at the start of the string works only for the correct keyword
        """
        markdown: str = "Idea: I want to test this."
        stripped_md: str = strip_preceding_keywords(markdown, "Intent:")
        self.assertEqual(stripped_md, markdown)

    def test_strip_starting_and_ending_characters(self):
        """
        Test if stripping certain keywords at the start of the string works
        """
        markdown: str = "I want to test this."
        stripped_md: str = strip_starting_and_ending_characters(markdown)
        correct_md: str = "I want to test this"
        self.assertEqual(stripped_md, correct_md)

    def test_strip_starting_and_ending_characters_not_there(self):
        """
        Test if stripping certain keywords at the start of the string works only for the correct keyword
        """
        markdown: str = "Intents: I want to test this"
        stripped_md: str = strip_starting_and_ending_characters(markdown)
        self.assertEqual(stripped_md, markdown)

    def test_strip_code_block_output(self):
        """
        Test if stripping the Markdown annotations from a code block works.
        """
        markdown1: str = """```This is a string with 
        multiple lines.```"""
        markdown2: str = """```sqlThis is a string with 
        multiple lines.```"""
        markdown3: str = """```cssThis is a string with 
        multiple lines.```"""
        markdown4: str = """```plaintext
        This is a string with 
        multiple lines.```"""
        stripped_md1: str = strip_code_block_output(markdown1)
        stripped_md2: str = strip_code_block_output(markdown2)
        stripped_md3: str = strip_code_block_output(markdown3)
        stripped_md4: str = strip_code_block_output(markdown4)
        control: str = """This is a string with 
        multiple lines."""
        self.assertEqual(stripped_md1, control)
        self.assertEqual(stripped_md2, control)
        self.assertEqual(stripped_md3, control)
        self.assertEqual(stripped_md4, control)

    def test_prepare_db_schema_for_prompt(self):
        """
        Test if the schema is correctly written as a string
        """
        test_schema: dict = {"table1": ["column1 t1", "column2 t2", "column3 t3"], "table2": ["column1 t1"],
                             "table3": ["column1 t1", "column2 t2"]}
        control_string: str = ("table1: column1 t1, column2 t2, column3 t3\n"
                               "table2: column1 t1\n"
                               "table3: column1 t1, column2 t2\n")
        test_string = prepare_db_schema_for_prompt(test_schema)
        self.assertEqual(test_string, control_string)

    def test_prepare_db_schema_for_prompt_including_fk(self):
        """
        Test if the schema is correctly written as a string including the foreign keys
        """
        test_schema: dict = {"table1": ["column1 t1", "column2 t2", "column3 t3"], "table2": ["column1 t1"],
                             "table3": ["column1 t1", "column2 t2"]}
        control_string: str = ("table1: column1 t1, column2 t2, column3 t3\n"
                               "No Foreign Keys\n"
                               "table2: column1 t1\n"
                               "No Foreign Keys\n"
                               "table3: column1 t1, column2 t2\n"
                               "No Foreign Keys\n")
        test_string = prepare_db_schema_for_prompt_including_fk(test_schema, dict())
        self.assertEqual(test_string, control_string)
        test_foreign_keys: dict = {'table1': ['FOREIGN KEY (column1) REFERENCES table2(column1)',
                                              'FOREIGN KEY (column2) REFERENCES table3(column1)'],
                                   'table3': ['FOREIGN KEY (column1) REFERENCES table2(column1)']}
        control_string2: str = ("table1: column1 t1, column2 t2, column3 t3\n"
                                "Foreign Keys: FOREIGN KEY (column1) REFERENCES table2(column1), "
                                "FOREIGN KEY (column2) REFERENCES table3(column1)\n"
                                "table2: column1 t1\n"
                                "No Foreign Keys\n"
                                "table3: column1 t1, column2 t2\n"
                                "Foreign Keys: FOREIGN KEY (column1) REFERENCES table2(column1)\n")
        test_string2: str = prepare_db_schema_for_prompt_including_fk(test_schema, test_foreign_keys)
        self.assertEqual(test_string2, control_string2)

