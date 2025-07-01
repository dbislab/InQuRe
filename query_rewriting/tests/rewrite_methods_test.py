"""
Test prefilter methods from the rewriting package.
"""
import unittest
import duckdb

import query_rewriting.config as config
from query_rewriting.rewrite_generator.prefilter_tables import simple_prefilter_via_scipy_embedding, add_joining_tables
from query_rewriting.utilities.duckdb_functions import get_tables_from_db


class TestRewriteMethods(unittest.TestCase):

    def setUp(self):
        """
        Set up.
        """
        pass

    def test_simple_prefilter_via_scipy_embedding(self):
        """
        Test if always the same tables are returned.
        """
        query: str = "SELECT count(*) FROM student;"
        schema: dict = {
            "students": ["matrnr", "name", "semester"],
            "studentLectures": ["matrnr", "lecture"],
            "lecture": ["lecture", "title", "description", "professor"],
            "professor": ["professor", "name", "office"],
            "exams": ["student", "professor", "lecture", "date"],
            "student_info": ["matrnr", "address", "birthDate"]
        }
        res: dict = simple_prefilter_via_scipy_embedding(query, schema)
        list1: list[str] = res.pop("students", [])
        list2: list[str] = res.pop("lecture", [])
        list3: list[str] = res.pop("professor", [])
        self.assertCountEqual(list1, schema["students"])
        self.assertCountEqual(list2, schema["lecture"])
        self.assertCountEqual(list3, schema["professor"])
        self.assertEqual(res, dict())

    def test_adding_joining_tables(self):
        """
        Test the method that adds tables that join two found tables together.
        Test if it adds tables correctly.
        """
        # Set up
        con = duckdb.connect(config.test_db_file)
        # Check that nothing happened without tables
        result: dict = add_joining_tables({'tbl1': ['i INTEGER', 'j VARCHAR']},
                                          {'tbl1': ['i INTEGER', 'j VARCHAR'],
                                           'tbl2': ['k INTEGER', 'l VARCHAR', 'm DATE']}, True)
        self.assertCountEqual(result.get('tbl1'), ['i INTEGER', 'j VARCHAR'])
        self.assertEqual(len(result), len({'tbl1': ['i INTEGER', 'j VARCHAR']}))
        # Tables connected via one other table with keys
        con.execute("CREATE TABLE IF NOT EXISTS tbl1 "
                    "(i INTEGER PRIMARY KEY, j VARCHAR);")
        con.execute("CREATE TABLE IF NOT EXISTS tbl2 "
                    "(k INTEGER PRIMARY KEY, l VARCHAR, m DATE);")
        con.execute("CREATE TABLE IF NOT EXISTS connector1 "
                    "(a INTEGER, b INTEGER, FOREIGN KEY (a) REFERENCES 'tbl1'(i), "
                    "FOREIGN KEY (b) REFERENCES 'tbl2' (k));")
        # Tables reference same table
        con.execute("CREATE TABLE IF NOT EXISTS connector2 "
                    "(a INTEGER PRIMARY KEY, b INTEGER UNIQUE)")
        con.execute('CREATE TABLE IF NOT EXISTS tbl3 '
                    '(i INTEGER, j VARCHAR, FOREIGN KEY (i) references "connector2"(a));')
        con.execute('CREATE TABLE IF NOT EXISTS tbl4 '
                    '(k INTEGER, l VARCHAR, m DATE, FOREIGN KEY (k) REfErences "connector2" (b));')
        # One table references, the other is referenced
        con.execute("CREATE TABLE IF NOT EXISTS tbl6 "
                    "(k INTEGER PRIMARY KEY, l VARCHAR, m DATE);")
        con.execute("CREATE TABLE IF NOT EXISTS connector3 "
                    "(a INTEGER PRIMARY KEY , b INTEGER, FOREIGN KEY (b) REFERENCES tbl6(k))")
        con.execute("CREATE TABLE IF NOT EXISTS tbl5 "
                    "(i INTEGER, j VARCHAR, FOREIGN KEY (i) REFERENCES connector3 (a));")
        # The other way around
        con.execute("CREATE TABLE IF NOT EXISTS tbl7 "
                    "(i INTEGER PRIMARY KEY , j VARCHAR);")
        con.execute("CREATE TABLE IF NOT EXISTS connector4 "
                    "(a INTEGER, b INTEGER PRIMARY KEY, FOREIGN KEY (a) REFERENCES tbl7(i))")
        con.execute("CREATE TABLE IF NOT EXISTS tbl8 "
                    "(k INTEGER, l VARCHAR, m DATE, FOREIGN KEY (k) REFERENCES connector4 (b));")
        # Test what happens if they reference the same column
        con.execute("CREATE TABLE IF NOT EXISTS connector5 "
                    "(a INTEGER PRIMARY KEY , b INTEGER)")
        con.execute("CREATE TABLE IF NOT EXISTS tbl9 "
                    "(k INTEGER PRIMARY KEY, l VARCHAR, m DATE, FOREIGN KEY (k) REFERENCES connector5 (a));")
        con.execute("CREATE TABLE IF NOT EXISTS tbl10 "
                    "(i INTEGER, j VARCHAR, FOREIGN KEY (i) REFERENCES connector5 (a));")
        con.close()
        # Create the cases for the function (db tables and found tables in prefilter)
        db_tables: dict = get_tables_from_db(True)
        prefiltered_tables_1: dict = {'tbl1': ['i INTEGER', 'j VARCHAR'], 'tbl2': ['k INTEGER', 'l VARCHAR', 'm DATE']}
        prefiltered_tables_2: dict = {'tbl3': ['i INTEGER', 'j VARCHAR'], 'tbl4': ['k INTEGER', 'l VARCHAR', 'm DATE']}
        prefiltered_tables_3: dict = {'tbl5': ['i INTEGER', 'j VARCHAR'], 'tbl6': ['k INTEGER', 'l VARCHAR', 'm DATE']}
        prefiltered_tables_4: dict = {'tbl7': ['i INTEGER', 'j VARCHAR'], 'tbl8': ['k INTEGER', 'l VARCHAR', 'm DATE']}
        prefiltered_tables_5: dict = {'tbl9': ['i INTEGER', 'j VARCHAR'], 'tbl10': ['k INTEGER', 'l VARCHAR', 'm DATE']}
        # Test the functions
        result1: dict = add_joining_tables(prefiltered_tables_1, db_tables, True)
        result2: dict = add_joining_tables(prefiltered_tables_2, db_tables, True)
        result3: dict = add_joining_tables(prefiltered_tables_3, db_tables, True)
        result4: dict = add_joining_tables(prefiltered_tables_4, db_tables, True)
        result5: dict = add_joining_tables(prefiltered_tables_5, db_tables, True)
        tables_after_addition1: dict = prefiltered_tables_1
        tables_after_addition1.update({'connector1': ['a INTEGER', 'b INTEGER']})
        tables_after_addition2: dict = prefiltered_tables_2
        tables_after_addition2.update({'connector2': ['a INTEGER', 'b INTEGER']})
        tables_after_addition3: dict = prefiltered_tables_3
        tables_after_addition3.update({'connector3': ['a INTEGER', 'b INTEGER']})
        tables_after_addition4: dict = prefiltered_tables_4
        tables_after_addition4.update({'connector4': ['a INTEGER', 'b INTEGER']})
        tables_after_addition5: dict = prefiltered_tables_5
        tables_after_addition5.update({'connector5': ['a INTEGER', 'b INTEGER']})
        # Checks
        self.assertTrue(all([set(result1.get(key)) == set(tables_after_addition1.get(key))
                             for key in result1.keys() if key in tables_after_addition1.keys()]))
        self.assertTrue(all([set(result1.get(key)) == set(tables_after_addition1.get(key))
                             for key in tables_after_addition1.keys() if key in result1.keys()]))
        self.assertTrue(all([set(result2.get(key)) == set(tables_after_addition2.get(key))
                             for key in result2.keys() if key in tables_after_addition2.keys()]))
        self.assertTrue(all([set(result2.get(key)) == set(tables_after_addition2.get(key))
                             for key in tables_after_addition2.keys() if key in result2.keys()]))
        self.assertTrue(all([set(result3.get(key)) == set(tables_after_addition3.get(key))
                             for key in result3.keys() if key in tables_after_addition3.keys()]))
        self.assertTrue(all([set(result3.get(key)) == set(tables_after_addition3.get(key))
                             for key in tables_after_addition3.keys() if key in result3.keys()]))
        self.assertTrue(all([set(result4.get(key)) == set(tables_after_addition4.get(key))
                             for key in result4.keys() if key in tables_after_addition4.keys()]))
        self.assertTrue(all([set(result4.get(key)) == set(tables_after_addition4.get(key))
                             for key in tables_after_addition4.keys() if key in result4.keys()]))
        self.assertTrue(all([set(result5.get(key)) == set(tables_after_addition5.get(key))
                             for key in result5.keys() if key in tables_after_addition5.keys()]))
        self.assertTrue(all([set(result5.get(key)) == set(tables_after_addition5.get(key))
                             for key in tables_after_addition5.keys() if key in result5.keys()]))
        # Tear down
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE connector1;")
        con.sql("DROP TABLE tbl1;")
        con.sql("DROP TABLE tbl2;")
        con.sql("DROP TABLE tbl3;")
        con.sql("DROP TABLE tbl4;")
        con.sql("DROP TABLE connector2;")
        con.sql("DROP TABLE tbl5;")
        con.sql("DROP TABLE connector3;")
        con.sql("DROP TABLE tbl6;")
        con.sql("DROP TABLE tbl8;")
        con.sql("DROP TABLE connector4;")
        con.sql("DROP TABLE tbl7;")
        con.sql("DROP TABLE tbl9;")
        con.sql("DROP TABLE tbl10;")
        con.sql("DROP TABLE connector5;")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()

    def test_adding_joined_tables_none_there(self):
        """
        Test the method that adds tables that join two found tables together.
        Test if it does not add tables that are not wanted.
        """
        # Set up
        con = duckdb.connect(config.test_db_file)
        # Test what happens if they reference different tables and those reference each other
        con.execute("CREATE TABLE IF NOT EXISTS connector3 "
                    "(a INTEGER, b INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE IF NOT EXISTS connector1 "
                    "(a INTEGER PRIMARY KEY, b INTEGER, FOREIGN KEY (b) REFERENCES connector3 (b))")
        con.execute("CREATE TABLE IF NOT EXISTS connector2 "
                    "(a INTEGER PRIMARY KEY, b INTEGER, FOREIGN KEY (b) REFERENCES connector3 (b))")
        con.execute("CREATE TABLE IF NOT EXISTS tbl1 "
                    "(k INTEGER PRIMARY KEY, l VARCHAR, m DATE, FOREIGN KEY (k) REFERENCES connector1 (a));")
        con.execute("CREATE TABLE IF NOT EXISTS tbl2 "
                    "(i INTEGER, j VARCHAR, FOREIGN KEY (i) REFERENCES connector2 (a));")
        con.close()
        # Create the cases for the function (db tables and found tables in prefilter)
        db_tables: dict = get_tables_from_db(True)
        prefiltered_tables_1: dict = {'tbl1': ['i INTEGER', 'j VARCHAR'], 'tbl2': ['k INTEGER', 'l VARCHAR', 'm DATE']}
        # Test the functions
        result1: dict = add_joining_tables(prefiltered_tables_1, db_tables, True)
        tables_after_addition1: dict = prefiltered_tables_1
        # Checks
        self.assertTrue(all([set(result1.get(key)) == set(tables_after_addition1.get(key))
                             for key in result1.keys() if key in tables_after_addition1.keys()]))
        self.assertTrue(all([set(result1.get(key)) == set(tables_after_addition1.get(key))
                             for key in tables_after_addition1.keys() if key in result1.keys()]))
        # Tear down
        con = duckdb.connect(config.test_db_file)
        con.sql("DROP TABLE tbl1;")
        con.sql("DROP TABLE tbl2;")
        con.sql("DROP TABLE connector1;")
        con.sql("DROP TABLE connector2;")
        con.sql("DROP TABLE connector3;")
        con.execute("SHOW ALL TABLES;")
        assert con.fetchone() is None
        con.close()

    def tearDown(self):
        """
        Tear down.
        """
        pass