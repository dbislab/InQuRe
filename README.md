## InQuRe - The Case for Intent-Based Query Rewriting

This repository contains the rewriting system implementation for intent-oriented rewriting from the Paper "The Case for Intent-Based Query Rewriting" by G. Nicolai, P. Hansert and S. Michel.

## Paper Abstract

With this work, we describe the concept of intent-based query rewriting and present a first viable solution. The aim is to allow rewrites to alter the structure and syntactic outcome of an original query while keeping the obtainable insights intact. This drastically differs from traditional query rewriting, which typically aims to decrease query evaluation time by using strict equivalence rules and optimization heuristics on the query plan.  Rewriting queries to queries that only provide a similar insight but otherwise can be entirely different can remedy inaccessible original data tables due to access control, privacy, or expensive data access regarding monetary cost or remote access. 
In this paper, we put forward INQURE, a system designed for INtent-based QUery REwriting. It uses access to a large language model (LLM) for the query understanding and human-like derivation of alternate queries. Around the LLM, INQURE employs upfront table filtering  and subsequent candidate rewrite pruning and ranking. We report on the results of an evaluation using a benchmark set of over 900 database table schemas and discuss the pros and cons of alternate approaches regarding runtime and quality of the rewrites of a user study.

## Setup

Python version: 3.12.5  
Requirements: Found in requirements.txt  
Main Packages:

- openai (for GPT)
- duckdb (as DB)
- sentence-transformers (for embeddings)
- nltk (as a toolkit to try out things, [Info on data](https://www.nltk.org/install.html))
- spacy (for word embeddings)
- sql-metadata (for SQL query parsing)

Downloads:

- nltk wordnet is downloaded during the tests for synonyms and antonyms
- en_core_web_lg

### Virtual environment

```python
# Create Environment myvenv
python -m venv myvenv
# Activate environment
myenv\Scripts\activate #Windows
source myenv/bin/activate #Linux
# Install dependencies
pip install package # Single package (optional with --upgrade to just upgrade)
pip install -r requirements.txt # All required packages
# Rewrite requirements.txt
pip freeze > requirements.txt
# Deactivate environment
deactivate
```

Download Word2Vec model from spacy (in the active virtual environment):

```python
python -m spacy download en_core_web_lg
```

This download should happen via the requirements.txt as well,
but if it does not you need to download it using the above python command.

### Database

A working DuckDB database is needed for the execution (has to be given with its path in the config).
It is possible to use any wanted schema in the database for this.  
The file has to end on '.db'.  
If there are no tables in the database, an error will be thrown, as no executable queries can be produced then.  
The used database from the paper can be found in resources/spider_with_prefixes.db.

### GPT API

- An API key for GPT is needed to execute the code
- API key must be set as environment variable via:

```python
setx OPENAI_API_KEY "your_api_key_here" #Windows PowerShell
export OPENAI_API_KEY="your_api_key_here" #Linux/macOS
```

- With environment variable (used in code):
  every new OpenAI object needs the parameter 'api_key=os.environ.get("OPENAI_API_KEY")'
    - This is the default mode from the OpenAI package
- Without environment variable:
  every new OpenAI object needs the parameter 'api_key="\<your OpenAI API key\>"' (not recommended)

## Usage

There are two main methods to run:
- main.py contains the execution method for rewriting
- tests/test_query_rewriting.py contains some test cases for multiple methods

### File Paths

File paths are set as relative paths in the config file with respect to the file where the main method is executed (e.g. for testing the relative path from "test_query_rewriting.py" to the test database).
These paths only work with the current structure of the repository.  
The paths can however be changed to, e.g., absolute paths, depending on where you put your data.

The main method (main.py) can be executed with multiple parameters.
It is possible to change those parameters in the config file (line 9 to 38).


Structure of the input file (example from the paper in resources/spider_with_prefixes_input.txt):

- Can contain both natural language requests (not yet supported) and SQL queries
- Every request is preceded by either a line containing the keyword 'SQL' or 'Natural Language'
- Requests are separated by an empty line

### Run Configurations

Prefilter kinds:

- 1: simple prefilter using a scipy vector embedding to compare similarity of the tables in the database and the query
  (take those above 0.4 similarity)
- 2: simple prefilter using the LLM: It is given all tables of the database and asked which ones are useful.
- 3: complex LLM version: Ask LLM for tables one could use and then search in the schema for them.

Rewrite kinds:

- 1: simple rewriter that uses a zero-shot prompt for the LLM
- 2: simple rewriter that first gets the intent of the SQL query and then uses it to generate alternative queries

Ranker kinds:

- 2: simple ranker using the defined intent similarity function
- 3: MMR using the defined string and intent similarities

Distance Function kinds:  
String similarity:

- 1: Inbuilt python comparison using difflib

Intent similarity:

- 1: Similarity using embeddings of the used tables
- 2: Similarity by asking the LLM for similarity values

Output:

- Used Parameters:
    - All the parameters given in the config file or as command line input
- For each query:
    - Query and Type
    - Result or error thrown at execution
    - Filtered tables in pre-filtering
    - Alternative queries
    - Whether queries needed correction and if so the corresponding prompt
    - Top-k rewrites with results
    - Prompt tokens (every time API call is made): Cumulative sum over all API calls of all queries until this point
      in this stage of the rewriting process
- Statistics:
    - Number of queries for different cases
    - Timings of different workflow parts
    - Tokens for the API calls used in total for this program run

There are more options for some of the parameters not mentioned in the paper. These were tested but deemed less interesting than those shown in the paper.

## Code Overview

Table Filtering and Rewriting: rewrite_generator  
Pruning and Ranking: ranking  
Distance Functions for Ranking: distance_measures  
Correction: postprocessor  
Automated tests: tests

## Evaluation

The code can be run in the different configurations from the paper.
Note that due to the LLM usage the code is not deterministic.

The data from the user study was not uploaded.
