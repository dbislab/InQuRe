"""
Returns statistics from the rewriter after the table filtering is done.
"""
# TODO use and note down with comment where each param is set to not lose oversight

class Phase1Statistics:

    # Runtime in seconds
    runtime: float = 0 # Set in generate_rewrites
    # Number of selected tables from the filter
    num_selected_tables: int = 0 # Set in generate_rewrites
    # The selected tables in the format {table:[column1 type1, column2 type2]}
    selected_tables: dict[str,list[str]] = dict() # Set in generate_rewrites
    # The number of input tokens used in this phase
    input_tokens: int = 0 # Set in prefilter_tables_llm for simple and complex filter
    # The number of output tokens used in this phase
    output_tokens: int = 0 # Set in prefilter_tables_llm for simple and complex filter
    # The number of requests sent to the SLLM filter (if it was used, otherwise it stays at 0)
    sllm_num_requests_llm: int = 0 # Set in prefilter_tables_llm for simple filter
    # All LLM prompts (one can be picked as an example)
    llm_prompts: list[str] = list() # Set in prefilter_tables_llm for simple and complex filter
    # All LLM answers (one can be picked as an example)
    llm_answers: list[str] = list() # Set in prefilter_tables_llm for simple and complex filter


    def __init__(self):
        pass