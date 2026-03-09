# TODO documentation, use
"""
Returns statistics from the rewriter after the table filtering is done.
"""

class Phase1Statistics:

    # Runtime in seconds
    runtime: float = 0
    # Number of selected tables from the filter
    num_selected_tables: int = 0
    # The selected tables in the format {table:[column1 type1, column2 type2]}
    selected_tables: dict[str,list[str]] = dict()
    # The number of input tokens used in this phase
    input_tokens: int = 0
    # The number of output tokens used in this phase
    output_tokens: int = 0
    # The number of requests sent to the SLLM filter (if it was used, otherwise it stays at 0)
    sllm_num_requests_llm: int = 0
    # All LLM prompts (one can be picked as an example)
    llm_prompts: list[str] = list()
    # All LLM answers (one can be picked as an example)
    llm_answers: list[str] = list()


    def __init__(self):
        pass