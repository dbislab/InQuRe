"""
Returns statistics from the rewriter after the whole process is finished.
"""

class Phase4Statistics:

    # Runtime in seconds
    runtime: float = 0 # Set in execute_query_rewriting in main
    # The number of input tokens used in this phase
    input_tokens: int = 0 # Set in gentle_self_correction in query_correction
    # The number of output tokens used in this phase
    output_tokens: int = 0 # Set in gentle_self_correction in query_correction
    # The number of rewrites that needed to be corrected
    num_queries_needing_correction: int = 0 # Set in execute_query_rewriting in main
    # The number of corrections needed in total for all rewrites
    num_corrections_rounds_in_total: int = 0 # Set in execute_query_rewriting in main
    # The number of queries that were not correctable
    num_non_correctable_queries: int = 0 # Set in execute_query_rewriting in main
    # The final rewrites (ordered)
    final_rewrites: list[str] = list() # Set in execute_query_rewriting in main
    # The error messages from the final rewrites in an ordered fashion (empty if the rewrite is executable)
    error_messages: list[str] = list() # Set in execute_query_rewriting in main
    # The results of the final rewrites (ordered), where the first list in the list describes the returned columns
    results_of_final_rewrites: list[list] = list() # Set in execute_query_rewriting in main
    # All LLM correction prompts (one can be picked as an example)
    llm_prompts: list[str] = list() # Set in gentle_self_correction in query_correction
    # All LLM answers (one can be picked as an example)
    llm_answers: list[str] = list() # Set in gentle_self_correction in query_correction

    def __init__(self):
        pass




