"""
Returns statistics from the rewriter after the whole process is finished.
"""
# TODO use and note down with comment where each param is set to not lose oversight

class Phase4Statistics:

    # Runtime in seconds
    runtime: float = 0
    # The number of input tokens used in this phase
    input_tokens: int = 0
    # The number of output tokens used in this phase
    output_tokens: int = 0
    # The number of rewrites that needed to be corrected
    num_queries_needing_correction: int = 0
    # The number of corrections needed in total for all rewrites
    num_corrections_rounds_in_total: int = 0
    # The number of queries that were not correctable
    num_non_correctable_queries: int = 0
    # The final rewrites (ordered)
    final_rewrites: list[str] = list()
    # The error messages from the final rewrites in an ordered fashion (empty if the rewrite is executable)
    error_messages: list[str] = list()
    # The results of the final rewrites (ordered), where the first list in the list describes the returned columns
    results_of_final_rewrites: list[list] = list()
    # All LLM correction prompts (one can be picked as an example)
    llm_prompts: list[str] = list()
    # All LLM answers (one can be picked as an example)
    llm_answers: list[str] = list()

    def __init__(self):
        pass




