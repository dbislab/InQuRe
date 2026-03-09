"""
Returns statistics from the rewriter after the ranking is done.
"""
# TODO use and note down with comment where each param is set to not lose oversight

class Phase3Statistics:

    # Runtime in seconds
    runtime: float = 0
    # The number of input tokens used in this phase
    input_tokens: int = 0
    # The number of output tokens used in this phase
    output_tokens: int = 0
    # The number of requests to the LLM needed for the LLMS similarity function
    llms_num_requests_to_LLM : int = 0
    # The number of queries that were pruned
    num_pruned_queries: int = 0
    # The current rewrites after this step (ordered)
    ranked_queries: list[str] = list()
    # All LLM prompts for the LLMS similarity function (one can be picked as an example)
    llm_prompts: list[str] = list()
    # All LLM answers (one can be picked as an example)
    llm_answers: list[str] = list()

    def __init__(self):
        pass