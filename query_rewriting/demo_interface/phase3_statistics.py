"""
Returns statistics from the rewriter after the ranking is done.
"""
# TODO use and note down with comment where each param is set to not lose oversight

class Phase3Statistics:
    # LLM is only used by the LLM-based similarity measure in this phase

    # Runtime in seconds
    runtime: float = 0 # Set in execute_query_rewriting in main
    # The number of input tokens used in this phase
    input_tokens: int = 0 # Set in sql_queries_comparison for llm_intent_similarity_measure
    # The number of output tokens used in this phase
    output_tokens: int = 0 # Set in sql_queries_comparison for llm_intent_similarity_measure
    # The number of requests to the LLM needed for the LLMS similarity function
    llms_num_requests_to_LLM : int = 0 # Set in sql_queries_comparison for llm_intent_similarity_measure
    # The number of queries that were pruned
    num_pruned_queries: int = 0 # Set in rank_alternatives
    # The current rewrites after this step (ordered)
    ranked_queries: list[str] = list() # Set in execute_query_rewriting in main
    # All LLM prompts for the LLMS similarity function (one can be picked as an example)
    llm_prompts: list[str] = list() # Set in sql_queries_comparison for llm_intent_similarity_measure
    # All LLM answers (one can be picked as an example)
    llm_answers: list[str] = list() # Set in sql_queries_comparison for llm_intent_similarity_measure

    def __init__(self):
        pass