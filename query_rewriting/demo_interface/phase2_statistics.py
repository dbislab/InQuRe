"""
Returns statistics from the rewriter after the rewriting is done.
"""

class Phase2Statistics:

    # Runtime in seconds
    runtime: float = 0 # Set in generate_rewrites
    # The number of input tokens used in this phase
    input_tokens: int = 0 # Set in generate_rewrites for simple rewriting and in generate_rewrites and find_metadata for NL rewriting
    # The number of output tokens used in this phase
    output_tokens: int = 0 # Set in generate_rewrites for simple rewriting and in generate_rewrites and find_metadata for NL rewriting
    # The rewrites produced after this phase (unordered)
    current_rewrites: list[str] = list() # Set in generate_rewrites
    # The number of rewrites found for the query
    num_produced_rewrites: int = 0 # Set in generate_rewrites
    # All LLM prompts (one/two (for NL rewriting) can be picked as an example)
    llm_prompts: list[str] = list() # Set in generate_rewrites for simple rewriting and in generate_rewrites and find_metadata for NL rewriting
    # All LLM answers (one/two (for NL rewriting) can be picked as an example)
    llm_answers: list[str] = list() # Set in generate_rewrites for simple rewriting and in generate_rewrites and find_metadata for NL rewriting

    def __init__(self):
        pass