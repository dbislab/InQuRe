"""
Returns statistics from the rewriter after the rewriting is done.
"""
# TODO use and note down with comment where each param is set to not lose oversight

class Phase2Statistics:

    # Runtime in seconds
    runtime: float = 0
    # The number of input tokens used in this phase
    input_tokens: int = 0
    # The number of output tokens used in this phase
    output_tokens: int = 0
    # The rewrites produced after this phase (unordered)
    current_rewrites: list[str] = list()
    # The number of rewrites found for the query
    num_produced_rewrites: int = 0
    # All LLM prompts (one/two (for NL rewriting) can be picked as an example)
    llm_prompts: list[str] = list()
    # All LLM answers (one/two (for NL rewriting) can be picked as an example)
    llm_answers: list[str] = list()

    def __init__(self):
        pass