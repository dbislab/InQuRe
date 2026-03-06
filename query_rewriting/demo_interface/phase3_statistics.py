# TODO documentation, use

class Phase3Statistics:

    runtime: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llms_num_requests_to_LLM : int = 0
    num_pruned_queries: int = 0
    ranked_queries: list[str] = list()
    llm_prompts: list[str] = list()
    llm_answers: list[str] = list()

    def __init__(self):
        pass