# TODO documentation, use

class Phase2Statistics:

    runtime: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    current_rewrites: list[str] = list()
    num_produced_rewrites: int = 0
    llm_prompts: list[str] = list()
    llm_answers: list[str] = list()

    def __init__(self):
        pass