# TODO documentation, use

class Phase1Statistics:

    runtime: float = 0
    num_selected_tables: int = 0
    selected_tables: dict[str,list[str]] = dict()
    input_tokens: int = 0
    output_tokens: int = 0
    sllm_num_requests_llm: int = 0
    llm_prompts: list[str] = list()
    llm_answers: list[str] = list()


    def __init__(self):
        pass