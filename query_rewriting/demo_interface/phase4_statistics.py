# TODO documentation, use

class Phase4Statistics:

    runtime: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    num_queries_needing_correction: int = 0
    num_corrections_rounds_in_total: int = 0
    num_non_correctable_queries: int = 0
    final_rewrites: list[str] = list()
    error_messages: list[str] = list()
    results_of_final_rewrites: list[list] = list()  # a list of all results, where the first list describes the returned columns
    llm_prompts: list[str] = list()
    llm_answers: list[str] = list()

    def __init__(self):
        pass




