"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""
from typing import Dict, Iterable, List, Any, Optional, Union, Tuple
from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""

import os
from tree_sitter import Language, Parser

language_build_path = os.path.join(os.path.dirname(__file__)+'/../../preprocessing/', 'py-tree-sitter.so')
PY_LANGUAGE = Language(language_build_path, 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

def assertion_to_test(assertion: str) -> str:
    """ get rid of the expected results in the assertion """
    program_bytes = bytes(assertion, 'utf-8')
    parsed_tree = parser.parse(program_bytes)

    root_node = parsed_tree.root_node
    assert len(root_node.children) == 1

    assert_stmt = root_node.children[0]
    assert assert_stmt.type == "assert_statement"
    # assert len(assert_stmt.children) == 2 # NOTE: it might break if something like "assert a == b,c"

    comparison_stmt = assert_stmt.children[1]
    assert comparison_stmt.type == "comparison_operator"
    assert len(comparison_stmt.children) == 3

    call_stmt = comparison_stmt.children[0]
    while call_stmt.type == "parenthesized_expression":
        assert len(call_stmt.children) == 3
        call_stmt = call_stmt.children[1]
    assert call_stmt.type == "call"

    call_str = program_bytes[call_stmt.start_byte:call_stmt.end_byte].decode("utf-8").strip()

    return call_str

def mbpp_example_to_demonstration(example: Dict[str, Any], 
                                  train: bool = True, 
                                  add_assertion_n: int = 0, 
                                  test_input_only: bool = False,
                                  ) -> str:
    # get the assertions
    if not test_input_only:
        assertion_header = '# These are the assertions for your function:\n'
        for test_case in example['test_list'][:add_assertion_n]:
            assertion_header += test_case + '\n'
    else:
        assertion_header = '# These are the calls for your function:\n'
        for test_case in example['test_list'][:add_assertion_n]:
            assertion_header += assertion_to_test(test_case) + '\n'

    # separate the function header and the function body
    #func_signature = example["func_signature"]
    #func_body = example["func_body"]

    func_comment = f'""" {example["text"]} """'

    header = assertion_header + '\n' + func_comment if add_assertion_n > 0 else func_comment

    if train:
        return f'### Task Start ###\n{header}\n{example["code"]}\n### Task End ###'
    else:
        return f'### Task Start ###\n{header}'

def saved_promptify_mbpp(
        prompt_file: str, 
        example: Dict[str, Any], 
        add_assertion_n: int,
        test_input_only: bool
        ) -> str:
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    return prompt + "\n\n" + mbpp_example_to_demonstration(example, train=False, add_assertion_n=add_assertion_n, 
                                                           test_input_only=test_input_only)

class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```",
                        "### Task End ###"], # NOTE: ### Task End ### is added to the stop words because of the prompt
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["train"]
        # the wrong split of mbpp can be loaded with old datasets cache
        #assert (
        #    len(dataset) == 500
        #), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        #description = doc["text"]
        #test_example = doc["test_list"][0]
        #prompt = f'"""\n{description}\n{test_example}\n"""\n'
        prompt_file = os.path.join(os.path.dirname(__file__)+'/../../prompt_files/', 'prompt_mbpp.txt')
        prompt = saved_promptify_mbpp(prompt_file, doc, add_assertion_n = 3, test_input_only = False)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])


    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        doc = self.dataset["train"][idx]
        # get prompt for few-shot case and take it out from the generation
        prompt_few_shot = self.get_prompt(doc)
        generation = generation[len(prompt_few_shot) :]
        # get prompt for zero-shot case (with only one test case) and add it to the generation
        description = doc["text"]
        test_example = doc["test_list"][0]
        prompt_zero_shot = f'"""\n{description}\n{test_example}\n"""\n'
        # add zero-shot prompt to the generation processed by elimiating stop words at the end
        return prompt_zero_shot + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
        )
        return results