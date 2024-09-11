"""
author: lmp-decaderan
email: ldecaderan@gmail.com

reviewed: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import random
import torch
import numpy as np

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from functools import partial
from pydantic import field_validator

from vllm.outputs import CompletionOutput, RequestOutput

# from mcts_math.agents.utils import math_is_equiv as is_equiv
from math_evaluation import is_equiv

def math_is_equiv(grt: Union[str, list[str]], prd: str):
    prd = remove_single_dollar(prd)
    if isinstance(grt, list):
        for g in grt:
            if is_equiv(remove_single_dollar(g), prd):
                return True
        return False
    else:
        return is_equiv(remove_single_dollar(grt), prd)
      
from nodes import MCTSNode
from constants import (
    TOO_MANY_CODE_ERRORS, 
    TOO_MANY_STEPS, 
    NO_VALID_CHILD, 
    SOLUTION_COLOR, 
    OBSERVATION_COLOR,
    WARNING_COLOR,
)

# from .tree import BaseTree, code_execution
# from .step_beam import SBSREACT


class MCTS(SBSREACT): 
    #SBSREACT는 REACT 클래스 wrapper, REACT 클래스는  BaseTree wrapper
    #BaseTree의 abstract 메소드에는 create_node, create_llm, generate함수
    
    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "mcts":
            raise ValueError(f"Wrong value for config mode.")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg

    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent, 
            additional_state_keys=self.REACT_NODE_KEYS,
            c_puct=self.config.c_puct,
        )

    def generate(self) -> None:
        self.search()

    @torch.inference_mode()
    def search(self) -> None:
        for idx in range(self.config.iterations):
            # node selection starting from root
            node = self.selection()
            # expansion_evaluation_backpropagation
            self.expansion_evaluation_backpropagation(node)

    def selection(self) -> Optional[Type[MCTSNode]]:
        node = self.root
        while node.has_children() or node.is_terminal:
            next_node = self.select_child(node)     # To encourage exploration, select from non-terminal children
            if next_node is None:                   # if None，it mean all children are terminal
                node.is_terminal = True
                break
            node = next_node
    
        return None if node.is_terminal else node

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        # TODO: implement multi-strategy
        # select the best child according to the puct
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue

            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        return random.choice(best_childs) if best_childs else None

    def expansion_evaluation_backpropagation(self, node: Type[MCTSNode]) -> None:
        """
        This function is only used for single example inference, required to set `create_local_llm` as True.
        """
        assert self.config.create_local_llm, "llm must be created within MCTS class."
        prompt = self.create_prompt()
        # expand and evaluate
        outputs, value_estimate = self.llm(prompt, n=self.n_generate_sample, stop=self.stop)
        if value_estimate is not None:  # input exceeds 4096, output '' and None
            self.expand_node(outputs, node)
        else:
            value_estimate = self.config.negative_reward
            node.is_terminal = True
        # backup
        node.update_recursive(value_estimate, self.root)

    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
        if self.config.remove_duplicate:
            dedup_outputs = []
            dedup_keys = set()
            for output in outputs:
                key = output.text.strip()
                if not key in dedup_keys:
                    dedup_keys.add(key)
                    dedup_outputs.append(output)
            outputs = dedup_outputs
        for idx, output in enumerate(outputs):
            prior_prob = np.exp(output.cumulative_logprob / len(output.token_ids))
            step_result, parser_result = self.step_unwrap(output.text.strip())
            self.create_child(step_result, parser_result, node, prior_prob, idx)

    def create_child(
        self, 
        step_result: str, 
        parser_result: Dict[str, str], 
        node: Type[MCTSNode],
        prior_prob: float,
        idx: int,
    ) -> None:
        if self.config.verbose:
            print(colored(f"{step_result}\n", SOLUTION_COLOR))
        
        # initialize a new node
        new_node = self.create_node(parent=node)
        new_node.tag = f"{node.tag}.{idx}"
        new_node.depth = node.depth + 1
        new_node.prior = prior_prob

        # update node state
        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["action"]:
            observation = code_execution(node, parser_result)
            observation = self.obs_wrap(observation)

            if self.config.verbose:
                print(colored(f"{observation}\n", OBSERVATION_COLOR))

            new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]

            if "Error" in observation:
                new_node.consecutive_errors = node.consecutive_errors + 1
                if new_node.consecutive_errors > self.config.errors_threshold:
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                    self.eval_final_answer(new_node)
        else:
            if self.config.verbose:
                print(colored(f"WARNING: '{step_result}' Cannot resolve\n", WARNING_COLOR))
            new_node.state["text"] = step_result

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            node.update_recursive(self.config.negative_reward, self.root)
            return 
        
        if self.ground_truth:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer)
            # backup
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
        else:
            # for testset, no ground_truth, put this node in candidate_nodes, then it will be evaluated by value model and backup in select_next_step().
            self.candidate_nodes.append(node)

    def select_next_step(self, outputs: Optional[List[RequestOutput]] = None) -> None:
        """process output from vllm
        e.g.,
        prompts = tree.create_prompt(is_value_only=True)
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.current_nodes = []
        if outputs is not None:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                # assert self.question in output.prompt
                # backup
                if candidate_node.is_terminal and self.ground_truth:
                    continue
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True
                candidate_node.update_recursive(value_estimate, self.root)
                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        selection_node = self.selection()
        if selection_node is not None:
            self.current_nodes.append(selection_node)
    
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            # assert self.question in output.prompt
            # current_step.value = output.value
            # expand n_generate_sample nodes
            value_estimate = output.value_estimate
            if value_estimate is not None:  # input exceeds 4096, output '' and None
                current_node.value = value_estimate
                self.expand_node(output.outputs, current_node)
            else:
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True
            # self.expand_node(output.outputs, current_node)
            # self.candidate_nodes.extend(current_node.children)

            # backup
            if self.config.update_leaf_value:
                # child node will be put into candidate_nodes, then all candidate_nodes will be evaluated by value model and backup in select_next_step().
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node)
            else:
                current_node.update_recursive(value_estimate, self.root)

    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["prior"] = node.prior
            states[node.tag]["visit_count"] = node.visit_count()
            if node.has_children():
                candidates.extend(node.children)
        return states


"""
author: lmp-decaderan
email: ldecaderan@gmail.com

reviewed: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import re

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from functools import partial
from vllm.outputs import RequestOutput

from nodes import BaseNode
from constants import (
    NO_VALID_CHILD, 
    TOO_MANY_STEPS, 
    TOO_MANY_CODE_ERRORS, 
    SOLUTION_COLOR, 
    OBSERVATION_COLOR,
)
# from .tree import BaseTree, code_execution
# from .react import REACT


class SBSREACT(REACT):
    """
    Step-level Beam Search
    """

    current_top_num: int = 1
    current_nodes: List[Type[BaseNode]] = []
    final_answer_nodes: List[Type[BaseNode]] = [] 
    candidate_nodes: List[Type[BaseNode]] = [] 

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.candidate_nodes.append(self.current_node)
        self.current_top_num = self.config.step_beam_width
        self.select_next_step()

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "sbs":
            raise ValueError(f"Wrong value for config mode, must be react")
        if not cfg.n_generate_sample >= 1:
            raise ValueError(f"Wrong value for config n_generate_sample, must be greater than 1")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg
    
    def create_llm(self) -> Callable[[...], List[str]]:
        # we only implement the batch inference
        pass

    def is_ignored_node(self, node: Type[BaseNode]) -> bool:
        return node.is_terminal or node.depth > self.config.max_depth

    def should_generate_next(self) -> bool:
        need_generate = False
        for step_node in self.current_nodes:
            if not self.is_ignored_node(step_node):
                need_generate = True
                break
        return need_generate

    def create_prompt(
        self,
        is_value_only: bool = False,
    ) -> str:
        """
        if is_value_only, the prompt is used to produce value estimate.
        """
        prompts = []
        current_nodes = self.candidate_nodes if is_value_only else self.current_nodes
        for current_node in current_nodes:
            if not is_value_only and self.is_ignored_node(current_node):
                continue
            partial_solution = self.collect_partial_solution(current_node)
            prompt = self.prompt_wrap(
                self.question, 
                partial_solution,
                self.config,
            )
            prompts.append(prompt)
        return prompts

    @staticmethod
    def is_valid_final_answer_node(node: Type[BaseNode]) -> bool:
        # by default, final_anwer = ""
        if node.is_terminal and node.state["final_answer"] and \
           node.state["final_answer"] not in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            return True
        return False

    def select_next_step(self, outputs: Optional[List[RequestOutput]] = None) -> None:
        """process output from vllm
        e.g.,
        prompts = tree.create_prompt(is_value_only=True)
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.current_nodes = []
        if outputs is not None:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                # assert self.question in output.prompt
                candidate_node.value = output.value_estimate if output.value_estimate is not None else -100
            
        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)
        self.current_nodes = self.candidate_nodes[:self.current_top_num]

        for current_node in self.current_nodes[:]:  # must shallow copy because of the remove in the loop 
            if self.__class__.is_valid_final_answer_node(current_node):
                self.final_answer_nodes.append(current_node)
                self.current_nodes.remove(current_node)
                self.current_top_num -= 1
            elif current_node.is_terminal or current_node.depth > self.config.max_depth:
                self.current_nodes.remove(current_node)
                self.current_top_num -= 1
    
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            # assert self.question in output.prompt
            # current_step.value = output.value
            # expand n_generate_sample nodes
            self.current_node = current_node
            current_output_texts = [otp.text.strip() for otp in output.outputs]
            if self.config.remove_duplicate:
                current_output_texts = set(current_output_texts)
            for idx, cur_output_text in enumerate(current_output_texts):
                step_result, parser_result = self.step_unwrap(cur_output_text)
                self._update_current_node(step_result, parser_result, idx)
            self.candidate_nodes.extend(current_node.children)

    def get_steps(self):
        final_answer_states = []
        for cur_node in self.final_answer_nodes:
            states = {
                "question": self.question,
                "ground_truth": self.ground_truth,
                "value": cur_node.value,
                "final_answer": cur_node.state["final_answer"],
                "solution": self.collect_partial_solution(cur_node),
                "tag": cur_node.tag,
            }
            final_answer_states.append(states)

        solutions = sorted(final_answer_states, key=lambda x: x['value'], reverse=True)
        return solutions

    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            if node.has_children():
                candidates.extend(node.children)
        states["solutions"] = self.get_steps()
        return states


"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
from abc import abstractmethod
from termcolor import colored
from typing import Optional, Any, Dict, List, Callable, Type, Tuple, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from omegaconf import DictConfig, OmegaConf

from timeout_decorator import timeout

from config import BaseConfig
from nodes import BaseNode
from tools import PythonInterpreter
from constants import TIMEOUT_SECONDS, TIMEOUT_MESSAGE, QUESTION_COLOR


def _python_ast_init():
    python = PythonInterpreter(globals=globals(), locals=None)
    return python


def tool_wrapper(tool):
    def _tool(query):
        return tool.run(query)
    return _tool


def no_action_wrapper(tool):
    def _tool(query):
        return "No action, no observation. Please continue to solve."
    return _tool


tools = {
    "python_interpreter": tool_wrapper(_python_ast_init()),
    "None": no_action_wrapper(_python_ast_init()),
}


class BaseTree(BaseModel):

    config: Any
    question: str

    ground_truth: Optional[Union[str, List[str]]] = None
    
    llm_model_id: str = None
    llm: Any = None

    root: Optional[Type[BaseNode]] = None
    current_node: Optional[Type[BaseNode]] = None 

    stop: Optional[List[str]] = None

    node_max_retry: int = 5

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.llm_model_id = self.config.model_dir

        if self.config.stop:
            # omegaconf.listconfig.ListConfig -> list
            self.stop = OmegaConf.to_object(self.config.stop)

        self.root = self.create_root()
        self.current_node = self.root

        if self.config.verbose and self.question:
            print(colored(f"Question: {self.question}\n", QUESTION_COLOR))

        if self.config.create_local_llm:
            self.llm = self.create_llm()
    
    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            if not os.path.exists(cfg.model_dir):
                raise ValueError(f"Model directory \"{cfg.model_dir}\" cannot be found.")
            return cfg

        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")
    
    def create_root(self) -> Type[BaseNode]:
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent: Optinal[Type[BaseNode]] = None) -> Type[BaseNode]:
        """
        subclass must implement
        """
    
    @abstractmethod
    def create_llm(self) -> Callable[[...], List[str]]:
        """
        subclass must implement
        """
    
    @abstractmethod
    def generate(self) -> None:
        """
        subclass must implement
        """
    
    def collect_partial_solution(self, node: Type[BaseNode]) -> str:
        # from leaf to root, and reverse
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            node = node.parent
        return self.config.step_delim.join(reversed(trajectory))
    
    def return_states(self) -> Dict[str, Dict[str, str]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            if node.has_children():
                candidates.extend(node.children)
        return states


def code_execution(
    node: Type[BaseNode], 
    parser_result: Dict[str, str],
) -> str:

    @timeout(TIMEOUT_SECONDS, exception_message=TIMEOUT_MESSAGE)
    def _code_execution(node: Type[BaseNode], parser_result: Dict[str, str]) -> str:
        # Define tool
        action = parser_result["action"]
        tool_func = tools[action]

        # Preventing variable update between different children
        # For each child, we re-run the historical code snippets with the same action (tool).
        history_action_inputs = collect_action_inputs(node, action)
        for history_ai in history_action_inputs:
            _ = tool_func(history_ai)
        
        # then, we execute current code snippets
        action_input = parser_result["action_input"]
        observation = str(tool_func(action_input)).strip()
        del tool_func
        return observation
    
    try:
        observation = _code_execution(node, parser_result)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))
    
    return observation


def collect_action_inputs(
    node: Type[BaseNode], 
    action: str,
) -> List[str]:
    action_inputs = []
    while node: 
        if node.state["action"] == action and \
            "TimeoutError" not in node.state["text"].split(node.state["action_input"])[-1]:
            action_inputs.append(node.state["action_input"])
        node = node.parent
    return action_inputs[::-1]


"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import re

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from functools import partial
from vllm import LLM, SamplingParams

from llms import local_vllm
from nodes import BaseNode
from constants import (
    NO_VALID_CHILD,
    SOLUTION_COLOR,
    OBSERVATION_COLOR,
    WARNING_COLOR,
)

# from .tree import BaseTree, code_execution


class REACT(BaseTree):

    REACT_NODE_KEYS: List[str] = ["action", "action_input", "final_answer"]
    prompt_wrap: Optional[Callable[[...], str]] = None
    obs_wrap: Optional[Callable[str, str]] = None
    step_unwrap: Optional[Callable[[...], Dict[str, str]]] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.config.prompt_wrap == "react":
            from .utils import react_prompt_wrap, react_obs_wrap, react_step_result_unwrap

            self.prompt_wrap = react_prompt_wrap
            self.obs_wrap = react_obs_wrap
            self.step_unwrap = react_step_result_unwrap

        elif self.config.prompt_wrap == "react_sft":
            from .utils import react_sft_prompt_wrap, react_sft_obs_wrap, react_sft_step_result_unwrap

            self.prompt_wrap = react_sft_prompt_wrap
            self.obs_wrap = react_sft_obs_wrap
            self.step_unwrap = react_sft_step_result_unwrap

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        super().validate_config(cfg)
        if not cfg.mode == "react":
            raise ValueError(f"Wrong value for config mode, must be react")
        if not cfg.n_generate_sample == 1:
            raise ValueError(f"Wrong value for config n_generate_sample, must be 1")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg
    
    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        return BaseNode(
            parent=parent, 
            additional_state_keys=self.REACT_NODE_KEYS,
        )

    def create_llm(self) -> Callable[[...], List[str]]:
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        llm = LLM(
            model=self.config.model_dir, 
            tensor_parallel_size=len(GPUS), 
            trust_remote_code=True,
            seed=self.config.seed,
            swap_space=self.config.swap_space,
        )
        sampling_params = SamplingParams(
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            use_beam_search=self.config.use_beam_search,
            best_of=self.config.best_of,
            max_tokens=self.config.max_tokens, 
            stop=self.stop,
            #seed=self.config.seed,
        )
        return partial(
            local_vllm,
            llm=llm,
            sampling_params=sampling_params,
            n=1,
            temperature=self.config.temperature,
        )

    def should_generate_next(self) -> bool:
        return not self.current_node.is_terminal and self.current_node.depth <= self.config.max_depth

    def generate(self) -> None:
        """
        generate as a linked list
        root -> x -> y -> z
        """
        while self.should_generate_next():
            step_result, parser_result = self.get_parsable_samples()
            self.update_current_node(step_result, parser_result)

    def update_current_node(
        self, 
        step_result: str,
        parser_result: Dict[str, str],
    ) -> None:
        self._update_current_node(step_result, parser_result)
        self.current_node = self.current_node.children[0]

    def _update_current_node(
        self,
        step_result: str,
        parser_result: Dict[str, str],
        idx: int = 0,
    ) -> None:
        if self.config.verbose:
            print(colored(f"{step_result}\n", SOLUTION_COLOR))

        # initialize a new node
        new_node = self.create_node(parent=self.current_node)
        new_node.tag = f"{self.current_node.tag}.{idx}"
        new_node.depth = self.current_node.depth + 1

        # update node state
        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
        elif parser_result["action"]:
            observation = code_execution(self.current_node, parser_result)
            observation = self.obs_wrap(observation)

            if self.config.verbose:
                    print(colored(f"{observation}\n", OBSERVATION_COLOR))

            new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
        else:
            if self.config.verbose:
                print(colored(f"WARNING: '{step_result}' Cannot resolve\n", WARNING_COLOR))
            new_node.state["text"] = step_result

        # update parent's children
        self.current_node.children.append(new_node)
    
    def get_parsable_samples(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = self.create_prompt()
        sampled_step_results = self.get_llm_samples(prompt)

        try:
            step_result = sampled_step_results[0]
            return self.step_unwrap(step_result)
        except Exception as e:
            n_samples = 3
            temperature = 0.7
            print(f"Exception: {e}. will retry {self.node_max_retry} times by setting temperature {temperature}, and generating {n_samples} samples in single run to save token counts")
        
            retry_cnt = 0
            while retry_cnt < self.node_max_retry:
                sampled_step_results = self.get_llm_samples(prompt, n_samples, temperature)
                for step_result in sampled_step_results:
                    try:
                        return self.step_unwrap(step_result)
                    except Exception as e:
                        retry_cnt += 1
                        print(f"Exception: {e}. Retry {retry_cnt} failed.")
                        continue
            return step_result, None

    def create_prompt(
        self,
    ) -> str:
        partial_solution = self.collect_partial_solution(self.current_node)
        prompt = self.prompt_wrap(
            self.question, 
            partial_solution,
            self.config,
        )
        return prompt
    
    def get_llm_samples(
        self, 
        prompt: str, 
        n: int = 1,
        temperature: Optional[float] = None,
    ) -> List[str]:
        if temperature is None:
            # default llm
            samples = self.llm(prompt, n=n)
        else:
            samples = self.llm(prompt, temperature=temperature, n=n)
        
        processed_samples = [sample.strip() for sample in set(samples)]
        return processed_samples
