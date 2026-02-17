import os
import logging
from collections import deque

from tarski.search import GroundForwardSearchModel
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io.fstrips import FstripsReader, FstripsWriter
from tarski.syntax.builtins import *
from tarski import fstrips as fs
from tarski.evaluators.simple import evaluate
from VLM.openai_api import *
from VLM.prompts import *
from planning.planning_utils import *


class SymbolicPlanner:
    def __init__(self, config:dict):
        self.config = config
        self.reader = FstripsReader(raise_on_error=True)
        self.parse_domain()
        self.starting_problem = self.parse_problem()
        self.max_depth = self.config['max_depth']

    def parse_domain(self):
        """parses the initial domain file
        """
        domain_file_path = self.config['planning_dir'] + os.sep + self.config['init_planning_domain']
        self.reader.parse_domain(domain_file_path)
    
    def parse_problem(self):
        """parses the problem file
        """
        problem_file_path = self.config['planning_dir'] + os.sep + self.config['planning_problem']
        problem = self.reader.parse_instance(problem_file_path)
        return problem
    
    
    def search(self) -> List[fs.Action]:
        """BFS ahead from the current node for a solution to the problem's goal

        Returns:
            List[fs.Action]: the plan found if any
        """
        # create obj to track state space
        space = SearchSpace()
        stats = SearchStats()

        model = GroundForwardSearchModel(self.starting_problem, ground_problem_schemas_into_plain_operators(self.starting_problem))
        open_list = deque()
        start_node = make_root_node(model.init())
        open_list.append(start_node)
        closed = {start_node}
        while open_list:
            stats.iterations += 1
            # logging.debug("dfs: Iteration {}, #unexplored={}".format(iteration, len(open_)))

            node = open_list.popleft()
            if model.is_goal(node.state): # found a plan
                stats.num_goals += 1
                plan = reverse_engineer_plan(node)
                logging.info(f"Goal found after {stats.nexpansions} expansions from node {start_node}. {stats.num_goals} goal states found.")
                logging.info(f"found plan from {start_node} to {node}. The plan is {plan}")
                return plan # early return if a plan is found since we only care about plan with the shortest length

            if 0 <= self.max_depth <= node.depth: # reached max depth
                logging.info(f"Max. expansions reached on one branch. # expanded: {stats.nexpansions} from node {start_node}, # goals: {stats.num_goals}.")
                continue
            else: # expand the node and add its children to the open list
                for operator, successor_state in model.successors(node.state):
                    if successor_state not in closed:
                        open_list.append(make_child_node(node, operator, successor_state))
                        closed.add(successor_state)
                        stats.nexpansions += 1

        logging.info(f"Search space exhausted. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
        space.complete = True
        return None


class LLMPlanner(SymbolicPlanner):
    """First iteration of a purely LLM planner"""

    def prompt_for_plan(self):
        """Generate a plan by calling the LLM
        """
        print("#"*20 + "The domain file" + "#"*20)
        print(self.reader.domain_text)
        print("#"*20 + "The problem file" + "#"*20)
        print(self.reader.problem_text)

        #TODO: implement the LLM planner by calling the LLM API
        # first. complete the prompt that is going to go into the LLM by filling in the domain and problem files
        print("#"*20 + "The current prompt" + "#"*20)
        print(plan_prompt) # feel free to modify this prompt for the best output
        
    
    def parse_plan(plan:str) -> List[str]:
        """Parse the plan string into a list of grounded operators

        Args:
            plan (str): the plan string (output of the prompt_for_plan method)

        Returns:
            List[str]: the list of grounded operators
        """
        #TODO: implement the parsing of the plan string


if __name__ == "__main__":
    # test the LLM planner
    planner = LLMPlanner(config=load_config('config.yaml')['planning']['nut_assembly'])
    plan = planner.prompt_for_plan()
    grounded_ops = planner.parse_plan(plan)
    print(grounded_ops)
