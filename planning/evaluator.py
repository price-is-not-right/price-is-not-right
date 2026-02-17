from typing import *
from VLM.prompts import *
from VLM.openai_api import *
from utils import *
from planner import *
from llm_planners import *
from tarski import fstrips as fs

def vlm_evaluate_success(obs:dict, operator:str):
    """evaluate whether the operator has been successfully executed in the environment

    Args:
        obs (dict): the observation of the environment
        operator (str): the grounded pddl representation of the operator
    """
    image:np.array = obs['agentview_image']
    base64_image = numpy_to_base64(image)
    # multimodal CoT
    rationale = chat_completion(rationale_generation_prompt, base64_image)
    full_eval_prompt = eval_prompt.format(rationale=rationale, grounded_operator=operator)
    response = chat_completion(full_eval_prompt)
    # parse response
    if response.lower() == 'yes':
        return True
    elif response.lower() == 'no':
        return False
    else:
        raise ValueError("Invalid response from the VLM")


def grounded_operator_repr(grounded_op:fs.Action) -> str:
    """Return a string representation of the grounded operator

    Args:
        grounded_op (fs.Action): the grounded operator
    Returns:
        str: the string representation of the grounded operator
    """
    effects_str:str = ' '.join(f'({eff})' for eff in grounded_op.effects)
    return f"{grounded_op.name}\nprecondition: {grounded_op.precondition.pddl_repr()}\neffects: and {effects_str}"


if __name__ == "__main__":
    # test the VLM evaluator
    # get a grounded operator from the planner. Use the first one in the plan
    planner = SymbolicPlanner(config=load_config('config.yaml')['planning']['nut_assembly'])
    plan = planner.search()
    grounded_op = grounded_operator_repr(plan[0])
    # load the example image
    image = load_image("images/agent_view.jpg")
    dummy_obs = {'agentview_image': image}
    # evaluate the success of the grounded operator
    success = vlm_evaluate_success(dummy_obs, grounded_op)