import os
import copy
import subprocess
from tarski import fstrips as fs

def grounded_operator_repr(grounded_op:fs.Action) -> str:
    """Return a string representation of the grounded operator

    Args:
        grounded_op (fs.Action): the grounded operator
    Returns:
        str: the string representation of the grounded operator
    """
    effects_str:str = ' '.join(f'({eff})' for eff in grounded_op.effects)
    return f"{grounded_op.name}\nprecondition: {grounded_op.precondition.pddl_repr()}\neffects: and {effects_str}"

def add_predicates_to_pddl(pddl_dir, init_predicates, pddl_name='problem_save.pddl', problem_name="problem_dummy.pddl"):
    '''
        Given a PDDL file, this function adds the predicates to the init section of the PDDL file.
        The new PDDL file is saved as "problem_dummy.pddl"
    '''
    # read the PDDL file
    pddl_file_path = pddl_dir + pddl_name
    with open(pddl_file_path, 'r') as file:
        lines = file.readlines()
        #print("Lines = ", lines)

    init_index = lines.index('  (:init \n')
    for predicate, value in init_predicates.items():
        if value:
            # first convert the predicate of the form "p1(o1,o1)" to "p1 o1 o1"
            predicate = predicate.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            # then add the predicate to the init section
            lines.insert(init_index + 1, f'({predicate})\n')

    # define new problem file path with the end file being named as "problem_dummy.pddl" (os.sep is used to handle the path separator)
    problem_path = pddl_dir + problem_name

    # overwrite the new problem file
    with open(problem_path, 'w') as file:
        file.writelines(lines)

def define_goal_in_pddl(pddl_dir, goal_predicates, problem_name="problem_dummy.pddl"):
    '''
        Given a PDDL file, this function adds the goal predicates to the goal section of the PDDL file.
        The new PDDL file is saved as "problem_dummy.pddl"
    '''
    # read the PDDL file
    problem_path = pddl_dir + problem_name
    with open(problem_path, 'r') as file:
        lines = file.readlines()

    goal_index_start = lines.index('  (:goal \n')
    goal_index_end = lines.index('  )\n', goal_index_start)

    # remove existing goal predicates
    del lines[goal_index_start + 1:goal_index_end]

    # add new goal predicates
    insert_pos = goal_index_start + 1
    lines.insert(insert_pos, '    (and\n')
    insert_pos += 1
    for predicate in goal_predicates:
        lines.insert(insert_pos, f'      ({predicate})\n')
        insert_pos += 1
    lines.insert(insert_pos, '    )\n')

    # overwrite the new problem file
    with open(problem_path, 'w') as file:
        file.writelines(lines)


def call_planner(pddl_dir, problem="problem_dummy.pddl", structure="pddl", mode=0):
    '''
        Given a domain and a problem file
        This function return the ffmetric Planner output.
        In the action format
    '''
    domain_path = pddl_dir + "domain.pddl"
    problem_path = pddl_dir + problem
    if structure == "pddl":
        run_script = f"./Metric-FF-v2.1/./ff -o {domain_path} -f {problem_path} -s {mode}"
        output = subprocess.getoutput(run_script)
        #print("Output = ", output)
        if "unsolvable" in output or "goal can be simplified to FALSE" in output:
            print("The planner failed because the problem is unsolvable: {}".format(output))
            return False, False
        try:
            output = output.split('ff: found legal plan as follows\n')[1]
            output = output.split('\ntime spent:')[0]
            # Remove empty lines
            output = os.linesep.join([s for s in output.splitlines() if s])
        except Exception as e:
            print("The planner failed because of: {}.\nThe output of the planner was:\n{}".format(e, output))

        plan, game_action_set = _output_to_plan(output, structure=structure)
        return plan, game_action_set

def _output_to_plan(output, structure):
    '''
    Helper function to perform regex on the output from the planner.
    ### I/P: Takes in the ffmetric output and
    ### O/P: converts it to a action sequence list.
    '''
    if structure == "pddl":
        action_set = []
        for action in output.split("\n"):
            #if action.startswith('step'):
            try:
                action_set.append(''.join(action.split(": ")[1]))
            except IndexError:
                return False, False
        
        # convert the action set to the actions permissable in the domain
        game_action_set = copy.deepcopy(action_set)

        #for i in range(len(game_action_set)):
        #   game_action_set[i] = applicator[game_action_set[i].split(" ")[0]]
        #for i in range(len(game_action_set)):
        #    for j in range(len(game_action_set[i])):
        #        if game_action_set[i][j] in applicator.keys():
        #            game_action_set[i][j] = applicator[game_action_set[i]]
        return action_set, game_action_set
