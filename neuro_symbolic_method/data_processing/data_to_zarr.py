import zarr
import numcodecs
import zipfile, pickle, copy, json, argparse, os
import numpy as np
from datasets import Dataset, Features, Value, ClassLabel, Sequence
from imitation.data.types import TrajectoryWithRew
import dataclasses

@dataclasses.dataclass(frozen=True, eq=False)
class TrajectoryWithKeypoint(TrajectoryWithRew):
    """A `Trajectory` that additionally includes reward information."""

    keypoint: np.ndarray
    """Reward, shape (trajectory_len, ). dtype float."""

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()


class GoalTrajectory(TrajectoryWithRew):
    """A `Trajectory` that additionally includes reward information."""

    desired_goals: np.ndarray

    achieved_goals: np.ndarray

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()

class KeypointTrajectory(TrajectoryWithRew):
    """A `Trajectory` that additionally includes reward information."""

    keypoint: np.ndarray

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()


def load_data_from_zip(dir_path):
    # List all zip files in the directory
    zip_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
    
    # Dictionary to hold the data extracted from each zip file
    data_buffers = {}

    # Iterate over each zip file
    for zip_file_name in zip_files:
        # Construct the full path to the zip file
        file_path = os.path.join(dir_path, zip_file_name)
        
        # Open the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            # Extract data.pkl
            with zip_file.open('data.pkl') as file:
                # Deserialize the data
                data = pickle.load(file)
                
                # Use the file name without extension as the key
                data_buffers[zip_file_name.split('.zip')[0]] = data

    return data_buffers

def load_buffer(file_path):
    # Read the bytes from the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        with zip_file.open('data.pkl', 'r') as file:
            data_bytes = file.read()

    # Convert the bytes back to a list of lists
    data_buffer = pickle.loads(data_bytes)

    return data_buffer

def convert_to_dict_format(data_buffer):
    dict_format_buffer = []
    for episode_buffer, symbolic_buffer, task in data_buffer:
        for transition in episode_buffer:
            obs, act, next_obs, reward, done = transition
            dict_format_buffer.append({
                "obs": obs,
                "acts": act,
                "next_obs": next_obs,
                "rews": reward,
                "dones": done,
                "infos": {"symbolic_trajectory": symbolic_buffer, "task": task}
            })
    return dict_format_buffer




features = Features({
    'observations': Sequence(Value('float32')),  # Assuming observations are vectors of floats
    'actions': Sequence(Value('float32')),  # Assuming actions are vectors of floats
    'next_observations': Sequence(Value('float32')),  # Same as observations
    'rewards': Sequence(Value('float32')),  # Rewards are floats
    'dones': Sequence(ClassLabel(names=["false", "true"]))  # Boolean done flags
})


def prepare_data_for_dataset(trajectories, args):
    trajectory_objects = []
    counter = 1
    for trajectory in trajectories:
        if not trajectory:
            continue
        episode = trajectory[0]
        # Assuming each step has observations, actions, next_obs, rewards, and done flags
        obs = [step[0] for step in episode]
        obs.append(episode[-1][2]) # add the `next_obs` from the last trajectory as the final obs
        obs = np.array(obs) # turn it into np.array
        acts = np.array([step[1] for step in episode])
        keypoint = np.array([step[2] for step in episode])#
        #print("obs: ", obs)
        rews = np.zeros(len(acts))  # Assuming zero rewards for all steps
        infos = np.array([{} for _ in episode])  # Assuming empty dicts for infos
        terminal = True #episode[-1][4]  # The 'done' flag of the last step
        # print("obs shape: ", np.array(obs).shape)
        # print("acts shape: ", np.array(acts).shape)
        # print("rews shape: ", np.array(rews).shape)
        # print("infos shape: ", np.array(infos).shape)
        # print("terminal shape: ", np.array(terminal).shape)
        # print("")
        # print("obs shape: ", obs[0])
        #if terminal:
        #    print("an episode that reached the goal")
        traj_obj = TrajectoryWithKeypoint(obs=obs, acts=acts, infos=infos, rews=rews, terminal=terminal, keypoint=keypoint)
        trajectory_objects.append(traj_obj)
        if args.num_demos > 0 and counter == args.num_demos:
            break
        counter += 1

    return trajectory_objects


def save_buffer_as_json(data_buffer, file_path):
    # Convert the data buffer to dictionary format
    dict_format_buffer = convert_to_dict_format(data_buffer)

    # Convert the dictionary format buffer to JSON
    json_data = json.dumps(dict_format_buffer)

    # Write the JSON data to a zip file
    with zipfile.ZipFile(file_path, 'w') as zip_file:
        with zip_file.open('data.json', 'w', force_zip64=True) as file:
            file.write(json_data.encode('utf-8'))

def load_json_buffer(file_path):
    # Open the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        # Open the JSON file
        with zip_file.open('data.json', 'r') as file:
            # Load the JSON data
            json_data = file.read().decode('utf-8')
            data_buffer = json.loads(json_data)

    return data_buffer

# Find indexes of the action space in the trajectories that are never used in the expert demonstrations or are always the same value
def find_constant_indexes(action):
    # Transpose the action array to get actions at each index
    action_t = np.transpose(action)

    # Find indexes where all values are the same or never used
    constant_indexes = [i for i, a in enumerate(action_t) if np.all(a == a[0])]

    return constant_indexes

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='Data Directory')
    parser.add_argument('--data_dir2', type=str, default="", help='Data Directory of second set of demonstrations')
    parser.add_argument('--data_dir3', type=str, default="", help='Data Directory of second set of demonstrations')
    parser.add_argument('--data_dir4', type=str, default="", help='Data Directory of second set of demonstrations')
    parser.add_argument('--data_dir5', type=str, default="", help='Data Directory of second set of demonstrations')
    parser.add_argument('--data_dir6', type=str, default="", help='Data Directory of second set of demonstrations')
    parser.add_argument('--filter_actions', type=bool, default=False, help='Filter actions')
    parser.add_argument('--num_demos', type=int, default=0, help='Number of demonstrations')
    parser.add_argument('--save_dir', type=str, default='', help='Save Directory')
    parser.add_argument('--lxm', action='store_true', help='Use lxm project format of data')
    args = parser.parse_args()

    # Load the buffer from the zip file
    data_buffers = load_data_from_zip(args.data_dir + '/traces/')
    if args.data_dir2 != "":
        data_buffers2 = load_data_from_zip(args.data_dir2 + '/traces/')
        for act, buffer in data_buffers2.items():
            if act in data_buffers:
                data_buffers[act] += buffer
            else:
                data_buffers[act] = buffer
    if args.data_dir3 != "":
        data_buffers3 = load_data_from_zip(args.data_dir3 + '/traces/')
        for act, buffer in data_buffers3.items():
            if act in data_buffers:
                data_buffers[act] += buffer
            else:
                data_buffers[act] = buffer
    if args.data_dir4 != "":
        data_buffers4 = load_data_from_zip(args.data_dir4 + '/traces/')
        for act, buffer in data_buffers4.items():
            if act in data_buffers:
                data_buffers[act] += buffer
            else:
                data_buffers[act] = buffer
    if args.data_dir5 != "":
        data_buffers5 = load_data_from_zip(args.data_dir5 + '/traces/')
        for act, buffer in data_buffers5.items():
            if act in data_buffers:
                data_buffers[act] += buffer
            else:
                data_buffers[act] = buffer
    if args.data_dir6 != "":
        data_buffers6 = load_data_from_zip(args.data_dir6 + '/traces/')
        for act, buffer in data_buffers6.items():
            if act in data_buffers:
                data_buffers[act] += buffer
            else:
                data_buffers[act] = buffer

    # Convert the buffer to a dict
    # dict_format_buffer = convert_to_dict_format(data_buffer)

    # Covert each buffer to a list of HuggingFace Trajectory
    demo_auto_trajectories = {}
    n_obj = 1
    constant_indexes = []
    for act, buffer in data_buffers.items():
        demo_trajectories_for_act = prepare_data_for_dataset(buffer, args) # a buffer is a list of trajectories for the high level act e.g reach_pick
        if args.filter_actions:
            # Find the indexes of the action space that are never used in the expert demonstrations or are always the same value
            constant_indexes = find_constant_indexes(np.array([demo_trajectories_for_act[0].acts]))
            print("Filtered action space for ", act, ":", constant_indexes, " index actions removed")

        # Save directory
        if args.save_dir == '':
            save_dir = args.data_dir + '/hf_traj/' + act + '/keypoint/'
        else:
            save_dir = args.save_dir + '/hf_traj/' + act + '/keypoint/'
        os.makedirs(save_dir, exist_ok=True)
        root = zarr.open(save_dir + 'keypoint.zarr', mode='w')
        data = root.create_group('data')

        total_timesteps = 0
        print(len(constant_indexes))
        print("Num of trajectories: ", len(demo_trajectories_for_act))
        print(demo_trajectories_for_act[0].acts)
        ee_dim = len(demo_trajectories_for_act[0].acts[0]) - len(constant_indexes)
        obs_dim = len(demo_trajectories_for_act[0].obs[0])
        try:
            keypoint_dim = len(demo_trajectories_for_act[0].keypoint[0])
        except:
            keypoint_dim = 1
        print("obs_dim: ", obs_dim, " keypoint_dim: ", keypoint_dim, " ee_dim: ", ee_dim)
        # Count total number of timesteps
        for traj in demo_trajectories_for_act:
            total_timesteps += len(traj.obs) - 1
        # Count total number of episodes
        n_episodes = len(demo_trajectories_for_act)

        action = data.create_dataset('action', shape=(total_timesteps, ee_dim), chunks=(201, ee_dim + len(constant_indexes)), dtype='f8')
        low_dim = data.create_dataset('keypoint', shape=(total_timesteps, n_obj, keypoint_dim), chunks=(201, n_obj, keypoint_dim),
                                    dtype='f8')
        state = data.create_dataset('state', shape=(total_timesteps, obs_dim), chunks=(201, obs_dim), dtype='f8')

        meta = root.create_group('meta')
        episodes_end = meta.create_dataset('episode_ends', shape=(n_episodes), chunks=(201), dtype='i8')
        data_cursor = 0
        print("Num of trajectories: ", len(demo_trajectories_for_act))
        # Create a Zarr group for each trajectory
        for traj_num, traj in enumerate(demo_trajectories_for_act):
            # Initialize a Zarr group
            #root = zarr.group(store=zarr.DirectoryStore(save_dir + str(i)))

            # Create a Zlib compressor
            #compressor = numcodecs.Zlib(level=1) 
            keypoint_save = None
            for i in range(len(traj.obs)-1):#
                #print(data_cursor)
                # Set the data for the current timestep
                if args.filter_actions:
                    action[data_cursor] = traj.acts[i][[i for i in range(ee_dim + len(constant_indexes)) if i not in constant_indexes]]
                else:
                    action[data_cursor] = traj.acts[i]
                try:
                    low_dim[data_cursor] = [traj.keypoint[i]]
                    keypoint_save = [traj.keypoint[i-1]]
                except:
                    low_dim[data_cursor] = keypoint_save
                
                state[data_cursor] = traj.obs[i]
                # If shape[0] of obs is not 15 then print the shape
                data_cursor += 1

            # Set the end of the episode
            episodes_end[traj_num] = data_cursor

            # Create Zarr arrays in the group for each data type
            #root.create_dataset('obs', data=obs, chunks=(1000, obs.shape[1]), compressor=compressor)
            #root.create_dataset('action', data=action, chunks=(1000, action.shape[1]), compressor=compressor)
            #root.create_dataset('next_obs', data=next_obs, chunks=(1000, next_obs.shape[1]), compressor=compressor)
            #root.create_dataset('keypoints', data=keypoints, chunks=(1000, keypoints.shape[1]), compressor=compressor)



# Assume obs, action, next_obs, and keypoints are numpy arrays
# obs, action, next_obs, keypoints = ...

