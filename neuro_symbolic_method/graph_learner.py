# Implement a GraphLearner class that learns a graph from (state, action, next_state) tuples.

import networkx as nx
import matplotlib.pyplot as plt

class GraphLearner:
    def __init__(self, file_path, display_graph=False):
        self.graph = nx.DiGraph()
        self.bisimulation = None
        self.display_graph = display_graph
        self.file_path = file_path
        self.state_mapping = {}
        self.state_set = set()

    def is_known_state(self, state):
        return frozenset(state.items()) in self.state_set

    def is_known_edge(self, state, action, next_state):
        state = self.add_state(state)
        next_state = self.add_state(next_state)
        return self.graph.has_edge(state, next_state) and self.graph.edges[state, next_state]['label'] == action

    def add_state(self, state):
        state_frozenset = frozenset(state.items())
        if state_frozenset not in self.state_set:
            state_name = str(len(self.state_mapping))
            self.state_mapping[state_name] = state
            self.state_set.add(state_frozenset)
        else:
            state_name = next(key for key, value in self.state_mapping.items() if value == state)
        return state_name

    def learn(self, state, action, next_state):
        #print("Learning the transition: ", state, action, next_state)
        state = self.add_state(state)
        next_state = self.add_state(next_state)
        if state is None or next_state is None:
            return
        #print(state, action, next_state)
        self.graph.add_edge(state, next_state, label=action)
        print(f"Edge attributes: {nx.get_edge_attributes(self.graph, 'label')}")  # Add this line
        self.bisimulation = self.compute_bisimulation_graph(self.graph)
        self.write_graph_to_file(self.file_path + "bisimulation.dfa", self.bisimulation)
        self.write_graph_to_file(self.file_path + "graph.dfa", self.graph)

    def compute_bisimulation_graph(self, G):
        # Compute the bisimulation partition
        partition = self.bisect(G)

        # Convert the partition to a NetworkX graph
        B = nx.DiGraph()
        for i, block in enumerate(partition):
            B.add_node(i)

        # Add an edge between two nodes if there is an edge between two elements of the corresponding equivalence classes
        for i, block1 in enumerate(partition):
            for j, block2 in enumerate(partition):
                for u in block1:
                    for v in block2:
                        if G.has_edge(u, v):
                            B.add_edge(i, j, label=G.edges[u, v]['label'])
                            break

        if self.display_graph:
            # Display the input graph and the bisimulation graph on the same plot
            plt.figure()
            plt.subplot(121)
            nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
            plt.subplot(122)
            nx.draw(B, with_labels=True, node_color='lightblue', font_weight='bold')
            plt.show()

        return B

    def bisect(self, G):
        # Initialize a partition of the states where each state is in its own block
        partition = [{node} for node in G.nodes()]

        # Initialize a worklist with all blocks in the partition
        worklist = partition.copy()

        while worklist:
            # Choose and remove a block B from the worklist
            B = worklist.pop()

            # For each possible edge label l
            edge_labels = set(edge_data['label'] for node in B for _, edge_data in G[node].items())
            for l in edge_labels:
                # Split the block B into two blocks: B1 containing the states in B with an outgoing edge labeled l, and B2 containing the other states
                B1 = {node for node in B if any(edge_data['label'] == l for _, edge_data in G[node].items())}
                B2 = B - B1

                if B1 and B2:
                    # Replace B in the partition with B1 and B2
                    partition.remove(B)
                    partition.extend([B1, B2])

                    # If B was in the worklist, add the smaller of B1 and B2 to the worklist
                    # Otherwise, add both B1 and B2 to the worklist
                    worklist.extend([B1, B2])

        return partition
        
    def parse_file_to_graph(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Create a directed graph
        G = nx.DiGraph()

        # Parse the number of states
        _, num_states, _ = lines[0].split()
        num_states = int(num_states)

        # Add the states to the graph
        G.add_nodes_from(range(num_states))

        # Parse the transitions
        for i in range(num_states):
            transitions = lines[i+3].split()[1:]
            for j in range(0, len(transitions), 2):
                label = transitions[j]
                target_state = int(transitions[j+1])
                # Add the edge to the graph with the label as an attribute
                G.add_edge(i, target_state, label=label)

        if self.display_graph:
            # Display the graph
            nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
            plt.show()

        self.graph = G
        return G

    def write_graph_to_file(self, file_path, G=None):
        if G is None:
            G = self.bisimulation
        with open(file_path, 'w') as file:
            # Write the first line
            file.write(f"dfa {G.number_of_nodes()} -1\n")

            # Get the set of labels
            labels = set(nx.get_edge_attributes(G, 'label').values())

            # Write the second line
            file.write(f"{len(labels)} {' '.join(labels)}\n")

            # Write the third line
            file.write("1 0\n")

            # Write the transitions
            for node in G.nodes:
                edges = G.out_edges(node, data=True)
                file.write(f"{len(edges)} {' '.join('{} {}'.format(data.get('label', 'MOVE'), v) for u, v, data in edges)}\n")
