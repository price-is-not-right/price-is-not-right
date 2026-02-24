from dataclasses import dataclass, field
from typing import Tuple, Dict, Set, List, Iterable
from typing import Dict, List, Set
import itertools
import networkx as nx
import numpy as np

@dataclass
class SceneObject:
    id: str
    position: Tuple[float, float, float]  # (x, y, z)
    metadata: Dict[str, Set[str]] = field(default_factory=dict)
    # metadata maps other_object_id -> set of relation strings, e.g. {"objB": {"left_of", "above"}}

    def set_relations(self, other_id: str, relations: Iterable[str]) -> None:
        """Set the relations (overwrite) describing this object w.r.t. other_id."""
        if relations:
            self.metadata[other_id] = set(relations)
        else:
            # store empty set to show explicit "none" or remove entry
            self.metadata.pop(other_id, None)

    def get_relations_to(self, other_id: str) -> Set[str]:
        """Return the set of relations this object has to other_id."""
        return self.metadata.get(other_id, set())

    def clear_metadata(self) -> None:
        self.metadata.clear()

    def __repr__(self) -> str:
        return f"SceneObject(id={self.id!r}, pos={self.position}, relations={len(self.metadata)} entries)"

# Helper mapping of inverse relationship labels
_INVERSE_REL = {
    "left_of": "right_of",
    "right_of": "left_of",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "above": "below",
    "below": "above",
    "x_aligned": "x_aligned",
    "y_aligned": "y_aligned",
    "z_aligned": "z_aligned",
}

def _compare_relations(a_pos: Tuple[float, float, float],
                       b_pos: Tuple[float, float, float],
                       eps: float):
    """
    Compare two positions and return relations of A with respect to B.
    Uses user-provided axis rules:
      - left/right: y (a left of b if a.y < b.y)
      - front/behind: x (a in front of b if a.x < b.x)
      - above/below: z (a above b if a.z > b.z)
    """
    ax, ay, az = a_pos
    bx, by, bz = b_pos
    rels = set()

    # Y axis -> left/right
    if ay < by - eps:
        rels.add("left_of")
    elif ay > by + eps:
        rels.add("right_of")
    else:
        rels.add("y_aligned")

    # X axis -> front/behind
    if ax < bx - eps:
        rels.add("in_front_of")
    elif ax > bx + eps:
        rels.add("behind")
    else:
        rels.add("x_aligned")

    # Z axis -> above/below (note: higher z => above)
    if az > bz + eps:
        rels.add("above")
    elif az < bz - eps:
        rels.add("below")
    else:
        rels.add("z_aligned")

    return rels

def update_object_metadata(objects: List[SceneObject], eps: float = 1e-6) -> None:
    """
    Update metadata for every object in `objects` describing its spatial relations
    with every other object.

    This updates each SceneObject.metadata in-place. For each ordered pair (A, B),
    we compute A's relations to B using the axis rules:
      - left/right by comparing y (A left of B if A.y < B.y)
      - in front / behind by comparing x (A in front of B if A.x < B.x)
      - above / below by comparing z (A above B if A.z > B.z)

    The inverse relation for B->A is stored consistently (e.g. if A is left_of B,
    B is right_of A).
    """
    # Clear existing metadata
    for obj in objects:
        obj.clear_metadata()

    # Build a lookup by id for convenience
    id_to_obj = {o.id: o for o in objects}

    # Pairwise comparisons
    n = len(objects)
    for i in range(n):
        a = objects[i]
        for j in range(i + 1, n):
            b = objects[j]
            a_rels = _compare_relations(a.position, b.position, eps)
            b_rels = _compare_relations(b.position, a.position, eps)

            # Normalize: ensure inverse mapping aligns with our _INVERSE_REL
            # Convert a_rels -> a_rels_normalized and b_rels_normalized via mapping
            # For axes we already return symmetric labels (e.g., y_aligned vs itself)
            # We store exactly those labels.
            a.set_relations(b.id, a_rels)
            b.set_relations(a.id, b_rels)

def pretty_print_scene(objects: List[SceneObject]) -> None:
    """Print readable relations for each object."""
    for obj in objects:
        print(f"Object {obj.id} at {obj.position}:")
        if not obj.metadata:
            print("  (no relations recorded)")
            continue
        for other_id, rels in obj.metadata.items():
            rel_list = ", ".join(sorted(rels))
            print(f"  -> {other_id}: {rel_list}")
        print()

def build_relation_graph(objects: List[SceneObject]) -> Dict[str, Dict[str, Set[str]]]:
    """
    Build a relation graph from the list of SceneObjects.
    Returns an adjacency-like dictionary:
      {
        objA: {
            objB: {"above", "left_of"},
            objC: {"below"},
            ...
        },
        ...
      }
    """
    graph = {}
    for obj in objects:
        graph[obj.id] = {}
        for other_id, rels in obj.metadata.items():
            graph[obj.id][other_id] = set(rels)
    return graph


def match_objects_by_relationships(set_obj1: List[SceneObject],
                                   set_obj2: List[SceneObject],
                                   relation_weight: float = 1.0,
                                   name_weight: float = 0.0):
    """
    Match set_obj1 ids to set_obj2 ids based on relational graph similarity.

    1. Builds relational graphs from both sets.
    2. Computes a similarity matrix between all pairs of nodes using
       overlap of relation types to corresponding neighbors.
    3. Solves optimal assignment using Hungarian algorithm for a global best match.

    Returns:
        mapping: Dict[obj1_id -> obj2_id]
    """
    from scipy.optimize import linear_sum_assignment
    import re

    def name_similarity(a, b):
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        a_tokens = set(re.findall(r"\w+", a))
        b_tokens = set(re.findall(r"\w+", b))
        overlap = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        return overlap / union if union else 0.0

    # Build relation graphs
    G1 = build_relation_graph(set_obj1)
    G2 = build_relation_graph(set_obj2)

    ids1 = list(G1.keys())
    ids2 = list(G2.keys())

    n1, n2 = len(ids1), len(ids2)
    sim_matrix = np.zeros((n1, n2))

    # Compute pairwise similarity
    for i, id1 in enumerate(ids1):
        for j, id2 in enumerate(ids2):
            rel_sim_scores = []
            # Compare how id1 and id2 relate to others in their own sets
            for nbr1, rels1 in G1[id1].items():
                for nbr2, rels2 in G2[id2].items():
                    rel_overlap = len(rels1 & rels2) / len(rels1 | rels2)
                    rel_sim_scores.append(rel_overlap)
            rel_score = np.mean(rel_sim_scores) if rel_sim_scores else 0.0
            name_score = name_similarity(id1, id2)
            sim_matrix[i, j] = relation_weight * rel_score + name_weight * name_score

    # Hungarian algorithm: maximize similarity → minimize negative similarity
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {ids1[i]: ids2[j] for i, j in zip(row_ind, col_ind)}
    return mapping



# Example usage
if __name__ == "__main__":
    objs = [
        SceneObject("A", (0.0, 0.0, 0.0)),
        SceneObject("B", (1.0, 1.0, 0.5)),
        SceneObject("C", (-1.0, -0.5, 1.5)),
    ]
    update_object_metadata(objs, eps=1e-8)
    pretty_print_scene(objs)

    # Query example:
    a = objs[0]
    print("A relations to B:", a.get_relations_to("B"))

