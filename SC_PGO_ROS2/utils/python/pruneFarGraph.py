"""Take the obstacles of people (and other things that moved) out of a far_planner visibility graph (.vgh).

far_planner builds its graph live, from terrain maps of the scans: a person it saw for long enough became an obstacle
polygon (contour nodes and edges) and stays in the saved graph. makeMergedMap.py finds what moved by looking at the
map from all the SC-PGO keyframes (dynamic_removal.py) and, with --dynamic-output, saves those points.

Here every obstacle polygon of the graph (the nodes that are not robot trajectory points (navpoints), joined by
contour edges) is judged whole, on what lies inside its outline on the floor plane: the convex hull of its nodes, or
discs of --radius around them for a polygon of one or two nodes. A ghost - at least --min-dynamic removed points
inside and fewer static map points than --static-ratio times them - goes with all its nodes, and they are taken out
of the edge lists of the others. Inside, not around: a box someone walked past has the removed points of the person
next to it, outside its outline, and few points of its own when the lidar barely sees it (a low box under a lidar
tilted 5 deg sees only its top edge); the polygon of a person's walk has them inside. The outline of a room holds
the walls' points too: it stays whatever walked inside.

Both clouds count only between --min-height and --max-height above the floor (the floor and ceiling are never
removed and would support every node). Graph frame: far_planner's map, which is camera_init in this stack, as the
SC-PGO map; SC-PGO loop closures can move the map from it by a few centimetres, --radius covers that.

usage: pruneFarGraph.py <graph.vgh> <map.pcd> <dynamic.pcd> [--output pruned.vgh]
       By default the graph is pruned in place and the original kept as <graph stem>_unpruned.vgh.
"""

import argparse
import os
import shutil
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, QhullError, cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dynamic_removal  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("graph")
parser.add_argument("map", help="map without the dynamic points (makeMergedMap.py output)")
parser.add_argument("dynamic", help="the removed points (makeMergedMap.py --dynamic-output)")
parser.add_argument("--output", default=None, help="pruned graph (default: in place, original kept as <stem>_unpruned.vgh)")
parser.add_argument("--radius", type=float, default=0.5, help="[m] discs around the nodes of a polygon of one or two (FAR nodes sit ~0.25 m off obstacles)")
parser.add_argument("--min-dynamic", type=int, default=3, help="removed points a ghost polygon needs inside")
parser.add_argument("--static-ratio", type=float, default=0.2, help="a ghost polygon has fewer static points inside than this times its removed ones")
parser.add_argument("--min-height", type=float, default=0.1, help="[m] above the floor")
parser.add_argument("--max-height", type=float, default=2.5, help="[m] above the floor")
args = parser.parse_args()


class Node:
    # graph_decoder line: id free_direct x y z surf_dir1(3) surf_dir2(3) covered frontier navpoint boundary connect_ids
    #                     | poly_ids | contour_ids | traj_ids
    HEAD = 15

    def __init__(self, line):
        first, poly, contour, traj = (line.rstrip("\n").split("|") + ["", "", ""])[:4]
        t = first.split()
        self.id = int(t[0])
        self.head = t[:self.HEAD]
        self.xy = np.array([float(t[2]), float(t[3])])
        self.navpoint = t[13] == "1"
        self.lists = [[int(x) for x in s.split()] for s in (" ".join(t[self.HEAD:]), poly, contour, traj)]

    def line(self):
        # as graph_decoder writes it: every value followed by a space, groups separated by "| "
        return " ".join(self.head) + " " + "| ".join("".join(f"{i} " for i in lst) for lst in self.lists) + "\n"


with open(args.graph) as f:
    nodes = {n.id: n for n in (Node(line) for line in f if line.strip())}


def band_xy(path):
    p = np.asarray(o3d.io.read_point_cloud(path).points)
    return p


static_points = band_xy(args.map)
dynamic_points = band_xy(args.dynamic)
floor_z = dynamic_removal.estimate_floor_z(static_points[:, 2])
lo, hi = floor_z + args.min_height, floor_z + args.max_height
static_xy = static_points[(static_points[:, 2] > lo) & (static_points[:, 2] < hi), :2]
dynamic_xy = dynamic_points[(dynamic_points[:, 2] > lo) & (dynamic_points[:, 2] < hi), :2] if len(dynamic_points) else np.zeros((0, 2))
print(f"graph {args.graph}: {len(nodes)} nodes; floor at z = {floor_z:.2f} m; {len(static_xy)} static and "
      f"{len(dynamic_xy)} removed points {args.min_height}-{args.max_height} m above it")

static_tree = cKDTree(static_xy)
dynamic_tree = cKDTree(dynamic_xy) if len(dynamic_xy) else None


def inside(component, xy, tree):
    """How many of the points (xy, its tree) lie inside the outline of the polygon."""
    if not len(xy):
        return 0
    corners = np.array([nodes[i].xy for i in component])
    if len(corners) >= 3:
        try:
            return int(np.sum(Delaunay(corners).find_simplex(xy) >= 0))
        except QhullError:  # all on a line
            pass
    near = set()
    for c in corners:
        near.update(tree.query_ball_point(c, args.radius))
    return len(near)


# obstacle polygons: components of the contour edges among the non-navpoint nodes
obstacle = {n.id for n in nodes.values() if not n.navpoint}
removed, seen, ghosts = set(), set(), []
for start in sorted(obstacle):
    if start in seen:
        continue
    component, stack = [], [start]
    seen.add(start)
    while stack:
        i = stack.pop()
        component.append(i)
        for j in nodes[i].lists[2]:
            if j in obstacle and j not in seen:
                seen.add(j)
                stack.append(j)
    n_dynamic = inside(component, dynamic_xy, dynamic_tree)
    if n_dynamic < args.min_dynamic:
        continue
    n_static = inside(component, static_xy, static_tree)
    if n_static < args.static_ratio * n_dynamic:
        removed.update(component)
        ghosts.append((component, n_dynamic, n_static))

for component, n_dynamic, n_static in ghosts:
    centre = np.mean([nodes[i].xy for i in component], axis=0)
    print(f"  polygon of {len(component)} nodes at ({centre[0]:.2f}, {centre[1]:.2f}): {n_dynamic} removed / "
          f"{n_static} static points inside; nodes {' '.join(map(str, sorted(component)))}")
for i in removed:
    del nodes[i]
for n in nodes.values():
    n.lists = [[j for j in lst if j not in removed] for lst in n.lists]

output = args.output
if output is None:
    stem, ext = os.path.splitext(args.graph)
    shutil.copyfile(args.graph, stem + "_unpruned" + ext)
    output = args.graph
with open(output, "w") as f:
    for n in nodes.values():
        f.write(n.line())
print(f"{len(removed)} nodes removed ({len(ghosts)} ghost polygons), {len(nodes)} left: {output}")
