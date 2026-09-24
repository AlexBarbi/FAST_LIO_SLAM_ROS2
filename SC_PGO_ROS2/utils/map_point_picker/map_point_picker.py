#!/usr/bin/env python3
"""Pick points on a PCD map from a top-down view in the browser and save them to a file.

    python3 map_point_picker.py [map.pcd] [--output selected_points.yaml] [--port 8765]

then open http://localhost:8765. Click the map to add a point, drag a point to move it,
right-click it to delete it, and press Save (or Ctrl+S). The points are in the frame of the
map (for SC-PGO's optimized_map.pcd, the frame of the first keyframe). Their z is the local
ground height under the click.

The page also overlays a FAR planner graph (.vgh saved by graph_decoder, e.g. warehouse.vgh from
the TARE exploration): its nodes and its trajectory, obstacle-contour and visibility edges. The
graph is drawn as saved, so it lines up only if it was built in the frame of the map, as the
exploration's graph and map are.

Live view: when rclpy can be imported (ROS 2 sourced, same ROS_DOMAIN_ID as the robot), the page
also shows where the robot is (FAST-LIO's lio_base), its trail, the path it is following (FAR's
/viz_path_topic and /way_point, TARE's /exploration_path) and OptiPessi's current plan (the MPC's
CoM trajectory and its goal). Everything is brought to the frame of the map with TF: global_map
when fast_lio_localization runs, else camera_init (the exploration, or a FAST-LIO started at the
spawn). --no-live turns it off.

Needs only numpy: the PCD reader handles ascii, binary and binary_compressed (LZF) files,
so the system python3 works (open3d is not needed).
"""
import argparse
import csv
import io
import json
import math
import os
import re
import struct
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORMATS = (".yaml", ".yml", ".json", ".csv", ".txt")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


# ---------------------------------------------------------------- PCD reading

def lzf_decompress(src, out_len):
    # LZF as written by PCL's binary_compressed: a control byte < 32 starts a literal run of
    # ctrl + 1 bytes, anything else is a back-reference of length (ctrl >> 5) + 2 (7 means
    # "read one more length byte") at distance ((ctrl & 0x1f) << 8 | next byte) + 1.
    out = bytearray(out_len)
    ip = op = 0
    n = len(src)
    while ip < n:
        ctrl = src[ip]
        ip += 1
        if ctrl < 32:
            ln = ctrl + 1
            out[op:op + ln] = src[ip:ip + ln]
            ip += ln
            op += ln
        else:
            ln = ctrl >> 5
            if ln == 7:
                ln += src[ip]
                ip += 1
            ref = op - ((ctrl & 0x1f) << 8) - src[ip] - 1
            ip += 1
            ln += 2
            d = op - ref
            if d >= ln:
                out[op:op + ln] = out[ref:ref + ln]
            else:  # the reference overlaps the output: it repeats with period d
                out[op:op + ln] = (out[ref:op] * (ln // d + 1))[:ln]
            op += ln
    if op != out_len:
        raise ValueError("corrupt binary_compressed data: got %d bytes, expected %d" % (op, out_len))
    return bytes(out)


def read_pcd_xyz(path):
    """Return the x, y, z columns of a PCD file as an (N, 3) float32 array, NaN points dropped."""
    with open(path, "rb") as f:
        raw = f.read()
    header = {}
    pos = 0
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("ascii", "replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        header[key.upper()] = value.split()
        if key.upper() == "DATA":
            break

    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    num = int(header["POINTS"][0])
    data = header["DATA"][0].lower()
    for axis in "xyz":
        if axis not in fields:
            raise ValueError("%s has no %s field (fields: %s)" % (path, axis, " ".join(fields)))

    kind = {"F": "f", "I": "i", "U": "u"}
    names = [name if name != "_" else "_pad%d" % i for i, name in enumerate(fields)]
    dtypes = [(n, "<%s%d" % (kind[t], s), (c,)) if c > 1 else (n, "<%s%d" % (kind[t], s))
              for n, t, s, c in zip(names, types, sizes, counts)]

    if data == "ascii":
        cols = [fields.index(a) for a in "xyz"]
        # COUNT > 1 fields take several columns; find the column where each axis starts
        starts = np.cumsum([0] + counts[:-1])
        text = raw[pos:].decode("ascii", "replace")
        xyz = np.loadtxt(io.StringIO(text), usecols=[starts[c] for c in cols], dtype=np.float64, ndmin=2)
        xyz = xyz[:num].astype(np.float32)
    elif data == "binary":
        cloud = np.frombuffer(raw, dtype=np.dtype(dtypes), count=num, offset=pos)
        xyz = np.stack([cloud[a].astype(np.float32) for a in "xyz"], axis=1)
    elif data == "binary_compressed":
        comp_size, raw_size = struct.unpack("<II", raw[pos:pos + 8])
        buf = lzf_decompress(raw[pos + 8:pos + 8 + comp_size], raw_size)
        # binary_compressed stores the fields one after the other (all x, then all y, ...)
        cols = {}
        off = 0
        for (name, t, s, c) in zip(names, types, sizes, counts):
            nbytes = s * c * num
            if name in ("x", "y", "z"):
                cols[name] = np.frombuffer(buf, dtype="<%s%d" % (kind[t], s), count=num * c, offset=off)[::c]
            off += nbytes
        xyz = np.stack([cols[a].astype(np.float32) for a in "xyz"], axis=1)
    else:
        raise ValueError("unsupported PCD DATA type: %s" % data)

    return xyz[np.isfinite(xyz).all(axis=1)]


def estimate_floor_z(z):
    # Same as makeMergedMap.py: the most populated 5 cm height band in the lower half of the map.
    z_low = z[z <= np.median(z)]
    num_bins = max(1, int(np.ceil((z_low.max() - z_low.min()) / 0.05)))
    counts, edges = np.histogram(z_low, bins=num_bins)
    peak = np.argmax(counts)
    return float(0.5 * (edges[peak] + edges[peak + 1]))


def height_above_floor(xyz, cell=1.0, min_count=10):
    """Height of each point above the floor around it.

    One floor height for the whole map is not enough: SLAM maps drift in z (the Sep 23 warehouse
    map's floor rises by about 0.4 m across 30 m). The local floor is the 5th percentile of z in
    each cell, then the lowest over the 3x3 neighbouring cells, so a cell that only sees the top
    of a shelf or a wall takes the floor next to it. Points with no dense cell nearby get +inf.
    """
    ij = np.floor(xyz[:, :2] / cell).astype(np.int64)
    ij -= ij.min(axis=0)
    nx, ny = ij.max(axis=0) + 1
    key = ij[:, 0] * ny + ij[:, 1]
    order = np.lexsort((xyz[:, 2], key))  # by cell, then by z inside the cell
    k_sorted, z_sorted = key[order], xyz[order, 2]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    counts = np.diff(np.r_[starts, len(k_sorted)])
    dense = counts >= min_count
    grid = np.full(nx * ny, np.inf)
    grid[k_sorted[starts[dense]]] = z_sorted[starts[dense] + (counts[dense] * 0.05).astype(np.int64)]
    pad = np.pad(grid.reshape(nx, ny), 1, constant_values=np.inf)
    floor = np.min([pad[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny] for dx in (-1, 0, 1) for dy in (-1, 0, 1)], axis=0)
    local = floor[ij[:, 0], ij[:, 1]]
    return np.where(np.isfinite(local), xyz[:, 2] - local, np.inf).astype(np.float32)


# ---------------------------------------------------------------- FAR graph

def read_vgh(path):
    """FAR planner visibility graph as saved by graph_decoder (GraphDecoder::SaveGraphCallBack).

    One node per line: id free_type x y z, the two surface directions (x y z each), is_covered
    is_frontier is_navpoint is_boundary, then the connected node ids, polygon ids, contour ids and
    trajectory ids, separated by "|". Edges are returned as pairs of indices into the node list,
    each once; trajectory edges only from navpoints, as graph_decoder draws them.
    """
    nodes, links = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) < 15:
                continue
            groups = [[int(v) for v in g.split()] for g in " ".join(tok[15:]).split("|")]
            groups += [[]] * (4 - len(groups))
            nodes.append({"id": int(tok[0]), "type": int(tok[1]),
                          "x": float(tok[2]), "y": float(tok[3]), "z": float(tok[4]),
                          "covered": tok[11] != "0", "frontier": tok[12] != "0",
                          "navpoint": tok[13] != "0", "boundary": tok[14] != "0"})
            links.append(groups)
    index = {n["id"]: i for i, n in enumerate(nodes)}
    edges = {"visibility": set(), "contour": set(), "trajectory": set()}
    for i, (connect, _polygon, contour, trajectory) in enumerate(links):
        for kind, ids in (("visibility", connect), ("contour", contour),
                          ("trajectory", trajectory if nodes[i]["navpoint"] else [])):
            for cid in ids:
                j = index.get(cid)
                if j is not None and j != i:
                    edges[kind].add((min(i, j), max(i, j)))
    return {"nodes": nodes, "edges": {k: sorted(v) for k, v in edges.items()}}


def list_graphs(graph_dir):
    try:
        return sorted(f for f in os.listdir(graph_dir) if f.endswith(".vgh") and NAME_RE.match(f))
    except OSError:
        return []


# ---------------------------------------------------------------- point files

def write_points(path, points, map_path):
    ext = os.path.splitext(path)[1].lower()
    rows = [(float(p["x"]), float(p["y"]), float(p["z"])) for p in points]
    if ext == ".json":
        text = json.dumps({"map": map_path, "points": [{"x": x, "y": y, "z": z} for x, y, z in rows]}, indent=2) + "\n"
    elif ext == ".csv":
        text = "x,y,z\n" + "".join("%.4f,%.4f,%.4f\n" % r for r in rows)
    elif ext == ".txt":
        text = "# x y z, map: %s\n" % map_path + "".join("%.4f %.4f %.4f\n" % r for r in rows)
    else:  # .yaml / .yml
        text = "map: %s\npoints:%s\n" % (json.dumps(map_path), "" if rows else " []") + \
               "".join("  - {x: %.4f, y: %.4f, z: %.4f}\n" % r for r in rows)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)  # never leave a half-written file behind


def read_points(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        text = f.read()
    if ext == ".json":
        pts = json.loads(text)["points"]
    elif ext == ".csv":
        pts = list(csv.DictReader(io.StringIO(text)))
    elif ext == ".txt":
        pts = [dict(zip("xyz", line.split())) for line in text.splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    else:
        import yaml  # only needed to load .yaml files back
        pts = (yaml.safe_load(text) or {}).get("points") or []
    return [{"x": float(p["x"]), "y": float(p["y"]), "z": float(p.get("z", 0.0))} for p in pts]


# ---------------------------------------------------------------- live robot data (ROS 2)

def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LiveBridge:
    """Keeps the latest robot data from ROS 2 and brings it to the frame of the map on request.

    Messages are kept in their own frame, as received, and moved with the latest TF when the page asks
    (like tare_path_to_opti_pessi_goal.py: planner and controller stamps come from different clocks).
    OptiPessi publishes in odom, which reaches the map through lio_odom_alignment's odom -> lio_map.
    """
    STALE = 5.0  # s without a message before a path or mark is no longer shown
    ROBOT_STALE = 3.0  # s without /Odometry before the robot is no longer shown

    def __init__(self, map_frame, robot_frame):
        import rclpy
        from geometry_msgs.msg import PointStamped
        from nav_msgs.msg import Odometry, Path
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener
        from visualization_msgs.msg import Marker, MarkerArray

        self.rclpy = rclpy
        self.latest_time = Time()  # "latest available" for TF lookups
        self.map_frame = map_frame  # None: global_map when it exists, else camera_init
        self.robot_frame = robot_frame
        self.lock = threading.Lock()
        self.latest = {}  # key -> (monotonic receive time, frame, (n, 3) array)
        self.odometry_time = None

        rclpy.init(args=[])
        self.node = rclpy.create_node("map_point_picker_live")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        def keep(key, frame, points):
            with self.lock:
                self.latest[key] = (time.monotonic(), frame, np.asarray(points, dtype=float).reshape(-1, 3))

        def xyz(p):
            return (p.x, p.y, p.z)

        def on_odometry(msg):
            self.odometry_time = time.monotonic()

        def on_far_path(msg):  # empty once the goal is reached
            keep("far", msg.header.frame_id, [xyz(p) for p in msg.points])

        def on_tare_path(msg):
            keep("tare", msg.header.frame_id, [xyz(p.pose.position) for p in msg.poses])

        def on_trajectory(msg):
            for m in msg.markers:
                if m.ns == "CoM Trajectory" and m.action == Marker.ADD:
                    keep("opti", m.header.frame_id, [xyz(p) for p in m.points])
                    return

        def on_plan(msg):
            found = {"opti_goal": None, "opti_detour": None}
            for m in msg.markers:
                if m.action == Marker.ADD and m.ns == "goal":
                    found["opti_goal"] = m
                elif m.action == Marker.ADD and m.ns == "detour_goal" and m.type == Marker.SPHERE:
                    found["opti_detour"] = m
            for key, m in found.items():  # absent from the message: no goal / no detour now
                keep(key, m.header.frame_id if m else "", [xyz(m.pose.position)] if m else [])

        self.node.create_subscription(Odometry, "/Odometry", on_odometry, 5)
        self.node.create_subscription(Marker, "/viz_path_topic", on_far_path, 5)
        self.node.create_subscription(PointStamped, "/way_point",
                                      lambda msg: keep("far_waypoint", msg.header.frame_id, [xyz(msg.point)]), 5)
        self.node.create_subscription(Path, "/exploration_path", on_tare_path, 5)
        self.node.create_subscription(MarkerArray, "/opti_pessi/optimizedStateTrajectory", on_trajectory, 1)
        self.node.create_subscription(MarkerArray, "/opti_pessi/plan", on_plan, 1)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def shutdown(self):
        self.executor.shutdown()
        self.rclpy.try_shutdown()

    def transform(self, target, source):
        """(R, t) taking points from source to target, with the latest TF."""
        if not source or source == target:
            return np.eye(3), np.zeros(3)
        tf = self.tf_buffer.lookup_transform(target, source, self.latest_time).transform
        return quat_to_matrix(tf.rotation), np.array([tf.translation.x, tf.translation.y, tf.translation.z])

    def target_frame(self):
        if self.map_frame:
            return self.map_frame
        try:
            self.transform("global_map", "camera_init")
            return "global_map"
        except Exception:
            return "camera_init"

    def snapshot(self):
        now = time.monotonic()
        frame = self.target_frame()
        out = {"ok": True, "frame": frame, "robot": None, "paths": {}, "marks": {}, "errors": []}
        if self.odometry_time is None or now - self.odometry_time > self.ROBOT_STALE:
            out["errors"].append("no /Odometry: FAST-LIO not running (or another ROS_DOMAIN_ID)")
        else:
            try:
                R, t = self.transform(frame, self.robot_frame)
                out["robot"] = {"x": t[0], "y": t[1], "z": t[2], "yaw": math.atan2(R[1, 0], R[0, 0])}
            except Exception as e:
                out["errors"].append("robot: %s" % e)
        with self.lock:
            latest = dict(self.latest)
        for key, (received, source, points) in latest.items():
            if now - received > self.STALE or len(points) == 0:
                continue
            try:
                R, t = self.transform(frame, source)
            except Exception as e:
                out["errors"].append("%s: %s" % (key, e))
                continue
            xy = np.round((points @ R.T + t)[:, :2], 3).tolist()
            if key in ("far", "tare", "opti"):
                out["paths"][key] = xy
            else:
                out["marks"][key] = xy[0]
        return out


# ---------------------------------------------------------------- HTTP server

def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            # Refuse DNS-rebinding requests when serving this machine only: the Host must be local.
            if not state["local_only"]:
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in LOCAL_HOSTS

        def points_path(self, name):
            # Only plain file names inside the output directory, with a known extension.
            if not name or not NAME_RE.match(name) or os.path.splitext(name)[1].lower() not in FORMATS:
                return None
            return os.path.join(state["out_dir"], name)

        def do_GET(self):
            if not self.host_ok():
                return self.send(HTTPStatus.FORBIDDEN, {"error": "bad Host header"})
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                with open(os.path.join(SCRIPT_DIR, "index.html"), "rb") as f:
                    return self.send(HTTPStatus.OK, f.read(), "text/html; charset=utf-8")
            if url.path == "/api/meta":
                # listed on every request: a new exploration run can add or rewrite graphs
                return self.send(HTTPStatus.OK, dict(state["meta"], graph_files=list_graphs(state["graph_dir"])))
            if url.path == "/api/cloud":
                return self.send(HTTPStatus.OK, state["cloud"], "application/octet-stream")
            if url.path == "/api/live":
                if state["live"] is None:
                    return self.send(HTTPStatus.OK, {"ok": False, "error": state["meta"]["live_error"]})
                return self.send(HTTPStatus.OK, state["live"].snapshot())
            if url.path == "/api/graph":
                name = parse_qs(url.query).get("name", [""])[0]
                if not NAME_RE.match(name) or not name.endswith(".vgh"):
                    return self.send(HTTPStatus.BAD_REQUEST, {"error": "graph must be a plain file name ending in .vgh"})
                path = os.path.join(state["graph_dir"], name)
                try:
                    return self.send(HTTPStatus.OK, dict(read_vgh(path), name=name, path=path))
                except FileNotFoundError:
                    return self.send(HTTPStatus.NOT_FOUND, {"error": "%s not found" % path})
                except Exception as e:
                    return self.send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "cannot read %s: %s" % (path, e)})
            if url.path == "/api/points":
                name = parse_qs(url.query).get("name", [state["meta"]["default_name"]])[0]
                path = self.points_path(name)
                if path is None:
                    return self.send(HTTPStatus.BAD_REQUEST, {"error": "file name must be a plain name ending in " + ", ".join(FORMATS)})
                if not os.path.exists(path):
                    return self.send(HTTPStatus.OK, {"name": name, "path": path, "exists": False, "points": []})
                try:
                    return self.send(HTTPStatus.OK, {"name": name, "path": path, "exists": True, "points": read_points(path)})
                except Exception as e:
                    return self.send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "cannot read %s: %s" % (path, e)})
            self.send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            if not self.host_ok():
                return self.send(HTTPStatus.FORBIDDEN, {"error": "bad Host header"})
            # A JSON content type forces a CORS preflight, which this server never answers, so
            # other web pages open in the browser cannot write files through it.
            if urlparse(self.path).path != "/api/points" or \
                    not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self.send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                path = self.points_path(body.get("name"))
                if path is None:
                    return self.send(HTTPStatus.BAD_REQUEST, {"error": "file name must be a plain name ending in " + ", ".join(FORMATS)})
                write_points(path, body["points"], state["meta"]["pcd"])
            except Exception as e:
                return self.send(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            print("saved %d points to %s" % (len(body["points"]), path), flush=True)
            self.send(HTTPStatus.OK, {"path": path, "count": len(body["points"])})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Top-down web viewer of a PCD map to pick points and save them to a file.")
    parser.add_argument("pcd", nargs="?", default="save_data/optimized_map.pcd",
                        help="map to show (default: save_data/optimized_map.pcd)")
    parser.add_argument("--output", default="selected_points.yaml",
                        help="points file; a relative path is next to the map. The extension picks the format: "
                             ".yaml/.yml, .json, .csv or .txt (default: selected_points.yaml). The page can save "
                             "under other names, always in this file's directory.")
    parser.add_argument("--graph", default=None,
                        help="FAR graph (.vgh, as saved by graph_decoder) to overlay; the page can switch to the other "
                             ".vgh files of its directory (default: warehouse.vgh, else the first .vgh next to the map)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default: 127.0.0.1, this machine only). The page can write files, "
                             "so only open it to a network you trust.")
    parser.add_argument("--max-points", type=int, default=2000000,
                        help="randomly thin larger maps to this many points for the browser (default: 2000000)")
    parser.add_argument("--no-live", action="store_true", help="do not show the robot live (no ROS 2)")
    parser.add_argument("--live-frame", default=None,
                        help="TF frame of the map for the live view (default: global_map when fast_lio_localization "
                             "publishes it, else camera_init)")
    parser.add_argument("--robot-frame", default="lio_base",
                        help="TF frame drawn as the robot (default: lio_base, FAST-LIO's base)")
    args = parser.parse_args()

    pcd = os.path.abspath(os.path.expanduser(args.pcd))
    if not os.path.isfile(pcd):
        sys.exit("map not found: %s" % pcd)
    output = os.path.expanduser(args.output)
    if not os.path.isabs(output):
        output = os.path.join(os.path.dirname(pcd), output)
    out_dir, default_name = os.path.split(output)
    if not NAME_RE.match(default_name) or os.path.splitext(default_name)[1].lower() not in FORMATS:
        sys.exit("--output must end in one of %s" % ", ".join(FORMATS))
    os.makedirs(out_dir, exist_ok=True)
    if args.graph:
        graph_dir, default_graph = os.path.split(os.path.abspath(os.path.expanduser(args.graph)))
        if not default_graph.endswith(".vgh") or not NAME_RE.match(default_graph):
            sys.exit("--graph must be a .vgh file")
    else:
        graph_dir = os.path.dirname(pcd)
        graphs = list_graphs(graph_dir)
        default_graph = "warehouse.vgh" if "warehouse.vgh" in graphs else (graphs[0] if graphs else "")

    print("loading %s ..." % pcd, flush=True)
    xyz = read_pcd_xyz(pcd)
    if len(xyz) == 0:
        sys.exit("map has no points")
    n_total = len(xyz)
    if n_total > args.max_points:
        keep = np.random.default_rng(0).choice(n_total, args.max_points, replace=False)
        xyz = xyz[np.sort(keep)]
    z = xyz[:, 2]
    meta = {
        "pcd": pcd,
        "n_points": int(len(xyz)),
        "n_total": int(n_total),
        "min": xyz.min(axis=0).tolist(),
        "max": xyz.max(axis=0).tolist(),
        "z_p01": float(np.percentile(z, 1)),
        "z_p995": float(np.percentile(z, 99.5)),
        "floor_z": estimate_floor_z(z),
        "out_dir": out_dir,
        "default_name": default_name,
        "formats": list(FORMATS),
        "graph_dir": graph_dir,
        "default_graph": default_graph,
    }
    live, meta["live_error"] = None, "turned off (--no-live)"
    if not args.no_live:
        try:
            live, meta["live_error"] = LiveBridge(args.live_frame, args.robot_frame), None
        except ImportError as e:
            meta["live_error"] = "no ROS 2 (%s): source /opt/ros/jazzy/setup.bash and the workspace" % e
    meta["live"] = live is not None
    # sent as x, y, z, height above the local floor (for the page's "Hide floor")
    cloud = np.column_stack([xyz, height_above_floor(xyz)])
    state = {"meta": meta, "cloud": np.ascontiguousarray(cloud, dtype="<f4").tobytes(), "live": live,
             "out_dir": out_dir, "graph_dir": graph_dir, "local_only": args.host in LOCAL_HOSTS}
    print("%d points (%d in the file), floor at z = %.2f m" % (len(xyz), n_total, meta["floor_z"]))
    print("live robot view: " + ("on, ROS_DOMAIN_ID=%s" % os.environ.get("ROS_DOMAIN_ID", "0") if live
                                 else meta["live_error"]))

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print("open http://%s:%d  (points are saved in %s)  Ctrl+C to quit"
          % ("localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host, args.port, out_dir), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if live is not None:
            live.shutdown()


if __name__ == "__main__":
    main()
