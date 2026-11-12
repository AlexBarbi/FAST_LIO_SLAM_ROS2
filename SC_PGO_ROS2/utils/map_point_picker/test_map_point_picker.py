"""The Start inspection button of the page: /api/inspection saves the points and tells far_inspection_goals.py to go.

    python3 -m pytest test_map_point_picker.py      (ROS 2 sourced, for the last test)
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import map_point_picker as picker

os.environ["ROS_DOMAIN_ID"] = "93"  # the last test publishes the start command: never on the domain of a robot

POINTS = [{"x": 4.0, "y": 1.0, "z": 0.05}, {"x": 6.0, "y": -2.0, "z": 0.04}]


class FakeLive:
    def __init__(self, error=None):
        self.error = error
        self.started = []

    def start_inspection(self, path):
        if self.error is None:
            self.started.append(path)
        return self.error

    def start_exit_inspection(self, path):
        return self.start_inspection(path)

    def exits_live(self):
        return {"verdicts": {"1": {"verdict": "WARNING", "reason": "live", "age": 0.5}}, "pass": {"state": "going"}}


@pytest.fixture
def page(tmp_path):
    """post(path, body) -> (status, json) on a server saving to tmp_path, with page.live to set."""
    state = {"meta": {"pcd": "/maps/optimized_map.pcd", "default_name": "selected_points.yaml",
                      "live_error": "turned off (--no-live)"},
             "live": None, "out_dir": str(tmp_path), "graph_dir": str(tmp_path), "local_only": True,
             "map_dir": str(tmp_path)}
    server = ThreadingHTTPServer(("127.0.0.1", 0), picker.make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def post(path, body):
        request = urllib.request.Request("http://127.0.0.1:%d%s" % (server.server_address[1], path),
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def get(path):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d%s" % (server.server_address[1], path)) as r:
                body = r.read()
                return r.status, json.loads(body) if r.headers.get_content_type() == "application/json" else body
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    post.state = state
    post.get = get
    yield post
    server.shutdown()
    server.server_close()


def test_start_saves_the_points_and_sends_their_file_to_the_inspection(page, tmp_path):
    page.state["live"] = live = FakeLive()
    status, answer = page("/api/inspection", {"name": "round_two.yaml", "points": POINTS})

    saved = str(tmp_path / "round_two.yaml")
    assert (status, answer) == (200, {"path": saved, "count": 2})
    assert picker.read_points(saved) == POINTS
    assert live.started == [saved]


def test_start_without_ros_says_so_and_still_saves(page, tmp_path):
    status, answer = page("/api/inspection", {"name": "selected_points.yaml", "points": POINTS})

    assert status == 503
    assert "turned off (--no-live)" in answer["error"]
    assert picker.read_points(str(tmp_path / "selected_points.yaml")) == POINTS


def test_start_with_no_inspection_node_listening_reports_it(page):
    page.state["live"] = FakeLive(error="nobody listens on /start_inspection")
    status, answer = page("/api/inspection", {"name": "selected_points.yaml", "points": POINTS})

    assert status == 503
    assert "nobody listens on /start_inspection" in answer["error"]


def test_start_needs_points_and_a_plain_file_name(page, tmp_path):
    page.state["live"] = live = FakeLive()

    assert page("/api/inspection", {"name": "selected_points.yaml", "points": []})[0] == 400
    assert page("/api/inspection", {"name": "../elsewhere.yaml", "points": POINTS})[0] == 400
    assert live.started == []
    assert os.listdir(tmp_path) == []


def test_saving_alone_starts_nothing(page):
    page.state["live"] = live = FakeLive()

    assert page("/api/points", {"name": "selected_points.yaml", "points": POINTS})[0] == 200
    assert live.started == []


def test_the_bridge_publishes_the_file_once_somebody_listens():
    rclpy = pytest.importorskip("rclpy")
    from std_msgs.msg import String

    bridge = picker.LiveBridge(None, "lio_base")
    try:
        error = bridge.start_inspection("/maps/selected_points.yaml")
        assert error is not None and "far_planner_launch.xml" in error  # tells what to start

        got = []
        listener = rclpy.create_node("far_inspection_goals_stand_in")
        listener.create_subscription(String, "/start_inspection", lambda msg: got.append(msg.data), 5)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not got:
            if bridge.start_inspection("/maps/selected_points.yaml") is None:
                end = time.monotonic() + 2.0
                while time.monotonic() < end and not got:
                    rclpy.spin_once(listener, timeout_sec=0.1)
            else:
                rclpy.spin_once(listener, timeout_sec=0.1)
        assert got == ["/maps/selected_points.yaml"]
    finally:
        bridge.shutdown()


# ---------------------------------------------------------------- exit inspection overlay

def write_exits(tmp_path, report=True):
    import yaml
    folder = tmp_path / "exit_inspection"
    folder.mkdir()
    exits = {"exits": [
        {"id": 0, "type": "route", "cls": "exit_arrow_side", "confirmed": True, "sign_center": [9.7, 0.8, 1.8],
         "normal": [-1.0, 0.0, 0.0]},
        {"id": 1, "type": "door", "cls": "exit_plain", "confirmed": True, "sign_center": [3.8, 5.95, 1.8],
         "normal": [0.0, -1.0, 0.0], "door_floor_center": [3.8, 5.95, -0.44], "door_width": 1.2}]}
    (folder / "exits.yaml").write_text(yaml.safe_dump(exits))
    if report:
        rep = {"finished": "2026-11-11T17:00:00", "summary": {"CLEAR": 0, "WARNING": 0, "BLOCKED": 1, "UNVERIFIED": 0},
               "params": {"margin": 0.5, "depth": 3.0},
               "exits": [{"id": 1, "verdict": "BLOCKED", "reason": "pallet", "viewpoint": {"x": 3.8, "y": 1.45,
                                                                                         "yaw": 1.57}}]}
        (folder / "exit_report.yaml").write_text(yaml.safe_dump(rep))
        for stamp in ("20261111-160000", "20261111-170000"):
            (folder / "reports" / stamp).mkdir(parents=True)
            (folder / "reports" / stamp / "report.html").write_text("<html>%s</html>" % stamp)
    return folder


def test_exits_come_with_the_verdicts_of_the_last_report(page, tmp_path):
    write_exits(tmp_path)
    status, j = page.get("/api/exits")
    assert status == 200 and j["live"] is None
    route, door = j["exits"]
    assert route["type"] == "route" and "verdict" not in route
    assert door["verdict"] == "BLOCKED" and door["viewpoint"]["y"] == 1.45
    assert j["report"]["summary"]["BLOCKED"] == 1 and j["report"]["html"].endswith("20261111-170000/report.html")
    status, body = page.get("/api/report")
    assert status == 200 and b"20261111-170000" in body


def test_no_exit_folder_means_no_exits(page):
    status, j = page.get("/api/exits")
    assert status == 200 and j["exits"] == [] and j["report"] is None
    assert page.get("/api/report")[0] == 404


def test_live_exit_verdicts_and_the_start_button(page, tmp_path):
    folder = write_exits(tmp_path, report=False)
    assert page("/api/exit_inspection", {})[0] == 503  # no ROS 2
    page.state["live"] = live = FakeLive()
    status, j = page.get("/api/exits")
    assert j["live"]["verdicts"]["1"]["verdict"] == "WARNING" and j["live"]["pass"]["state"] == "going"
    status, answer = page("/api/exit_inspection", {})
    assert status == 200 and live.started == [str(folder / "exits.yaml")]
    page.state["live"] = FakeLive(error="nobody listens on /start_exit_inspection")
    assert page("/api/exit_inspection", {})[0] == 503


def test_the_start_button_needs_an_exits_file(page):
    page.state["live"] = live = FakeLive()
    assert page("/api/exit_inspection", {})[0] == 400 and live.started == []
