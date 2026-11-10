"""dynamic_removal on simulated keyframes: a 10 x 10 x 3 m room with a thin post, seen by an Airy-like lidar (every 4th
column, 96 rings over a 0-90 deg dome, tilted 5 deg forward) from keyframes 1 m apart; a person stands in front of the
first two keyframes only, then is gone; the keyframes go along the room and come back.

Run: <workspace>/.venv-yolo/bin/python -m pytest SC_PGO_ROS2/utils/python/test_dynamic_removal.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dynamic_removal as dr  # noqa: E402

FLOOR = -0.47           # in the lidar frame of a standing robot
HALF = 5.0              # room half size
CEILING = 2.53
POST = (2.0, -2.0, 0.04)        # x, y, radius: a shelf leg
PERSON = (2.5, 1.0, 0.25)       # x, y, radius, from the floor to 1.75 m
PERSON_TOP = FLOOR + 1.75


def beams():
    tilt = np.radians(5.0)
    az = np.radians(np.arange(0.0, 360.0, 1.6))
    el = np.radians(np.linspace(0.0, 90.0, 96))
    A, E = np.meshgrid(az, el)
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)
    # dome around the tilted sensor axis: rotate about y by +5 deg (pitch forward)
    R = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    return d @ R.T


def cylinder_hit(o, d, cx, cy, radius, z_top):
    """Range to a vertical cylinder from the floor to z_top, inf if missed."""
    ox, oy = o[0] - cx, o[1] - cy
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox ** 2 + oy ** 2 - radius ** 2
    disc = b * b - 4 * a * c
    t = np.full(len(d), np.inf)
    ok = (disc >= 0) & (a > 1e-12)
    t_ok = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
    z = o[2] + t_ok * d[ok, 2]
    t_ok[(t_ok <= 0) | (z < FLOOR) | (z > z_top)] = np.inf
    t[ok] = t_ok
    return t


def scan(o, person):
    d = beams()
    t = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis, bound in ((0, HALF), (0, -HALF), (1, HALF), (1, -HALF), (2, FLOOR), (2, CEILING)):
            ti = (bound - o[axis]) / d[:, axis]
            t = np.where((ti > 0) & (ti < t), ti, t)
    t = np.minimum(t, cylinder_hit(o, d, *POST, CEILING))
    if person:
        t = np.minimum(t, cylinder_hit(o, d, *PERSON, PERSON_TOP))
    world = o + d * t[:, None]
    return world[np.isfinite(t) & (t > 0.1) & (t < 60.0)]


def path():
    """Keyframes 1 m apart: along the room and back on a parallel line, as an exploration that turns around."""
    out = [np.array([x, -0.5, 0.0]) for x in np.arange(-3.0, 3.01, 1.0)]
    return out + [np.array([x, -1.5, 0.0]) for x in np.arange(3.0, -3.01, -1.0)]


def keyframes():
    scans, poses, person_masks = [], [], []
    for k, o in enumerate(path()):
        world = scan(o, person=k < 2)  # the person leaves after the first two keyframes
        T = np.eye(4)
        T[:3, 3] = o
        scans.append(world - o)
        poses.append(T)
        r_xy = np.hypot(world[:, 0] - PERSON[0], world[:, 1] - PERSON[1])
        person_masks.append((r_xy < PERSON[2] + 0.02) & (world[:, 2] < PERSON_TOP + 0.02) & (k < 2))
    return scans, poses, person_masks


def test_a_person_who_left_is_removed_and_the_room_stays():
    scans, poses, person_masks = keyframes()
    masks = dr.find_dynamic(scans, poses, log=lambda *_: None)
    removed = np.concatenate(masks)
    person = np.concatenate(person_masks)
    world = np.vstack([s + T[:3, 3] for s, T in zip(scans, poses)])
    post = np.hypot(world[:, 0] - POST[0], world[:, 1] - POST[1]) < POST[2] + 0.02

    body = person & (world[:, 2] > FLOOR + 0.2)  # the feet stay: nothing under min_height (0.1 m) is removed
    assert body.sum() > 50
    assert removed[body].mean() > 0.9
    assert post.sum() > 20
    assert removed[post].mean() < 0.01  # thin, between the beams of other keyframes: no see-through votes
    assert removed[~person].mean() < 0.001


def test_a_person_seen_by_every_keyframe_stays():
    scans, poses = [], []
    for o in path():
        T = np.eye(4)
        T[:3, 3] = o
        scans.append(scan(o, person=True) - o)
        poses.append(T)
    masks = dr.find_dynamic(scans, poses, log=lambda *_: None)
    assert np.concatenate(masks).mean() < 0.001


def test_floor_estimate_ignores_a_bigger_layer_above_it():
    rng = np.random.default_rng(0)
    floor = rng.normal(-0.5, 0.01, 15000)
    shelf_tops = rng.normal(0.65, 0.01, 140000)
    ceiling = rng.normal(8.7, 0.02, 3000000)
    noise = rng.uniform(-0.62, -0.55, 50)
    z = np.concatenate([floor, shelf_tops, ceiling, noise])
    assert abs(dr.estimate_floor_z(z) + 0.5) < 0.05
