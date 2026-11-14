"""Find the points of things that moved (people, carts, forklifts) in a stack of SC-PGO keyframes.

Visibility check, as Removert (Kim & Kim, IROS 2020) does it on the same SC-PGO keyframes: every map point is looked
at from every keyframe. In the range image of a keyframe (its own scan, lidar frame), a map point that lies in front
of what the keyframe saw in that direction - the beams went through where the point is, to something farther - is
empty space for that keyframe: a vote for "dynamic". A map point at about the range the keyframe saw there is a vote
for "static". Points with enough dynamic votes, and several times more of them than static ones, are removed.

What the keyframe saw in a direction is the nearest return in a small window of pixels around it (min filter), not
the one beam through the point: a beam that grazes the floor or a wall passes close to the surface points seen from
elsewhere, but its neighbour beams hit that surface nearer, so the window's minimum keeps those points. The window
also stops gaps between the sparse beams from counting as empty space.

It does not need a detector: a person the people filter missed (far, behind the robot) is removed when another
keyframe sees through the place where they stood. Someone standing still at the same place for every keyframe that
sees it stays, as anything static.

Votes are cast per voxel (the map points in a voxel share them); points of a removed voxel all go.

Params tuned 2026-11-10 on the simulated warehouse (Airy tilted 5 deg, two walking actors, ground truth from their
scripts), 174 keyframes of a 330 s exploration: 87 % of the person points removed from unfiltered keyframes, 68 % of
those the people filter had left, no static point removed; 44 SC-PGO keyframes of another run: 92 % of the leaked
person points, 28 static points (shelf legs). Live empty-room exploration (people filter on, 205 keyframes): 95 % of
the 1390 leaked person points, no static point; open-floor points at person height 1326 -> 21. What stays: someone
standing against a wall (closer to it than the margin), feet (under min_height), people standing still for most of
the keyframes that see them.
"""

import numpy as np
from scipy.ndimage import minimum_filter


class Params:
    voxel = 0.1            # [m] votes are per voxel of this size
    res_azimuth = 1.6      # [deg] range image columns: the Airy columns that reach SC-PGO (every 4th of 900)
    res_elevation = 1.0    # [deg] range image rows: the Airy rings are 0.94 deg apart
    window = (3, 3)        # pixels (rows, columns) of the min filter: what a keyframe saw around a direction
    near = 0.7             # [m] map points closer than this to a keyframe get no vote from it (robot body, blind)
    far = 30.0             # [m] nor farther than this (range image too coarse to tell)
    margin = 0.3           # [m] a point is in front of the seen surface when nearer than it by more than
    margin_ratio = 0.04    #     max(margin, margin_ratio * range): pose and range errors grow with range
    beam_tolerance = 0.15  # [m] an empty-space vote also needs the beam of the pixel to pass this close to the voxel:
                           # thin posts (shelf legs) between the sparse beams are not seen through; 0: no check
    min_dynamic = 3        # votes needed to remove a voxel ...
    dynamic_ratio = 3.0    # ... and at least this many times its static votes
    # Only voxels this high above the floor can go: people and vehicles stand on it. Below, the floor at the edge of
    # what a tilted lidar sees; above, the ceiling, whose points move with the drift of FAST-LIO in z between
    # keyframes far apart in time and would get votes as if they had moved.
    min_height = 0.1       # [m]
    max_height = 2.5       # [m]


def _spherical(points):
    r = np.linalg.norm(points, axis=1)
    elevation = np.degrees(np.arcsin(np.clip(points[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    return r, elevation, azimuth


class RangeImage:
    """Nearest return per pixel of one keyframe scan (lidar frame), and its minimum over a window."""

    def __init__(self, scan_local, p):
        self.p = p
        r, el, az = _spherical(scan_local)
        self.el_min = -90.0
        self.rows = int(np.ceil(180.0 / p.res_elevation))
        self.cols = int(np.ceil(360.0 / p.res_azimuth))
        row, col = self.pixel(el, az)
        image = np.full((self.rows, self.cols), np.inf)
        np.minimum.at(image, (row, col), r)
        # direction of the nearest beam of each pixel (farthest written first, the nearest last and kept)
        order = np.argsort(-r)
        self.direction = np.zeros((self.rows, self.cols, 3))
        self.direction[row[order], col[order]] = scan_local[order] / np.maximum(r[order], 1e-9)[:, None]
        # columns wrap around at +-180 deg
        pad = p.window[1] // 2
        wrapped = np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1) if pad else image
        self.nearest = minimum_filter(wrapped, size=p.window, mode="constant", cval=np.inf)[:, pad:pad + self.cols]
        self.image = image
        # lowest row with a return in each column: the window has no beam under it to hit what the row's beams pass
        # just over (a box whose top is at the height of the lidar, seen by the lowest beams of another keyframe)
        valid = np.isfinite(image)
        self.lowest_row = np.where(valid.any(axis=0), np.argmax(valid, axis=0), self.rows)

    def pixel(self, el, az):
        row = np.clip(((el - self.el_min) / self.p.res_elevation).astype(int), 0, self.rows - 1)
        col = ((az + 180.0) / self.p.res_azimuth).astype(int) % self.cols
        return row, col


def estimate_floor_z(z):
    """The lowest 5 cm height band, within 0.3 m above the lowest points of the map, holding at least half as many
    points as the fullest band there. A lidar tilted down only a little sees the floor far ahead only: its layer can be
    much smaller than shelf tops or the ceiling (warehouse: 15 k floor points against a 140 k shelf-top layer), or
    hardly more than the walls in each band, but nothing real is under it."""
    bottom = np.percentile(z, 0.05)  # a few points under the floor: noise, reflections
    counts, edges = np.histogram(z[(z >= bottom) & (z < bottom + 0.3)], bins=6, range=(bottom, bottom + 0.3))
    band = int(np.argmax(counts >= 0.5 * counts.max()))
    # the fullest centimetre of that band
    fine, fine_edges = np.histogram(z[(z >= edges[band]) & (z < edges[band + 1])], bins=5, range=(edges[band], edges[band + 1]))
    peak = int(np.argmax(fine))
    return 0.5 * (fine_edges[peak] + fine_edges[peak + 1])


def find_dynamic(scans_local, poses, p=None, log=print):
    """scans_local: list of (N_i, 3) keyframe scans in their lidar frame; poses: list of 4x4 lidar-to-map.
    Returns a list of boolean masks, True for the points of each scan that belong to something that moved."""
    p = p or Params()
    world = [s @ T[:3, :3].T + T[:3, 3] for s, T in zip(scans_local, poses)]
    all_points = np.vstack(world)
    floor_z = estimate_floor_z(all_points[:, 2])
    keys = np.floor(all_points / p.voxel).astype(np.int64)
    _, voxel_of_point, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    voxel_of_point = voxel_of_point.ravel()
    centers = np.zeros((len(counts), 3))
    np.add.at(centers, voxel_of_point, all_points)
    centers /= counts[:, None]
    dynamic = np.zeros(len(counts), np.int32)
    static = np.zeros(len(counts), np.int32)
    height = centers[:, 2] - floor_z
    candidate = np.nonzero((height >= p.min_height) & (height <= p.max_height))[0]
    log(f"dynamic removal: {len(all_points)} points in {len(counts)} voxels of {p.voxel} m, {len(scans_local)} "
        f"keyframes; floor at z = {floor_z:.2f} m, {len(candidate)} voxels {p.min_height}-{p.max_height} m above it")

    for scan, T in zip(scans_local, poses):
        image = RangeImage(scan, p)
        local = (centers[candidate] - T[:3, 3]) @ T[:3, :3]
        r, el, az = _spherical(local)
        use = (r > p.near) & (r < p.far)
        idx = candidate[use]
        r, el, az = r[use], el[use], az[use]
        row, col = image.pixel(el, az)
        nearest = image.nearest[row, col]
        seen = np.isfinite(nearest)
        idx, r_v, nearest, row, col, v_local = idx[seen], r[seen], nearest[seen], row[seen], col[seen], local[use][seen]
        margin = np.maximum(p.margin, p.margin_ratio * r_v)
        empty = (r_v < nearest - margin) & (row > image.lowest_row[col])
        if p.beam_tolerance > 0:
            # distance from the voxel to the beam line of its pixel
            beam = image.direction[row, col]
            along = np.sum(v_local * beam, axis=1)
            off_beam = np.linalg.norm(v_local - along[:, None] * beam, axis=1)
            empty &= off_beam < p.beam_tolerance
        dynamic[idx[empty]] += 1
        static[idx[np.abs(r_v - nearest) <= margin]] += 1

    removed_voxel = (dynamic >= p.min_dynamic) & (dynamic >= p.dynamic_ratio * static)
    removed = removed_voxel[voxel_of_point]
    log(f"dynamic removal: {removed_voxel.sum()} voxels, {removed.sum()} points ({100.0 * removed.mean():.2f} %) removed")
    masks, start = [], 0
    for s in scans_local:
        masks.append(removed[start:start + len(s)])
        start += len(s)
    return masks
