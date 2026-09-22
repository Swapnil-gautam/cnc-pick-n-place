"""Suction grasp planning: where to put the 4-cup tool on a flat region.

A cup only seals if its whole disc lands on wood -- not over an edge, the kerf
gap, or a drilled hole. Candidates are scored by distance from the region's
center (closest to the center of mass = least peel moment when lifting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .assets import CUP_OFFSETS_YZ, CUP_RADIUS

EDGE_MARGIN = 0.003


@dataclass(frozen=True)
class Region:
    """A flat pickable area in its object's frame: rounded rectangle minus holes."""

    center: tuple[float, float]
    half_length: float  # along object x
    half_width: float
    corner_radius: float = 0.0
    holes: tuple[tuple[float, float, float], ...] = ()  # (x, y, radius), object frame


@dataclass(frozen=True)
class Grasp:
    xy: tuple[float, float]  # tool center, object frame
    yaw: float  # tool yaw relative to the object (tool long axis vs object x)


def cup_centers(xy: tuple[float, float], yaw: float) -> np.ndarray:
    """Cup centers for the tool at `xy` with long axis at `yaw` (same frame as xy)."""
    long_axis = np.array([math.cos(yaw), math.sin(yaw)])
    short_axis = np.array([-math.sin(yaw), math.cos(yaw)])
    return np.array([np.asarray(xy) + y * short_axis + z * long_axis for y, z in CUP_OFFSETS_YZ])


def _inside_rounded_rect(p: np.ndarray, region: Region, shrink: float) -> bool:
    hx, hy = region.half_length - shrink, region.half_width - shrink
    if hx <= 0 or hy <= 0:
        return False
    r = max(region.corner_radius - shrink, 0.0)
    dx, dy = abs(p[0] - region.center[0]), abs(p[1] - region.center[1])
    if dx > hx or dy > hy:
        return False
    cx, cy = hx - r, hy - r
    if dx > cx and dy > cy:
        return (dx - cx) ** 2 + (dy - cy) ** 2 <= r * r
    return True


def cups_fit(region: Region, xy: tuple[float, float], yaw: float) -> bool:
    for c in cup_centers(xy, yaw):
        if not _inside_rounded_rect(c, region, CUP_RADIUS + EDGE_MARGIN):
            return False
        for hx, hy, hr in region.holes:
            if math.hypot(c[0] - hx, c[1] - hy) < CUP_RADIUS + hr + EDGE_MARGIN:
                return False
    return True


def plan_grasp(region: Region, search: float = 0.08, step: float = 0.005) -> Grasp | None:
    """Best cup placement on `region`, or None if the tool can't seal anywhere on it."""
    offsets = np.arange(-search, search + 1e-9, step)
    best, best_d = None, float("inf")
    for yaw in (0.0, math.pi / 2):
        for dx in offsets:
            for dy in offsets:
                xy = (region.center[0] + dx, region.center[1] + dy)
                d = math.hypot(dx, dy)
                if d < best_d and cups_fit(region, xy, yaw):
                    best, best_d = Grasp(xy, yaw), d
    return best
