"""Jobs (parts to cut), stock planks, and the geometry both imply.

Plank-local frame: origin at the plank center, +x along its length, +y across
its width, top face at z = +thickness/2. The corner registered against the CNC
datum stops is local (-L/2, +W/2) -- with the plank lying along world +Y
(yaw = +90 deg) that corner is the front-left one.

The part is cut MARGIN in from that corner. Beyond the part, the blank keeps a
GRIP_STRIP of solid wood so the robot still has somewhere to put its suction
cups on the skeleton after the part is cut out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MARGIN = 0.025
GRIP_STRIP = 0.11  # fits the vacuum tool's 60 mm cup spread plus cup radius, with room for the kerf
TOOL_DIAMETER = 0.006  # 6 mm end mill
THICKNESS_TOLERANCE = 0.0015


@dataclass(frozen=True)
class Job:
    name: str
    length: float  # part outline, along plank length
    width: float
    thickness: float
    corner_radius: float
    holes: tuple[tuple[float, float, float], ...] = field(default=())  # (x, y, diameter), part-centered

    @property
    def blank_length(self) -> float:
        return self.length + MARGIN + GRIP_STRIP

    @property
    def blank_width(self) -> float:
        return self.width + 2 * MARGIN

    def fits(self, length: float, width: float, thickness: float) -> bool:
        return (length >= self.blank_length and width >= self.blank_width
                and abs(thickness - self.thickness) <= THICKNESS_TOLERANCE)

    def waste_area(self, length: float, width: float) -> float:
        return length * width - self.length * self.width

    def part_center_in_plank(self, length: float, width: float) -> tuple[float, float]:
        return (-length / 2 + MARGIN + self.length / 2, width / 2 - MARGIN - self.width / 2)


def rounded_rect(length: float, width: float, radius: float, per_corner: int = 12) -> np.ndarray:
    """CCW outline, centered at the origin. Shape (N, 2)."""
    r = min(radius, length / 2, width / 2)
    hx, hy = length / 2 - r, width / 2 - r
    pts = []
    for (cx, cy), a0 in (((hx, hy), 0.0), ((-hx, hy), 0.5 * math.pi), ((-hx, -hy), math.pi), ((hx, -hy), 1.5 * math.pi)):
        for k in range(per_corner + 1):
            a = a0 + 0.5 * math.pi * k / per_corner
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.asarray(pts)


def offset_outline(outline: np.ndarray, distance: float) -> np.ndarray:
    """Offset a convex CCW outline outward by `distance` (vertex-normal offset)."""
    prev, nxt = np.roll(outline, 1, axis=0), np.roll(outline, -1, axis=0)
    tangent = nxt - prev
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return outline + distance * normal


# Parts this cell makes. Holes sit away from the part center, where the cups land.
JOBS = {
    "shelf_bracket": Job("shelf_bracket", 0.30, 0.14, 0.018, 0.030, ((-0.11, 0.0, 0.010), (0.11, 0.0, 0.010))),
    # 96 mm handle-hole spacing (a standard cabinet-hardware pitch).
    "drawer_front": Job("drawer_front", 0.38, 0.12, 0.018, 0.012, ((-0.048, 0.0, 0.008), (0.048, 0.0, 0.008))),
    "tray_base": Job("tray_base", 0.24, 0.18, 0.012, 0.040, ((-0.08, -0.05, 0.006), (0.08, 0.05, 0.006))),
}
