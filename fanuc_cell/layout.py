"""Cell layout: every station's pose and size, in meters (world frame).

The robot stands on a pedestal at the origin, facing +X toward the CNC router;
the stations sit in a ring ~0.85 m around it. With the vacuum tool pointing
down, the CRX's wrist rides ~0.27 m above the pick point, so the pedestal puts
the shoulder near table height -- floor-mounted, table picks were out of reach.

Top view:
                      +Y
        scrap bin     output table
                  ROBOT              CNC router  -> +X
        staging       infeed table
                      -Y
"""

from __future__ import annotations

from dataclasses import dataclass

TABLE_TOP_Z = 0.75
PEDESTAL_HEIGHT = 0.70
ROBOT_BASE = (0.0, 0.0, PEDESTAL_HEIGHT)


@dataclass(frozen=True)
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def top_z(self) -> float:
        return self.center[2] + self.size[2] / 2

    def contains_xy(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (abs(x - self.center[0]) <= self.size[0] / 2 - margin
                and abs(y - self.center[1]) <= self.size[1] / 2 - margin)


def table(cx: float, cy: float, sx: float, sy: float, top_z: float = TABLE_TOP_Z) -> Box:
    return Box((cx, cy, top_z / 2), (sx, sy, top_z))


PEDESTAL = Box((0.0, 0.0, PEDESTAL_HEIGHT / 2), (0.36, 0.36, PEDESTAL_HEIGHT))

INFEED_TABLE = table(0.0, -0.85, 0.85, 0.55)
STAGING_TABLE = table(-0.80, -0.35, 0.50, 0.70)
OUTPUT_TABLE = table(0.0, 0.85, 0.85, 0.55)

# Scrap bin: open-top box on the floor, walls only.
SCRAP_BIN_CENTER = (-0.75, 0.45)
SCRAP_BIN_SIZE = (0.50, 0.60, 0.55)
SCRAP_BIN_WALL = 0.02

# CNC router. The machine body is a solid block; the MDF spoilboard sits on top.
CNC_BODY = Box((0.95, 0.0, 0.40), (1.00, 1.40, 0.80))
SPOILBOARD = Box((0.95, 0.0, 0.825), (0.90, 1.30, 0.05))
BED_TOP_Z = SPOILBOARD.top_z

# Datum stops the blank is registered against: a fence along Y at the front
# (robot side) and a stop along X on the left. Blank's front-left corner = DATUM_XY.
DATUM_XY = (0.56, -0.58)
FENCE_HEIGHT = 0.02
FRONT_FENCE = Box((0.545, -0.20, BED_TOP_Z + FENCE_HEIGHT / 2), (0.03, 0.80, FENCE_HEIGHT))
SIDE_STOP = Box((0.75, -0.595, BED_TOP_Z + FENCE_HEIGHT / 2), (0.40, 0.03, FENCE_HEIGHT))

# Gantry travels along X (away from the robot) on rails down both sides of the
# bed; the beam spans Y, the carriage runs along it and the spindle plunges in Z.
# It parks at the back, so the robot's side of the machine is completely open.
RAIL_Y = (-0.68, 0.68)
RAIL_X_RANGE = (0.45, 1.45)
RAIL_TOP_Z = BED_TOP_Z + 0.05
GANTRY_BEAM_Z = BED_TOP_Z + 0.45
GANTRY_PARK_X = 1.38
CARRIAGE_DX = 0.13  # tool sits this far in front (-X) of the beam
SPINDLE_PARK_Z = BED_TOP_Z + 0.25

# Overhead depth camera, looking straight down; sees the infeed and staging tables.
CAMERA_POS = (-0.35, -0.62, 2.25)

# Robot "home": tool down, tucked above the pedestal between infeed and staging.
HOME_JOINTS = {"J1": -1.5708, "J2": 0.0, "J3": -0.5, "J4": 0.0, "J5": 2.0708, "J6": 0.0}
