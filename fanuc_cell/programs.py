"""Reusable robot programs: pick / place / drop with planned suction grasps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .grasp import Grasp
from .motion import ArmController, Program, Vacuum

HOVER = 0.15
APPROACH_GAP = 0.003  # cup lips stop this far above the surface; the suction ray closes the rest
RELEASE_GAP = 0.004


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _rot(xy, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


@dataclass(frozen=True)
class Hold:
    """How an object sits in the tool: grasp point (object frame) and tool yaw minus object yaw."""

    grasp_xy: tuple[float, float]
    tool_minus_object_yaw: float


def grasp_to_world(obj_xy, obj_yaw: float, grasp: Grasp) -> tuple[np.ndarray, float]:
    return np.asarray(obj_xy) + _rot(grasp.xy, obj_yaw), obj_yaw + grasp.yaw


def pick(arm: ArmController, vac: Vacuum, obj_xy, obj_yaw: float, top_z: float, grasp: Grasp,
         label: str = "") -> Program:
    """Approach from above, seal, lift. Returns a Hold, or None if there's no seal."""
    dev = arm.device
    tcp_xy, tool_yaw = grasp_to_world(obj_xy, obj_yaw, grasp)
    tool_yaw = arm.closest_equivalent_yaw(tool_yaw)  # the cup pattern is 180-deg symmetric
    target = torch.tensor([tcp_xy[0], tcp_xy[1], top_z], device=dev, dtype=torch.float32)
    up = torch.tensor([0.0, 0.0, HOVER], device=dev)
    yield from arm.move_to(target + up, tool_yaw, 3.0)
    yield from arm.move_linear(target + torch.tensor([0.0, 0.0, APPROACH_GAP], device=dev), tool_yaw, 1.2)
    yield from vac.grip()
    held = vac.holding
    yield from arm.move_linear(target + up, tool_yaw, 1.2)
    held = held and vac.holding
    print(f"[robot] pick {label}: {'sealed' if held else 'NO SEAL'}")
    return Hold(grasp.xy, _wrap(tool_yaw - obj_yaw)) if held else None


def place(arm: ArmController, vac: Vacuum, hold: Hold, obj_xy, obj_yaw: float, top_z: float,
          label: str = "", allow_flip: bool = True) -> Program:
    """Put the held object down so its center lands at obj_xy with heading obj_yaw.

    With allow_flip, obj_yaw + 180 deg is also acceptable (rectangular stock/parts);
    whichever needs less wrist rotation wins.
    """
    dev = arm.device
    headings = (obj_yaw, obj_yaw + math.pi) if allow_flip else (obj_yaw,)
    obj_yaw = min(headings, key=lambda a: abs(_wrap(a + hold.tool_minus_object_yaw - arm.tcp_yaw())))
    tcp_xy = np.asarray(obj_xy) + _rot(hold.grasp_xy, obj_yaw)
    tool_yaw = obj_yaw + hold.tool_minus_object_yaw
    target = torch.tensor([tcp_xy[0], tcp_xy[1], top_z], device=dev, dtype=torch.float32)
    up = torch.tensor([0.0, 0.0, HOVER], device=dev)
    yield from arm.move_to(target + up, tool_yaw, 3.5)
    yield from arm.move_linear(target + torch.tensor([0.0, 0.0, RELEASE_GAP], device=dev), tool_yaw, 1.5)
    yield from vac.release()
    yield from arm.move_linear(target + up, tool_yaw, 1.0)
    print(f"[robot] placed {label}")


def drop(arm: ArmController, vac: Vacuum, xy, release_z: float, label: str = "") -> Program:
    dev = arm.device
    target = torch.tensor([xy[0], xy[1], release_z], device=dev, dtype=torch.float32)
    yield from arm.move_to(target, arm.tcp_yaw(), 3.0)
    yield from vac.release(0.6)
    print(f"[robot] dropped {label}")
