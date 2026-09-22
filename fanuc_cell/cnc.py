"""CNC router: moving gantry, a real toolpath, and what cutting leaves behind.

Machining sequence for one blank (like a router post-processor would emit):
  1. probe the blank's datum corner (real routers use a touch probe / pushers),
  2. drill every hole (rapid over, plunge through, retract),
  3. profile the part outline -- tool-radius offset, ramped lead-in, two depth
     passes, the second breaking through into the spoilboard,
  4. park the gantry at the far end so the robot can reach in.

The groove, drill holes and sawdust are instanced visual markers (no physics).
When the profile closes, the blank is swapped for the actual cut part and the
leftover skeleton (pre-authored meshes), lying exactly where the cut happened.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from . import layout as L
from .jobs import TOOL_DIAMETER, Job, offset_outline, rounded_rect

Program = Generator[None, None, None]

RAPID = 0.35  # m/s
FEED = 0.12
PLUNGE = 0.03
BIT_LENGTH = 0.05
TOOL_PARK = (L.GANTRY_PARK_X - L.CARRIAGE_DX, 0.0, L.SPINDLE_PARK_Z)
SAFE_CLEARANCE = 0.03
KERF_SPACING = 0.003
FAR_AWAY = (0.0, 0.0, -50.0)
MAX_KERF, MAX_HOLES, MAX_DUST = 900, 8, 900


def _q_yaw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]  # (x, y, z, w)


def yaw_of(quat_xyzw) -> float:
    x, y, z, w = quat_xyzw
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class Workpiece:
    """A blank registered on the bed, in the machine's (probed) plank frame."""

    xy: np.ndarray  # plank center, world
    yaw: float  # plank-local +x in world; chosen so local (-L/2, +W/2) is the datum corner
    length: float
    width: float
    thickness: float

    @property
    def top_z(self) -> float:
        return L.BED_TOP_Z + self.thickness

    def to_world(self, local_xy) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y = local_xy
        return self.xy + np.array([c * x - s * y, s * x + c * y])

    def contains(self, x: float, y: float) -> bool:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.xy[0], y - self.xy[1]
        return abs(c * dx + s * dy) <= self.length / 2 and abs(-s * dx + c * dy) <= self.width / 2


@dataclass
class CutResult:
    part_xy: np.ndarray
    skeleton_xy: np.ndarray
    yaw: float
    workpiece: Workpiece


class CncRouter:
    def __init__(self, scene, physics_dt: float, effects: bool = True) -> None:
        self.effects = effects
        self.scene = scene
        self.dt = physics_dt
        self.device = scene["robot"].device
        self.tool = np.array(TOOL_PARK)
        self.busy = False
        self.workpiece: Workpiece | None = None

        wood_dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.10, 0.05), roughness=1.0)
        self.markers = None if not effects else VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/CncCuts",
            markers={
                "kerf": sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=wood_dark),
                "hole": sim_utils.CylinderCfg(radius=0.5, height=1.0, axis="Z", visual_material=wood_dark),
                "dust": sim_utils.SphereCfg(
                    radius=0.5, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.66))
                ),
            },
        ))
        self.kerf: list[tuple[float, float, float, float]] = []  # x, y, z, yaw
        self.holes: list[tuple[float, float, float, float]] = []  # x, y, z, diameter
        self.dust_pos = np.zeros((MAX_DUST, 3))
        self.dust_vel = np.zeros((MAX_DUST, 3))
        self.dust_size = np.zeros(MAX_DUST)
        self.dust_state = np.zeros(MAX_DUST, dtype=np.int8)  # 0 unused, 1 flying, 2 settled
        self._dust_next = 0
        self._rng = np.random.default_rng(7)
        self._dirty = True
        self._tick_count = 0
        self._apply_gantry()

    # -- gantry kinematics -----------------------------------------------------

    def _apply_gantry(self) -> None:
        x, y, z = self.tool
        gx = x + L.CARRIAGE_DX
        poses = {
            "gantry_upright_left": (gx, L.RAIL_Y[0], L.RAIL_TOP_Z + 0.225),
            "gantry_upright_right": (gx, L.RAIL_Y[1], L.RAIL_TOP_Z + 0.225),
            "gantry_beam": (gx, 0.0, L.GANTRY_BEAM_Z),
            "carriage": (x, y, L.GANTRY_BEAM_Z - 0.03),
            "spindle": (x, y, z + BIT_LENGTH + 0.13),
            "bit": (x, y, z + BIT_LENGTH / 2),
        }
        for name, pos in poses.items():
            pose = torch.tensor([[*pos, 0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32)
            self.scene[name].write_root_pose_to_sim(pose)

    # -- effects ---------------------------------------------------------------

    def _emit_dust(self, count: int, spray_dir: np.ndarray | None) -> None:
        for _ in range(count):
            i = self._dust_next
            self._dust_next = (self._dust_next + 1) % MAX_DUST
            a = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(0.3, 1.1)
            v = np.array([math.cos(a) * speed, math.sin(a) * speed, self._rng.uniform(0.2, 0.9)])
            if spray_dir is not None:  # chips fly off the cutting side of the bit
                v[:2] += 0.6 * spray_dir
            self.dust_pos[i] = self.tool + np.array([0.0, 0.0, 0.004])
            self.dust_vel[i] = v
            self.dust_size[i] = self._rng.uniform(0.0015, 0.0035)
            self.dust_state[i] = 1

    def _surface_z_many(self, xy: np.ndarray) -> np.ndarray:
        z = np.zeros(len(xy))
        body = (np.abs(xy[:, 0] - L.CNC_BODY.center[0]) <= L.CNC_BODY.size[0] / 2) & \
               (np.abs(xy[:, 1] - L.CNC_BODY.center[1]) <= L.CNC_BODY.size[1] / 2)
        z[body] = L.CNC_BODY.top_z
        bed = (np.abs(xy[:, 0] - L.SPOILBOARD.center[0]) <= L.SPOILBOARD.size[0] / 2) & \
              (np.abs(xy[:, 1] - L.SPOILBOARD.center[1]) <= L.SPOILBOARD.size[1] / 2)
        z[bed] = L.BED_TOP_Z
        if self.workpiece is not None:
            wp = self.workpiece
            c, s = math.cos(wp.yaw), math.sin(wp.yaw)
            d = xy - wp.xy
            on = (np.abs(c * d[:, 0] + s * d[:, 1]) <= wp.length / 2) & (np.abs(-s * d[:, 0] + c * d[:, 1]) <= wp.width / 2)
            z[on] = wp.top_z
        return z

    def tick(self, draw_every: int = 4) -> None:
        """Advance sawdust; redraw markers every few steps if anything changed. Call every physics step."""
        flying = np.nonzero(self.dust_state == 1)[0]
        if len(flying):
            self.dust_vel[flying, 2] -= 9.81 * self.dt
            self.dust_vel[flying] *= 0.985  # air drag on light chips
            self.dust_pos[flying] += self.dust_vel[flying] * self.dt
            floor = self._surface_z_many(self.dust_pos[flying, 0:2]) + self.dust_size[flying] / 2
            landed = self.dust_pos[flying, 2] <= floor
            self.dust_pos[flying[landed], 2] = floor[landed]
            self.dust_state[flying[landed]] = 2
            self._dirty = True
        self._tick_count += 1
        if self._dirty and self._tick_count % draw_every == 0:
            self._draw()
            self._dirty = False

    def _draw(self) -> None:
        if not self.effects:
            return
        total = MAX_KERF + MAX_HOLES + MAX_DUST
        pos = np.tile(np.asarray(FAR_AWAY, dtype=np.float32), (total, 1))  # spares parked out of sight
        quat = np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (total, 1))
        scale = np.full((total, 3), 1e-4, dtype=np.float32)
        idx = np.full(total, 2, dtype=np.int32)

        kerf = np.asarray(self.kerf[-MAX_KERF:], dtype=np.float32).reshape(-1, 4)
        n = len(kerf)
        pos[:n] = kerf[:, 0:3]
        quat[:n, 2], quat[:n, 3] = np.sin(kerf[:, 3] / 2), np.cos(kerf[:, 3] / 2)
        scale[:n] = (0.0045, TOOL_DIAMETER, 0.0012)
        idx[:n] = 0

        holes = np.asarray(self.holes[-MAX_HOLES:], dtype=np.float32).reshape(-1, 4)
        h0, h1 = MAX_KERF, MAX_KERF + len(holes)
        pos[h0:h1] = holes[:, 0:3]
        scale[h0:h1, 0] = scale[h0:h1, 1] = holes[:, 3]
        scale[h0:h1, 2] = 0.0012
        idx[h0:h1] = 1

        live = np.nonzero(self.dust_state > 0)[0]
        d0 = MAX_KERF + MAX_HOLES
        pos[d0:d0 + len(live)] = self.dust_pos[live]
        scale[d0:d0 + len(live)] = self.dust_size[live, None]
        self.markers.visualize(translations=pos, orientations=quat, scales=scale, marker_indices=idx)

    def blow_off_bed(self) -> None:
        """Air-blast the spoilboard clean between cycles (removes settled chips)."""
        self.dust_state[:] = 0
        self._dirty = True

    # -- motion ---------------------------------------------------------------

    def _travel(self, goal: np.ndarray, speed: float, on_step=None) -> Program:
        start = self.tool.copy()
        dist = float(np.linalg.norm(goal - start))
        steps = max(1, math.ceil(dist / (speed * self.dt)))
        for k in range(1, steps + 1):
            prev = self.tool.copy()
            self.tool = start + (goal - start) * (k / steps)
            self._apply_gantry()
            if on_step is not None:
                on_step(prev, self.tool)
            yield

    def park(self) -> Program:
        yield from self._travel(np.array([self.tool[0], self.tool[1], L.SPINDLE_PARK_Z]), RAPID)
        yield from self._travel(np.array(TOOL_PARK), RAPID)

    # -- machining --------------------------------------------------------------

    def probe(self, plank, length: float, width: float, thickness: float) -> Workpiece:
        """Find the blank's datum corner. Of the two equivalent frames of a rectangle,
        use the one whose local (-L/2, +W/2) corner is nearest the datum stops."""
        xy = plank.data.root_pos_w.torch[0, 0:2].cpu().numpy().copy()
        yaw = yaw_of(plank.data.root_quat_w.torch[0].tolist())
        best = None
        for cand in (yaw, yaw + math.pi):
            wp = Workpiece(xy, cand, length, width, thickness)
            d = np.linalg.norm(wp.to_world((-length / 2, width / 2)) - np.asarray(L.DATUM_XY))
            if best is None or d < best[0]:
                best = (d, wp)
        return best[1]

    def machine(self, plank_name: str, plank_dims: tuple[float, float, float], job: Job,
                part_name: str, skeleton_name: str, park: dict[str, tuple[float, float, float]],
                result: list) -> Program:
        """Cut `job` out of the blank on the bed; appends a CutResult to `result`."""
        self.busy = True
        length, width, thickness = plank_dims
        wp = self.probe(self.scene[plank_name], length, width, thickness)
        self.workpiece = wp
        top, safe = wp.top_z, wp.top_z + SAFE_CLEARANCE
        c = np.asarray(job.part_center_in_plank(length, width))
        print(f"[cnc] probed blank corner at {np.round(wp.to_world((-length / 2, width / 2)), 4).tolist()}, "
              f"cutting '{job.name}' ({job.length * 1000:.0f} x {job.width * 1000:.0f} mm)")

        def pt(local_xy, z):
            w = wp.to_world(local_xy)
            return np.array([w[0], w[1], z])

        # 1) drill holes
        for hx, hy, d in job.holes:
            above = pt(c + (hx, hy), safe)
            yield from self._travel(above, RAPID)
            yield from self._travel(pt(c + (hx, hy), top + 0.002), RAPID)

            def drilling(prev, cur):
                if cur[2] < top:
                    self._emit_dust(3, None)

            yield from self._travel(pt(c + (hx, hy), top - thickness - 0.001), PLUNGE, drilling)
            self.holes.append((above[0], above[1], top + 0.0006, d))
            self._dirty = True
            yield from self._travel(above, RAPID)

        # 2) profile: tool center runs one tool radius outside the part outline
        path = offset_outline(rounded_rect(job.length, job.width, job.corner_radius, per_corner=16),
                              TOOL_DIAMETER / 2) + c
        pts = [pt(p, 0.0) for p in np.vstack([path, path[:1]])]
        seg_len = [np.linalg.norm(pts[i + 1][:2] - pts[i][:2]) for i in range(len(pts) - 1)]
        yield from self._travel(np.array([*pts[0][:2], safe]), RAPID)
        yield from self._travel(np.array([*pts[0][:2], top + 0.002]), RAPID)
        last_mark = [None]

        def cutting(prev, cur, mark_kerf=True):
            if cur[2] >= top - 1e-4:
                return
            move = cur[:2] - prev[:2]
            n = np.linalg.norm(move)
            if n > 1e-9:
                self._emit_dust(2, np.array([-move[1], move[0]]) / n)
            if not mark_kerf:
                return
            if last_mark[0] is None or np.linalg.norm(cur[:2] - last_mark[0]) >= KERF_SPACING:
                yaw = math.atan2(move[1], move[0]) if n > 1e-9 else 0.0
                self.kerf.append((cur[0], cur[1], top + 0.0005, yaw))
                self._dirty = True
                last_mark[0] = cur[:2].copy()

        ramp_len = 0.04  # first pass ramps down along the path instead of plunging straight
        second_pass = lambda prev, cur: cutting(prev, cur, mark_kerf=False)
        for depth, first in ((top - thickness / 2, True), (top - thickness - 0.0005, False)):
            travelled = 0.0
            for i, seg in enumerate(seg_len):
                travelled += seg
                z = top - (top - depth) * min(1.0, travelled / ramp_len) if first else depth
                yield from self._travel(np.array([*pts[i + 1][:2], z]), FEED, cutting if first else second_pass)
            print(f"[cnc] profile pass to {1000 * (top - depth):.1f} mm deep complete")

        # 3) retract and park, then the cut is done: swap the blank for part + skeleton
        yield from self.park()
        yaw_q = torch.tensor(_q_yaw(wp.yaw), device=self.device)
        part_xy = wp.to_world(c)

        def place(name, xyz, q):
            pose = torch.tensor([[*xyz, *q.tolist()]], device=self.device, dtype=torch.float32)
            self.scene[name].write_root_pose_to_sim(pose)
            self.scene[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))

        on_bed = L.BED_TOP_Z + thickness / 2
        place(plank_name, park[plank_name], torch.tensor(_q_yaw(0.0)))
        place(skeleton_name, (wp.xy[0], wp.xy[1], on_bed), yaw_q)
        place(part_name, (part_xy[0], part_xy[1], on_bed), yaw_q)
        self.kerf.clear()
        self.holes.clear()
        self._dirty = True
        self.busy = False
        result.append(CutResult(part_xy, wp.xy.copy(), wp.yaw, wp))
        yield
