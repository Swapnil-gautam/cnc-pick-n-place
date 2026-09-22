"""Phase 2 check: load a blank, machine a part out of it, unload part + skeleton.

    ./isaaclab.sh -p /workspace/cnc_tending/fanuc_phase2_test.py --headless
    ./isaaclab.sh -p /workspace/cnc_tending/fanuc_phase2_test.py --livestream 2 --viz kit
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--job", default="shelf_bracket")
parser.add_argument("--hold", action="store_true", help="Keep simulating after the cycle (for the live viewer).")
parser.add_argument("--no-effects", action="store_true", help="Skip groove/sawdust markers.")
parser.add_argument("--start-delay", type=float, default=0.0, help="Sim seconds to wait before starting (viewer).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")  # Isaac Lab's surface gripper is CPU-only
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from fanuc_cell import layout as L
from fanuc_cell.assets import build_robot_usd
from fanuc_cell.geometry import build_part_usd, build_skeleton_usd, part_area
from fanuc_cell.grasp import Region, plan_grasp
from fanuc_cell.jobs import JOBS, TOOL_DIAMETER
from fanuc_cell.motion import ArmController, Vacuum
from fanuc_cell.programs import drop, pick, place
from fanuc_cell.scene import WOOD_DENSITY

build_robot_usd()
from fanuc_cell.cnc import CncRouter, yaw_of  # noqa: E402
from fanuc_cell.scene import CellSceneCfg, PlankSpec, add_planks, add_scrap_bin, add_wood_usd  # noqa: E402

JOB = JOBS[args_cli.job]
PLANK = PlankSpec("Plank0", round(JOB.blank_length + 0.04, 3), round(JOB.blank_width + 0.02, 3), JOB.thickness,
                  (0.05, -0.85, L.TABLE_TOP_Z + JOB.thickness / 2), yaw=0.4)
PARK = {"Plank0": (5.0, -5.0, 0.05), "Part0": (5.0, 5.0, 0.05), "Skeleton0": (5.5, 5.0, 0.05)}
DATUM_GAP = 0.003
RENDER_EVERY = 8  # 120 Hz physics, 15 Hz rendering
PERF_EVERY = 1200  # print a timing breakdown every 10 sim seconds
assert JOB.fits(PLANK.length, PLANK.width, PLANK.thickness), "test plank too small for the job"


def cell_program(arm: ArmController, vac: Vacuum, cnc: CncRouter, scene: InteractiveScene):
    T = PLANK.thickness
    yield from arm.hold(0.5 + args_cli.start_delay)

    # --- load: pick the blank, register it against the datum stops -----------
    plank = scene["Plank0"]
    p_xy = plank.data.root_pos_w.torch[0, 0:2].cpu().numpy().copy()
    p_yaw = yaw_of(plank.data.root_quat_w.torch[0].tolist())
    g = plan_grasp(Region((0.0, 0.0), PLANK.length / 2, PLANK.width / 2))
    hold = yield from pick(arm, vac, p_xy, p_yaw, L.TABLE_TOP_Z + T, g, "blank")
    if hold is None:
        return
    bed_xy = (L.DATUM_XY[0] + PLANK.width / 2 + DATUM_GAP, L.DATUM_XY[1] + PLANK.length / 2 + DATUM_GAP)
    yield from place(arm, vac, hold, bed_xy, math.pi / 2, L.BED_TOP_Z + T, "blank on CNC bed")
    yield from arm.move_to(torch.tensor([0.25, -0.35, 1.25], device=arm.device), arm.tcp_yaw(), 2.0)  # clear of the machine

    # --- machine ---------------------------------------------------------------
    result = []
    yield from cnc.machine("Plank0", (PLANK.length, PLANK.width, T), JOB, "Part0", "Skeleton0", PARK, result)
    cut = result[0]
    cnc.blow_off_bed()
    yield from arm.hold(0.3)

    # --- unload the part -> output table --------------------------------------
    part_region = Region((0.0, 0.0), JOB.length / 2, JOB.width / 2, JOB.corner_radius,
                         tuple((hx, hy, d / 2) for hx, hy, d in JOB.holes))
    g = plan_grasp(part_region)
    print(f"[plan] part grasp offset {np.round(g.xy, 3).tolist()} m, tool rotated {math.degrees(g.yaw):.0f} deg vs part")
    hold = yield from pick(arm, vac, cut.part_xy, cut.yaw, L.BED_TOP_Z + T, g, "finished part")
    if hold is None:
        return
    out_xy = (L.OUTPUT_TABLE.center[0], L.OUTPUT_TABLE.center[1])
    yield from place(arm, vac, hold, out_xy, 0.0, L.TABLE_TOP_Z + T, "part on output table")

    # --- unload the skeleton -> scrap bin --------------------------------------
    c = np.asarray(JOB.part_center_in_plank(PLANK.length, PLANK.width))
    strip_start = c[0] + JOB.length / 2 + TOOL_DIAMETER  # far edge of the kerf
    strip = Region(((strip_start + PLANK.length / 2) / 2, 0.0), (PLANK.length / 2 - strip_start) / 2, PLANK.width / 2)
    g = plan_grasp(strip)
    if g is None:
        print("[plan] skeleton: no place on the handling strip fits all 4 cups")
        return
    if (yield from pick(arm, vac, cut.skeleton_xy, cut.yaw, L.BED_TOP_Z + T, g, "skeleton")) is None:
        return
    yield from drop(arm, vac, L.SCRAP_BIN_CENTER, 0.95, "skeleton into scrap bin")
    yield from arm.move_to(torch.tensor([0.25, 0.35, 1.25], device=arm.device), arm.tcp_yaw(), 2.0)
    yield from arm.hold(1.0)

    # --- verify ------------------------------------------------------------------
    part = scene["Part0"].data.root_pos_w.torch[0].tolist()
    part_yaw = math.degrees(yaw_of(scene["Part0"].data.root_quat_w.torch[0].tolist()))
    skel = scene["Skeleton0"].data.root_pos_w.torch[0].tolist()
    in_bin = (abs(skel[0] - L.SCRAP_BIN_CENTER[0]) < L.SCRAP_BIN_SIZE[0] / 2
              and abs(skel[1] - L.SCRAP_BIN_CENTER[1]) < L.SCRAP_BIN_SIZE[1] / 2 and skel[2] < L.SCRAP_BIN_SIZE[2])
    print(f"[verify] part on output table: off by {1000 * math.hypot(part[0] - out_xy[0], part[1] - out_xy[1]):.1f} mm, "
          f"z={part[2]:.3f} (expect {L.TABLE_TOP_Z + T / 2:.3f}), yaw={part_yaw:.1f} deg")
    print(f"[verify] skeleton in scrap bin: {in_bin} (at {np.round(skel, 3).tolist()})")


def main() -> None:
    part_usd = build_part_usd(JOB)
    skel_usd = build_skeleton_usd(JOB, PLANK.length, PLANK.width, PLANK.thickness)
    part_mass = part_area(JOB) * JOB.thickness * WOOD_DENSITY
    skel_mass = (PLANK.length * PLANK.width - part_area(JOB)) * PLANK.thickness * WOOD_DENSITY

    sim = sim_utils.SimulationContext(
        # 120 Hz physics for stable suction contact (rendering is throttled in the loop, see RENDER_EVERY).
        # Suction grippers raycast for the object; scene queries are only on by default with a GUI.
        sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device, enable_scene_query_support=True)
    )
    sim.set_camera_view(eye=[2.3, -1.6, 2.1], target=[0.5, -0.1, 0.8])
    cfg = CellSceneCfg(num_envs=1, env_spacing=5.0)
    add_scrap_bin(cfg)
    add_planks(cfg, [PLANK])
    add_wood_usd(cfg, "Part0", part_usd, part_mass, PARK["Part0"])
    add_wood_usd(cfg, "Skeleton0", skel_usd, skel_mass, PARK["Skeleton0"])
    scene = InteractiveScene(cfg)
    sim.reset()
    dt = sim.get_physics_dt()

    robot = scene["robot"]
    q0 = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=q0)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
    robot.set_joint_position_target_index(target=q0)
    robot.reset()
    for _ in range(30):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    arm = ArmController(scene, dt)
    vac = Vacuum(scene, arm)
    cnc = CncRouter(scene, dt, effects=not args_cli.no_effects)
    prog = cell_program(arm, vac, cnc, scene)
    wall0, steps = time.time(), 0
    cost = {"program": 0.0, "write": 0.0, "step": 0.0, "update": 0.0, "tick": 0.0}
    while simulation_app.is_running():
        steps += 1
        if steps % PERF_EVERY == 0:
            wall = time.time() - wall0
            parts = ", ".join(f"{k} {1000 * v / PERF_EVERY:.1f}ms" for k, v in cost.items())
            print(f"[perf] step {steps}: sim {steps * dt:.1f}s / wall {wall:.1f}s | per step: {parts}", flush=True)
            cost = dict.fromkeys(cost, 0.0)
        t = time.time()
        try:
            next(prog)
        except StopIteration:
            if not args_cli.hold:
                break
            prog = arm.hold(3600.0)
        t1 = time.time(); cost["program"] += t1 - t
        scene.write_data_to_sim()
        t2 = time.time(); cost["write"] += t2 - t1
        sim.step(render=steps % RENDER_EVERY == 0)  # sim.step renders every call unless told not to
        t3 = time.time(); cost["step"] += t3 - t2
        scene.update(dt)
        t4 = time.time(); cost["update"] += t4 - t3
        cnc.tick()
        cost["tick"] += time.time() - t4
    print("[test] done")


if __name__ == "__main__":
    main()
    simulation_app.close()
