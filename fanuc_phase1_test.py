"""Phase 1 check: FANUC CRX-10iA/L + vacuum picks one plank (known pose) and
registers it against the CNC bed's datum stops.

    ./isaaclab.sh -p /workspace/cnc_tending/fanuc_phase1_test.py --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")  # Isaac Lab's surface gripper is CPU-only
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from fanuc_cell import layout as L
from fanuc_cell.assets import build_robot_usd
from fanuc_cell.motion import ArmController, Vacuum

build_robot_usd()
from fanuc_cell.scene import CellSceneCfg, PlankSpec, add_planks, add_scrap_bin  # noqa: E402  (needs the USD)

PLANK = PlankSpec("Plank0", 0.60, 0.20, 0.018, (0.05, -0.85, L.TABLE_TOP_Z + 0.009), yaw=0.4)
HOVER = 0.15
GAP = 0.004  # blank placed this far off each datum stop


def program(arm: ArmController, vac: Vacuum, scene: InteractiveScene):
    dev = arm.device
    plank = scene["Plank0"]
    top = lambda: plank.data.root_pos_w.torch[0].clone() + torch.tensor([0, 0, PLANK.thickness / 2], device=dev)

    yield from arm.hold(0.5)
    pick = top()
    yaw = arm.closest_equivalent_yaw(PLANK.yaw)
    print(f"[test] plank top at {[round(v, 3) for v in pick.tolist()]}, tcp at {[round(v, 3) for v in arm.tcp_pos().tolist()]}")

    yield from arm.move_to(pick + torch.tensor([0, 0, HOVER], device=dev), yaw, 3.0)
    yield from arm.move_linear(pick + torch.tensor([0, 0, 0.003], device=dev), yaw, 1.5)
    yield from vac.grip()
    print(f"[test] vacuum holding: {vac.holding}")
    yield from arm.move_linear(pick + torch.tensor([0, 0, HOVER], device=dev), yaw, 1.2)
    carried_offset = top() - arm.tcp_pos()
    print(f"[test] plank-top vs tcp after lift: {[round(v, 4) for v in carried_offset.tolist()]}")

    # Long axis along world Y against the front fence, corner at the datum.
    cx = L.DATUM_XY[0] + PLANK.width / 2 + GAP
    cy = L.DATUM_XY[1] + PLANK.length / 2 + GAP
    place = torch.tensor([cx, cy, L.BED_TOP_Z + PLANK.thickness], device=dev)
    yaw = arm.closest_equivalent_yaw(math.pi / 2)
    yield from arm.move_to(place + torch.tensor([0, 0, HOVER], device=dev), yaw, 3.5)
    yield from arm.move_linear(place + torch.tensor([0, 0, 0.004], device=dev), yaw, 1.5)
    yield from vac.release()
    yield from arm.move_linear(place + torch.tensor([0, 0, HOVER], device=dev), yaw, 1.0)
    yield from arm.hold(1.0)

    final = plank.data.root_pos_w.torch[0]
    target_center = torch.tensor([cx, cy, L.BED_TOP_Z + PLANK.thickness / 2], device=dev)
    err = (final - target_center).tolist()
    q = plank.data.root_quat_w.torch[0].tolist()  # xyzw
    final_yaw = math.degrees(math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
    print(f"[test] placement error xyz (mm): {[round(e * 1000, 1) for e in err]}, final yaw {final_yaw:.1f} deg (target +/-90)")


def main() -> None:
    # Suction grippers raycast for the object; scene queries are only on by default with a GUI.
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device, enable_scene_query_support=True)
    )
    sim.set_camera_view(eye=[2.6, -2.4, 2.3], target=[0.3, -0.2, 0.8])
    cfg = CellSceneCfg(num_envs=1, env_spacing=5.0)
    add_scrap_bin(cfg)
    add_planks(cfg, [PLANK])
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
    prog = program(arm, vac, scene)
    while simulation_app.is_running():
        try:
            next(prog)
        except StopIteration:
            break
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
    print("[test] done")


if __name__ == "__main__":
    main()
    simulation_app.close()
