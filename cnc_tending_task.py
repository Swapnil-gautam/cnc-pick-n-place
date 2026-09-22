"""Simulates a Franka arm tending a CNC machine: load raw stock onto the bed,
wait for the machine to finish its cycle, then separate the finished part
from the waste and drop each into its own bin.

No vision is used. The bed, the stock, and the fixture are all at fixed,
known poses, so every pick/place target is hard-coded -- the
"deterministic layout, no CV needed" case. The one thing standing in for
real hardware is the CNC "done" signal: here it's a step counter
(`--cnc-cycle-steps`), but `_step_wait_cnc()` is the single place you'd swap
in a real handshake (an M-code-driven digital input, a PLC flag, or a status
read over FOCAS/OPC-UA/Modbus/GRBL-serial).

Built on Isaac Lab 3.0-beta's native API (InteractiveScene +
DifferentialIKController), like scripts/tutorials/05_controllers/run_diff_ik.py.

Usage (from /workspace/isaaclab):

    ./isaaclab.sh -p /workspace/cnc_tending/cnc_tending_task.py --livestream 2 --viz kit

Add --cycles N to run more than one load/machine/unload cycle, --test to
exit after the first cycle completes.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Robotic arm CNC tending cell (no vision, fixed poses).")
parser.add_argument("--test", action="store_true", help="Exit after the first load/machine/unload cycle.")
parser.add_argument("--cycles", type=int, default=3, help="Number of stock pieces to run through the cell.")
parser.add_argument(
    "--cnc-cycle-steps",
    type=int,
    default=150,
    help="How many physics steps the simulated CNC 'machining' cycle takes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

GRIPPER_OPEN = (0.04, 0.04)
GRIPPER_CLOSED = (0.0, 0.0)
# Isaac Lab 3.0 quaternions are (x, y, z, w). This is 180 deg about X: gripper pointing straight down.
DOWNWARD_QUAT = (1.0, 0.0, 0.0, 0.0)
# The IK target is the panda_hand frame origin; the fingertip pads sit ~0.1034m below it.
HAND_TO_FINGERTIP = 0.1034
HOVER_HEIGHT = 0.15
RELEASE_CLEARANCE = 0.005

STOCK_SIZE = 0.05
PART_SIZE = 0.05
WASTE_SIZE = 0.025
BED_TOP_Z = 0.10
BIN_TOP_Z = 0.02
BIN_HALF_WIDTH = 0.125
# Raw stock is presented on a small infeed table at bed height. A floor-level
# pick at the old spot put the arm near full extension, where the IK diverged.
INFEED_XY = (0.4, 0.35)

GRIP_MATERIAL_CFG = sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.0)

# This Isaac Sim 6.0.1 install resolves cloud assets against content-bucket
# version "6.0", but the Franka USD isn't published there yet (404); "5.1"
# has it and is asset-compatible, so pin the path to that instead.
_FRANKA_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
_FRANKA_CFG.spawn.usd_path = _FRANKA_CFG.spawn.usd_path.replace("/Isaac/6.0/", "/Isaac/5.1/")

# Where objects wait while they aren't part of the current step: behind the
# default camera and far outside the robot's reach. Parking is used instead of
# toggling USD visibility, because hiding a rigid body leaves its collider in
# place, and editing USD mid-simulation is a likely cause of the physics view
# invalidation seen in long runs.
PARK_STOCK = (4.5, 4.0)
PARK_PART = (4.5, 4.3)
PARK_WASTE = (4.5, 4.6)


def _cube(size: float, mass: float, color: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=(size, size, size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        physics_material=GRIP_MATERIAL_CFG,
    )


def _static_box(size: tuple[float, float, float], color: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


@configclass
class CncCellSceneCfg(InteractiveSceneCfg):
    """Franka arm, a CNC machine (bed + housing), two bins, and three cubes."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    )

    robot = _FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Blue-gray work table (top at BED_TOP_Z) with an orange housing behind it.
    cnc_bed = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CncBed",
        spawn=_static_box((0.4, 0.35, 0.08), (0.25, 0.35, 0.45)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, BED_TOP_Z - 0.04)),
    )
    cnc_housing = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CncHousing",
        spawn=_static_box((0.15, 0.55, 0.55), (0.85, 0.55, 0.1)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.775, 0.0, 0.275)),
    )
    infeed_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/InfeedTable",
        spawn=_static_box((0.14, 0.14, BED_TOP_Z), (0.5, 0.5, 0.5)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*INFEED_XY, BED_TOP_Z / 2)),
    )
    output_bin = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OutputBin",
        spawn=_static_box((0.25, 0.25, 0.02), (0.2, 0.7, 0.3)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.5, BIN_TOP_Z - 0.01)),
    )
    scrap_bin = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ScrapBin",
        spawn=_static_box((0.25, 0.25, 0.02), (0.75, 0.2, 0.2)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.5, BIN_TOP_Z - 0.01)),
    )

    stock = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stock",
        spawn=_cube(STOCK_SIZE, 0.05, (0.55, 0.35, 0.2)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*INFEED_XY, BED_TOP_Z + STOCK_SIZE / 2)),
    )
    good_part = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoodPart",
        spawn=_cube(PART_SIZE, 0.04, (0.75, 0.55, 0.3)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*PARK_PART, PART_SIZE / 2)),
    )
    waste_chip = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WasteChip",
        spawn=_cube(WASTE_SIZE, 0.01, (0.35, 0.35, 0.35)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*PARK_WASTE, WASTE_SIZE / 2)),
    )


class CncTendingCell:
    """Phase-based state machine for one Franka arm tending one fixed CNC bed.

    A cycle is: pick stock -> place on bed, wait for the CNC, pick finished
    part -> output bin, pick waste -> scrap bin. Each pick-place is 8 phases.
    Every arm motion interpolates smoothly from where the hand is at the start
    of the phase to a goal fixed at that moment, then holds the goal to settle.
    """

    PHASES = [
        # (name, duration in physics steps)
        ("move above pick", 90),
        ("approach pick", 60),
        ("close gripper", 30),
        ("lift", 50),
        ("move above place", 110),
        ("lower to place", 60),
        ("open gripper", 25),
        ("retract", 45),
    ]
    # Fraction of a move phase spent travelling; the rest holds at the goal.
    TRAVEL_FRACTION = 0.7

    def __init__(self, scene: InteractiveScene, device: str, cnc_cycle_steps: int) -> None:
        self.scene = scene
        self.device = device
        self.cnc_cycle_steps = cnc_cycle_steps
        self.robot = scene["robot"]

        def rest(x: float, y: float, surface_z: float, size: float) -> torch.Tensor:
            return torch.tensor([x, y, surface_z + size / 2], device=device)

        # Resting center positions of each cube at each station.
        self.infeed_rest = rest(*INFEED_XY, BED_TOP_Z, STOCK_SIZE)
        self.bed_rest = rest(0.5, 0.0, BED_TOP_Z, STOCK_SIZE)
        self.part_on_bed_rest = rest(0.5, 0.0, BED_TOP_Z, PART_SIZE)
        self.waste_on_bed_rest = rest(0.5, 0.14, BED_TOP_Z, WASTE_SIZE)
        self.output_bin_rest = rest(0.0, 0.5, BIN_TOP_Z, PART_SIZE)
        self.scrap_bin_rest = rest(0.0, -0.5, BIN_TOP_Z, WASTE_SIZE)
        self.park = {
            "stock": rest(*PARK_STOCK, 0.0, STOCK_SIZE),
            "good_part": rest(*PARK_PART, 0.0, PART_SIZE),
            "waste_chip": rest(*PARK_WASTE, 0.0, WASTE_SIZE),
        }

        self.arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.arm_cfg.resolve(scene)
        self.gripper_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
        self.gripper_cfg.resolve(scene)
        self.ee_jacobi_idx = self.arm_cfg.body_ids[0] - 1 if self.robot.is_fixed_base else self.arm_cfg.body_ids[0]
        self.jacobi_joint_ids = [j + self.robot.num_base_dofs for j in self.arm_cfg.joint_ids]

        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=device)
        self.ik_command = torch.zeros(scene.num_envs, self.ik_controller.action_dim, device=device)
        self.ik_command[:, 3:7] = torch.tensor(DOWNWARD_QUAT, device=device)

        self._segments = [
            dict(kind="pick_place", obj="stock", place=self.bed_rest, label="stock -> bed"),
            dict(kind="wait_cnc"),
            dict(kind="pick_place", obj="good_part", place=self.output_bin_rest, label="good part -> output bin"),
            dict(kind="pick_place", obj="waste_chip", place=self.scrap_bin_rest, label="waste chip -> scrap bin"),
        ]
        self._seg_idx = 0
        self._sub_phase = 0
        self._sub_step = 0
        self._wait_step = 0
        self._pick_rest = None
        self._move_start = None
        self._move_goal = None
        self._hold_target = None
        self.cycles_done = 0
        self.misses = 0

    # -- robot helpers ------------------------------------------------------

    def _ee_pos_b(self) -> torch.Tensor:
        ee_pose_w = self.robot.data.body_pose_w.torch[:, self.arm_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w.torch
        ee_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return ee_pos_b[0]

    def _command_hand(self, hand_pos: torch.Tensor) -> None:
        self.ik_command[:, 0:3] = hand_pos
        self.ik_controller.set_command(self.ik_command)
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.ee_jacobi_idx, :, self.jacobi_joint_ids]
        ee_pose_w = self.robot.data.body_pose_w.torch[:, self.arm_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w.torch
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos = self.robot.data.joint_pos.torch[:, self.arm_cfg.joint_ids]
        joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=self.arm_cfg.joint_ids)
        self._hold_target = hand_pos

    def _set_gripper(self, opened: bool) -> None:
        target = torch.tensor([GRIPPER_OPEN if opened else GRIPPER_CLOSED], device=self.device)
        self.robot.set_joint_position_target_index(target=target, joint_ids=self.gripper_cfg.joint_ids)

    def _hold(self) -> None:
        if self._hold_target is not None:
            self._command_hand(self._hold_target)

    def _move(self, goal_fingertip: torch.Tensor, duration: int) -> None:
        """Interpolate the hand toward a fingertip goal fixed at phase entry.

        Interpolates in cylindrical coordinates around the robot base so long
        moves sweep around the base instead of cutting close to it.
        """
        if self._sub_step == 0:
            self._move_start = self._ee_pos_b().clone()
            self._move_goal = goal_fingertip + torch.tensor([0.0, 0.0, HAND_TO_FINGERTIP], device=self.device)
        t = min(1.0, self._sub_step / (self.TRAVEL_FRACTION * duration))
        s = t * t * (3.0 - 2.0 * t)

        start, goal = self._move_start, self._move_goal
        r0, r1 = torch.linalg.norm(start[0:2]), torch.linalg.norm(goal[0:2])
        a0, a1 = torch.atan2(start[1], start[0]), torch.atan2(goal[1], goal[0])
        da = torch.atan2(torch.sin(a1 - a0), torch.cos(a1 - a0))
        r = r0 + (r1 - r0) * s
        a = a0 + da * s
        z = start[2] + (goal[2] - start[2]) * s
        self._command_hand(torch.stack([r * torch.cos(a), r * torch.sin(a), z]))

    # -- object helpers -----------------------------------------------------

    def _object_pos(self, name: str) -> torch.Tensor:
        return self.scene[name].data.root_pos_w.torch[0]

    def _teleport(self, name: str, pos: torch.Tensor) -> None:
        pose = torch.zeros(1, 7, device=self.device)
        pose[0, 0:3] = pos
        pose[0, 6] = 1.0  # identity quaternion, (x, y, z, w)
        self.scene[name].write_root_pose_to_sim(pose)
        self.scene[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))

    # -- CNC handshake ------------------------------------------------------
    # Real cell: replace this method's timer with a read of the actual signal
    # (an M-code-driven digital input, a PLC flag over Modbus/OPC-UA, or a
    # GRBL/LinuxCNC/Fanuc-FOCAS status query). Nothing else needs to change.

    def _step_wait_cnc(self) -> None:
        self._hold()
        if self._wait_step == 0:
            print(f"[cycle {self.cycles_done + 1}] CNC: cycle start (stock loaded, arm clear of envelope)")
        self._wait_step += 1
        if self._wait_step >= self.cnc_cycle_steps:
            print(f"[cycle {self.cycles_done + 1}] CNC: job-complete signal received")
            # Deterministic outcome: the stock becomes a finished part and a
            # waste chip at fixed, known poses on the bed -- no vision needed.
            self._teleport("stock", self.park["stock"])
            self._teleport("good_part", self.part_on_bed_rest)
            self._teleport("waste_chip", self.waste_on_bed_rest)
            self._advance_segment()

    # -- state machine ------------------------------------------------------

    def start(self) -> None:
        self._teleport("stock", self.infeed_rest)
        self._teleport("good_part", self.park["good_part"])
        self._teleport("waste_chip", self.park["waste_chip"])
        self._set_gripper(opened=True)
        self._seg_idx = 0
        self._sub_phase = 0
        self._sub_step = 0
        self._wait_step = 0
        self.ik_controller.reset()

    def _report_placement(self, segment: dict) -> None:
        actual = self._object_pos(segment["obj"])
        target = segment["place"]
        xy_err = torch.linalg.norm(actual[0:2] - target[0:2]).item()
        z_err = abs(actual[2].item() - target[2].item())
        # The bed is a fixture, so it needs a tight placement; a bin only needs
        # the object to land inside it.
        xy_tol = 0.03 if segment["obj"] == "stock" else BIN_HALF_WIDTH - 0.02
        cycle = self.cycles_done + 1
        if xy_err < xy_tol and z_err < 0.02:
            print(f"[cycle {cycle}] {segment['label']}: placed OK (off by {xy_err * 100:.1f} cm)")
        else:
            self.misses += 1
            pos = ", ".join(f"{v:.3f}" for v in actual.tolist())
            print(f"[cycle {cycle}] {segment['label']}: MISSED -- object ended at ({pos})")

    def _advance_segment(self) -> None:
        self._seg_idx += 1
        self._sub_phase = 0
        self._sub_step = 0
        self._wait_step = 0
        if self._seg_idx >= len(self._segments):
            self.cycles_done += 1

    def step(self) -> None:
        if self.is_cycle_done():
            return

        segment = self._segments[self._seg_idx]
        if segment["kind"] == "wait_cnc":
            self._step_wait_cnc()
            return

        name, duration = self.PHASES[self._sub_phase]
        if self._sub_step == 0:
            print(f"[cycle {self.cycles_done + 1}] {segment['label']}: {name}")
            if self._sub_phase == 0:
                self._pick_rest = self._object_pos(segment["obj"]).clone()

        above = torch.tensor([0.0, 0.0, HOVER_HEIGHT], device=self.device)
        release = torch.tensor([0.0, 0.0, RELEASE_CLEARANCE], device=self.device)
        place = segment["place"]

        if self._sub_phase == 0:
            self._set_gripper(opened=True)
            self._move(self._pick_rest + above, duration)
        elif self._sub_phase == 1:
            self._move(self._pick_rest, duration)
        elif self._sub_phase == 2:
            self._hold()
            self._set_gripper(opened=False)
        elif self._sub_phase == 3:
            self._move(self._pick_rest + above, duration)
        elif self._sub_phase == 4:
            self._move(place + above, duration)
        elif self._sub_phase == 5:
            self._move(place + release, duration)
        elif self._sub_phase == 6:
            self._hold()
            self._set_gripper(opened=True)
        elif self._sub_phase == 7:
            self._move(place + above, duration)

        self._sub_step += 1
        if self._sub_step >= duration:
            self._sub_step = 0
            self._sub_phase += 1
            if self._sub_phase >= len(self.PHASES):
                self._report_placement(segment)
                self._advance_segment()

    def is_cycle_done(self) -> bool:
        return self._seg_idx >= len(self._segments)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, 1.4, 1.2], target=[0.4, 0.0, 0.1])

    scene = InteractiveScene(CncCellSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    sim_dt = sim.get_physics_dt()

    # sim.reset() leaves the arm at all-zero joints (standing straight up, a
    # singular pose the IK can't escape cleanly). Start from the configured
    # bent "ready" pose instead, then let it settle.
    robot = scene["robot"]
    joint_pos = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.set_joint_position_target_index(target=joint_pos)
    robot.reset()
    for _ in range(30):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    cell = CncTendingCell(scene, device=sim.device, cnc_cycle_steps=args_cli.cnc_cycle_steps)
    cell.start()

    print(f"Starting CNC tending cell: {args_cli.cycles} cycle(s)")
    while simulation_app.is_running():
        cell.step()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if cell.is_cycle_done():
            print(f"Cycle {cell.cycles_done} complete ({cell.misses} missed placement(s) so far)")
            if args_cli.test or cell.cycles_done >= args_cli.cycles:
                break
            cell.start()

    print("Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
