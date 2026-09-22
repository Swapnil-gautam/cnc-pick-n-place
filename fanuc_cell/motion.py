"""Arm + vacuum control for the CRX-10iA/L, written as step generators.

A robot program is a generator; every `yield` advances the simulation by one
physics step. Programs compose with `yield from`, like TP programs calling
sub-programs.
"""

from __future__ import annotations

import math
from collections.abc import Generator

import torch
import warp as wp

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from .assets import TOOL_LENGTH

Program = Generator[None, None, None]


def _q_about_z(angle: float, device) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)], device=device)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class ArmController:
    """Moves the vacuum TCP (center of the cup lips) with differential IK.

    The tool always points straight down; `yaw` rotates the cup array about the
    vertical. yaw=0 puts the cup array's long axis along world +X.
    """

    TRAVEL_FRACTION = 0.75

    def __init__(self, scene: InteractiveScene, physics_dt: float) -> None:
        self.robot = scene["robot"]
        self.device = self.robot.device
        self.dt = physics_dt
        self.arm_cfg = SceneEntityCfg("robot", joint_names=["J[1-6]"], body_names=["vacuum_tool"])
        self.arm_cfg.resolve(scene)
        self.body_id = self.arm_cfg.body_ids[0]
        self.jacobi_body_idx = self.body_id - 1  # fixed base: root body has no Jacobian row
        self.jacobi_joint_ids = [j + self.robot.num_base_dofs for j in self.arm_cfg.joint_ids]

        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik = DifferentialIKController(cfg, num_envs=1, device=self.device)
        self._command = torch.zeros(1, 7, device=self.device)
        # Tool +X (approach axis) -> world -Z: +90 deg about Y.
        s = math.sin(math.pi / 4)
        self._q_down = torch.tensor([0.0, s, 0.0, s], device=self.device)
        self._tcp_offset = torch.tensor([TOOL_LENGTH, 0.0, 0.0], device=self.device)
        self.target_pos = None
        self.target_yaw = 0.0

    # -- state (all public positions are world frame) -------------------------

    def _tool_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        body = self.robot.data.body_pose_w.torch[:, self.body_id]
        root = self.robot.data.root_pose_w.torch
        pos, quat = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], body[:, 0:3], body[:, 3:7])
        return pos, quat

    def tcp_pos(self) -> torch.Tensor:
        body = self.robot.data.body_pose_w.torch[:, self.body_id]
        return (body[:, 0:3] + quat_apply(body[:, 3:7], self._tcp_offset.unsqueeze(0)))[0]

    def base_pos(self) -> torch.Tensor:
        return self.robot.data.root_pose_w.torch[0, 0:3]

    def tcp_yaw(self) -> float:
        _, quat = self._tool_pose_b()
        # Cup array long axis is tool +Z; project it onto the floor.
        axis = quat_apply(quat, torch.tensor([[0.0, 0.0, 1.0]], device=self.device))[0]
        return math.atan2(axis[1].item(), axis[0].item())

    # -- low level ------------------------------------------------------------

    def _tool_quat(self, yaw: float) -> torch.Tensor:
        return quat_mul(_q_about_z(yaw, self.device).unsqueeze(0), self._q_down.unsqueeze(0))[0]

    def command_tcp(self, tcp: torch.Tensor, yaw: float) -> None:
        """tcp: world-frame position. The robot base is only translated, never rotated."""
        quat = self._tool_quat(yaw)
        tool_pos = tcp - quat_apply(quat.unsqueeze(0), self._tcp_offset.unsqueeze(0))[0]
        self._command[0, 0:3] = tool_pos - self.base_pos()
        self._command[0, 3:7] = quat
        self.ik.set_command(self._command)
        pos, cur_quat = self._tool_pose_b()
        jac = self.robot.data.body_link_jacobian_w.torch[:, self.jacobi_body_idx, :, self.jacobi_joint_ids]
        q = self.robot.data.joint_pos.torch[:, self.arm_cfg.joint_ids]
        q_des = self.ik.compute(pos, cur_quat, jac, q)
        self.robot.set_joint_position_target_index(target=q_des, joint_ids=self.arm_cfg.joint_ids)
        self.target_pos, self.target_yaw = tcp.clone(), yaw

    def closest_equivalent_yaw(self, yaw: float) -> float:
        """The cup array is symmetric under 180 deg; pick whichever is nearer the current yaw."""
        cur = self.tcp_yaw()
        return min((yaw, yaw + math.pi, yaw - math.pi), key=lambda a: abs(_wrap(a - cur)))

    # -- programs -------------------------------------------------------------

    def hold(self, seconds: float) -> Program:
        for _ in range(max(1, round(seconds / self.dt))):
            if self.target_pos is not None:
                self.command_tcp(self.target_pos, self.target_yaw)
            yield

    def move_to(self, goal: torch.Tensor, yaw: float, seconds: float) -> Program:
        """Smooth move: sweeps around the base (cylindrical interpolation), then settles."""
        start = self.tcp_pos().clone()
        yaw0 = self.tcp_yaw()
        dyaw = _wrap(yaw - yaw0)
        r0, r1 = torch.linalg.norm(start[0:2]).item(), torch.linalg.norm(goal[0:2]).item()
        a0, a1 = math.atan2(start[1].item(), start[0].item()), math.atan2(goal[1].item(), goal[0].item())
        da = _wrap(a1 - a0)
        steps = max(1, round(seconds / self.dt))
        for k in range(steps):
            t = min(1.0, k / (self.TRAVEL_FRACTION * steps))
            s = t * t * (3.0 - 2.0 * t)
            r, a = r0 + (r1 - r0) * s, a0 + da * s
            z = start[2].item() + (goal[2].item() - start[2].item()) * s
            tcp = torch.tensor([r * math.cos(a), r * math.sin(a), z], device=self.device)
            self.command_tcp(tcp, yaw0 + dyaw * s)
            yield

    def move_linear(self, goal: torch.Tensor, yaw: float, seconds: float) -> Program:
        """Straight-line move, for short approaches/retracts where the path must be exact."""
        start = self.tcp_pos().clone()
        yaw0 = self.tcp_yaw()
        dyaw = _wrap(yaw - yaw0)
        steps = max(1, round(seconds / self.dt))
        for k in range(steps):
            t = min(1.0, k / (self.TRAVEL_FRACTION * steps))
            s = t * t * (3.0 - 2.0 * t)
            self.command_tcp(start + (goal - start) * s, yaw0 + dyaw * s)
            yield


class Vacuum:
    """Isaac Lab SurfaceGripper wrapper. state: -1 open, 0 closing, 1 closed (holding)."""

    def __init__(self, scene: InteractiveScene, arm: ArmController) -> None:
        self.gripper = scene["vacuum"]
        self.arm = arm

    def _set(self, value: float) -> None:
        self.gripper.set_grippers_command_index(torch.tensor([value], device=self.gripper.device))

    @property
    def holding(self) -> bool:
        return int(wp.to_torch(self.gripper.state)[0].item()) == 1

    def grip(self, timeout: float = 1.0) -> Program:
        """Vacuum on. Returns once a seal forms (or the timeout passes -- check `holding`)."""
        self._set(1.0)
        for _ in range(max(1, round(timeout / self.arm.dt))):
            yield from self.arm.hold(self.arm.dt)
            if self.holding:
                break
        yield from self.arm.hold(0.2)  # let the seal settle before lifting

    def release(self, seconds: float = 0.4) -> Program:
        self._set(-1.0)
        yield from self.arm.hold(seconds)
