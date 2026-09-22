"""Scene config for the FANUC CNC-tending cell."""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.assets import SurfaceGripperCfg

from . import layout as L
from .assets import ROBOT_USD

WOOD_DENSITY = 500.0  # kg/m^3, pine


@dataclass(frozen=True)
class PlankSpec:
    name: str
    length: float  # along the plank's local X
    width: float
    thickness: float
    pos: tuple[float, float, float]  # center
    yaw: float
    tint: tuple[float, float, float] = (0.66, 0.46, 0.26)  # pine


def _static(size, color, rough=0.6) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=rough),
    )


def _box_asset(path: str, box: L.Box, color, rough=0.6) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{path}",
        spawn=_static(box.size, color, rough),
        init_state=AssetBaseCfg.InitialStateCfg(pos=box.center),
    )


def _kinematic(path: str, size, color, pos, collide: bool = True) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{path}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.3, roughness=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    import math

    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def plank_cfg(spec: PlankSpec) -> RigidObjectCfg:
    mass = spec.length * spec.width * spec.thickness * WOOD_DENSITY
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
        spawn=sim_utils.CuboidCfg(
            size=(spec.length, spec.width, spec.thickness),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.tint, roughness=0.8),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.pos, rot=yaw_quat_xyzw(spec.yaw)),
    )


CRX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=L.ROBOT_BASE, joint_pos=L.HOME_JOINTS),
    actuators={
        # Torque limits from the CRX asset's drives; stiff position control for task-space IK.
        "base": ImplicitActuatorCfg(joint_names_expr=["J[1-2]"], stiffness=6000.0, damping=400.0, effort_limit_sim=400.0),
        "elbow": ImplicitActuatorCfg(joint_names_expr=["J3"], stiffness=5000.0, damping=300.0, effort_limit_sim=200.0),
        "wrist": ImplicitActuatorCfg(joint_names_expr=["J[4-6]"], stiffness=2000.0, damping=100.0, effort_limit_sim=60.0),
    },
)


_RAIL_LEN = L.RAIL_X_RANGE[1] - L.RAIL_X_RANGE[0]
_RAIL_CX = sum(L.RAIL_X_RANGE) / 2


@configclass
class CellSceneCfg(InteractiveSceneCfg):
    """Static cell. Planks are added per run with add_planks()."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=450.0))
    sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(intensity=900.0, angle=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2, 0.3, 0.0, 0.93)),
    )

    robot = CRX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    vacuum = SurfaceGripperCfg(
        prim_path="{ENV_REGEX_NS}/Robot/vacuum_tool/SurfaceGripper",
        max_grip_distance=0.02,
        coaxial_force_limit=600.0,
        shear_force_limit=400.0,
        retry_interval=0.2,
    )

    pedestal = _box_asset("Pedestal", L.PEDESTAL, (0.18, 0.18, 0.20), rough=0.4)
    infeed_table = _box_asset("InfeedTable", L.INFEED_TABLE, (0.35, 0.36, 0.40))
    staging_table = _box_asset("StagingTable", L.STAGING_TABLE, (0.30, 0.32, 0.36))
    output_table = _box_asset("OutputTable", L.OUTPUT_TABLE, (0.28, 0.45, 0.35))

    cnc_body = _box_asset("CncBody", L.CNC_BODY, (0.78, 0.80, 0.83), rough=0.35)
    spoilboard = _box_asset("CncSpoilboard", L.SPOILBOARD, (0.42, 0.34, 0.24), rough=0.9)
    front_fence = _box_asset("CncFrontFence", L.FRONT_FENCE, (0.7, 0.72, 0.75), rough=0.3)
    side_stop = _box_asset("CncSideStop", L.SIDE_STOP, (0.7, 0.72, 0.75), rough=0.3)
    rail_left = _box_asset("CncRailLeft", L.Box((_RAIL_CX, L.RAIL_Y[0], L.BED_TOP_Z + 0.025), (_RAIL_LEN, 0.04, 0.05)),
                           (0.2, 0.2, 0.22), rough=0.3)
    rail_right = _box_asset("CncRailRight", L.Box((_RAIL_CX, L.RAIL_Y[1], L.BED_TOP_Z + 0.025), (_RAIL_LEN, 0.04, 0.05)),
                            (0.2, 0.2, 0.22), rough=0.3)

    # Moving gantry (kinematic): two uprights on the side rails, a beam across the
    # bed, a carriage on the beam and a spindle that plunges. Starts parked at the back.
    # (Initial poses are placeholders; CncRouter sets the real ones on startup.)
    gantry_upright_left = _kinematic("CncUprightLeft", (0.14, 0.08, 0.45), (0.95, 0.45, 0.05),
                                     (L.GANTRY_PARK_X, L.RAIL_Y[0], L.RAIL_TOP_Z + 0.225))
    gantry_upright_right = _kinematic("CncUprightRight", (0.14, 0.08, 0.45), (0.95, 0.45, 0.05),
                                      (L.GANTRY_PARK_X, L.RAIL_Y[1], L.RAIL_TOP_Z + 0.225))
    gantry_beam = _kinematic("CncBeam", (0.16, 1.44, 0.12), (0.95, 0.45, 0.05),
                             (L.GANTRY_PARK_X, 0.0, L.GANTRY_BEAM_Z))
    carriage = _kinematic("CncCarriage", (0.10, 0.16, 0.22), (0.25, 0.26, 0.28),
                          (L.GANTRY_PARK_X - L.CARRIAGE_DX, 0.0, L.GANTRY_BEAM_Z - 0.03))
    # Spindle and bit don't collide: the bit plunges into the wood, and as a
    # kinematic collider it would shove the blank off the bed instead of cutting.
    spindle = _kinematic("CncSpindle", (0.09, 0.09, 0.26), (0.80, 0.80, 0.82),
                         (L.GANTRY_PARK_X - L.CARRIAGE_DX, 0.0, L.SPINDLE_PARK_Z + 0.18), collide=False)
    bit = _kinematic("CncBit", (0.006, 0.006, 0.05), (0.75, 0.75, 0.78),
                     (L.GANTRY_PARK_X - L.CARRIAGE_DX, 0.0, L.SPINDLE_PARK_Z + 0.025), collide=False)


def add_planks(cfg: CellSceneCfg, specs: list[PlankSpec]) -> None:
    for spec in specs:
        setattr(cfg, spec.name, plank_cfg(spec))


def add_wood_usd(cfg: CellSceneCfg, name: str, usd_path: str, mass: float, park: tuple[float, float, float]) -> None:
    """A cut part or skeleton (USD from geometry.py), spawned parked out of the way."""
    setattr(cfg, name, RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=park),
    ))


def add_scrap_bin(cfg: CellSceneCfg) -> None:
    (cx, cy), (sx, sy, sz), t = L.SCRAP_BIN_CENTER, L.SCRAP_BIN_SIZE, L.SCRAP_BIN_WALL
    color = (0.15, 0.30, 0.55)
    walls = {
        "Floor": ((cx, cy, t / 2), (sx, sy, t)),
        "WallXNeg": ((cx - sx / 2 + t / 2, cy, sz / 2), (t, sy, sz)),
        "WallXPos": ((cx + sx / 2 - t / 2, cy, sz / 2), (t, sy, sz)),
        "WallYNeg": ((cx, cy - sy / 2 + t / 2, sz / 2), (sx, t, sz)),
        "WallYPos": ((cx, cy + sy / 2 - t / 2, sz / 2), (sx, t, sz)),
    }
    for name, (center, size) in walls.items():
        setattr(cfg, f"scrap_{name.lower()}", _box_asset(f"ScrapBin{name}", L.Box(center, size), color))
