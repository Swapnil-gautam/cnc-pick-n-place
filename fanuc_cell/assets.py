"""Authors USD assets the cell needs that don't exist off the shelf.

Must be called after the Kit app has launched (pxr + the cloud asset resolver).
"""

from __future__ import annotations

import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

CRX_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/Robots/Fanuc/crx10ia_l/crx10ia_l.usd"
)

# Measured from the CRX asset: J6_link origin -> flange is 0.16 m along J6 +X,
# and at zero joint angles the flange sits at (0.70, -0.15, 0.955) with identity rotation.
FLANGE_OFFSET = 0.16
FLANGE_POS_AT_ZERO = (0.70, -0.15, 0.955)

# Vacuum tool, in its own frame: +X is the approach axis (out of the cup faces).
TOOL_LENGTH = 0.105  # flange -> cup lip
CUP_RADIUS = 0.020
# 2x2 cups, 60 mm x 140 mm centers; tool Z is the long axis. Sized so all four
# cups fit on the narrowest part and on the skeleton's handling strip.
CUP_OFFSETS_YZ = [(y, z) for y in (-0.03, 0.03) for z in (-0.07, 0.07)]

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
ROBOT_USD = os.path.join(GENERATED_DIR, "crx10ia_l_vacuum.usda")

_Q_JOINT_Z_TO_TOOL_X = Gf.Quatf(0.70710678, 0.0, 0.70710678, 0.0)  # (w, x, y, z): rotate +90 deg about Y


def _color(prim: Usd.Prim, rgb: tuple[float, float, float]) -> None:
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _add_collider(prim: Usd.Prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)


def _cylinder_x(stage: Usd.Stage, path: str, radius: float, height: float, x_center: float,
                y: float = 0.0, z: float = 0.0, rgb=(0.2, 0.2, 0.2)) -> None:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("X")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(x_center, y, z))
    _color(cyl.GetPrim(), rgb)
    _add_collider(cyl.GetPrim())


def _add_attachment_joint(stage: Usd.Stage, path: str, tool_path: str, anchor_path: str,
                          y: float, z: float) -> Usd.Prim:
    """One suction cup: a compliant D6 joint the surface gripper re-targets onto the gripped object.

    Mirrors isaacsim.robot.surface_gripper's reference asset. body1 points at J6_link, which
    is rigidly fixed to the tool, so the placeholder constraint is redundant until a grasp.
    """
    joint = UsdPhysics.Joint.Define(stage, path)
    prim = joint.GetPrim()
    prim.AddAppliedSchema("IsaacAttachmentPointAPI")
    for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        UsdPhysics.LimitAPI.Apply(prim, axis)
    for axis in ("transZ", "rotX", "rotY", "rotZ"):
        UsdPhysics.DriveAPI.Apply(prim, axis)

    # transX/transY locked (high < low), a little plunge + tilt compliance like a bellows cup.
    for axis in ("transX", "transY"):
        prim.GetAttribute(f"limit:{axis}:physics:low").Set(1.0)
        prim.GetAttribute(f"limit:{axis}:physics:high").Set(-1.0)
    prim.GetAttribute("limit:transZ:physics:low").Set(0.0)
    prim.GetAttribute("limit:transZ:physics:high").Set(0.01)
    for axis in ("rotX", "rotY", "rotZ"):
        prim.GetAttribute(f"limit:{axis}:physics:low").Set(-3.0)
        prim.GetAttribute(f"limit:{axis}:physics:high").Set(3.0)
    prim.GetAttribute("drive:rotX:physics:stiffness").Set(100.0)
    prim.GetAttribute("drive:rotY:physics:stiffness").Set(100.0)
    prim.GetAttribute("drive:rotZ:physics:stiffness").Set(10000.0)
    prim.GetAttribute("drive:transZ:physics:stiffness").Set(5000.0)
    prim.GetAttribute("drive:transZ:physics:damping").Set(100.0)

    prim.CreateAttribute("isaac:forwardAxis", Sdf.ValueTypeNames.Token).Set("Z")
    # The suction ray starts this far past the cup lip (to skip the cup's own collider).
    # Keep it below the approach gap: a ray that starts inside the plank never hits it.
    prim.CreateAttribute("isaac:clearanceOffset", Sdf.ValueTypeNames.Float).Set(0.001)
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateJointEnabledAttr(True)
    joint.CreateBreakForceAttr(3.4028235e38)
    joint.CreateBreakTorqueAttr(3.4028235e38)

    joint.CreateBody0Rel().SetTargets([Sdf.Path(tool_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(anchor_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(TOOL_LENGTH, y, z))
    joint.CreateLocalRot0Attr(_Q_JOINT_Z_TO_TOOL_X)
    joint.CreateLocalPos1Attr(Gf.Vec3f(TOOL_LENGTH + FLANGE_OFFSET, y, z))
    joint.CreateLocalRot1Attr(_Q_JOINT_Z_TO_TOOL_X)
    return prim


def build_robot_usd(path: str = ROBOT_USD) -> str:
    """FANUC CRX-10iA/L with a 4-cup vacuum tool and an IsaacSurfaceGripper on the flange."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    robot = stage.DefinePrim("/Robot", "Xform")
    robot.GetReferences().AddReference(CRX_USD)
    stage.SetDefaultPrim(robot)

    # The tool is its own rigid body, fixed to J6_link at the flange.
    tool_path = "/Robot/vacuum_tool"
    tool = UsdGeom.Xform.Define(stage, tool_path)
    UsdGeom.XformCommonAPI(tool).SetTranslate(Gf.Vec3d(*FLANGE_POS_AT_ZERO))
    UsdPhysics.RigidBodyAPI.Apply(tool.GetPrim())
    UsdPhysics.MassAPI.Apply(tool.GetPrim()).CreateMassAttr(1.2)

    mount = UsdPhysics.FixedJoint.Define(stage, f"{tool_path}/mount_joint")
    mount.CreateBody0Rel().SetTargets([Sdf.Path("/Robot/J6_link")])
    mount.CreateBody1Rel().SetTargets([Sdf.Path(tool_path)])
    mount.CreateLocalPos0Attr(Gf.Vec3f(FLANGE_OFFSET, 0.0, 0.0))
    mount.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.CreateLocalRot0Attr(Gf.Quatf(1.0))
    mount.CreateLocalRot1Attr(Gf.Quatf(1.0))

    # Geometry: adapter flange -> aluminium plate -> 4 rubber cups.
    _cylinder_x(stage, f"{tool_path}/adapter", 0.035, 0.05, 0.025, rgb=(0.15, 0.15, 0.17))
    plate = UsdGeom.Cube.Define(stage, f"{tool_path}/plate")
    plate.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(plate)
    xf.SetTranslate(Gf.Vec3d(0.056, 0.0, 0.0))
    xf.SetScale(Gf.Vec3f(0.012, 0.10, 0.19))
    _color(plate.GetPrim(), (0.75, 0.77, 0.80))
    _add_collider(plate.GetPrim())
    cup_height = TOOL_LENGTH - 0.062
    for i, (y, z) in enumerate(CUP_OFFSETS_YZ):
        _cylinder_x(stage, f"{tool_path}/cup_{i}", CUP_RADIUS, cup_height, 0.062 + cup_height / 2,
                    y=y, z=z, rgb=(0.05, 0.05, 0.05))

    joints = [
        _add_attachment_joint(stage, f"/Robot/suction_joints/cup_{i}", tool_path, "/Robot/J6_link", y, z).GetPath()
        for i, (y, z) in enumerate(CUP_OFFSETS_YZ)
    ]

    gripper = stage.DefinePrim(f"{tool_path}/SurfaceGripper", "IsaacSurfaceGripper")
    gripper.CreateAttribute("isaac:status", Sdf.ValueTypeNames.Token).Set("Open")
    gripper.CreateAttribute("isaac:maxGripDistance", Sdf.ValueTypeNames.Float).Set(0.02)
    gripper.CreateAttribute("isaac:coaxialForceLimit", Sdf.ValueTypeNames.Float).Set(600.0)
    gripper.CreateAttribute("isaac:shearForceLimit", Sdf.ValueTypeNames.Float).Set(400.0)
    gripper.CreateAttribute("isaac:retryInterval", Sdf.ValueTypeNames.Float).Set(0.2)
    gripper.CreateRelationship("isaac:attachmentPoints").SetTargets(joints)
    gripper.CreateRelationship("isaac:grippedObjects")

    stage.GetRootLayer().Save()
    return path
