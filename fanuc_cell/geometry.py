"""Authors USD for what a cut produces: the finished part and the leftover skeleton.

Visual meshes are exact (rounded outline, kerf gap, drilled holes). Colliders
are simpler: a convex hull for the part, and four boxes around the hole for the
skeleton (PhysX dynamic bodies can't use concave triangle meshes).
"""

from __future__ import annotations

import math
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from .assets import GENERATED_DIR
from .jobs import TOOL_DIAMETER, Job, offset_outline, rounded_rect

WOOD = (0.66, 0.46, 0.26)  # pine, matches scene.PlankSpec.tint
WOOD_CUT_EDGE = (0.48, 0.32, 0.17)  # routed edges read slightly darker
HOLE = (0.10, 0.07, 0.04)


class _MeshBuilder:
    """Accumulates flat-shaded faces (each face group gets its own vertices)."""

    def __init__(self) -> None:
        self.points: list[tuple[float, float, float]] = []
        self.counts: list[int] = []
        self.indices: list[int] = []

    def tri(self, a, b, c) -> None:
        base = len(self.points)
        self.points += [tuple(a), tuple(b), tuple(c)]
        self.counts.append(3)
        self.indices += [base, base + 1, base + 2]

    def quad(self, a, b, c, d) -> None:
        self.tri(a, b, c)
        self.tri(a, c, d)

    def define(self, stage: Usd.Stage, path: str, rgb) -> UsdGeom.Mesh:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in self.points]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(self.counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(self.indices))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        return mesh


def _walls(mb: _MeshBuilder, loop: np.ndarray, z0: float, z1: float, outward: bool) -> None:
    n = len(loop)
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        if not outward:
            a, b = b, a
        mb.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1))


def _new_stage(path: str, root: str) -> Usd.Stage:
    """A stage whose default prim is a rigid body (Isaac Lab only tunes, never adds, RigidBodyAPI)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)
    stage.SetDefaultPrim(root_prim)
    return stage


def build_part_usd(job: Job) -> str:
    """Finished part, centered at its own origin, +x along its length."""
    path = os.path.join(GENERATED_DIR, f"part_{job.name}.usda")
    stage = _new_stage(path, "/Part")
    t = job.thickness
    outline = rounded_rect(job.length, job.width, job.corner_radius)

    faces = _MeshBuilder()
    n = len(outline)
    for i in range(n):  # convex: fan from the center
        a, b = outline[i], outline[(i + 1) % n]
        faces.tri((0, 0, t / 2), (a[0], a[1], t / 2), (b[0], b[1], t / 2))
        faces.tri((0, 0, -t / 2), (b[0], b[1], -t / 2), (a[0], a[1], -t / 2))
    body = faces.define(stage, "/Part/faces", WOOD)
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(body.GetPrim()).CreateApproximationAttr("convexHull")

    edges = _MeshBuilder()
    _walls(edges, outline, -t / 2, t / 2, outward=True)
    edges.define(stage, "/Part/routed_edge", WOOD_CUT_EDGE)

    # Drilled holes: dark bores that break through both faces.
    for i, (hx, hy, d) in enumerate(job.holes):
        bore = UsdGeom.Cylinder.Define(stage, f"/Part/hole_{i}")
        bore.CreateAxisAttr("Z")
        bore.CreateRadiusAttr(d / 2)
        bore.CreateHeightAttr(t + 0.0008)
        UsdGeom.XformCommonAPI(bore).SetTranslate(Gf.Vec3d(hx, hy, 0.0))
        bore.CreateDisplayColorAttr([Gf.Vec3f(*HOLE)])

    stage.GetRootLayer().Save()
    return path


def _rect_hit(center: np.ndarray, direction: np.ndarray, half_l: float, half_w: float) -> np.ndarray:
    """Where a ray from `center` (inside the rectangle) leaves the plank outline."""
    ts = []
    for axis, half in ((0, half_l), (1, half_w)):
        if abs(direction[axis]) > 1e-12:
            bound = half if direction[axis] > 0 else -half
            ts.append((bound - center[axis]) / direction[axis])
    return center + min(ts) * direction


def build_skeleton_usd(job: Job, plank_length: float, plank_width: float, thickness: float) -> str:
    """The plank with the part-plus-kerf cut out, in plank-local coordinates."""
    key = f"{job.name}_{round(plank_length * 1000)}x{round(plank_width * 1000)}x{round(thickness * 1000)}"
    path = os.path.join(GENERATED_DIR, f"skeleton_{key}.usda")
    stage = _new_stage(path, "/Skeleton")
    t, hl, hw = thickness, plank_length / 2, plank_width / 2
    center = np.asarray(job.part_center_in_plank(plank_length, plank_width))

    hole = offset_outline(rounded_rect(job.length, job.width, job.corner_radius, per_corner=16), TOOL_DIAMETER / 2)
    hole = hole + center
    # Pair every hole vertex (and every outer corner) with the matching point on the
    # plank outline along the same ray from the part center -> a quad strip.
    angles = [math.atan2(p[1] - center[1], p[0] - center[0]) for p in hole]
    corner_angles = [math.atan2(cy - center[1], cx - center[0]) for cx, cy in ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw))]
    all_angles = sorted(set(round(a, 9) for a in angles + corner_angles))
    hole_angles = np.unwrap(np.asarray(angles))
    order = np.argsort(hole_angles)
    hole_sorted, ang_sorted = hole[order], hole_angles[order]

    def inner_at(a: float) -> np.ndarray:
        a = (a - ang_sorted[0]) % (2 * math.pi) + ang_sorted[0]
        x = np.append(ang_sorted, ang_sorted[0] + 2 * math.pi)
        px = np.append(hole_sorted[:, 0], hole_sorted[0, 0])
        py = np.append(hole_sorted[:, 1], hole_sorted[0, 1])
        return np.array([np.interp(a, x, px), np.interp(a, x, py)])

    inner = np.array([inner_at(a) for a in all_angles])
    outer = np.array([_rect_hit(center, np.array([math.cos(a), math.sin(a)]), hl, hw) for a in all_angles])

    faces = _MeshBuilder()
    n = len(all_angles)
    for i in range(n):
        j = (i + 1) % n
        o0, o1, i0, i1 = outer[i], outer[j], inner[i], inner[j]
        faces.quad((i0[0], i0[1], t / 2), (o0[0], o0[1], t / 2), (o1[0], o1[1], t / 2), (i1[0], i1[1], t / 2))
        faces.quad((i1[0], i1[1], -t / 2), (o1[0], o1[1], -t / 2), (o0[0], o0[1], -t / 2), (i0[0], i0[1], -t / 2))
    faces.define(stage, "/Skeleton/faces", WOOD)
    sides = _MeshBuilder()
    _walls(sides, np.array([(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]), -t / 2, t / 2, outward=True)
    sides.define(stage, "/Skeleton/sawn_edge", WOOD)
    routed = _MeshBuilder()
    _walls(routed, hole_sorted, -t / 2, t / 2, outward=False)
    routed.define(stage, "/Skeleton/routed_edge", WOOD_CUT_EDGE)

    # Colliders: four boxes framing the hole's bounding box (invisible).
    x0, y0 = hole[:, 0].min(), hole[:, 1].min()
    x1, y1 = hole[:, 0].max(), hole[:, 1].max()
    strips = {
        "left": ((-hl, x0), (-hw, hw)),
        "right": ((x1, hl), (-hw, hw)),
        "front": ((x0, x1), (-hw, y0)),
        "back": ((x0, x1), (y1, hw)),
    }
    for name, ((ax, bx), (ay, by)) in strips.items():
        if bx - ax < 0.002 or by - ay < 0.002:
            continue
        box = UsdGeom.Cube.Define(stage, f"/Skeleton/collider_{name}")
        box.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(box)
        xf.SetTranslate(Gf.Vec3d((ax + bx) / 2, (ay + by) / 2, 0.0))
        xf.SetScale(Gf.Vec3f(bx - ax, by - ay, t))
        box.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    stage.GetRootLayer().Save()
    return path


def part_area(job: Job) -> float:
    o = rounded_rect(job.length, job.width, job.corner_radius)
    return 0.5 * abs(np.dot(o[:, 0], np.roll(o[:, 1], -1)) - np.dot(o[:, 1], np.roll(o[:, 0], -1)))
