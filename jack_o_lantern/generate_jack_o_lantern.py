#!/usr/bin/env python3
"""Generate a 3D-printable jack-o'-lantern (OBJ + MTL for Blender, STL for Bambu Studio).

The pumpkin is described as an implicit surface (signed-distance-like field):
  * a body of revolution (superellipse dome, rounded underside) around a
    centre point, with 10 lobes (wider ones at the face, narrower at the back),
    a soft stem dimple and a flat base;
  * a carved face (smiling eyes, nose, toothy grin) wrapped cylindrically around
    the front and cut 6.5 mm deep with rounded rims.  The ceilings of all
    openings slope at >=45 deg so the model prints standing on its base
    without supports.
The field is sampled on a voxel grid, meshed with marching cubes, simplified and
written out.  Units: millimetres.

    pip install numpy scipy scikit-image trimesh fast-simplification pillow
    python3 generate_jack_o_lantern.py
"""
import os
import numpy as np
import trimesh
import fast_simplification
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NAME = "jack_o_lantern"

# ---------------------------------------------------------------- body shape
A = 50.0            # radius of the lobe crests at the equator (width = 100 mm)
Z_EQ = 24.5         # height of the equator above the base
B_TOP = 45.5        # dome height above the equator
N_TOP = 2.05        # superellipse exponent of the dome (>2 = fuller shoulders)
R_BASE = 39.0       # radius of the underside profile at the base plane
P_LOW = 2.5         # underside profile exponent (overhang at the base ~48 deg)
DIMPLE = 0.09       # stem dimple depth (fraction of the dome radius)
DIMPLE_W = 0.28     # stem dimple angular width (rad)
N_LOBES = 10
LOBE_SKEW = 0.2     # front lobes 36/(1-0.2) = 45 deg wide, back lobes 30 deg
GROOVE = 0.065      # groove depth (fraction of radius)
GROOVE_SOFT = 0.08  # rounding of the groove bottom (fraction of a lobe half-width)

# ---------------------------------------------------------------- face carving
RD = 46.0           # radius at which the flat face design is wrapped
FACE_Z0 = 10.5      # height of the lowest point of the mouth
DEPTH = 6.5         # carving depth (horizontal)
K_CEIL = 1.2        # ceiling slope inside the openings (1 = 45 deg)
RIM_R = 1.0         # rounding of the outer rim of the openings
INNER_R = 0.8       # rounding of the inner corners of the openings

# ---------------------------------------------------------------- meshing
H = 0.25            # voxel size, mm
TARGET_FACES = 300_000

# Face design in (u, v) mm: u = arc length around the front (0 = centre),
# v = height above FACE_Z0.  Traced from the reference photo and cleaned up.
EYE_R = [(25.44, 28.21), (25.90, 28.27), (26.47, 28.49), (26.91, 28.80), (27.21, 29.28), (27.31, 29.84),
         (27.29, 30.62), (27.19, 30.97), (26.99, 31.25), (26.84, 31.65), (26.81, 31.92), (26.66, 32.38),
         (26.44, 32.86), (26.01, 33.46), (25.96, 33.73), (25.29, 35.01), (24.89, 35.66), (24.54, 36.06),
         (24.49, 36.28), (24.19, 36.83), (23.94, 37.11), (23.71, 37.50), (23.29, 37.98), (23.11, 38.30),
         (22.89, 38.56), (22.81, 38.81), (22.46, 39.17), (22.21, 39.62), (21.81, 39.95), (21.16, 40.84),
         (20.90, 41.05), (20.48, 41.35), (20.33, 41.39), (19.83, 41.35), (19.45, 41.18), (19.24, 40.99),
         (18.94, 40.56), (18.49, 40.15), (17.74, 39.10), (16.54, 37.85), (16.06, 37.24), (15.84, 37.04),
         (15.54, 36.58), (14.99, 36.00), (13.29, 33.73), (11.89, 31.45), (11.44, 30.47), (11.16, 29.67),
         (11.19, 28.93), (11.41, 28.60), (11.85, 28.31), (12.38, 28.20), (12.85, 28.23), (13.48, 28.37),
         (16.33, 29.61), (17.73, 30.07), (18.60, 30.24), (20.20, 30.24), (21.08, 30.11), (21.98, 29.81),
         (22.77, 29.44), (24.70, 28.36), (25.28, 28.20)]
NOSE = [(3.78, 20.42), (4.29, 20.54), (4.66, 20.81), (4.71, 21.36), (4.57, 22.17), (3.90, 23.83),
        (3.07, 25.55), (1.42, 28.50), (1.06, 28.85), (0.66, 28.93), (-0.08, 29.00), (-1.08, 28.85),
        (-1.44, 28.50), (-3.07, 25.59), (-3.92, 23.84), (-4.59, 22.17), (-4.73, 21.36), (-4.69, 20.81),
        (-4.32, 20.54), (-3.60, 20.40), (-2.23, 20.59), (-0.11, 20.63), (2.20, 20.59), (3.53, 20.40)]
MOUTH_HALF = 32.0   # half-width of the grin
MOUTH_CORNER = 18.0
LIP_MID = 12.7      # height of the upper lip between the teeth
TEETH_UP = [(-18.5, -11.1, 9.9), (8.8, 16.4, 9.9)]  # (u0, u1, lowest v)
TOOTH_LOW = (1.0, 10.1, 4.4)                       # (u0, u1, highest v)


def smax(a, b, k):
    """Polynomial smooth maximum (rounds the intersection edge with radius ~k)."""
    h = np.clip(k - np.abs(a - b), 0.0, None) / k
    return np.maximum(a, b) + h * h * k * 0.25


# ------------------------------------------------------------------ the body
def _profile_table():
    """Radius of the (lobe-less) body of revolution as a function of the polar
    angle theta measured from the centre (0, 0, Z_EQ); theta = 0 points up."""
    t = np.linspace(0.0, np.pi / 2, 2000)
    rho_top = A * np.cos(t) ** (2 / N_TOP)
    z_top = Z_EQ + B_TOP * np.sin(t) ** (2 / N_TOP)
    k = (A - R_BASE) / Z_EQ ** P_LOW           # rounded underside
    z_low = np.linspace(Z_EQ, -6.0, 2000)
    rho_low = A - k * (Z_EQ - z_low) ** P_LOW
    rho_cap = np.linspace(rho_low[-1], 0.0, 500)
    rho = np.concatenate([rho_top[::-1], rho_low[1:], rho_cap[1:]])
    z = np.concatenate([z_top[::-1], z_low[1:], np.full(499, -6.0)])
    th = np.arctan2(rho, z - Z_EQ)
    r = np.hypot(rho, z - Z_EQ)
    assert np.all(np.diff(th) > 0), "profile must be star-shaped around the centre"
    return th, r


PROF_TH, PROF_R = _profile_table()


def body_field(x, y, z):
    dz = z - Z_EQ
    rho = np.hypot(x, y)
    r = np.sqrt(rho * rho + dz * dz)
    th = np.arctan2(rho, dz)
    R = np.interp(th, PROF_TH, PROF_R)
    R *= 1.0 - DIMPLE * np.exp(-(th / DIMPLE_W) ** 2)
    al = np.arctan2(x, -y)                     # 0 = front (-Y), a groove runs down the face
    w = N_LOBES / (2 * np.pi) * (al - LOBE_SKEW * np.sin(al)) + 0.5
    q = 1.0 - 2.0 * np.abs(w - np.round(w))    # 1 on a crest, 0 in a groove
    q = np.sqrt(q * q + GROOVE_SOFT ** 2) - GROOVE_SOFT
    sin_th = rho / np.maximum(r, 1e-9)
    g = GROOVE * (1.0 - np.exp(-(sin_th / 0.35) ** 2))   # grooves fade into the dimple
    # the groove between the two face lobes is softer across the face
    g = g * (1.0 - 0.5 * np.exp(-(al / 0.3) ** 2 - ((z - FACE_Z0 - 20.0) / 26.0) ** 4))
    return r - R * (1.0 - g * (1.0 - q) ** 2)


# ------------------------------------------------------------------ the face
U0, U1, V0, V1, FPX = -42.0, 42.0, -8.0, 56.0, 0.025


def face_sdf_table():
    nu = int(round((U1 - U0) / FPX)) + 1
    nv = int(round((V1 - V0) / FPX)) + 1
    u = U0 + FPX * np.arange(nu)
    v = V0 + FPX * np.arange(nv)
    uu, vv = np.meshgrid(u, v, indexing="xy")   # rows = v, cols = u

    # grin: crescent between the lower edge and the upper lip, minus teeth
    t = np.clip(np.abs(uu) / MOUTH_HALF, 0, 1)
    v_lo = MOUTH_CORNER * (1.0 - (1.0 - t * t) ** 0.707)
    v_up = LIP_MID + (MOUTH_CORNER + 0.2 - LIP_MID) * t ** 2.2
    mouth = (np.abs(uu) < MOUTH_HALF) & (vv > v_lo) & (vv < v_up)
    for a, b, vmin in TEETH_UP:
        mouth &= ~((uu > a) & (uu < b) & (vv > vmin))
    a, b, vmax = TOOTH_LOW
    mouth &= ~((uu > a) & (uu < b) & (vv < vmax))

    img = Image.new("L", (nu, nv), 0)
    draw = ImageDraw.Draw(img)
    for poly in (EYE_R, [(-p, q) for p, q in EYE_R], NOSE):
        draw.polygon([((p - U0) / FPX, (q - V0) / FPX) for p, q in poly], fill=1)
    eyes_nose = np.asarray(img).astype(bool)

    def sdf(m):
        return (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)) * FPX

    def rounded_sdf(m, r_open=0.7, r_close=0.9):  # round convex / concave corners
        m = sdf(sdf(m) < -r_open) < r_open
        m = sdf(sdf(m) < r_close) < -r_close
        return sdf(m).astype(np.float32)

    # Two groups, each crossed at most once by any vertical line (needed by the
    # sloped-ceiling test in cutter_field): eyes + nose, and the grin.
    return [rounded_sdf(eyes_nose), rounded_sdf(mouth)]


def sample2d(table, u, v):
    return ndimage.map_coordinates(table, [(v - V0) / FPX, (u - U0) / FPX], order=1, mode="nearest")


# horizontal distance from the axis to the body surface, tabulated over (alpha, z)
AL0, AL1, DAL = -1.3, 1.3, 0.002
ZT0, ZT1, DZT = -1.0, 75.0, 0.05


def surface_radius_table():
    al = np.arange(AL0, AL1 + DAL / 2, DAL)
    z = np.arange(ZT0, ZT1 + DZT / 2, DZT)
    AL, Z = np.meshgrid(al, z, indexing="ij")
    lo = np.zeros_like(AL)
    hi = np.full_like(AL, A + 2.0)
    inside_axis = body_field(np.zeros_like(Z) + 1e-6, np.zeros_like(Z), Z) < 0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f = body_field(mid * np.sin(AL), -mid * np.cos(AL), Z)
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    return np.where(inside_axis, 0.5 * (lo + hi), 0.0)


def cutter_field(x, y, z, face, rsurf):
    """Signed field of the carving volume (negative inside); also returns the
    horizontal depth below the outer surface."""
    al = np.arctan2(x, -y)
    rho = np.hypot(x, y)
    rs = ndimage.map_coordinates(rsurf, [(al - AL0) / DAL, (z - ZT0) / DZT], order=1, mode="nearest")
    s = rs - rho
    u = RD * al
    v = z - FACE_Z0
    v_up = v + K_CEIL * np.maximum(s, 0.0)    # the ceiling drops as the cut goes deeper
    c = np.min([np.maximum(sample2d(t, u, v), sample2d(t, u, v_up)) for t in face], axis=0)
    c = smax(c, s - DEPTH, INNER_R)
    c = np.where(np.abs(al) < 1.1, c, 50.0)
    return c, s


# ------------------------------------------------------------------ build
def build():
    face = face_sdf_table()
    rsurf = surface_radius_table()

    xs = np.arange(-A - 1.0, A + 1.0 + H / 2, H)
    ys = xs.copy()
    zs = np.arange(-0.625, Z_EQ + B_TOP + 1.5, H)    # no sample exactly on the base plane
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    front = (Y < 2.0) & (np.abs(X) < A)
    for k, zk in enumerate(zs):
        Z = np.full_like(X, zk)
        f = body_field(X, Y, Z)
        if FACE_Z0 - 4 < zk < FACE_Z0 + 48:
            c, _ = cutter_field(X[front], Y[front], Z[front], face, rsurf)
            f[front] = smax(f[front], -c, RIM_R)
        vol[:, :, k] = np.maximum(f, -zk)       # flat base at z = 0

    verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(H, H, H), allow_degenerate=False)
    verts += np.array([xs[0], ys[0], zs[0]])
    print(f"marching cubes: {len(verts)} vertices, {len(faces)} faces")

    reduction = 1.0 - TARGET_FACES / len(faces)
    if reduction > 0:
        verts, faces = fast_simplification.simplify(verts, faces, target_reduction=reduction, agg=6)
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    mesh.vertices[np.abs(mesh.vertices[:, 2]) < 1e-3, 2] = 0.0   # keep the base exactly flat
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()

    # mark the faces inside the carving (for the dark material in the OBJ)
    cen = mesh.triangles_center
    c, s = cutter_field(cen[:, 0], cen[:, 1], cen[:, 2], face, rsurf)
    inside = (cen[:, 1] < 0) & (s > 1.1) & (c < 0.5)
    return mesh, inside


def write_outputs(mesh, inside):
    stl_path = os.path.join(OUT_DIR, NAME + ".stl")
    mesh.export(stl_path)

    # OBJ: Y-up / -Z forward convention (Blender's default import settings
    # turn it back into Z-up with the face looking at the Front view).
    v = mesh.vertices
    n = mesh.vertex_normals
    v_obj = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])
    n_obj = np.column_stack([n[:, 0], n[:, 2], -n[:, 1]])
    mtl_name = NAME + ".mtl"
    with open(os.path.join(OUT_DIR, mtl_name), "w") as fh:
        fh.write("# Jack-o'-lantern materials\n"
                 "newmtl pumpkin_skin\nKa 0 0 0\nKd 0.95 0.42 0.05\nKs 0.25 0.20 0.15\nNs 40\nd 1\nillum 2\n\n"
                 "newmtl pumpkin_inside\nKa 0 0 0\nKd 0.22 0.08 0.02\nKs 0.02 0.02 0.02\nNs 5\nd 1\nillum 2\n")
    lines = ["# Jack-o'-lantern, units: millimetres, Y up, face towards +Z",
             f"mtllib {mtl_name}", f"o {NAME}"]
    lines += [f"v {a:.4f} {b:.4f} {c:.4f}" for a, b, c in v_obj]
    lines += [f"vn {a:.4f} {b:.4f} {c:.4f}" for a, b, c in n_obj]
    lines.append("s 1")
    for mat, sel in (("pumpkin_skin", ~inside), ("pumpkin_inside", inside)):
        lines.append(f"usemtl {mat}")
        lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in mesh.faces[sel] + 1]
    with open(os.path.join(OUT_DIR, NAME + ".obj"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    mesh, inside = build()
    print("watertight:", mesh.is_watertight, "winding consistent:", mesh.is_winding_consistent)
    print("size (mm):", np.round(mesh.extents, 2), "volume (cm3):", round(mesh.volume / 1000, 1))
    print("faces:", len(mesh.faces), "carved faces:", int(inside.sum()))
    write_outputs(mesh, inside)
