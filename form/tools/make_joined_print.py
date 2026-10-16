"""
Print layout for the two shell halves as ONE stable part: both halves stand together like a closed cup
(3 mm apart at the seams) and are held by thin break-away tie walls outside the seam ends.
The halves themselves are exactly mold_shell_A.stl / mold_shell_B.stl (only moved +-1.5 mm apart);
ties touch only the outer end faces of the lip / lugs (never the mating faces) through small dots.

  python3 make_joined_print.py --height 80   -> print/mold_shells_AB_joined.stl
  python3 make_joined_print.py --height 60   -> print_60mm/mold_shells_AB_joined.stl
"""
import os, sys, argparse
import numpy as np, trimesh, manifold3d as mf
from shapely.geometry import box as sbox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_mold as B

GAP = 3.0          # gap between the mating faces while printing (ridge of A stays 1.5 mm away from B)
TIE_T = 0.8        # tie wall thickness (2 lines)
TIE_GAP = 0.5      # air gap between tie wall and part
DOT_H, DOT_W = 1.2, 1.0
DOT_Y0 = 2.4       # dots sit 2.4..3.4 mm from the mating face (clear of the V-notch chamfer)
DOT_STEP = 9.0     # min spacing of dots along z
DOT_MAX_GAP = 4.8  # longest horizontal bridge a dot may span

def load(path):
    t = trimesh.load(path, process=True)
    return t, mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))

def box(x0, x1, y0, y1, z0, z1): return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

def seam_contour(meshes, zs, side):
    """outermost |x| of the lip/lugs at the seam (|y| <= LIP_T) for each z, on side +1 (x>0) or -1"""
    out = []
    for z in zs:
        best = 0.0
        for m in meshes:
            s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if s is None: continue
            for q in s.to_2D(to_2D=np.eye(4))[0].polygons_full:
                c = q.intersection(sbox(0, -B.LIP_T, 300, B.LIP_T) if side > 0 else sbox(-300, -B.LIP_T, 0, B.LIP_T))
                if not c.is_empty: best = max(best, c.bounds[2] if side > 0 else -c.bounds[0])
        out.append(best)
    return np.array(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--height', type=int, default=80, choices=sorted(B.SIZES))
    h = ap.parse_args().height
    B.configure(h)
    tA, A = load(os.path.join(B.OUT, 'mold_shell_A.stl'))
    tB, Bm = load(os.path.join(B.OUT, 'mold_shell_B.stl'))
    Z_RIM = tA.bounds[1][2]
    z_mid_top = B.BOLT_Z[1] + B.LUG_H                       # top of the middle lugs
    dz = 0.25
    zs = np.arange(dz/2, Z_RIM, dz)
    half = GAP/2
    ymax = half + B.LIP_T - 0.5                              # tie walls stay inside the lip thickness
    parts = [A.translate([0, half, 0]), Bm.translate([0, -half, 0])]
    log = []
    for side in (1, -1):
        c = seam_contour([tA, tB], zs, side)
        segs = [(0.0, z_mid_top), (z_mid_top, Z_RIM)]
        for k, (z0, z1) in enumerate(segs):
            sel = (zs >= z0) & (zs <= z1)
            xw = c[sel].max() + TIE_GAP                       # vertical tie wall just outside the contour
            wall = box(xw, xw + TIE_T, -ymax, ymax, z0, z1)
            if k == 1:   # upper wall stands on the middle lugs: keep its foot 1 mm away from the mating faces
                wall = wall - box(xw - 1, xw + TIE_T + 1, -(half + 1.0), half + 1.0, z0 - 1, z0 + 0.6)
            if side < 0: wall = wall.mirror([1, 0, 0])
            parts.append(wall)
            # dots: short bridges from the part's end face to the wall
            last = -1e9; nd = 0; dz_list = []
            cand = [(z, x) for z, x in zip(zs[sel], c[sel]) if z0 + 1.5 < z < z1 - DOT_H - 1.0]
            for z, _ in cand:
                if z - last < DOT_STEP: continue
                w = (zs >= z) & (zs <= z + DOT_H)
                cmin, cmax = c[w].min(), c[w].max()
                if xw - cmin > DOT_MAX_GAP or cmax - cmin > 0.6: continue
                for sy in (1, -1):
                    y0, y1 = half + DOT_Y0, half + DOT_Y0 + DOT_W
                    d = box(cmin - 0.3, xw + 0.2, min(sy*y0, sy*y1), max(sy*y0, sy*y1), z, z + DOT_H)
                    parts.append(d.mirror([1, 0, 0]) if side < 0 else d)
                last = z; nd += 1; dz_list.append(round(float(z), 1))
            log.append(f"side {'+' if side > 0 else '-'}x, z {z0:.0f}..{z1:.0f}: tie wall at |x|={xw:.2f}, {nd} dot pairs at z={dz_list}")
    joined = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    o = joined.to_mesh(); V = np.asarray(o.vert_properties)[:, :3].astype(np.float64); F = np.asarray(o.tri_verts)
    chk = trimesh.Trimesh(V.astype(np.float32).astype(np.float64), F, process=True)
    assert chk.is_watertight and chk.is_winding_consistent, 'joined mesh not watertight'
    halves = parts[0] + parts[1]
    extra = joined.volume() - halves.volume()
    assert abs((joined ^ halves).volume() - halves.volume()) < 1e-3, 'halves changed'
    out = os.path.join(B.OUT, 'mold_shells_AB_joined.stl')
    trimesh.Trimesh(V, F, process=False).export(out)
    for l in log: print(l)
    print(f"{out}: {len(F)} tris, size {np.round(chk.extents, 1).tolist()} mm, components {len(chk.split(only_watertight=False))}, "
          f"ties {extra:.2f} mm3 (halves untouched)")

if __name__ == '__main__':
    main()
