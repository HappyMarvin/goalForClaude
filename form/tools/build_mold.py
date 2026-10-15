"""
Builds all printable files from form/forForm.zip (forForm.blend):
  print/pumpkin_for_mold.stl      - master for the silicone mold (80 mm tall, mounting holes in the bottom)
  print/pumpkin_no_holes.stl      - the same figure without mounting holes
  print/mold_shell_A.stl          - shell half A (with ridge on the flange faces)
  print/mold_shell_B.stl          - shell half B (with groove)
  print/mold_base.stl             - base plate: plinth for the pumpkin, centering boss + pin, ears for the bottom bolts

Requires: bpy (Blender as a python module), numpy, scipy, trimesh, manifold3d, pymeshlab, shapely
Run:  python3 build_mold.py            (from any directory)
"""
import os, sys, zipfile, tempfile, json
import numpy as np
import trimesh, manifold3d as mf, pymeshlab
from scipy.ndimage import gaussian_filter1d
from shapely.geometry.polygon import orient

HERE = os.path.dirname(os.path.abspath(__file__))
FORM = os.path.dirname(HERE)
OUT = os.path.join(FORM, 'print')
CACHE = os.path.join(tempfile.gettempdir(), 'pumpkin_mold_cache')
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

# ---------------------------------------------------------------- parameters (mm)
HEIGHT = 80.0               # pumpkin height
TARGET_TRIS = 700_000       # mesh reduction target (max deviation measured ~0.005 mm)
CX, CY = 0.5, -0.5          # mold axis in the leveled pumpkin frame (minimises silicone)
T_SIL = 10.0                # minimum silicone thickness
TOP_SIL = 11.0              # silicone above the pumpkin top (= rim / fill level)
H_PLAT = 2.0                # plinth under the pumpkin (gives the mold opening a 2 mm wall instead of a feather edge)
BOSS_D, BOSS_H = 14.0, 3.0  # centering boss on the plinth
PIN_D, PIN_X = 6.0, 12.0    # anti-rotation pin, toward the face (+X in the assembly frame)
REC_CLR, REC_DEPTH = 0.25, 3.4
PILOT_D, PILOT_TOP = 3.4, 30.0   # pilot hole for the M4 bolt that holds the pumpkin on the base
WALL = 3.0                  # shell wall
FL_T, FL_X = 5.0, 68.0      # flange thickness per half, flange outer edge
FIN_T, FIN_Y = 4.0, 64.0    # middle fins (stiffness + the shell stands on them upside down)
BOLT_D, BOLT_X = 4.5, 60.0  # M4 clearance holes
BOLT_Z = [8.0, 50.0, 86.0]  # the lowest one also goes through the base ears
RIDGE_IN, RIDGE_OUT, RIDGE_H = 4.0, 6.0, 1.5
GROOVE_CLR, GROOVE_D = 0.3, 1.9
BASE_T, BASE_X, BASE_Y, BASE_R = 6.0, 70.0, 66.0, 8.0
EAR_T, EAR_GAP, EAR_X0, EAR_X1, EAR_H = 4.0, 0.2, 55.5, 67.0, 15.0
CB_D, CB_DEPTH = 10.0, 4.5  # counterbore for the bolt head under the base
SEG = 240

def log(*a): print(*a, flush=True)

# ---------------------------------------------------------------- 1. read the blend file
def load_blend():
    cache = os.path.join(CACHE, 'orig.npz')
    if os.path.exists(cache):
        d = np.load(cache); return d['V'], d['F']
    import bpy
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(os.path.join(FORM, 'forForm.zip')) as z: z.extract('forForm.blend', tmp)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(tmp, 'forForm.blend'))
    obs = [o for o in bpy.data.objects if o.type == 'MESH']
    assert len(obs) == 1, obs
    o = obs[0]
    me = o.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    me.calc_loop_triangles()
    V = np.zeros(len(me.vertices)*3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
    F = np.zeros(len(me.loop_triangles)*3, dtype=np.int32); me.loop_triangles.foreach_get('vertices', F); F = F.reshape(-1, 3)
    mw = np.array(o.matrix_world); V = V @ mw[:3, :3].T + mw[:3, 3]
    np.savez(cache, V=V, F=F)
    return V, F

# ---------------------------------------------------------------- 2. level the (0.44 deg tilted) flat bottom, scale to 80 mm
def level_and_scale(V0, F):
    m0 = trimesh.Trimesh(V0, F, process=False)
    n, c, a = m0.face_normals, m0.triangles_center, m0.area_faces
    sel = (n[:, 2] < -0.995) & (c[:, 2] < V0[:, 2].min() + 0.07)
    P, w = c[sel], np.sqrt(a[sel])
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    coef, *_ = np.linalg.lstsq(A*w[:, None], P[:, 2]*w, rcond=None)
    nb = np.array([coef[0], coef[1], -1.0]); nb /= np.linalg.norm(nb)
    t = np.array([0, 0, -1.0]); v = np.cross(nb, t); s = np.linalg.norm(v); ct = nb @ t
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - ct)/s**2)
    V1 = V0 @ R.T
    z0 = V1[:, 2].min(); sc = HEIGHT/(V1[:, 2].max() - z0)
    V1 = (V1 - [0, 0, z0])*sc
    log(f"bottom tilt corrected: {np.degrees(np.arccos(ct)):.3f} deg, scale {sc:.4f} mm/unit")
    return V1, dict(R=R, z0=z0, s=sc)

# ---------------------------------------------------------------- 3. reduce triangle count (quadric collapse), check deviation
def decimate(V, F):
    cache = os.path.join(CACHE, f'dec_{TARGET_TRIS}.npz')
    if os.path.exists(cache):
        d = np.load(cache); return d['V'], d['F']
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F.astype(np.int32)), 'orig')
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F.astype(np.int32)), 'dec')
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=TARGET_TRIS, qualitythr=0.5, preserveboundary=True,
        preservenormal=True, preservetopology=True, optimalplacement=True, planarquadric=True, autoclean=True)
    m = ms.current_mesh(); Vd, Fd = m.vertex_matrix(), m.face_matrix()
    h1 = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, samplenum=2_000_000, sampleface=True, samplevert=True)
    h2 = ms.get_hausdorff_distance(sampledmesh=1, targetmesh=0, samplenum=2_000_000, sampleface=True, samplevert=True)
    log(f"decimated {len(F)} -> {len(Fd)} tris; deviation max {max(h1['max'], h2['max']):.4f} mm, mean {h1['mean']:.5f} mm")
    np.savez(cache, V=Vd, F=Fd)
    return Vd, Fd

def to_manifold(V, F): return mf.Manifold(mf.Mesh(vert_properties=np.asarray(V, np.float32), tri_verts=np.asarray(F, np.uint32)))
def from_manifold(m):
    o = m.to_mesh(); return np.asarray(o.vert_properties)[:, :3].astype(np.float64), np.asarray(o.tri_verts)
def cyl(d, h, z0=0.0, x=0.0, y=0.0, d_top=None):
    return mf.Manifold.cylinder(h, d/2, (d_top if d_top else d)/2, 128).translate([x, y, z0])
def box(x0, x1, y0, y1, z0, z1): return mf.Manifold.cube([x1-x0, y1-y0, z1-z0]).translate([x0, y0, z0])

def save_stl(V, F, name):
    t = trimesh.Trimesh(V, F, process=False)
    chk = trimesh.Trimesh(V.astype(np.float32).astype(np.float64), F, process=True)   # as a slicer sees the STL
    assert chk.is_watertight and chk.is_winding_consistent and (chk.area_faces > 1e-10).all(), name
    t.export(os.path.join(OUT, name))
    log(f"  {name}: {len(F)} tris, size {np.round(t.extents, 1).tolist()} mm, volume {t.volume/1000:.1f} cm3")

# ---------------------------------------------------------------- 4. pumpkin master (mounting recesses in the flat bottom)
def make_pumpkin(Vd, Fd):
    V = Vd - [CX, CY, 0]
    P = to_manifold(V, Fd)
    e = 0.5
    cut = cyl(BOSS_D + 2*REC_CLR, REC_DEPTH + e, -e)
    cut += cyl(BOSS_D + 2*REC_CLR + 1.0 + 2*e, 0.5 + e, -e, d_top=BOSS_D + 2*REC_CLR)
    cut += cyl(PIN_D + 2*REC_CLR, REC_DEPTH + e, -e, x=PIN_X)
    cut += cyl(PIN_D + 2*REC_CLR + 1.0 + 2*e, 0.5 + e, -e, x=PIN_X, d_top=PIN_D + 2*REC_CLR)
    cut += cyl(PILOT_D, PILOT_TOP + e, -e)
    Vm, Fm = from_manifold(P - cut)
    return V, Fd, Vm, Fm

# ---------------------------------------------------------------- 5. shell profile: revolution surface >= T_SIL from every point of the pumpkin
def shell_profile(PVa, Z_RIM):
    dz = 0.05
    zb = np.arange(0, Z_RIM + dz/2, dz)
    rho = np.hypot(PVa[:, 0], PVa[:, 1])
    bi = np.clip(np.round(PVa[:, 2]/dz).astype(int), 0, len(zb)-1)
    rmax = np.full(len(zb), -np.inf); np.maximum.at(rmax, bi, rho)
    k = int(round(T_SIL/dz)); offs = np.arange(-k, k+1)
    add = np.sqrt(np.maximum(T_SIL**2 - (offs*dz)**2, 0))
    req = np.zeros(len(zb))
    for i in range(len(zb)):
        j = i - offs; ok = (j >= 0) & (j < len(zb)); v = rmax[j[ok]] + add[ok]; v = v[np.isfinite(v)]
        req[i] = max(0.0, v.max()) if len(v) else 0.0
    def lean_limit(r):     # <= 45 deg lean both ways: shell halves print upright without supports
        r = r.copy()
        for _ in range(2):
            for i in range(1, len(r)): r[i] = max(r[i], r[i-1] - dz)
            for i in range(len(r)-2, -1, -1): r[i] = max(r[i], r[i+1] - dz)
        return r
    def closing(f, R):     # morphological closing with a disk: fills the small dips between lobe arcs
        kk = int(round(R/dz)); t = np.arange(-kk, kk+1); c = np.sqrt(np.maximum(R**2 - (t*dz)**2, 0))
        def pad(g):
            m = int(round(1.0/dz)); s0 = (g[m]-g[0])/m; s1 = (g[-1]-g[-1-m])/m
            return np.concatenate([g[0] - s0*np.arange(kk, 0, -1), g, g[-1] + s1*np.arange(1, kk+1)])
        fp = pad(f); dil = np.array([np.max(fp[i:i+2*kk+1] + c) for i in range(len(f))])
        dp = pad(dil); return np.array([np.min(dp[i:i+2*kk+1] - c) for i in range(len(f))])
    r = lean_limit(req)
    r = np.maximum(r, closing(r, 20.0))
    r = lean_limit(np.maximum(req, gaussian_filter1d(r, 0.6/dz, mode='nearest')))
    r = np.maximum(r, req)
    return zb, r, req

def teardrop_y(x, z, dia, y0=-80.0, y1=80.0):     # horizontal hole along Y, pointed top (prints without support)
    rr = dia/2; q = rr*np.sqrt(0.5)
    cs = (mf.CrossSection.circle(rr, 64) + mf.CrossSection([np.array([[-q, q], [q, q], [0, rr*np.sqrt(2)]])])).hull()
    return mf.Manifold.extrude(cs, y1 - y0).rotate([90, 0, 0]).translate([x, y1, z])

def make_mold(PV, PF, PVm, PFm):
    Z_P = H_PLAT; Z_RIM = Z_P + HEIGHT + TOP_SIL
    PVa = PV + [0, 0, Z_P]
    zb, r_in, req = shell_profile(PVa, Z_RIM)
    zz = np.append(zb[::10], Z_RIM) if zb[::10][-1] < Z_RIM else zb[::10]
    rr = np.interp(zz, zb, r_in)
    rev = lambda poly: mf.Manifold.revolve(mf.CrossSection([np.asarray(poly, float)]), SEG)
    cav_poly = [(0, zz[0])] + list(zip(rr, zz)) + [(0, zz[-1])]
    cavity = rev([(0, -2.0), (rr[0], -2.0)] + list(zip(rr, zz)) + [(rr[-1], Z_RIM + 2.0), (0, Z_RIM + 2.0)])
    cs_cav = mf.CrossSection([np.asarray(cav_poly, float)])
    # wall band reaches 0.5 mm into the cavity so the 3D cavity cut is transversal (no coincident surfaces)
    cs_wall = (cs_cav.offset(WALL, mf.JoinType.Round, 2.0, 64) - cs_cav.offset(-0.5, mf.JoinType.Round, 2.0, 64)) ^ mf.CrossSection.square([300, Z_RIM + 2]).translate([0, -1])
    wall = mf.Manifold.revolve(cs_wall, SEG)
    shell = (wall + box(-FL_X, FL_X, -FL_T, FL_T, -1, Z_RIM + 1) + box(-FIN_T/2, FIN_T/2, -FIN_Y, FIN_Y, -1, Z_RIM + 1)) - cavity
    holes = None
    for sx in (-1, 1):
        for bz in BOLT_Z:
            h = teardrop_y(sx*BOLT_X, bz, BOLT_D); holes = h if holes is None else holes + h
    notch = None
    for sx in (-1, 1):   # V-notch at the outer flange edge along the seam (to pry the halves apart)
        n_ = mf.Manifold.cube([3.0, 3.0, Z_RIM + 2], True).rotate([0, 0, 45]).translate([sx*FL_X, 0, Z_RIM/2])
        notch = n_ if notch is None else notch + n_
    shell = shell - holes - notch
    def band(off0, off1, y0, y1, z0=-1.0, z1=Z_RIM + 1.0):
        out = None
        for sx in (-1, 1):
            a = [(sx*(np.interp(z, zb, r_in) + off0), z) for z in zz]
            b = [(sx*(np.interp(z, zb, r_in) + off1), z) for z in zz[::-1]]
            cs = mf.CrossSection([np.array(a + b, float)], mf.FillRule.NonZero) ^ mf.CrossSection.square([400, z1 - z0]).translate([-200, z0])
            m = mf.Manifold.extrude(cs, y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])
            out = m if out is None else out + m
        return out
    ridge = band(RIDGE_IN, RIDGE_OUT, -RIDGE_H, 0.5)
    groove = band(RIDGE_IN - GROOVE_CLR, RIDGE_OUT + GROOVE_CLR, -GROOVE_D, 0.5, -2.0, Z_RIM + 2.0)
    trim = box(-200, 200, -200, 200, 0, Z_RIM)
    halfA = (((shell ^ box(-200, 200, 0, 200, -10, 200)) + ridge - holes) ^ trim).simplify(1e-4)
    halfB = (((shell ^ box(-200, 200, -200, 0, -10, 200)) - groove) ^ trim).simplify(1e-4)
    # base
    plate_cs = mf.CrossSection.square([2*(BASE_X-BASE_R), 2*(BASE_Y-BASE_R)]).translate([-(BASE_X-BASE_R), -(BASE_Y-BASE_R)]).offset(BASE_R, mf.JoinType.Round, 2.0, 96)
    plate = mf.Manifold.extrude(plate_cs, BASE_T).translate([0, 0, -BASE_T])
    sec = trimesh.Trimesh(PV, PF, process=False).section(plane_origin=[0, 0, 0.1], plane_normal=[0, 0, 1]).to_2D(to_2D=np.eye(4))[0]
    foot = max(sec.polygons_full, key=lambda p: p.area)
    foot_in = orient(foot.buffer(-0.3, join_style=1).simplify(0.01), 1.0)
    platform = mf.Manifold.extrude(mf.CrossSection([np.asarray(foot_in.exterior.coords[:-1], float)]), H_PLAT + 0.5).translate([0, 0, -0.5])
    ch = 0.6
    boss = cyl(BOSS_D, BOSS_H - ch + 0.1, H_PLAT - 0.1) + cyl(BOSS_D, ch, H_PLAT + BOSS_H - ch, d_top=BOSS_D - 2*ch)
    pin = cyl(PIN_D, BOSS_H - ch + 0.1, H_PLAT - 0.1, x=PIN_X) + cyl(PIN_D, ch, H_PLAT + BOSS_H - ch, x=PIN_X, d_top=PIN_D - 2*ch)
    ears = None
    for sx in (-1, 1):
        for sy in (-1, 1):
            y0 = FL_T + EAR_GAP
            e = box(min(sx*EAR_X0, sx*EAR_X1), max(sx*EAR_X0, sx*EAR_X1), min(sy*y0, sy*(y0+EAR_T)), max(sy*y0, sy*(y0+EAR_T)), -0.5, EAR_H)
            ears = e if ears is None else ears + e
    base = plate + platform + boss + pin + ears
    base = base - cyl(BOLT_D, 40, -20) - cyl(CB_D, CB_DEPTH + 0.01, -BASE_T - 0.01)
    for sx in (-1, 1): base = base - teardrop_y(sx*BOLT_X, BOLT_Z[0], BOLT_D)
    # ---- checks
    P_clean = to_manifold(PVa, PF)
    P_master = to_manifold(PVm + [0, 0, Z_P], PFm)
    plinth_solid = mf.Manifold.extrude(mf.CrossSection([np.asarray(foot_in.exterior.coords[:-1], float)]), H_PLAT + 0.7).translate([0, 0, -0.5])
    sil = (cavity ^ box(-200, 200, -200, 200, 0, Z_RIM)) - P_clean - plinth_solid   # (printed bottom is flat: no film under it)
    log(f"shell: rim (= silicone fill level) at z={Z_RIM:.1f} mm, top opening D={2*rr[-1]:.1f} mm, max inner D={2*r_in.max():.1f} mm")
    log(f"silicone volume {sil.volume()/1000:.1f} cm3 (~{sil.volume()/1000*1.1:.0f} g at 1.1 g/cm3)")
    for n, a, b in [('master/shellA', P_master, halfA), ('master/shellB', P_master, halfB), ('master/base', P_master, base),
                    ('A/B', halfA, halfB), ('A/base', halfA, base), ('B/base', halfB, base)]:
        v = (a ^ b).volume(); log(f"  interference {n}: {v:.4f} mm3"); assert v < 1e-3
    rho = np.hypot(PVa[:, 0], PVa[:, 1]); z = PVa[:, 2]; best = np.full(len(PVa), np.inf)
    for zq, rq in zip(zb[::2], r_in[::2]):
        s = np.abs(z - zq) < 16; best[s] = np.minimum(best[s], np.hypot(rq - rho[s], zq - z[s]))
    best = np.minimum(best, Z_RIM - z)
    log("silicone thickness over the pumpkin: min %.2f, 5%% %.2f, median %.2f mm" % tuple(np.percentile(best, [0, 5, 50])))
    json.dump(dict(Z_RIM=Z_RIM, Z_P=Z_P, top_opening_d=2*rr[-1], silicone_cm3=sil.volume()/1000), open(os.path.join(CACHE, 'mold.json'), 'w'))
    return dict(halfA=halfA, halfB=halfB, base=base, silicone=sil)

def main():
    V0, F0 = load_blend()
    log(f"source mesh: {len(V0)} verts, {len(F0)} tris")
    V1, tr = level_and_scale(V0, F0)
    Vd, Fd = decimate(V1, F0)
    PV, PF, PVm, PFm = make_pumpkin(Vd, Fd)
    # print orientation: face (+X in the assembly frame) turned to the front of the bed (-Y), so "seam: back" hides the seam
    rot = lambda V: np.c_[V[:, 1], -V[:, 0], V[:, 2]]
    log("writing STL files:")
    save_stl(rot(PVm), PFm, 'pumpkin_for_mold.stl')
    save_stl(rot(PV), PF, 'pumpkin_no_holes.stl')
    parts = make_mold(PV, PF, PVm, PFm)
    for name, key in [('mold_shell_A.stl', 'halfA'), ('mold_shell_B.stl', 'halfB'), ('mold_base.stl', 'base')]:
        V, F = from_manifold(parts[key]); V = V - [0, 0, V[:, 2].min()]
        save_stl(V, F, name)
    # assembly-frame meshes for previews
    for key in ['halfA', 'halfB', 'base', 'silicone']:
        V, F = from_manifold(parts[key]); np.savez(os.path.join(CACHE, f'asm_{key}.npz'), V=V, F=F)
    np.savez(os.path.join(CACHE, 'asm_pumpkin.npz'), V=PVm + [0, 0, H_PLAT], F=PFm)
    np.savez(os.path.join(CACHE, 'level.npz'), **tr)

if __name__ == '__main__':
    main()
