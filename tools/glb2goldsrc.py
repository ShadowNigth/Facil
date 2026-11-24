#!/usr/bin/env python3
"""
glb2goldsrc.py -- convert glTF 2.0 (.glb) Counter-Strike weapon models into
GoldSrc / Counter-Strike 1.6 studio model sources.

Per model it writes, into --workdir:
    smd/<name>_ref.smd            bind pose: mesh + full skeleton
    smd/<name>_<animation>.smd    one SMD per glTF animation (inspections included)
    smd/<Material>.bmp            8-bit paletted texture per material
    qc/<name>.qc                  $body / $sequence list ready for studiomdl

GoldSrc constraints handled here
--------------------------------
* 2048 vertices and 20000 triangles per sub-model (engine allocation sizes), so the
  source mesh is cluster decimated (vertices only merge inside one bone).
* No linear blend skinning: every triangle is bound 100% to its dominant joint.
* Animation channels are stored as offsets from the bone's rest position, with a
  per-bone scale derived from the largest offset.  The node hierarchy of these CS
  rips is rigid (local offsets never change), so every bone keeps its rest offset
  and animation is pure rotation, except the moving root which also translates.
* Angles are GoldSrc AngleQuaternion() angles (R = Rz*Ry*Rx, radians).
* glTF (Y up, -Z forward) -> GoldSrc (X forward, Z up, Y left) is a +90 deg yaw,
  applied to vertices and bone frames as (x, y, z) -> (-y, x, z).
"""
from __future__ import annotations

import io
import json
import math
import os
import struct

import numpy as np
from PIL import Image

MAX_VERTS = 2048
MAX_TRIS = 20000
MAX_BONES = 128
SAFE_VERTS = 2020
SAFE_TRIS = 19600
MAX_FRAMES = 480

CT = {5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2),
      5123: ('H', 2), 5125: ('I', 4), 5126: ('f', 4)}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


# --------------------------------------------------------------------------- glTF
class GLB:
    def __init__(self, path):
        self.path = path
        data = open(path, 'rb').read()
        if data[:4] != b'glTF':
            raise ValueError('not a GLB file: %s' % path)
        off, self.j, self.bin = 12, None, b''
        while off < len(data):
            clen, ctype = struct.unpack('<II', data[off:off + 8])
            chunk = data[off + 8:off + 8 + clen]
            if ctype == 0x4E4F534A:
                self.j = json.loads(chunk)
            elif ctype == 0x004E4942:
                self.bin = chunk
            off += 8 + clen
        self.nodes = self.j.get('nodes', [])
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get('children', []):
                self.parent[c] = i

    def acc(self, idx):
        a = self.j['accessors'][idx]
        bv = self.j['bufferViews'][a['bufferView']]
        fmt, sz = CT[a['componentType']]
        n = NC[a['type']]
        start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = bv.get('byteStride') or sz * n
        cnt = a['count']
        if stride == sz * n:
            return np.frombuffer(self.bin, dtype=np.dtype(fmt), count=cnt * n,
                                 offset=start).reshape(cnt, n)
        out = np.empty((cnt, n), dtype=np.dtype(fmt))
        for k in range(cnt):
            out[k] = np.frombuffer(self.bin, dtype=np.dtype(fmt), count=n,
                                   offset=start + k * stride)
        return out

    def image_bytes(self, idx):
        im = self.j['images'][idx]
        bv = self.j['bufferViews'][im['bufferView']]
        o = bv.get('byteOffset', 0)
        return self.bin[o:o + bv['byteLength']]


# --------------------------------------------------------------------------- math
def quat_to_mat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                         (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                     0.25 * s, (m[1, 0] - m[0, 1]) / s])


_ANGLE_CACHE = {}


def _wrap(x):
    """Wrap an angle into (-pi, pi]."""
    return (x + math.pi) % (2 * math.pi) - math.pi


def angle_candidates(m):
    """All Euler triples (Rz*Ry*Rx convention) that produce matrix ``m``.

    The mapping angles -> matrix is many-to-one: besides adding 2*pi to single
    components there is a second family ``(pi-x, pi-y, z+pi ...)``.  GoldSrc adds
    animation deltas to the stored angles component-wise, so a wrong branch would
    produce a completely different pose even when it is the "same" matrix.
    """
    target = np.asarray(m, float)
    x, y, z, w = mat_to_quat(target)
    sy = max(-1.0, min(1.0, 2 * (w * y + x * z)))
    ax = math.asin(sy)
    ay = math.atan2(-2 * (w * z - x * y), 1 - 2 * (x * x + y * y))
    az = math.atan2(-2 * (w * x - y * z), 1 - 2 * (y * y + z * z))
    out = [np.array([ax, ay, az])]
    cy = math.hypot(target[2, 1], target[2, 2])
    if cy > 1e-9:
        for sgn in (1.0, -1.0):
            yy = math.atan2(-sgn * target[2, 0], sgn * cy)
            xx = math.atan2(sgn * target[2, 1], sgn * target[2, 2])
            m0 = angle_mat(np.array([xx, yy, 0.0]))
            zz = math.atan2(target[1, 0], target[0, 0]) - \
                math.atan2(m0[1, 0], m0[0, 0])
            for dz in (0.0, math.pi, -math.pi):
                out.append(np.array([xx, yy, zz + dz]))
    good = [c for c in out if np.abs(angle_mat(c) - target).max() < 1e-6]
    return good or out


def mat_to_angles(m, hint=None):
    """Angles (radians) for GoldSrc's AngleQuaternion() R(t) = Rz*Ry*Rx.

    With ``hint`` the branch closest to that triple is returned, which keeps the
    engine's additive ``value + delta`` within one continuous representation.
    """
    target = np.asarray(m, float)
    cands = angle_candidates(target)
    if hint is None:
        best = cands[0]
    else:
        hint = np.asarray(hint, float)
        # shift every candidate by whole turns so it sits next to the hint
        best, bd = cands[0], None
        for c in cands:
            c = hint + np.array([_wrap(c[k] - hint[k]) for k in range(3)])
            d = float(np.abs(c - hint).max())
            if bd is None or d < bd:
                best, bd = c, d
        if len(_ANGLE_CACHE) < 400000:
            _ANGLE_CACHE[target.tobytes()] = best
        return best.copy()
    if len(_ANGLE_CACHE) < 400000:
        _ANGLE_CACHE[target.tobytes()] = best
    return best.copy()


def angle_mat(a):
    """GoldSrc AngleQuaternion(): R = Rz*Ry*Rx."""
    ax, ay, az = a
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx]])


def project_rotation(m):
    """Closest proper rotation (SVD polar decomposition); mirrored frames coming from
    the source rigs collapse to their nearest rotation, which keeps GoldSrc happy and
    animation continuous."""
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def slerp(qa, qb, t):
    qa = np.asarray(qa, float)
    qb = np.asarray(qb, float)
    d = float(np.dot(qa, qb))
    if d < 0:
        qb, d = -qb, -d
    if d > 0.9995:
        q = qa + t * (qb - qa)
        return q / (np.linalg.norm(q) or 1.0)
    th0 = math.acos(max(-1.0, min(1.0, d)))
    q2 = qb - qa * d
    n = np.linalg.norm(q2)
    if n < 1e-12:
        return qa
    return qa * math.cos(th0 * t) + (q2 / n) * math.sin(th0 * t)


def wrap_pi(a):
    a = np.asarray(a, float).copy()
    a[a >= math.pi] = a[a >= math.pi] % (2 * math.pi) - 2 * math.pi
    a[a < -math.pi] = a[a < -math.pi] % (2 * math.pi)
    return a


# glTF -> GoldSrc basis: +90 deg yaw, (x, y, z) -> (-y, x, z); det = +1 so triangle
# winding is untouched.
R_GOLD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R_GOLD_T = R_GOLD.T


def to_gold(v):
    return np.asarray(v, float) @ R_GOLD_T


def gold_rot(m):
    return R_GOLD @ np.asarray(m, float) @ R_GOLD_T


# --------------------------------------------------------------------------- nodes
def decompose(n):
    if 'matrix' in n:
        m = np.array(n['matrix'], float).reshape(4, 4).T
        T = m[:3, 3].copy()
        R = m[:3, :3].copy()
        s = np.linalg.norm(R, axis=0)
        s[s == 0] = 1.0
        if np.linalg.det(R) < 0:      # mirrored matrix: keep it in the rotation
            R[:, 0] *= -1
            s[0] *= -1
        return T, R / np.abs(s), s
    T = np.array(n.get('translation', [0, 0, 0]), float)
    R = quat_to_mat(np.array(n.get('rotation', [0, 0, 0, 1]), float))
    S = np.array(n.get('scale', [1, 1, 1]), float)
    return T, R, S


def local_matrix(n):
    T, R, S = decompose(n)
    m = np.eye(4)
    m[:3, :3] = R * S
    m[:3, 3] = T
    return m


class Rig:
    """Flattened node hierarchy used for skinning and animation.

    node[i]   glTF node index of rig node i
    parent[i] rig index of its parent (-1 for a root)
    """

    def __init__(self, g):
        self.g = g
        joints = list(g.j['skins'][0]['joints']) if g.j.get('skins') else []
        animated = set()
        for a in g.j.get('animations', []):
            for ch in a['channels']:
                animated.add(ch['target']['node'])
        seeds = set(joints) | animated
        if not seeds:
            seeds = {0}
        keep = set()
        for s in seeds:
            n = s
            while n is not None and n not in keep:
                keep.add(n)
                n = g.parent.get(n)
        self.keep = keep
        self.joints = joints
        self.node, self.parent, self.ix = [], [], {}

        def add(ni, par):
            here = len(self.node)
            self.ix[ni] = here
            self.node.append(ni)
            self.parent.append(par)
            for c in g.nodes[ni].get('children', []):
                if c in keep:
                    add(c, here)

        roots = [i for i in range(len(g.nodes))
                 if i in keep and g.parent.get(i) is None]
        # a child whose parent frame is missing from the kept set becomes a root
        for i in range(len(g.nodes)):
            if i in keep:
                par = g.parent.get(i)
                if par is not None and par not in keep:
                    roots.append(i)
        for r in roots:
            add(r, -1)
        assert len(self.node) == len(set(self.node)), 'duplicated rig node'
        # nodes whose parent frame was filtered out must start a new chain
        for i, ni in enumerate(self.node):
            if self.parent[i] < 0 and g.parent.get(ni) is not None:
                # climb to the closest ancestor that survived
                anc = g.parent.get(ni)
                while anc is not None and anc not in self.ix:
                    anc = g.parent.get(anc)
                self.parent[i] = self.ix[anc] if anc is not None else -1
        self.world = []
        for i, ni in enumerate(self.node):
            m = local_matrix(g.nodes[ni])
            self.world.append((self.world[self.parent[i]] @ m)
                              if self.parent[i] >= 0 else m)
        self.joint_of_node, self.node_of_joint = {}, {}
        for j, ni in enumerate(joints):
            if ni in self.ix:
                self.joint_of_node[self.ix[ni]] = j
                self.node_of_joint[j] = self.ix[ni]
        # inverse bind matrices (glTF stores them column-major)
        self.ibm = {}
        if g.j.get('skins') and 'inverseBindMatrices' in g.j['skins'][0]:
            raw = g.acc(g.j['skins'][0]['inverseBindMatrices']).reshape(-1, 4, 4)
            for j, ni in enumerate(joints):
                if j < len(raw) and ni in self.ix:
                    self.ibm[self.ix[ni]] = np.transpose(raw[j])

    def world_fix(self, rig_index, world=None):
        """(world . IBM)^-1 . world : maps the stored skin-space geometry into world."""
        W = self.world[rig_index] if world is None else world
        ibm = self.ibm.get(rig_index)
        if ibm is None:
            return np.eye(4)
        return np.linalg.inv(W @ ibm) @ W

    def skin_matrix(self, rig_index, world=None):
        """``world[joint] . inverseBindMatrix``: the skinning matrix of a joint.

        The mesh of these CS ports stores its vertices in skin space, so a bone's
        actual geometry transform is the joint world matrix composed with its
        inverse bind matrix -- using the joint matrix alone scrambles the model.
        """
        W = self.world[rig_index] if world is None else world
        ibm = self.ibm.get(rig_index)
        return W if ibm is None else W @ ibm

    def name(self, i):
        return self.g.nodes[self.node[i]].get('name') or ('node%d' % self.node[i])


# --------------------------------------------------------------------------- mesh
def collect_primitives(g, rig):
    owner = {}
    for ni, nd in enumerate(g.nodes):
        if 'mesh' in nd:
            owner.setdefault(nd['mesh'], ni)
    skin = bool(g.j.get('skins'))
    joints = list(g.j['skins'][0]['joints']) if skin else []
    njoints = len(joints)
    prims = []
    for mi, mesh in enumerate(g.j['meshes']):
        M = np.eye(4)
        # glTF spec: the transform of a node holding a *skinned* mesh is ignored;
        # the geometry is already expressed in skin space.  Honouring it here is
        # what pushed the pistol away from the hands.
        if owner.get(mi) is not None and not skin:
            M = local_matrix(g.nodes[owner[mi]])
        for pr in mesh['primitives']:
            at = pr['attributes']
            P = g.acc(at['POSITION']).astype(np.float64)
            UV = (g.acc(at['TEXCOORD_0']).astype(np.float64)
                  if 'TEXCOORD_0' in at else np.zeros((len(P), 2)))
            N = (g.acc(at['NORMAL']).astype(np.float64) if 'NORMAL' in at else None)
            idx = (g.acc(pr['indices']).reshape(-1).astype(np.int64)
                   if 'indices' in pr else np.arange(len(P), dtype=np.int64))
            if 'JOINTS_0' in at:
                JI = g.acc(at['JOINTS_0']).astype(np.int64)
                Wt = g.acc(at['WEIGHTS_0']).astype(np.float64)
                # JOINTS_0 holds indices into skin.joints, not node numbers
                JI = JI.copy()
                if njoints:
                    ok = JI < njoints
                    JI = np.where(ok, np.asarray(joints, np.int64)[np.clip(JI, 0, njoints - 1)],
                                  -1)
                    JI[~ok] = -1
                Wt = np.where(JI >= 0, Wt, 0.0)
            else:
                JI = np.full((len(P), 1), -1, np.int64)
                Wt = np.zeros((len(P), 1))
            if skin:
                # linear blend skinning with the *rest* pose: at rest every skin
                # matrix is ~identity, so this maps skin space -> world.  The
                # skeleton frames were pre-multiplied by ``scale``, but the inverse
                # bind matrices were not, so skinning must use the unscaled world
                # or ``W . IBM`` is no longer the identity.
                world = getattr(rig, 'world_unscaled', rig.world)
                Pw = np.zeros_like(P)
                Nw = np.zeros_like(P) if N is not None else None
                for k in range(JI.shape[1]):
                    for j in np.unique(JI[:, k]):
                        if j < 0:
                            continue
                        sel = JI[:, k] == j
                        node = rig.ix.get(int(j))
                        Sm = (rig.skin_matrix(node, world=world[node])
                              if node is not None else np.eye(4))
                        w = Wt[sel, k:k + 1]
                        Pw[sel] += w * (P[sel] @ Sm[:3, :3].T + Sm[:3, 3])
                        if N is not None:
                            Nw[sel] += w * (N[sel] @ Sm[:3, :3].T)
                if N is not None:
                    N = Nw
            else:
                R3 = M[:3, :3]
                Pw = (R3 @ P.T).T + M[:3, 3]
                Nw = None if N is None else (np.linalg.inv(R3).T @ N.T).T
            if abs(getattr(rig, 'scale', 1.0) - 1.0) > 1e-9:
                Pw = Pw * rig.scale
            prims.append(dict(
                P=Pw, N=Nw, UV=UV, idx=idx.reshape(-1, 3), J=JI, W=Wt,
                mat=int(pr.get('material', 0)), skinned=skin))
    return prims


def weld(prims, rig):
    q = 2000.0
    key, pos, bone, nrm, uvs = {}, [], [], [], []
    tris = []
    for pr in prims:
        P = pr['P']
        if pr['skinned']:
            # JOINTS_0 was resolved to glTF *node* numbers; bone_of_node is keyed by
            # rig index, so translate back through rig.ix.
            dom = pr['J'][np.arange(len(P)), np.argmax(pr['W'], axis=1)]
            dom_node = np.array([rig.ix.get(int(j), 0) if j >= 0 else 0 for j in dom])
        else:
            dom_node = np.zeros(len(P), np.int64)
        vmap = np.empty(len(P), np.int64)
        for i in range(len(P)):
            k = (int(round(P[i, 0] * q)), int(round(P[i, 1] * q)),
                 int(round(P[i, 2] * q)), int(dom_node[i]))
            vi = key.get(k)
            if vi is None:
                vi = key[k] = len(pos)
                pos.append(P[i])
                bone.append(int(dom_node[i]))
                nrm.append(np.zeros(3))
                uvs.append({})
            vmap[i] = vi
            if pr['N'] is not None:
                nrm[vi] += pr['N'][i]
            if pr['mat'] not in uvs[vi]:
                uvs[vi][pr['mat']] = (float(pr['UV'][i, 0]) % 1.0,
                                      float(pr['UV'][i, 1]) % 1.0)
        for t in vmap[pr['idx']]:
            if t[0] != t[1] and t[1] != t[2] and t[0] != t[2]:
                tris.append((int(t[0]), int(t[1]), int(t[2]), pr['mat']))
    N = np.array(nrm)
    ln = np.linalg.norm(N, axis=1)
    ln[ln == 0] = 1.0
    return (np.array(pos), np.array(bone), N / ln[:, None], uvs,
            np.array(tris, np.int64))


def decimate(verts, bones, nrm, uvs, tris, target_tris, target_verts=None):
    lo = verts.min(axis=0)
    diag = float(np.linalg.norm(verts.max(axis=0) - lo)) or 1.0
    for k in range(1, 400):
        step = 0.0035 * (1.09 ** k) * diag
        if step > 0.6 * diag:
            break
        cells = np.floor((verts - lo) / step).astype(np.int64)
        key, rep, remap = {}, [], np.empty(len(verts), np.int64)
        for i in range(len(verts)):
            kk = (cells[i, 0], cells[i, 1], cells[i, 2], int(bones[i]))
            r = key.get(kk)
            if r is None:
                key[kk] = r = len(key)
                rep.append(i)
            remap[i] = r
        t2 = remap[tris[:, :3]]
        ok = ((t2[:, 0] != t2[:, 1]) & (t2[:, 1] != t2[:, 2]) &
              (t2[:, 0] != t2[:, 2]))
        nverts = len(key)
        if ok.sum() <= target_tris and (target_verts is None
                                        or nverts <= target_verts):
            t2 = t2[ok]
            mats = tris[ok][:, 3]
            used = np.unique(t2.reshape(-1))
            inv = {int(u): i for i, u in enumerate(used)}
            t3 = np.vectorize(inv.get)(t2).astype(np.int64)
            rep = np.array(rep, np.int64)
            return (verts[rep[used]], bones[rep[used]], nrm[rep[used]],
                    [uvs[rep[u]] for u in used],
                    np.hstack([t3, mats[:, None]]))
    return verts, bones, nrm, uvs, tris


# --------------------------------------------------------------------------- anim
def bake_clip(g, anim, rig, fps, rest_frame=False, max_frames=None):
    """Sample one glTF animation into per-node local TRS for every frame.

    rest_frame=True returns a single "frame 0 is the bind pose" sample (used to check
    and undo constant offsets baked into the clips)."""
    tracks, tmax = {}, 0.0
    for ch in anim['channels']:
        s = anim['samplers'][ch['sampler']]
        t = g.acc(s['input']).reshape(-1).astype(float)
        v = g.acc(s['output']).astype(float)
        ni = rig.ix.get(ch['target']['node'])
        if ni is None:
            continue
        tracks.setdefault(ni, {})[ch['target']['path']] = (t, v)
        if len(t):
            tmax = max(tmax, float(t[-1]))
    if tmax <= 0:
        tmax = 1.0 / fps
    nframes = int(round(tmax * fps)) + 1
    fps_used = fps
    cap = MAX_FRAMES if max_frames is None else min(MAX_FRAMES, max_frames)
    if nframes > cap:
        nframes = cap
        fps_used = (nframes - 1) / tmax
    times = np.arange(nframes) / fps_used
    if rest_frame:
        times = times[:1]
        nframes = 1
    frames = []
    for f in range(nframes):
        pose = []
        for i, ni in enumerate(rig.node):
            n = g.nodes[ni]
            T, R, S = decompose(n)
            tr = tracks.get(i, {})
            if 'translation' in tr and not rest_frame:
                T = sample(tr['translation'], times[f], False)
            if 'rotation' in tr and not rest_frame:
                R = quat_to_mat(sample(tr['rotation'], times[f], True))
            if 'scale' in tr:
                S = sample(tr['scale'], times[f], False)
            pose.append((T, mat_to_quat(R), S))
        frames.append(pose)
    return frames, nframes, fps_used


def sample(track, t, is_quat):
    ts, vs = track
    if len(ts) == 1:
        return vs[0].astype(float)
    i = int(np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 2))
    frac = float(np.clip((t - ts[i]) / max(ts[i + 1] - ts[i], 1e-9), 0, 1))
    if is_quat:
        return slerp(vs[i], vs[i + 1], frac)
    return vs[i] * (1 - frac) + vs[i + 1] * frac


def poses_to_world(rig, pose):
    world = []
    for i, (T, q, S) in enumerate(pose):
        m = np.eye(4)
        m[:3, :3] = quat_to_mat(q) * S
        m[:3, 3] = T
        world.append((world[rig.parent[i]] @ m) if rig.parent[i] >= 0 else m)
    return world


# --------------------------------------------------------------------------- files
def write_smd(path, bones, frame0, triangles):
    L = ['version 1', 'nodes']
    L += ['%d "%s" %d' % (i, b['name'], b['parent']) for i, b in enumerate(bones)]
    L += ['end', 'skeleton', 'time 0']
    L += ['%d %.6f %.6f %.6f %.6f %.6f %.6f' %
          (i, p[0], p[1], p[2], r[0], r[1], r[2])
          for i, (p, r) in enumerate(frame0)]
    L += ['end', 'triangles']
    for mat, verts in triangles:
        L.append(mat)
        for bone, p, n, u, v in verts:
            L.append('%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f' %
                     (bone, p[0], p[1], p[2], n[0], n[1], n[2], u, v))
    L.append('end')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def write_anim_smd(path, bones, frames):
    L = ['version 1', 'nodes']
    L += ['%d "%s" %d' % (i, b['name'], b['parent']) for i, b in enumerate(bones)]
    L += ['end', 'skeleton']
    for fi, fr in enumerate(frames):
        L.append('time %d' % fi)
        for bi, (p, r) in enumerate(fr):
            L.append('%d %.6f %.6f %.6f %.6f %.6f %.6f' %
                     (bi, p[0], p[1], p[2], r[0], r[1], r[2]))
    L.append('end')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def safe_name(name, limit=48):
    out = ''.join(ch if (ch.isalnum() or ch in '_.-') else '_' for ch in name)
    return (out.strip('._') or 'x')[:limit]


def write_bmp8(path, pal, idx, w, h):
    row = (w + 3) & ~3
    data = bytearray()
    for y in range(h - 1, -1, -1):
        data += idx[y, :w].tobytes()
        data += b'\x00' * (row - w)
    palbytes = bytearray()
    for i in range(256):
        palbytes += bytes((pal[i * 3 + 2], pal[i * 3 + 1], pal[i * 3], 0))
    off = 14 + 40 + len(palbytes)
    with open(path, 'wb') as f:
        f.write(struct.pack('<2sIHHI', b'BM', off + len(data), 0, 0, off))
        f.write(struct.pack('<IiiHHIIiiII', 40, w, h, 1, 8, 0, len(data),
                            2835, 2835, 256, 256))
        f.write(bytes(palbytes))
        f.write(bytes(data))


def material_image(g, mi):
    mats = g.j['materials']
    mat = mats[mi] if mi < len(mats) else {}
    name = mat.get('name') or ('material%d' % mi)
    tex = mat.get('pbrMetallicRoughness', {}).get('baseColorTexture')
    alpha_mode = mat.get('alphaMode', 'OPAQUE')
    if tex is None:
        return Image.new('RGB', (32, 32), (180, 180, 180)), name, alpha_mode
    src = g.j['textures'][tex['index']]['source']
    return Image.open(io.BytesIO(g.image_bytes(src))), name, alpha_mode


def write_texture(g, mi, out_dir):
    img, name, alpha_mode = material_image(g, mi)
    name = name[:-4] if name.lower().endswith('.bmp') else name
    tname = safe_name(name)
    img = img.convert('RGBA')
    arr = np.asarray(img)
    alpha = arr[..., 3]
    masked = alpha_mode in ('MASK', 'BLEND') or bool((alpha < 250).any())
    rgb = Image.fromarray(arr[..., :3], 'RGB')
    q = rgb.quantize(colors=255 if masked else 256, method=Image.MEDIANCUT,
                     dither=Image.FLOYDSTEINBERG)
    pal = q.getpalette()[:768]
    pal += [0] * (768 - len(pal))
    idx = np.array(q, dtype=np.uint8)
    if masked:
        idx[alpha < 128] = 255
        pal[255 * 3:255 * 3 + 3] = [0, 0, 0]
    write_bmp8(os.path.join(out_dir, tname + '.bmp'), pal, idx,
               img.size[0], img.size[1])
    return tname, masked


# --------------------------------------------------------------------------- model
LOOPING = ('idle',)
SKIP = ('transfix',)


class Converter:
    #: GoldSrc quantises every SMD vertex to 1/100 of a unit, so source meshes in
    #: metres would only be ~50 quanta wide and come out shredded.  Viewmodels are
    #: authored at roughly 40-60 units for a rifle, so the whole rig is scaled up.
    TARGET_UNITS = 45.0

    def __init__(self, glb_path, smd_dir, qc_dir, fps=30.0, log=print, cd=None,
                 scale=None):
        self.cd = cd
        self.g = GLB(glb_path)
        self.base = os.path.splitext(os.path.basename(glb_path))[0]
        self.smd_dir, self.qc_dir = smd_dir, qc_dir
        self.fps = fps
        self.log = log
        self.scale = scale
        os.makedirs(smd_dir, exist_ok=True)
        os.makedirs(qc_dir, exist_ok=True)

    # ------------------------------------------------------------------ build
    def convert(self):
        g = self.g
        rig = self.rig = Rig(g)
        # uniform scale: GoldSrc vertices snap to 0.01 units, so a metre-scale mesh
        # (a 0.5 m rifle = 50 quanta) would be shredded.  Scale positions only.
        if self.scale is None:
            lo = min(float(rig.world[i][:3, 3].min()) for i in range(len(rig.world)))
            hi = max(float(rig.world[i][:3, 3].max()) for i in range(len(rig.world)))
            span = max(hi - lo, 1e-6)
            self.scale = self.TARGET_UNITS / span if span < 2.0 else 1.0
        rig.scale = self.scale
        rig.world_unscaled = list(rig.world)
        # ``rig.world`` stays in source units: the animation sampler rebuilds it
        # from the node transforms on every frame, so pre-scaling it only scaled
        # the bind pose and left the clips 1x too small (and the geometry pass
        # multiplies by ``scale`` itself, which used to double-apply it).
        prims = collect_primitives(g, rig)
        verts, bone_node, nrm, uvs, tris = weld(prims, rig)
        self.log('  welded %d verts %d tris (source)' % (len(verts), len(tris)))
        # One decimation pass aimed straight at the real limit.  Looping with a
        # 0.8x target over-shrank the mesh (each pass re-grids the *already*
        # reduced set), which once collapsed a pistol to 144 vertices.
        nv = len(np.unique(tris[:, :3].reshape(-1)))
        if nv > SAFE_VERTS or len(tris) > SAFE_TRIS:
            verts, bone_node, nrm, uvs, tris = decimate(
                verts, bone_node, nrm, uvs, tris, SAFE_TRIS, SAFE_VERTS)
        used = np.unique(tris[:, :3].reshape(-1))
        inv = {int(u): i for i, u in enumerate(used)}
        tris[:, :3] = np.vectorize(inv.get)(tris[:, :3])
        verts, bone_node, nrm = verts[used], bone_node[used], nrm[used]
        uvs = [uvs[u] for u in used]
        self.log('  final %d verts %d tris' % (len(verts), len(tris)))

        # ---------------- skeleton -------------------------------------
        parent = rig.parent
        needed = set(int(b) for b in bone_node) | {0}
        for a in g.j.get('animations', []):
            for ch in a['channels']:
                ni = rig.ix.get(ch['target']['node'])
                if ni is not None:
                    needed.add(ni)
        # every ancestor of a needed bone is also needed
        for b in list(needed):
            n = b
            while n >= 0:
                needed.add(n)
                n = parent[n]
        def depth(n):
            d = 0
            while parent[n] >= 0:
                n = parent[n]
                d += 1
            return d
        root = min(needed, key=lambda n: (depth(n), n))
        # keep only the subtree below the root
        def in_subtree(n):
            # includes the root itself
            while True:
                if n == root:
                    return True
                if n < 0:
                    return False
                n = parent[n]
        needed = {n for n in needed if in_subtree(n)}
        # pre-order renumbering, root = 0
        order = []

        def walk(n):
            order.append(n)
            for c in range(len(parent)):
                if parent[c] == n and c in needed:
                    walk(c)
        walk(root)
        newix = {n: i for i, n in enumerate(order)}
        qc_parent = [-1 if order[i] == root else newix[parent[order[i]]]
                     for i in range(len(order))]
        names, used_names = [], set()
        for n in order:
            nm = rig.name(n)
            low = nm.lower()
            if low.endswith('hand_r'):
                nm = 'r_hand'
            elif low.endswith('hand_l'):
                nm = 'l_hand'
            base = safe_name(nm, 31)
            cand, k = base, 1
            while cand.lower() in used_names:
                k += 1
                cand = (base[:27] + '_%02d' % k)[:31]
            used_names.add(cand.lower())
            names.append(cand)
        self.order = order
        self.bone_of_node = {n: i for i, n in enumerate(order)}
        self.smd_bones = [dict(name=names[i], parent=qc_parent[i])
                          for i in range(len(order))]
        self.root = 0
        # sanity: every bone must sit below a single root, no duplicates
        assert len(set(order)) == len(order), 'duplicated bone in QC order'
        bad = [i for i in range(len(order))
               if qc_parent[i] >= i]
        assert not bad, 'bone ordering is not topological: %s' % bad[:5]
        self.log('  bones: %d (%s -> %s ...)' %
                 (len(order), names[0], ', '.join(names[1:2])))

        # ---------------- bind pose (GoldSrc space) ---------------------
        # These source rigs are *globally* mirrored: every joint of the weapon
        # viewmodels carries a negative-determinant matrix.  GoldSrc bone frames
        # must be proper rotations, and reflecting individual subtrees is exactly
        # what tore the mesh apart.  The mirror is uniform across the skeleton,
        # so the motion of a bone relative to its own rest pose is proper even
        # though the absolute matrix is not:
        #
        #     G_b(f) = W_b(f) . W0_b^-1 . F_b          (det G_b = +1)
        #
        # with F_b the rest frame stored in the model and W_b the true world
        # matrix of the source (both in GoldSrc axes).  At the rest pose
        # G_b = F_b for every bone, so the engine reproduces the source mesh
        # exactly and no geometry is mirrored.
        # Option 1: the whole rig is mirrored, so mirror the whole model *once*
        # instead of projecting each bone (which reflected individual subtrees and
        # tore the mesh apart).  After one global reflection every bone matrix is a
        # proper rotation, the mesh is a coherent mirror image of the source, and
        # nothing has to be projected anywhere.
        gold_w = []
        for n in order:
            W = np.eye(4)
            W[:3, :3] = gold_rot(rig.world[n][:3, :3])
            W[:3, 3] = to_gold(rig.world[n][:3, 3]) * self.scale
            gold_w.append(W)
        dets = [float(np.linalg.det(W[:3, :3])) for W in gold_w]
        if sum(1 for d in dets if d < 0) > len(dets) / 2:
            self.mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
            self.log('  rig is mirrored -> applied one global reflection')
        else:
            self.mirror = np.eye(4)
        self.W0 = [self.mirror @ W for W in gold_w]
        self.F = []
        for i in range(len(order)):
            F = self.W0[i].copy()
            # safety net: only project if a bone is still improper after the
            # global reflection (mixed-handedness rigs)
            if np.linalg.det(F[:3, :3]) < 0:
                F[:3, :3] = project_rotation(F[:3, :3])
            self.F.append(F)
        # cached pieces used per frame
        self.W0rot = [W[:3, :3].copy() for W in self.W0]
        self.W0invF = [np.linalg.inv(self.W0[i]) @ self.F[i]
                       for i in range(len(order))]
        self.bind_local = []      # stored frames: proper rotation + local offset
        for i in range(len(order)):
            p = qc_parent[i]
            L = self.F[i] if p < 0 else np.linalg.inv(self.F[p]) @ self.F[i]
            self.bind_local.append((L[:3, 3], mat_to_angles(L[:3, :3])))
        # Vertices are written to the SMD in world space; studiomdl converts them
        # into bone space itself, using the rest pose in the reference skeleton.
        self.write_reference(verts, bone_node, nrm, uvs, tris)
        seqs = self.write_animations()
        self.write_qc(verts, seqs)
        return dict(name=self.base, bones=len(order), verts=len(verts),
                    tris=len(tris), sequences=len(seqs),
                    materials=len(set(int(m) for m in tris[:, 3])))

    # ------------------------------------------------------------------ mesh
    def write_reference(self, verts, bone_node, nrm, uvs, tris):
        V, N = to_gold(verts), to_gold(nrm)
        # move vertices into each bone's stored (proper) frame
        # Vertices stay in *world* space: studiomdl (Grab_Triangles ->
        # build_reference) moves every vertex into its bone's space itself, using the
        # rest pose written to the skeleton block.  Pre-transforming here would
        # apply that twice.
        M3 = self.mirror[:3, :3]
        V, N = (M3 @ V.T).T, (M3 @ N.T).T
        Vb, Nb = V.copy(), N.copy()
        self.used_mats = sorted(set(int(m) for m in tris[:, 3]))
        self.tex = {mi: write_texture(self.g, mi, self.smd_dir)
                    for mi in self.used_mats}
        entries = []
        for mi in self.used_mats:
            tname = self.tex[mi][0]
            sel = tris[tris[:, 3] == mi]
            for t in sel:
                # GoldSrc binds each vertex to a single bone.  When a triangle's
                # corners belong to different bones the engine tears it apart
                # (long black sheets), so the whole triangle is bound to the bone
                # most of its vertices use.  Vertices stay in world space, so the
                # rest pose is bit-identical -- only the animation assignment
                # changes, which is what keeps the pieces rigid.
                votes = [int(bone_node[int(vi)]) for vi in t[:3]]
                bone_idx = self.bone_of_node[max(set(votes), key=votes.count)]
                e = []
                for vi in t[:3]:
                    uv = uvs[int(vi)].get(mi, (0.0, 0.0))
                    e.append((bone_idx, Vb[int(vi)], Nb[int(vi)], uv[0], uv[1]))
                entries.append((tname + '.bmp', e))
        # The SMD skeleton holds the *rest* values (+0 deg delta); the triangles are
        # written in the identity (additive) pose so that studiomdl's
        # W(p) = R(value) p + value gives exactly the original rest geometry.
        # studiomdl takes every bone's rest position/rotation from this skeleton and
        # the animations only store *deltas* on top of it
        rest = [(np.array(self.bind_local[i][0], float),
                 np.array(self.bind_local[i][1], float))
                for i in range(len(self.order))]
        self.ref_path = os.path.join(self.smd_dir, self.base + '_ref.smd')
        write_smd(self.ref_path, self.smd_bones, rest, entries)

    # ------------------------------------------------------------------ anims
    def write_animations(self):
        """Bake every glTF animation into one SMD sequence.

        GoldSrc stores, per bone, a rest frame (position + rotation) and per sequence
        a translation and an *additive* angle triple: the engine shows
        ``value + delta`` component-wise, so deltas are absolute angles minus rest.

        Vertices were stored in bone space (see ``write_reference``) so that the
        chain ``C_b(f) = project(W_b(f))`` -- the nearest *proper* rotation to the
        true world matrix, which is what a mirrored rig requires -- maps them back
        onto the source mesh.  Frames are therefore evaluated in that projected
        world space instead of multiplying per-node locals (projection is not
        multiplicative, so ``project(A^-1 B) != project(A)^-1 project(B)``).
        """
        g, rig = self.g, self.rig
        seqs = []
        if not g.j.get('animations'):
            return seqs
        n = len(self.order)
        for ai, anim in enumerate(g.j['animations']):
            name = anim.get('name') or ('anim%d' % ai)
            if any(s in name.lower() for s in SKIP):
                continue
            # GoldSrc stores animation channel offsets as 16-bit values relative to
            # the start of the sequence, so numframes*numbones*6*2 must stay < 64K
            maxfr = max(2, 60000 // (6 * 2 * max(1, n)))
            frames, nframes, fps = bake_clip(g, anim, rig, self.fps,
                                             max_frames=maxfr)
            out = []
            for fr in frames:
                world = poses_to_world(rig, fr)
                # engine chain A_b(f) = project(W_b(f)): proper, equals the true
                # world matrix for normal bones, keeps mirrored subtrees rigid
                # G_b(f) = W_b(f) . W0_b^-1 . F_b  -- proper for every bone,
                # and equal to F_b at the rest pose.
                # full mirrored world matrix -- the same construction used for
                # the bind frames, so A(rest) is bit-identical to F and the
                # engine's rest + delta lands exactly on the bind pose.  (An
                # offset-local formula here left the translation at zero and
                # every clip exploded.)
                A = []
                for i, nd in enumerate(self.order):
                    M = np.eye(4)
                    M[:3, :3] = self.mirror[:3, :3] @ gold_rot(world[nd][:3, :3])
                    M[:3, 3] = self.mirror[:3, :3] @ (to_gold(world[nd][:3, 3])
                                                      * self.scale)
                    # the global reflection makes every *mirrored* bone proper,
                    # but a bone that was already proper in the source (only the
                    # Armature root here) comes out improper.  Its local frame then
                    # poisons every descendant's delta, so project it back exactly
                    # like the bind frames do.
                    if np.linalg.det(M[:3, :3]) < 0:
                        M[:3, :3] = project_rotation(M[:3, :3])
                    A.append(M)
                row = []
                prev = out[-1] if out else None
                for i in range(n):
                    pn = self.smd_bones[i]['parent']
                    L = A[i] if pn < 0 else np.linalg.inv(A[pn]) @ A[i]
                    # SMD skeleton frames are absolute; studiomdl stores the
                    # difference against the rest frame as the engine's delta, and
                    # the engine adds them component-wise, so every frame must stay
                    # on the same Euler branch as the previous one
                    # rows hold *absolute* angles, so the continuity hint is the
                    # previous frame's absolute triple (adding bind_local here
                    # shifted the hint by a whole branch and produced +-pi jumps)
                    hint = (self.bind_local[i][1] if prev is None
                            else prev[i][1])
                    # SMD animation frames hold the *absolute* local transform.
                    # studiomdl subtracts the rest frame itself
                    # (``panim.pos - bonetable.pos`` in find scales), and the engine
                    # then shows ``value + delta``; subtracting bind_local here as
                    # well applied the rest offset twice and blew the pose up.
                    row.append((L[:3, 3].copy(),
                                mat_to_angles(L[:3, :3], hint)))
                out.append(row)
            for f in range(1, nframes):
                for i in range(n):
                    prev = out[f - 1][i][1]
                    cur = out[f][i][1]
                    for k in range(3):
                        while cur[k] - prev[k] > math.pi:
                            cur[k] -= 2 * math.pi
                        while cur[k] - prev[k] < -math.pi:
                            cur[k] += 2 * math.pi
            # GoldSrc's channel decoder reads ``panimvalue[valid + 2]`` when the
            # requested frame is the last one of a run; with a run that spans the
            # whole clip that read escapes into the next channel.  Repeating the
            # final frame makes the run one frame longer and keeps the tail valid.
            if out:
                out.append([(tuple(p), tuple(a)) for p, a in out[-1]])
                nframes += 1
            smd = '%s_%s.smd' % (self.base, safe_name(name, 40))
            write_anim_smd(os.path.join(self.smd_dir, smd), self.smd_bones, out)
            seqs.append(dict(name=safe_name(name, 40), smd=smd, fps=fps,
                             frames=nframes,
                             loop=any(l in name.lower() for l in LOOPING)))
        self.log('  sequences: %d' % len(seqs))
        return seqs

    # ------------------------------------------------------------------ qc
    def write_qc(self, verts, seqs):
        V = to_gold(verts)
        mn, mx = V.min(axis=0), V.max(axis=0)
        L = ['// %s -- generated by tools/glb2goldsrc.py' % self.base,
             '$modelname "cstrike/models/%s.mdl"' % self.base,
             '$cd "%s"' % (self.cd or '.'),
             '$bbox -16 -16 -16 16 16 16',
             '$cbox -16 -16 -16 16 16 16',
             '$keepallbones',
             '$body studio "%s_ref"' % self.base,
             '$origin 0 0 0',
             '$scale 1.0',
             # keep every sequence in the main .mdl: GoldSrc's sequence-group
             # auto-split renumbers sequences and scrambles the per-sequence
             # animation offsets (the main file must stay the single group)
             '$sequencegroupsize 65535']
        for mi in self.used_mats:
            tname, masked = self.tex[mi]
            if masked:
                L.append('$texrendermode "%s.bmp" masked' % tname)
        for s in seqs:
            smd = s['smd'][:-4] if s['smd'].endswith('.smd') else s['smd']
            # NOTE: no ``fps`` option -- studiomdl's sequence ``fps`` is the *time*
            # between frames in seconds, not a frame rate.  Omitting it makes the
            # engine play one baked frame per tick, which is what the SMD holds.
            L.append('$sequence "%s" {\n\t"%s"\n%s}' %
                     (s['name'], smd, '\tloop\n' if s['loop'] else ''))
        self.qc_path = os.path.join(self.qc_dir, self.base + '.qc')
        with open(self.qc_path, 'w') as f:
            f.write('\n'.join(L) + '\n')


def convert(glb_path, smd_dir, qc_dir, fps=30.0, log=print, cd=None):
    return Converter(glb_path, smd_dir, qc_dir, fps, log, cd).convert()
