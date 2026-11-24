#!/usr/bin/env python3
"""
mdlpreview.py -- read a GoldSrc (Counter-Strike 1.6) .mdl and render frames of its
sequences with a small software rasterizer, so converted models can be checked
without the game.

Layouts follow the CS 1.6 engine structures (engine/studio.h); animation values are
evaluated exactly like CStudioModelRenderer::StudioCalcBoneQuaternion/Position.
"""
from __future__ import annotations

import math
import os
import struct
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- structures
HDR = [('id', 4, 'i'), ('version', 4, 'i'), ('name', 64, 's'), ('length', 4, 'i'),
       ('eyeposition', 12, 'f'), ('min', 12, 'f'), ('max', 12, 'f'),
       ('bbmin', 12, 'f'), ('bbmax', 12, 'f'), ('flags', 4, 'i'),
       ('numbones', 4, 'i'), ('boneindex', 4, 'i'),
       ('numbonecontrollers', 4, 'i'), ('bonecontrollerindex', 4, 'i'),
       ('numhitboxes', 4, 'i'), ('hitboxindex', 4, 'i'),
       ('numseq', 4, 'i'), ('seqindex', 4, 'i'),
       ('numseqgroups', 4, 'i'), ('seqgroupindex', 4, 'i'),
       ('numtextures', 4, 'i'), ('textureindex', 4, 'i'),
       ('texturedataindex', 4, 'i'),
       ('numskinref', 4, 'i'), ('numskinfamilies', 4, 'i'), ('skinindex', 4, 'i'),
       ('numbodyparts', 4, 'i'), ('bodypartindex', 4, 'i'),
       ('numattachments', 4, 'i'), ('attachmentindex', 4, 'i'),
       ('soundtable', 4, 'i'), ('soundindex', 4, 'i'), ('soundgroups', 4, 'i'),
       ('soundgroupindex', 4, 'i'), ('numtransitions', 4, 'i'),
       ('transitionindex', 4, 'i')]

BONE = [('name', 32, 's'), ('parent', 4, 'i'), ('flags', 4, 'i'),
        ('bonecontroller', 24, 'i'), ('value', 24, 'f'), ('scale', 24, 'f')]

SEQDESC = [('label', 32, 's'), ('fps', 4, 'f'), ('flags', 4, 'i'),
           ('activity', 4, 'i'), ('actweight', 4, 'i'),
           ('numevents', 4, 'i'), ('eventindex', 4, 'i'),
           ('numframes', 4, 'i'), ('numpivots', 4, 'i'), ('pivotindex', 4, 'i'),
           ('motiontype', 4, 'i'), ('motionbone', 4, 'i'),
           ('linearmovement', 12, 'f'), ('automoveposindex', 4, 'i'),
           ('automoveangleindex', 4, 'i'), ('bbmin', 12, 'f'), ('bbmax', 12, 'f'),
           ('numblends', 4, 'i'), ('animindex', 4, 'i'),
           ('blendtype', 8, 'i'), ('blendstart', 8, 'f'), ('blendend', 8, 'f'),
           ('blendparent', 4, 'i'), ('seqgroup', 4, 'i'),
           ('entrynode', 4, 'i'), ('exitnode', 4, 'i'), ('nodeflags', 4, 'i'),
           ('nextseq', 4, 'i')]

TEXTURE = [('name', 64, 's'), ('flags', 4, 'i'), ('width', 4, 'i'),
           ('height', 4, 'i'), ('index', 4, 'i')]

BODYPART = [('name', 64, 's'), ('nummodels', 4, 'i'), ('base', 4, 'i'),
            ('modelindex', 4, 'i')]

MODEL = [('name', 64, 's'), ('type', 4, 'i'), ('boundingradius', 4, 'f'),
         ('nummesh', 4, 'i'), ('meshindex', 4, 'i'), ('numverts', 4, 'i'),
         ('vertinfoindex', 4, 'i'), ('vertindex', 4, 'i'), ('numnorms', 4, 'i'),
         ('norminfoindex', 4, 'i'), ('normindex', 4, 'i'), ('numgroups', 4, 'i'),
         ('groupindex', 4, 'i')]

MESH = [('numtris', 4, 'i'), ('triindex', 4, 'i'), ('skinref', 4, 'i'),
        ('numnorms', 4, 'i'), ('normindex', 4, 'i')]


def upack(data, base, layout):
    out = {}
    off = base
    for name, size, kind in layout:
        raw = data[off:off + size]
        if kind == 'i':
            vals = struct.unpack('<%di' % (size // 4), raw)
            out[name] = vals[0] if len(vals) == 1 else list(vals)
        elif kind == 'f':
            vals = struct.unpack('<%df' % (size // 4), raw)
            out[name] = vals[0] if len(vals) == 1 else list(vals)
        else:
            out[name] = raw.split(b'\x00')[0].decode('latin-1')
        off += size
    return out


def struct_size(layout):
    return sum(s for _, s, _ in layout)


class MDL:
    def __init__(self, path):
        self.path = path
        self.data = open(path, 'rb').read()
        d = self.data
        self.h = upack(d, 0, HDR)
        assert self.h['id'] == 0x54534449, 'not a GoldSrc studio model'
        h = self.h
        self.bones = [upack(d, h['boneindex'] + i * struct_size(BONE), BONE)
                      for i in range(h['numbones'])]
        self.seqs = [upack(d, h['seqindex'] + i * struct_size(SEQDESC), SEQDESC)
                     for i in range(h['numseq'])]
        self.textures = [upack(d, h['textureindex'] + i * struct_size(TEXTURE),
                               TEXTURE) for i in range(h['numtextures'])]
        self.bodyparts = [upack(d, h['bodypartindex'] + i * struct_size(BODYPART),
                                BODYPART) for i in range(h['numbodyparts'])]
        self.models = []
        for bp in self.bodyparts:
            for i in range(bp['nummodels']):
                self.models.append(upack(
                    d, bp['modelindex'] + i * struct_size(MODEL), MODEL))
        self.skinref = struct.unpack('<%dh' % h['numskinref'],
                                     d[h['skinindex']:h['skinindex'] + 2 * h['numskinref']])
        self.meshes = []
        for m in self.models:
            for i in range(m['nummesh']):
                self.meshes.append((m, upack(
                    d, m['meshindex'] + i * struct_size(MESH), MESH)))
        self.palettes = []
        for t in self.textures:
            raw = d[t['index']:t['index'] + t['width'] * t['height']]
            pal = d[t['index'] + t['width'] * t['height']:
                    t['index'] + t['width'] * t['height'] + 768]
            pal = pal + b'\x00' * (768 - len(pal))
            self.palettes.append((np.frombuffer(pal, np.uint8).reshape(256, 3),
                                  np.frombuffer(raw, np.uint8).reshape(
                                      t['height'], t['width'])))

    # ------------------------------------------------------------- animation
    def _raw(self, off, kind='H'):
        return struct.unpack('<' + kind, self.data[off:off + struct.calcsize('<' + kind)])[0]

    def anim_values(self, panim_off, k, frame):
        """Verbatim port of the engine's ``StudioCalcBoneQuaterion`` channel walk.

        ``mstudioanim_t`` is ``unsigned short offset[6]`` and each channel offset is
        relative to that entry.  A channel is a chain of 2-byte ``mstudioanimvalue_t``
        unions: ``{byte valid, byte total}`` followed by ``valid`` sample shorts.
        Returns ``(sample, next_sample)`` in *raw* units, or ``None`` for a constant
        channel (the caller then uses the bone's rest value).
        """
        data = self.data
        off = self._raw(panim_off + 2 * k, 'H')
        if off == 0:
            return None
        p = panim_off + off
        valid, total = struct.unpack('<BB', data[p:p + 2])
        k0 = frame
        if total < valid:                      # engine DEBUG guard
            k0 = 0
        guard = 0
        while total <= k0 and guard < 4096:
            k0 -= total
            p += 2 * (valid + 1)
            valid, total = struct.unpack('<BB', data[p:p + 2])
            if total < valid:
                k0 = 0
            guard += 1

        def val(i):
            return struct.unpack('<h', data[p + 2 * i:p + 2 * i + 2])[0]

        if valid > k0:
            a = val(k0 + 1)
            if valid > k0 + 1:
                b = val(k0 + 2)
            elif total > k0 + 1:
                b = a
            else:
                b = val(valid + 2)
        else:
            a = val(valid)
            b = a if total > k0 + 1 else val(valid + 2)
        return a, b

    def bone_pose(self, pseq, frame, blend=0):
        nb = self.h['numbones']
        nb_blends = max(1, pseq.get('numblends', 1))
        panim = pseq['animindex'] + (blend % nb_blends) * nb * 12
        pos = np.zeros((nb, 3))
        ang = np.zeros((nb, 3))
        for j in range(nb):
            bone = self.bones[j]
            v = bone['value']
            sc = bone['scale']
            for k in range(3):
                pos[j, k] = v[k]
                # sample the *current* frame (s = 0): the engine feeds the two
                # samples to a slerp, and only the first one matters at an exact
                # frame boundary.  Averaging them would blend in the next frame.
                res = self.anim_values(panim + j * 12, k, frame)
                if res is not None:
                    pos[j, k] += res[0] * sc[k]
                ang[j, k] = v[k + 3]
                res = self.anim_values(panim + j * 12, k + 3, frame)
                if res is not None:
                    ang[j, k] += res[0] * sc[k + 3]
        return pos, ang

    def bone_world(self, pos, ang):
        world = []
        for j in range(len(self.bones)):
            ax, ay, az = ang[j]
            cx, sx = math.cos(ax), math.sin(ax)
            cy, sy = math.cos(ay), math.sin(ay)
            cz, sz = math.cos(az), math.sin(az)
            # R = Rz*Ry*Rx  (GoldSrc AngleQuaternion)
            R = np.array([
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx]])
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = R, pos[j]
            p = self.bones[j]['parent']
            world.append((world[p] @ m) if p >= 0 else m)
        return world

    def vertices(self, world):
        """World-space triangles per texture, decoding the tri-strip commands the
        same way the engine renderer does (strip when count > 0, fan when < 0)."""
        out = []
        d = self.data
        for m, mesh in self.meshes:
            tail = d[mesh['triindex']:mesh['triindex'] + mesh['numtris'] * 3 * 8 + 8192]
            cmd = np.frombuffer(tail, np.int16)
            verts = np.frombuffer(d[m['vertindex']:m['vertindex'] + m['numverts'] * 12],
                                  np.float32).reshape(-1, 3)
            norms = np.frombuffer(d[m['normindex']:m['normindex'] + m['numnorms'] * 12],
                                  np.float32).reshape(-1, 3)
            vbone = d[m['vertinfoindex']:m['vertinfoindex'] + m['numverts']]
            tex = self.skinref[mesh['skinref']]
            tris = []
            i = 0
            while i < len(cmd):
                cnt = int(cmd[i])
                i += 1
                if cnt == 0:
                    break
                n = abs(cnt)
                chunk = cmd[i:i + n * 4].reshape(-1, 4)
                i += n * 4
                for t in range(2, n):
                    if cnt > 0:
                        sel = [t - 2, t - 1, t] if t % 2 == 0 else [t - 1, t - 2, t]
                    else:
                        sel = [0, t - 1, t]
                    tris.append(chunk[sel])
            if not tris:
                continue
            arr = np.array(tris, dtype=np.int64)
            P = np.zeros((len(arr), 3, 3))
            N = np.zeros((len(arr), 3, 3))
            UV = np.zeros((len(arr), 3, 2))
            for t in range(len(arr)):
                for c in range(3):
                    vi, ni, s, tt = arr[t, c]
                    vb = int(vbone[vi])
                    M = world[vb]
                    P[t, c] = M[:3, :3] @ verts[vi] + M[:3, 3]
                    N[t, c] = M[:3, :3] @ norms[ni]
                    UV[t, c] = (s / float(self.textures[tex]['width']),
                                tt / float(self.textures[tex]['height']))
            out.append((tex, P, N, UV))
        return out


def render_frame(mdl, tris, res=700, eye=None, fwd=None):
    """Simple painter with texture + shading; camera framed on the model bbox."""
    allP = np.concatenate([t[1].reshape(-1, 3) for t in tris])
    lo, hi = allP.min(0), allP.max(0)
    center = (lo + hi) / 2
    radius = max(np.linalg.norm(hi - lo) / 2, 1e-3)
    if eye is None:
        eye = center + np.array([0.0, -1.0, 0.25]) * radius * 2.6
    if fwd is None:
        fwd = center - eye
    fwd = np.asarray(fwd, float)
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = 1 / math.tan(math.radians(50) / 2)
    img = np.ones((res, res, 3))
    zbuf = np.full((res, res), 1e9)
    depth = []
    for tex, P, N, UV in tris:
        cam = np.stack([(P - eye) @ right, (P - eye) @ up, (P - eye) @ fwd], axis=2)
        for i in range(len(P)):
            if np.any(cam[i, :, 2] <= 0.01):
                continue
            sx = f * cam[i, :, 0] / cam[i, :, 2]
            sy = f * cam[i, :, 1] / cam[i, :, 2]
            px = (sx / 2 + 0.5) * (res - 1)
            py = (1 - (sy / 2 + 0.5)) * (res - 1)
            x0, x1 = int(max(0, math.floor(px.min()))), int(min(res - 1, math.ceil(px.max())))
            y0, y1 = int(max(0, math.floor(py.min()))), int(min(res - 1, math.ceil(py.max())))
            if x1 < x0 or y1 < y0:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            den = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0])
            if abs(den) < 1e-9:
                continue
            w0 = ((px[1] - X) * (py[2] - Y) - (px[2] - X) * (py[1] - Y)) / den
            w1 = ((px[2] - X) * (py[0] - Y) - (px[0] - X) * (py[2] - Y)) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            z = w0 * cam[i, 0, 2] + w1 * cam[i, 1, 2] + w2 * cam[i, 2, 2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            mask = inside & (z < sub)
            if not mask.any():
                continue
            u = w0 * UV[i, 0, 0] + w1 * UV[i, 1, 0] + w2 * UV[i, 2, 0]
            v = w0 * UV[i, 0, 1] + w1 * UV[i, 1, 1] + w2 * UV[i, 2, 1]
            pal, pix = mdl.palettes[tex]
            h, w = pix.shape
            tx = np.clip((u * (w - 1)).astype(int), 0, w - 1)
            ty = np.clip((v * (h - 1)).astype(int), 0, h - 1)
            col = pal[pix[ty, tx]] / 255.0
            n = N[i, 0] * w0[..., None] + N[i, 1] * w1[..., None] + N[i, 2] * w2[..., None]
            ln = np.linalg.norm(n, axis=-1, keepdims=True)
            ln[ln == 0] = 1
            n = n / ln
            light = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
            shade = 0.45 + 0.65 * np.clip(n @ light, 0, 1)
            col = np.clip(col * shade[..., None], 0, 1)
            dst = img[y0:y1 + 1, x0:x1 + 1]
            dst[mask] = col[mask]
            img[y0:y1 + 1, x0:x1 + 1] = dst
            # NOTE: fancy-index assignment on a slice works on a copy, so the depth
            # buffer has to be written back explicitly or the z-test never sticks.
            zs = zbuf[y0:y1 + 1, x0:x1 + 1]
            zs[mask] = z[mask]
            zbuf[y0:y1 + 1, x0:x1 + 1] = zs
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def contact_sheet(path, out_png, seqs=None, frames_per_seq=3, res=340):
    mdl = MDL(path)
    names = [s['label'] for s in mdl.seqs]
    pick = seqs or list(range(min(len(names), 8)))
    tiles = []
    for si in pick:
        seq = mdl.seqs[si]
        n = max(1, seq['numframes'])
        step = max(1, (n - 1) // frames_per_seq) if n > 1 else 1
        for fi in range(0, n, step)[:frames_per_seq]:
            pos, ang = mdl.bone_pose(seq, fi)
            world = mdl.bone_world(pos, ang)
            tris = mdl.vertices(world)
            im = render_frame(mdl, tris, res=res)
            tiles.append((names[si], fi, n, im))
    cols = frames_per_seq
    rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new('RGB', (cols * res, rows * res), 'white')
    for i, (name, fi, n, im) in enumerate(tiles):
        sheet.paste(im, ((i % cols) * res, (i // cols) * res))
    sheet.save(out_png)
    return names, len(tiles)


if __name__ == '__main__':
    mdl_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'sheet.png'
    m = MDL(mdl_path)
    print('%s: %d bones, %d seqs, %d textures, %d bodyparts' %
          (os.path.basename(mdl_path), m.h['numbones'], m.h['numseq'],
           m.h['numtextures'], m.h['numbodyparts']))
    print('sequences:', ', '.join('%s(%d)' % (s['label'], s['numframes'])
                                  for s in m.seqs))
    print('textures:', ', '.join(t['name'] for t in m.textures))
    names, n = contact_sheet(mdl_path, out)
    print('wrote %s (%d tiles)' % (out, n))
