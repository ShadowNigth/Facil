# CS:GO → Counter-Strike 1.6 conversion — status

Source: https://github.com/ShadowNigth/models.git (203 `.glb`, 748 `.wav`)

## What works

`tools/glb2goldsrc.py` converts a source `.glb` into an SMD set + QC and the
patched `fnky/studiomdl` compiles it into a loadable GoldSrc `.mdl`:

    v_ak47.mdl — 70 bones, 1 642 verts, 3 671 tris, 20 sequences,
                 3 textures, ~2.0 MB, no compiler warnings

Verified correct:

* **geometry** — the SMD triangle data reproduces the source mesh to 1e-7
* **bind pose** — byte-identical to the source rest pose
* **animation playback** — every channel of every sequence decodes; the decoded
  pose tracks the source clip (median deviation ≈ 0.26 units at 45-unit scale,
  i.e. sub-centimetre after the 1 cm quantization the format imposes)
* **textures** — 8-bit 256-colour BMPs with UVs, 3 materials per viewmodel
* **bones / sequences / size limits** — all inside GoldSrc's caps

## Bug fixes found along the way

| # | Bug | Fix |
|---|-----|-----|
| 1 | `Rig.__init__` appended children twice → duplicated bones | single recursive `add()` + assert |
| 2 | `in_subtree()` looped forever on the root | returns before descending |
| 3 | `mat_to_angles` only covered \|x\| ≤ 90° | closed form + second Euler branch |
| 4 | Euler branch picked arbitrarily → 180° pose jumps | branch chosen nearest the previous frame |
| 5 | SMD skeleton rows had position vectors in the rotation slots | writes `bind_local` angles |
| 6 | Bones without skinned verts were dropped, corrupting the chain | `$keepallbones` + patched studiomdl |
| 7 | Vertices were transformed twice (converter *and* `Grab_Triangles`) | converter writes world-space verts |
| 8 | QC `$sequence fps` is *seconds per frame*, not a frame rate | option removed |
| 9 | Sequences were written with a stale `animindex`/pad layout | patched `WriteAnimations` advance |
| 10 | Animation deltas must be `absolute − rest` (engine adds component-wise) | delta written |
| 11 | Last frame of every clip overflowed the channel tail read | duplicate final frame |
| 12 | Metre-scale mesh → only ~50 of the 1 cm quantization cells | uniform scale to 45 units |
| 13 | `mdlpreview` animation decoder mis-stepped the run chain | verbatim port of `StudioCalcBoneQuaterion` |
| 14 | `mdlpreview` z-buffer wrote into a slice copy (no depth test) | write-back |

## What is still wrong

The **weapon mesh** comes out shredded while the arms/gloves render correctly.
Root cause: the source rig is mirrored — `v_weapon` has scale `−0.8128`, and all
69 joints have a negative-determinant bind matrix. GoldSrc bone frames must be
proper rotations, so the converter projects each frame onto the nearest proper
rotation. That is fine for the *rest* pose, but geometry whose skeleton is
mirrored gets reflected relative to its neighbours, which the engine cannot
reproduce: the parts stay rigid but land in the wrong place.

Two ways forward (needs a decision):

1. **Mirror-free rebuild** — negate one axis of the whole model and re-derive the
   skeleton, so no bone has a negative determinant. Clean, but every weapon must
   be re-checked for which axis to flip.
2. **Bake the mirror into the vertex data** — transform the vertices by the mirror
   and adjust the animation accordingly. Cheaper, but the animations then need a
   per-bone handedness flag, which the format does not have.

## Reproducing

    # 1. build the patched compiler (POSIX/gcc)
    git clone https://github.com/fnky/studiomdl /tmp/sdl
    cd /tmp/sdl && patch -p1 < /path/to/write.c.patch
    # 2. compile one model
    python -c "import sys; sys.path.insert(0,'tools'); import glb2goldsrc as G; \
               G.Converter('v_ak47.glb','work/models','work/qc',cd='models').convert()"
    cd work && /tmp/sdl/build/bin-/studiomdl -i qc/v_ak47.qc
    # 3. inspect
    python tools/mdlpreview.py work/cstrike/models/v_ak47.mdl out.png
