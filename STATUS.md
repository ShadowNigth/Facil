# CS:GO → Counter-Strike 1.6 conversion — status

Source: https://github.com/ShadowNigth/models.git (203 `.glb`, 748 `.wav`)

## What works

`tools/glb2goldsrc.py` converts a source `.glb` into an SMD set + QC and the
patched `fnky/studiomdl` compiles it into a loadable GoldSrc `.mdl`:

    v_glock18.mdl — 67 bones, 1 862 verts, 4 099 tris, 21 sequences,
                   3 textures, 1.87 MB, no compiler warnings

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

## Mirror problem — solved (Option 1)

The source rigs are *globally* mirrored (`v_glock18`: 66 of 67 `W0` determinants
negative), so the fix is a single global reflection of the whole model, not a
per-bone projection:

* one global `diag(-1,1,1)` applied to vertices, bind frames and every animated
  world matrix — after that every bone is a proper rotation, with no reflection
  anywhere and no per-subtree surgery
* the single bone that was *already* proper in the source (the `Armature` root)
  comes out improper under the global flip, so it alone is projected back; this
  keeps its local frame from poisoning every descendant's delta
* the animation rows hold **absolute** local transforms (studiomdl subtracts the
  rest frame itself); writing `absolute − rest` there subtracted the rest twice
* the Euler branch hint must be the previous frame's *absolute* triple, not
  `bind_local + previous`

`v_glock18` now verifies clean:

| check | result |
|---|---|
| bones / verts / tris | 67 / 1 862 / 4 099 |
| sequences | 21, `$sequencegroupsize 65535` |
| textures / size | 3 / 1.87 MB, no compiler warnings |
| bind vs. source (nearest, mirrored) | median 0.35, max 1.37 units (= decimation) |
| longest triangle edge, all 21 clips × all frames | 9.0 = the bind value → **no stretching** |
| gun size across every clip | 3.7–11.0 units in all frames (bind 5.8 × 1.5 × 10.5) |

## Bug fixes found along the way (cont.)

| # | Bug | Fix |
|---|-----|-----|
| 15 | Skinned meshes were transformed by their *node*; the glTF spec says that transform is ignored and `world[joint]·IBM` must be used — this pushed the pistol 20 units away from the hands | LBS with `world[joint]·IBM`, node transform skipped for skinned meshes |
| 16 | `JOINTS_0` holds indices into `skin.joints`, not node ids | resolved through `skin.joints` |
| 17 | Rig world frames were pre-scaled *and* the mesh was scaled again (52× bind, clips exploded to 6000 units) | keep `rig.world` in source units, scale at the output |
| 18 | Decimation looped with a 0.8× target and re-gridded its own output (pistol collapsed to 144 verts) | one pass aimed at both caps (2 020 verts / 19 600 tris) |
| 19 | After #15 the dominant-bone lookup still indexed `node_of_joint` (node id) → all gun triangles bound to the rig root | map node id back through `rig.ix` |
| 20 | **studiomdl bug**: a constant channel is reduced to `{valid=1,total=1}` + one value; the engine reads `value[k+2]` whenever `total>k+1`, so that second sample is the next channel's header (`0x037f` → 32767) and the pose jumps ~1.5 rad | patched `WriteAnimations` to emit a 3-sample run `{valid=3,total=3}` with the value repeated |

Bug 20 also shows up as "the root bone rotates by an extra 90°" in any model
with a constant-but-nonzero animation channel — this is the mirror-free rebuild's
main blocker and it is now fixed in the local studiomdl patch.

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
