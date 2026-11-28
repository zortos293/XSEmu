# Mesh-shader execution: design note

## Decision: host-native mesh shaders first, the two-pass prepass only as a fallback

The console mesh program runs as a real host mesh shader (ms_6_6) in the title's mesh pipeline,
and a DispatchMesh or a mesh ExecuteIndirect is one host call with the title's group counts.
The two-pass path (compute prepass + pass-through mesh shader) stays for the pipelines the
native form cannot hold, listed under "Fallback".

Why native:

1. Indirect calls. The group counts of a mesh ExecuteIndirect are written by the GPU. A native
   pipeline takes the (sanitised) argument buffer as it is. The two-pass path has to size a
   record buffer on the CPU for a count it cannot know, and its prepass and mesh pass cannot tell
   the commands of one multi-command call apart (D3D12 gives a mesh shader no draw index), so
   it needs a group cap and one prepass + draw per command.
2. Cost. Both forms run the same translated program body. Two-pass adds, per draw: a second
   pipeline and root-signature bind, a compute dispatch, a UAV barrier, 16 B x (1 + attributes)
   x 128 vertices of record written and read back per group (10 to 45 KB a group in this
   title), and a pass-through shader. Native adds nothing.
3. Semantics are the same in both: the program body, its 64-bit masks and its exports are
   unchanged; only where the exports land differs.

What "native" means here. The console container holds no usable HLSL or DXIL, only RDNA2 NGG
machine code, so "native" is still the RDNA2 -> HLSL translation, emitted as a mesh shader
instead of a compute shader:

- exp pos / exp param / exp prim write per-thread locals (lane n of the group exports vertex n
  and primitive n, as the hardware NGG stage does);
- a group of several console waves gives a host wave its console wave from an LDS ticket: a
  counter each host wave's first lane increments once when the group starts, so `gi / 32` (which
  a host need not align to waves: NVIDIA fills 2-D groups from 16 x 2 tiles) is not used
  (VIctory-NVIDIA 854f57f, Vice-Tay c125c0c);
- s_sendmsg GS_ALLOC_REQ (M0 = primitives << 12 | vertices) sets the counts of the wave that
  sent it (the lane the console lets send, then lane 0 of that wave). The console's request is
  the group's, so after the program every wave adds its counts (0 for a wave with no request) to
  two groupshared dwords;
- after a group barrier every thread calls SetMeshOutputCounts with the group's counts (D3D12
  takes the values of the group's first active thread, which is the console thread that sent the
  request), and after a second barrier writes its own vertex and primitive, at its own console
  thread index;
- a null primitive (bit 31 of the export) or one with an index past the vertex count becomes
  the degenerate triangle (0, 0, 0): the console culls with the primitive export only;
- the output struct is the one the pass-through declares (what the translated pixel shader
  reads), so the mesh and pixel signatures link exactly as they do today.

## What the title's shaders say (1,831 mesh programs captured on the PC)

- all are NGG programs, wave64, numthreads 128, 128 vertices / 128 primitives, triangles;
- one GS_ALLOC_REQ and two s_barrier each (2 programs: three), no dispatcher loop, no early
  return;
- no per-primitive attributes (PSG1 empty in all 1,831), no payload (PSV0 payload size 0);
- no amplification shader at all: the 262 containers in the stream's "AS" slot are mesh
  shaders (DXIL kind 13) - the console numbers AS / MS subobjects 23 / 24, the PC 24 / 25;
- 2 programs use LDS (3,072 B of groupshared), 1,829 none;
- primitive exports use channel 0 only (indices + null bit); 32 programs export pos1.

## Console-semantics limits of the native form (-> fallback)

| limit | why | how common here |
|---|---|---|
| the group spans more than one console wave | the native form packs one console wave into one 32-lane host wave, so it accepted a group of one console wave only: at most 32 threads for a wave32 program, 64 for a wave64 one. A host group can hold several waves, and `XCL_MESH_MULTI_WAVE=1` gives a group of two or more console waves one mesh shader (`pso::mesh_native_host_threads`). Off by default | every mesh program of this title, whose groups are 128 threads (two console waves). play-124 ran 0 native and 417 two-pass pipelines and 431,054 two-pass calls; play-125 ran 0 native and 3,119,263 two-pass calls. The fallback line said so each time: "the group has more than one console wave" |
| the host's wave size is not fixed at 32 | HLSL has no WaveSize for mesh shaders; the translation needs waves it knows (wave64 programs run as two 32-lane halves, docs/wave64-on-wave32.md). NVIDIA reports 32..32. AMD (32..64) and Intel pick per shader | never on the RTX 3080, which reports 32..32. This limit did not cause the fallback there |
| more than 15 pixel-shader inputs | every input is a float4 in the link signature: 16 B x (1 + n) x 128 vertices must fit D3D12's 32 KB mesh output limit; the pass-through splits such records over several mesh groups, a native shader cannot | at most 69 of 1,831 programs (17+ outputs) |
| the program uses LDS | xcl_lds is 32 KB, the mesh stage allows 28 KB of groupshared | 2 of 1,831 |
| more than 128 hardware lanes (vertices or primitives > 128) | a host mesh group has at most 128 threads | none |
| DXC or the driver rejects the mesh shader | - | not seen |

XCL_MESH=two-pass forces the fallback (A/B runs, and how the fallback is tested on the RTX).

Until `XCL_MESH_MULTI_WAVE=1`, the RTX 3080 ran every mesh pipeline in two passes: the wave-size
condition held, the group-width condition did not. The native tests use a one-console-wave
program, so they passed on that card while the title's two-wave groups fell back.

## Driver pipelines per pair

A pair costs one driver pipeline as a host mesh shader and two in the fallback, the raster pass
and the compute pass. The depth prepass record observer adds one, or two with its paired
program, and no ordinary run builds it.
`hle::d3d12x::mesh_pso::mesh_pipelines` is that list: the build lines print its count, and the
caches line reports `mesh pipelines: N native, M two-pass, K driver compiles`, where K counts the
creates that reached the driver and leaves out a kept blob or a pipeline library hit.

## Fallback: two passes

Record memory: one XCL buffer of 32 MB (TWO_PASS_SCRATCH_BYTES) that every two-pass mesh call
writes and reads from its start. The GPU runs a queue's calls in order, and the prepass empties
what it does not export (each thread's primitive is null and the group's counts are 0 until the
program exports them), so no call needs memory of its own and nothing is zeroed on the CPU. The
title issues its indirect mesh calls one record at a time, thousands a frame (seen by the
VIctory fork too): worst-case memory per call would not fit.

Direct DispatchMesh: the prepass runs x * y * z groups with the graphics root arguments (io) on
the compute root signature, a UAV barrier, then the mesh pipeline over groups * split. More
records than the buffer holds come from the list's unified memory.

In both forms y and z fold into x: a console program finds its group in one register (v5 =
group * threads + thread), so a call of (x, y, z) runs x * y * z groups numbered on one axis.
The title's programs take x groups only (VIctory runs them that way).

Indirect: the sanitised copy of the arguments drives both passes. Each command k gets its own
record range (the res root view moves by k * capacity) and runs as a single-command indirect
dispatch (XCL's dispatch signature of the same stride) followed by a single-command mesh call.
The group count is unknown on the CPU, so each command is clamped to a capacity (the record
buffer split over the call's commands: 700 to 3,200 groups for a one-command call in this
title); what is over it is not drawn. This is the limit that made native the first choice.

Console geometry shaders are also built "in two passes" (the same record and pass-through,
`TwoPassPso::kind` "GS"), but they are drawn by Draw calls, not mesh dispatches. Running those
draws is not part of this change: they were refused before and are still skipped and counted
(the title has no such pipeline).

## Amplification shaders

Not implemented, on purpose: no console AS exists in the captured set, so the console ABI
between an AS program and its mesh program (where DispatchMesh's counts and the payload go) is
unknown. A pipeline with a real console AS still fails at creation with its own error line.
DispatchMeshX (list #98 / #100) with indirect arguments is logged and skipped for the same
reason; a payload address is logged once and not passed on (a mesh program without an AS
declares no payload). A mesh pipeline whose AS and MS are PC DXIL is not translated at all:
its DispatchMesh is the PC call with the title's own root signature, payload included.

## HLE surface

- command list #127 (older layout) / #129 (newer layout): DispatchMesh(x, y, z);
- #98 / #100: DispatchMeshX(x, flags, indirect, payload) - the direct form only;
- #59 / #61 ExecuteIndirect with a mesh signature: through the producer check and the
  sanitiser like every other indirect call;
- the refusal guard and its counter are replaced by counts of mesh calls run natively, run in
  two passes, and skipped.

## Tests

gpu::recomp (real GPU; the host-mesh-shader ones need an adapter with 32-lane waves only, so
they skip on WARP, AMD and Intel):

- gpu_mesh_mode_native_draws_what_two_passes_draw: a console-style NGG program (GS_ALLOC_REQ,
  pos / param / primitive exports, two wave64s per group) as a host mesh shader draws the known
  pixels, the same pixels as the two passes, and the same again when its vertices are made for
  a pixel shader's inputs;
- gpu_mesh_mode_culls_null_primitives: a null primitive draws nothing, in both forms;
- gpu_mesh_mode_native_indirect_takes_the_gpu_count: ExecuteIndirect with a mesh signature
  takes group counts and the command count from a GPU buffer;
- mesh_native_refuses_what_the_host_mesh_stage_cannot_hold: the fallback conditions;
- mesh_containers_compile_as_host_mesh_shaders (ignored; XCL_SAMPLE_MESH=<directory of dumped
  MS containers>): every captured mesh program through the translator and DXC.

hle::d3d12x:

- the_sanitiser_plans_a_mesh_call_in_two_passes, the_sanitiser_empties_the_records_past_the_gpu_count
  (real GPU): the argument copy both passes of the fallback read, with a GPU count;
- mesh_calls_reach_the_pc_list_only_with_a_mesh_pipeline: routing of DispatchMesh,
  DispatchMeshX, draws and indirect calls by pipeline and signature kind.

Not covered by a test: the fallback's per-command prepass + mesh call (bind::two_pass_direct /
two_pass_indirect) end to end - it needs the title-side list state and unified memory, which
the test harness does not have. XCL_MESH=two-pass in a game run exercises it.

## Follow-ups (not in this patch)

- Lane-independent lowering of mesh programs that have no cross-lane data flow (as VIctory
  does): no exchange between the two 32-lane halves, any host wave size, AMD and Intel hosts
  native too. This is translator work (emit.rs).
- Pixel-shader inputs declared with their real component counts, so the 32 KB limit matches
  the console's own and the 69 wide programs go native.
- Per-primitive attributes (the console's param table marks them; none in this title).
