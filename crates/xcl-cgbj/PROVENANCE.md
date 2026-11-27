# Provenance of crates/xcl-cgbj

This crate is the shader translator of CGBJ, copied into xsemu on 2026-11-27 for ZOR-632 (CAPA-86).

- Source: CGBJ, the cgbj-reupload repository, commit f4b48ad766ddbec1a1867022cd42f7645b0dd55e, folder crates/victory-shader.
- CGBJ is a fork of VIctory (Marvisak/VIctory) through fxceless/victory-fork and avatarmanz/VIctory-7900XTX.
- Licence: GPL-3.0-or-later. LICENSE in this folder is CGBJ's LICENSE file, unchanged.
- The rest of xsemu was GPL-2.0-only when this crate was copied; the user allowed the copy for the private repository on 2026-11-26. On 2026-11-27 xsemu moved to GPL-3.0-or-later, following upstream XCL's relicense (xcl-dev/xcl 06f90c1), so this crate and the rest of the tree are now under the same licence.

## What was copied

Every .rs file under crates/victory-shader/src (38 files) and crates/victory-shader/build.rs, each at the same relative path here. Each file has three comment lines added at the top (SPDX line, source commit and path, licence pointer). Apart from the six files listed under "Changes to copied files" (src/lib.rs, src/emit/mod.rs, src/emit/shader.rs, src/emit/lower.rs, src/resource.rs, src/structure.rs), each is otherwise byte for byte the file at f4b48ad. PROVENANCE.sha256 lists the sha256 of each original file. To check an unmodified file, drop its first three lines and hash the rest:

    tail -n +4 src/emit/lower.rs | sha256sum

Line numbers here are the original's plus 3.

Third-party parts inside the copied files keep their own notices: src/gfx10/opcodes.rs is generated from Mesa's aco_opcodes.py (Valve, MIT).

## What was not copied

- crates/victory-shader/benches/guard_runs.rs (a timing bench).
- Everything outside crates/victory-shader: CGBJ's host crates, its tools, docs and launcher.

## What is new here

- Cargo.toml: the package is renamed xcl-cgbj (library xcl_cgbj). The dependency versions and lints of CGBJ's workspace are written out, because this crate is its own workspace like crates/xcl-dxil. CGBJ's build profiles are not carried over. Cargo.lock pins the dependency versions of CGBJ's own lock.
- src/xcl.rs: written for xsemu. The wave policy the host sets (set_wave_exact) and the wave form a translation took (take_form). Since M1a also the XCL binding's API: XclConfig, XclShader, XclStats, translate_compute, BINDING_VERSION. M1 took the size probe out again (set_size_probe), added two counts to XclStats (page_groups, grouped_loads) and made BINDING_VERSION 2. M1 v2 made it 3 (the numeric class of an image). M1d added the packed form's label and numthreads (PACKED_FORM, packed_threads) and made it 5.
- src/bin/xcl-cgbj-census.rs: written for xsemu, after the measuring tool CAPA-85 wrote. It translates a folder of shader containers and reports forms, refusals and sizes. Since M1a it takes --binding victory|xcl (M1a also had xcl-size-probe, which M1 removed).
- src/emit/xcl.rs: written for xsemu (M1a). The pipeline of a translation under the XCL binding. Since M1 also Refusals: which refusal is reported when no wave form of a program translates.
- src/emit/lower/xcl.rs: written for xsemu (M1a). The XCL binding's lowering: entry state, the address resolver, and the memory, buffer, image and sampler instructions on register values. It is a child module of the copied src/emit/lower.rs so that it can use the emitter's state. Since M1 also page groups (the two 64 KiB pages the scalar loads of one base lie in, resolved once) and the forms that take their format from the V#. Since M1b also the format forms of halves, fmax and fmin done as a compare-exchange loop, and image stores of some channels.
- src/emit/lower/xcl_formats.rs: written for xsemu (M1). The buffer formats a V# names: constant tables of the 77 formats, and the code that decodes and encodes a component by them. A child module of src/emit/lower/xcl.rs.
- src/emit/lower/xcl_tests.rs: written for xsemu (M1a, more in M1). Tests of the binding.
- src/emit/lower/packed.rs: written for xsemu (M1d). The packed form of a console wave64 compute program on a host of 32-lane waves: one host wave runs one console wave, and each host lane holds console lanes i and i + 32. A vector instruction is lowered once per half and a scalar one once, and what crosses lanes (masks as data, wave-wide decisions, lane reads, permutes, whole-wave sections) takes both halves. It is a child module of the copied src/emit/lower.rs, like lower/xcl.rs.

## Changes to copied files

Every change is marked with a comment starting `xsemu:`, and a changed file's header says "Modified for xsemu". Each later change is added to this list with its patch.

capa86-m0-xcl-cgbj-crate-v2:
- src/lib.rs: declares the new module xcl.
- src/emit/mod.rs, in emit() and translation_fingerprint():
  - The exact wave policy (xcl::set_wave_exact). When it is on, the approximate local-wave forms are refused as under VICTORY_WAVE_LOCAL=idiom, and wave-wide branches are refused where a console wave is two host waves. The wave-wide branch of a program whose waves are the host's has no barrier and stays allowed. The policy is part of the fingerprint. With the policy off, behaviour is CGBJ's except for the next point.
  - CGBJ's list of two title shader container digests, which switched the wave-wide form on for those two programs, is removed. It was a title-specific choice. VICTORY_WAVE_WIDE=1 still turns the form on for every program when the policy is off.
  - The form a translation is kept in is noted for the caller (xcl::take_form).

capa86-m1a-xcl-binding (the XCL binding, a second resource binding beside CGBJ's own, selected per call by xcl::translate_compute; with the binding not selected every change below does nothing, and the census of the 93 test containers gives the same sizes and instruction counts as before):
- src/emit/mod.rs: declares the new module emit::xcl.
- src/resource.rs: track() is CGBJ's function under the name track_plain, and track() passes its result to emit::xcl::tracked, which under the binding turns a failure of the tracer into empty results (the binding traces no descriptor).
- src/emit/shader.rs: three new fields of Shader (view_space, sampler_space, sampler_base) take the place of the constants ROOT_CBV_SPACE and SAMPLER_SPACE in root_uav_handle, sampler_handle and finish, so the binding can put io at u0 space 0 and its static samplers in space 1 from a first register. Their defaults are CGBJ's values. finish() also hands the module's instruction counts to xcl::note_census under the binding.
- src/emit/lower.rs: declares the child module xcl and calls into it at these seams: lower() (no host thread count under the binding), the Lower struct and Lower::new (the binding's state), run() (the entry code, and the statistics at the end), inputs() (an entry SGPR is its io dword), inst() (refusals by name, and the flat forms reach flat()), flat(), smem(), buffer() and handle() (each hands over to the binding at its top, smem() and flat() after the loads from the code), format_load() (takes the raw words from the binding), and image() (a descriptor is the registers' values, in two places).

capa86-m1-xcl-binding (page groups, format forms and the reported refusal; with the binding not selected the census of the 93 test containers gives the same results, sizes, refusal reasons, forms and instruction counts as at M0):
- src/emit/lower.rs: run() calls the binding's x_ready before the instructions of a block and after each instruction, where it resolves the page groups whose bases have just been made. Without the binding x_ready returns at once.
- src/emit/mod.rs: emit() hands the result of each retry in another wave form to emit::xcl::Refusals, and where it returned the first attempt's refusal, or the jump-threaded form's, it returns what Refusals reports: the last refusal of a retry that was not about the wave's form, when the refusal that would be returned is about the wave's form. The words of the refusal "no wave32 lowering for this wave64 program" moved to the constant emit::xcl::NO_WAVE32.
- src/structure.rs: a new kind of lane atom, Atom::Launch (the lane is inside the dispatch's thread limits). Under the XCL binding with thread limits, EXEC at launch is that atom instead of every lane, so lanes past the limits run with EXEC clear as on the console. Without the binding the atom is never made.
- src/emit/lower.rs: atom() lowers Atom::Launch to the binding's test of the limits (x_launch).
- src/emit/bitcode/opt.rs is unmodified. A change to it was built and measured in M1 (reads of views without the UAV bit reused across blocks) and taken out again, because it changed no container.

capa86-m1v2-xcl-binding (the numeric class of an image, and in src/emit/lower/xcl.rs the byte offset of the second to fourth piece of a grouped s_load_dwordx8 or x16; with the binding not selected nothing changes):
- src/emit/lower.rs: image() hands image_store, image_load* and the gathers that stay gathers to the binding (x_image_store, x_image_read), which picks the view's component type from the T#'s format at run time as xsemu's translator does. Under the binding image_atomic_smin and smax take a view of signed integers.
- src/xcl.rs: BINDING_VERSION 3.

capa87-cgbj-upstream-wave-fixes (fixes DoubleGhosty/cool_fox, the continuation of CGBJ, made to the same files after f4b48ad; each hunk is marked `xsemu: from DoubleGhosty/cool_fox <commit>`; tests from those commits that need code this crate lacks are left out):
- src/structure.rs, src/emit/lower.rs: 9a4aa8d3ea9fb6af2af1e54a040daaa1abc7443e (DoubleGhosty, DoubleGhosty/cool_fox#66) and bd8b4f524a1c63546ba0211c4d2c56c93605bcaf (steven-seagull27, #78). A branch on the SCC of a lane-mask operation (`s_and_b64 s, vcc, exec` then `s_cbranch_scc0/1`) whose arms keep EXEC is the whole console wave's, and stores in its arms keep their EXEC guard. Before, structure split it per lane: on a host of 32-lane waves 188 of 576 words were wrong (RTX 5080, WARP, RTX A3000), and with the store in the arms 127 of 576. Under the exact wave policy such a program now takes a retry form or is refused for XCL's translator, where it translated split per lane in the first form before (src/emit/xcl_scc_tests.rs).
- src/emit/lower.rs: caec503 (an array texture query's unused LOD slot is undefined), 87ed9d3 (full-width unsigned integer conversions round once), e7a40a4 (SDWA 16-bit sign extension; its BC7 part is not in this crate).
- src/emit/wave.rs: 13e600d (inline float constants of v_add_f16 decoded as f16 in the whole-wave simulation, not as f32 bits); its test leaves out the identity case, which needs Shape::Half.
- src/emit/mod.rs: declares the test module emit::xcl_scc_tests (new: fixtures and a test from bd8b4f52's wave64_tests.rs, CGBJ's native() helper, and an exact-policy test of xsemu's).

capa86-m1b-xcl-binding (format forms of halves, fmax and fmin as a compare-exchange loop, image stores of some channels; with the binding not selected nothing changes, and every compute program M1 v2 translates gives the same container byte for byte):
- src/emit/lower.rs: image() lets an image store of some channels through to the binding's x_image_store under the binding. Without the binding it is refused as before, at the same place.

capa86-m1c-xcl-binding (an S# word made by s_brev_b32 of a constant is a constant, as xsemu's translator takes it, so such a sampler is static and not read from the sampler heap; under the binding a wave64 program is not translated as two independent host waves unless VICTORY_XCL_LOCAL_WAVES=1):
- src/emit/lower/xcl.rs: x_const follows s_brev_b32 of an inline constant or a literal.
- src/emit/mod.rs: under the binding emit() skips the local waves forms (with_local_waves, with_local_waves_any) unless asked for, and reports "a wave64 program as two independent host waves under the XCL binding" where the idiom form would have applied. Without the binding nothing changes.
- src/emit/xcl.rs: the refusal and the switch (local_waves).
- src/xcl.rs: BINDING_VERSION 4.

capa86-m1d-xcl-binding (the packed form of a wave64 compute program on 32-lane waves, under the binding only; with the binding not selected nothing changes, and every compute program M1c translates gives the same container byte for byte):
- src/emit/mod.rs: emit() tries the packed form (emit::xcl::packed) as the last retry, before the exact policy stops, and reports it as "packed wave64". A first attempt refused for a lane mask read as data goes through the retries under the binding (emit::xcl::packed_after).
- src/emit/xcl.rs: packed() structures every block with a wave-wide branch as the console wave's (structure::structure_wide), then lowers in the packed form. It refuses a group that ends inside a console wave when the host gives no thread limits, and a packed container of 1 MiB or more ("packed wave64 output over 1 MiB"; VICTORY_XCL_PACKED_MAX_BYTES sets the cap, 0 for none). translate() reports the packed form's numthreads.
- src/emit/lower.rs: declares the child module packed and calls into it at these seams: the Lower struct and Lower::new (the form's state, set up last), run() (the high half's inputs and joins, and inst() and section_inst() through the form), the exchange setup of two host waves (skipped in the form), mask_dword() (the ballot of both halves), the SCC of a mask operation (some lane of either half), wave_wide() (one wave op over both halves), lane_bit(), thread_id() and the lane index (the console lane of the half), read_lane(), ds_bpermute_b32, v_readlane_b32 and v_readfirstlane_b32 (a lane in either half), the waterfall read and compare (run as on the console), a lane-read of a section's value, the high word of the ballot in v_permlane16_b32 and v_permlanex16_b32 (a half's own 32 lanes), and the exit of a block (a branch that is still per lane is refused).
- src/emit/lower/xcl.rs: the thread limits are a function (x_limits), made once per half in the packed form, where a console lane past the group is outside them.
- src/emit/shader.rs: a new field host_threads, the declared numthreads when they differ from the console's, used for the metadata and PSV0.
- src/emit/xcl_scc_tests.rs: tests of the packed form on the SCC fixtures, a ballot over both halves, a lane read past 32 and the declared numthreads.
- src/xcl.rs: BINDING_VERSION 5, PACKED_FORM, packed_threads.

## Known content

- Comments in several copied files name title shader programs.
