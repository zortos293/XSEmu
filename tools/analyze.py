#!/usr/bin/env python3
"""Bottleneck analyzer for XSEmu run logs (master plan Phase A, stdlib only).

Reads one run directory per launch (~/.capy/work/runs/play-*): xcl.log.gz, xcl.log, or
SUMMARY.txt when the run's own log is gone (marked partial), plus memwatch.log,
ramfloor.tsv and gpu.csv. Prints one markdown section per run, a comparison table and a
top-5 verdict.

Usage: python tools/analyze.py [--runs-dir DIR] [--out FILE] [run ...]
"""
import argparse, csv, glob, gzip, os, re, statistics, sys
from collections import Counter, defaultdict

PREFIX = re.compile(r"^\[\s*([0-9.]+)\]\[\s*[0-9]+\]\[[A-Z ]+\]\s?(.*)$")
STATUS = re.compile(r"^status (\d+) s: ([0-9.]+) fps \| pipelines (\d+) of (\d+) built, (\d+) pending, (\d+) failed \| RAM ([0-9.]+) GB committed, ([0-9.]+) GB free \| VRAM (n/a|([0-9.]+) of ([0-9.]+) GB)")
CENSUS = re.compile(r"^draw census: frame (\d+): (\d+) draws \((\d+) skipped\), (\d+) with a depth-writing pipeline, (\d+) without, (\d+) unknown; (\d+) indirect, (\d+) dispatches")
FAIL_GROUP = re.compile(r"^D3D12\.X: failed pipelines at (\d+) s: (\d+) in all: (.*)$")
FAIL_ONE = re.compile(r"^D3D12\.X: failed pipeline #(\d+) \(([a-z]+), key ([0-9a-f]+), (\w+) ([0-9a-f]*)\): (.*)$")
SKIP_ONE = re.compile(r"^D3D12\.X: (\w+\(Dispatch\)|draw|dispatch) skipped \(#(\d+) of pipeline (0x[0-9a-f]+), key ([0-9a-f]+), (\w+) ([0-9a-f]+)\): (.*)$")
ASYNC = re.compile(r"^D3D12\.X: async pipelines: (\d+) pending \(oldest ([0-9.]+) s\), (\d+) built, (\d+) failed; (\d+) draws/dispatches skipped in the last 2 s \((\d+) so far\)")
MESH = re.compile(r"^mesh pipelines: (\d+) native, (\d+) two-pass, (\d+) driver compiles; mesh calls: (\d+) native, (\d+) two-pass, (\d+) skipped")
TWOPASS = re.compile(r"^D3D12\.X: indirect mesh calls in two passes: (\d+) commands over their room of (\d+) thread groups so far \((\d+) groups cut; the most one command asked for (\d+); (\d+) calls checked")
CACHES = re.compile(r"^caches: translation (\d+) from disk, (\d+) made; DXC (\d+) hits, (\d+) compiled in (\d+) thread-s; driver (\d+) pipelines in (\d+) thread-s; direct DXIL backend (\d+) compiled and validated, (\d+) left to DXC \((\d+) unsupported, (\d+) failed, (\d+) invalid\)")
MEMORY = re.compile(r"^memory: (\d+) MB committed by the process; video memory (\d+) MB local, (\d+) MB non-local; target pages (\d+) MB")
PRESENT = re.compile(r"^present: (frame|plane0) (\d+) \((\d+)x(\d+), format (\d+)\).*?(\d+) of (\d+) samples not black")
WAITED = re.compile(r"^D3D12\.X: a waited-for (\w+) pipeline: (\d+) ms in the queue, (\d+) ms to build.*?; (\w+) ([0-9a-f]+) (\d+) B of DXIL; slot wait (\d+) ms, driver (\d+) ms")
STALL = re.compile(r"^live stall: driver build \(slot held\) ([0-9.]+)s")
SUBMIT_WATCH = re.compile(r"^submit watch: (q\d) submission #(\d+) has not finished ([0-9.]+) s")
SHADER_CODE = re.compile(r"^\[SHADER\] (\w+) code ([0-9a-f]+): (\S+), (\d+) guest instructions")
MEMDIAG = re.compile(r"^memdiag: ([a-z][a-z /]{3,40}):")
STANDIN = re.compile(r"^D3D12\.X: (title patch [^()]{3,90})")
ACTIVITY = re.compile(r"^activity \((\d+) s\): (\d+) presents \((\d+) dropped[^)]*\), (\d+) submits, (\d+) draws, (\d+) dispatches")
XGAME = re.compile(r"^XGameRuntime call #(\d+): (\w+)")
TEXTURE_ALIAS = re.compile(r"^texture aliases: (\d+) placements at (\d+) shared addresses now")
PSO_CREATE = re.compile(r"^D3D12\.X: (\w+) PSO with the translated console (\w+) ([0-9a-f]+) \((\d+) B DXIL, (\d+) ms in the driver, (\d+) ms waiting for a slot")
NUM = re.compile(r"0x[0-9a-f]+|[0-9a-f]{16}|\d+")

def norm(reason):
    return NUM.sub("#", reason).strip()[:120]

class Run:
    def __init__(self, path):
        self.path = path
        self.tag = os.path.basename(path)
        self.partial = False
        self.summary_text = None
        self.status, self.census = [], []
        self.fail_groups, self.fails, self.skips = [], [], []
        self.async_lines, self.mesh, self.twopass, self.caches = [], [], [], []
        self.memory, self.presents, self.waited = [], [], []
        self.stalls, self.submit_watch = [], []
        self.shaders, self.shader_examples = Counter(), {}
        self.memdiag, self.memdiag_last = Counter(), {}
        self.standins, self.standin_first = Counter(), {}
        self.ray_off = 0
        self.device_removed = 0
        self.dred = 0
        self.xgame = Counter()
        self.alias, self.activity, self.pso_create = [], [], []
        self.log_bytes = self.log_lines = 0
        self.last_t = 0.0
        self.first_t = None
        self.version = None
        self.examples = defaultdict(list)
        self.untouched = Counter()

    def read(self):
        log = None
        for name in ("xcl.log.gz", "xcl.log"):
            p = os.path.join(self.path, name)
            if os.path.exists(p):
                log = p
                break
        if log is None:
            s = os.path.join(self.path, "SUMMARY.txt")
            if os.path.exists(s):
                self.partial = True
                self.summary_text = open(s, encoding="utf-8", errors="replace").read()
            return
        opener = gzip.open if log.endswith(".gz") else open
        with opener(log, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                self.log_bytes += len(line)
                self.log_lines += 1
                m = PREFIX.match(line)
                if not m:
                    continue
                try:
                    t = float(m.group(1))
                except ValueError:
                    continue
                self.last_t = t
                if self.first_t is None:
                    self.first_t = t
                self.line(t, m.group(2).strip())

    def world_entry(self):
        for c in self.census:
            if c[2] >= 300:
                return c
        return None

    def world_t(self):
        """World entry: the census volume rule, else the first stand-in, else the first present
        more than a quarter of which is not black. The rule that fired is reported."""
        w = self.world_entry()
        if w:
            return w[0]
        for t, kind, idx, ww, hh, nz, tot in self.presents:
            if tot and nz * 4 >= tot:
                return t
        if self.standin_first:
            return min(v[0] for v in self.standin_first.values())
        return None

    def world_rule(self):
        if self.world_entry():
            return "census volume (300 draws)"
        for t, kind, idx, ww, hh, nz, tot in self.presents:
            if tot and nz * 4 >= tot:
                return "first present over a quarter non-black"
        if self.standin_first:
            return "first title stand-in"
        return "none"

    def inworld_census(self):
        wt = self.world_t()
        return [c for c in self.census if wt is not None and c[0] >= wt]

    def inworld_status(self):
        wt = self.world_t()
        st = [s for s in self.status if wt is not None and s[0] >= wt]
        return st or self.status

    def fps_stats(self):
        vals = [s[1] for s in self.inworld_status() if s[1] > 0]
        if not vals:
            return None
        ms = [1000.0 / v for v in vals]
        p95 = sorted(vals)[max(0, int(len(vals) * 0.95) - 1)]
        return {"n": len(vals), "fps_median": statistics.median(vals), "fps_p95": p95,
                "fps_max": max(vals), "ms_median": statistics.median(ms),
                "ms_p95": sorted(ms)[max(0, int(len(ms) * 0.95) - 1)]}

    def peaks(self):
        p = {}
        if self.status:
            p["ram_commit_peak_gb"] = max(s[6] for s in self.status)
            p["vram_peak_gb"] = max((s[8] or 0) for s in self.status)
            p["vram_budget_gb"] = max((s[9] or 0) for s in self.status)
        if self.memory:
            p["commit_peak_mb"] = max(m[0] for m in self.memory)
            p["vram_local_peak_mb"] = max(m[1] for m in self.memory)
            p["vram_nonlocal_peak_mb"] = max(m[2] for m in self.memory)
        return p

    def drop_totals(self):
        rt, other = Counter(), Counter()
        for stage, key, h, reason in self.fails:
            (rt if "ray tracing off" in reason else other)[norm(reason)] += 1
        for kind, idx, pso, key, stage, h, reason in self.skips:
            (rt if "ray tracing off" in reason else other)[norm(reason)] += 1
        return rt, other

    def line(self, t, msg):
        if msg.startswith("draw census: frame"):
            m = CENSUS.match(msg)
            if m:
                g = m.groups()
                self.census.append((t, int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]), int(g[6]), int(g[7])))
            return
        if msg.startswith("draw census:") or msg.startswith("GPU dispatch capture") or msg.startswith("GPU timing"):
            return
        if msg.startswith("status "):
            m = STATUS.match(msg)
            if m:
                g = m.groups()
                self.status.append((t, float(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]),
                                    float(g[6]), float(g[7]),
                                    float(g[9]) if g[9] else 0.0,
                                    float(g[10]) if g[10] else 0.0))
            return
        if msg.startswith("D3D12.X: failed pipeline"):
            m = FAIL_ONE.match(msg)
            if m:
                g = m.groups()
                self.fails.append((g[1], g[2], g[4], g[5]))
                return
            m = FAIL_GROUP.match(msg)
            if m:
                self.fail_groups.append((t, int(m.group(1)), int(m.group(2)), m.group(3)))
            return
        if msg.startswith("D3D12.X: "):
            if " skipped (#" in msg:
                m = SKIP_ONE.match(msg)
                if m:
                    g = m.groups()
                    self.skips.append((g[0], int(g[1]), g[2], g[3], g[4], g[5], g[6]))
                return
            if msg.startswith("D3D12.X: async pipelines:"):
                m = ASYNC.match(msg)
                if m:
                    self.async_lines.append((t,) + tuple(float(x) for x in m.groups()))
                return
            if msg.startswith("D3D12.X: a waited-for"):
                m = WAITED.match(msg)
                if m:
                    g = m.groups()
                    self.waited.append((int(g[1]), int(g[2]), g[3], g[4], int(g[5]), int(g[6]), int(g[7])))
                return
            if msg.startswith("D3D12.X: title patch"):
                m = STANDIN.match(msg)
                if m:
                    name = m.group(1)
                    self.standins[name] += 1
                    self.standin_first.setdefault(name, (t, msg[:200]))
                return
            if msg.startswith("D3D12.X: indirect mesh calls in two passes:"):
                m = TWOPASS.match(msg)
                if m:
                    self.twopass.append(tuple(int(x) for x in m.groups()))
                return
            if " PSO with the translated console " in msg:
                m = PSO_CREATE.match(msg)
                if m:
                    g = m.groups()
                    self.pso_create.append((g[0], g[1], g[2], int(g[3]), int(g[4]), int(g[5])))
                return
        if msg.startswith("mesh pipelines:"):
            m = MESH.match(msg)
            if m:
                self.mesh.append(tuple(int(x) for x in m.groups()))
            return
        if msg.startswith("caches: translation"):
            m = CACHES.match(msg)
            if m:
                self.caches.append(tuple(int(x) for x in m.groups()))
            return
        if msg.startswith("memory: "):
            m = MEMORY.match(msg)
            if m:
                self.memory.append(tuple(int(x) for x in m.groups()))
            return
        if msg.startswith("present: "):
            m = PRESENT.match(msg)
            if m:
                g = m.groups()
                self.presents.append((t, g[0], int(g[1]), int(g[2]), int(g[3]), int(g[5]), int(g[6])))
            return
        if msg.startswith("live stall: driver build"):
            m = STALL.match(msg)
            if m:
                self.stalls.append((t, float(m.group(1)), msg[:160]))
            return
        if msg.startswith("submit watch:"):
            m = SUBMIT_WATCH.match(msg)
            if m:
                self.submit_watch.append((t, float(m.group(3)), msg[:180]))
            return
        if msg.startswith("[SHADER] "):
            m = SHADER_CODE.match(msg)
            if m:
                self.shaders[(m.group(1), m.group(3))] += 1
                self.shader_examples.setdefault((m.group(1), m.group(3)), msg[:180])
            return
        if msg.startswith("memdiag: "):
            m = MEMDIAG.match(msg)
            if m:
                self.memdiag[m.group(1)] += 1
                self.memdiag_last[m.group(1)] = msg[:220]
            return
        if "ray tracing off" in msg:
            self.ray_off += 1
        if "device removed" in msg.lower():
            self.device_removed += 1
        if msg.startswith("D3D12: DRED"):
            self.dred += 1
        if msg.startswith("XGameRuntime call"):
            m = XGAME.match(msg)
            if m:
                self.xgame[m.group(2)] += 1
            return
        if msg.startswith("texture aliases: "):
            m = TEXTURE_ALIAS.match(msg)
            if m:
                self.alias.append((t, int(m.group(1)), int(m.group(2))))
            return
        if msg.startswith("activity ("):
            m = ACTIVITY.match(msg)
            if m:
                g = m.groups()
                self.activity.append((t, int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5])))
            return
        if msg.startswith("XCL "):
            if self.version is None:
                self.version = msg[:60]
            return
        low = msg.lower()
        for phrase in ("untranslated", "not translated", "inexact", "unimplemented", "no translator"):
            if phrase in low:
                self.untouched[phrase] += 1
                if len(self.examples[phrase]) < 3:
                    self.examples[phrase].append(msg[:200])
                break

FENCE = chr(96) * 3

def aux(run):
    """memwatch, ramfloor and gpu.csv for a run directory, when they exist."""
    out = {}
    p = os.path.join(run.path, "memwatch.log")
    if os.path.exists(p):
        low = None
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.search(r"min free (\d+) MB", line)
            if m:
                v = int(m.group(1))
                low = v if low is None else min(low, v)
        out["memwatch_min_mb"] = low
    p = os.path.join(run.path, "ramfloor.tsv")
    if os.path.exists(p):
        vals = []
        for line in open(p, encoding="utf-8", errors="replace"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[1].isdigit():
                vals.append(int(f[1]))
        if vals:
            out["ramfloor_min_mb"] = min(vals)
            out["ramfloor_samples"] = len(vals)
    p = os.path.join(run.path, "gpu.csv")
    if os.path.exists(p):
        util, used = [], []
        try:
            for row in csv.DictReader(open(p, encoding="utf-8", errors="replace")):
                u = row.get(" utilization.gpu [%]") or row.get("utilization.gpu [%]")
                mm = row.get(" memory.used [MiB]") or row.get("memory.used [MiB]")
                if u and u.strip() not in ("", "[N/A]"):
                    try:
                        util.append(float(u.strip().rstrip(" %")))
                    except ValueError:
                        pass
                if mm and mm.strip() not in ("", "[N/A]"):
                    try:
                        used.append(float(mm.strip()))
                    except ValueError:
                        pass
        except Exception:
            pass
        if util:
            out["gpu_util_median"] = statistics.median(util)
            out["gpu_util_p95"] = sorted(util)[max(0, int(len(util) * 0.95) - 1)]
        if used:
            out["gpu_mem_peak_mib"] = max(used)
    return out

def lane_env(tag, work):
    """The XCL_* a run's lane asked for: play128 -> <work>/play128-run/launch.env and start83.sh.

    The lane directories live beside the runs directory (both under ~/.capy/work), not beside
    this file: the tool is in the repository and the runs are not."""
    lane = os.path.join(work, tag.replace("-", "") + "-run")
    env = {}
    for name in ("launch.env", "start83.sh"):
        p = os.path.join(lane, name)
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"(XCL_[A-Z0-9_]+)=([^\s\"';]+)", text):
            env.setdefault(m.group(1), m.group(2))
    return env

FORBIDDEN = {"XCL_IMAGE_WRITEBACK": "0"}

def section(r, auxd, env):
    L = []
    a = L.append
    a("## " + r.tag + (" (partial: log missing, SUMMARY.txt only)" if r.partial else ""))
    a("")
    if r.partial:
        a("The run's own log is gone; this section quotes its SUMMARY.txt and the auxiliary files.")
        a("")
        a(FENCE)
        a(r.summary_text.strip())
        a(FENCE)
        a("")
        return "\n".join(L)
    a("- log: %.1f MB, %d lines, %.0f s of run time, %s" % (r.log_bytes / 1e6, r.log_lines, r.last_t - (r.first_t or 0), r.version or "version line missing"))
    if env:
        items = sorted(env.items())
        a("- lane settings: " + ", ".join("%s=%s" % (k, v) for k, v in items[:18]) + (" (+%d more)" % (len(items) - 18) if len(items) > 18 else ""))
        for k, want in FORBIDDEN.items():
            if k in env and env[k] != want:
                a("- **WARNING** %s=%s (expected %s)" % (k, env[k], want))
    w = r.world_entry()
    wt = r.world_t()
    if w:
        a("- world entry: first census frame with 300 or more draws at %.1f s (frame %d, %d draws)" % (w[0], w[1], w[2]))
    elif wt is not None:
        a("- world entry: inferred at %.1f s from %s" % (wt, r.world_rule()))
    else:
        a("- world entry: **not seen**")
    if w or wt is not None:
        a("- world entry rule: %s" % r.world_rule())
    for name, n in r.standins.most_common(6):
        a("- stand-in engaged %dx: %s (first at %.0f s)" % (n, name[:110], r.standin_first[name][0]))
    fs = r.fps_stats()
    iw = r.inworld_census()
    if fs:
        a("- in-world steady state: %d status samples, fps median %.1f, p95 %.1f, max %.1f; ms/frame median %.0f, p95 %.0f" % (fs["n"], fs["fps_median"], fs["fps_p95"], fs["fps_max"], fs["ms_median"], fs["ms_p95"]))
    if iw:
        draws = [c[2] for c in iw]
        disp = [c[8] for c in iw]
        a("- in-world census: %d frames, draws median %d max %d, dispatches median %d max %d, skipped %d" % (len(iw), statistics.median(draws), max(draws), statistics.median(disp), max(disp), sum(c[3] for c in iw)))
    rt, other = r.drop_totals()
    a("- drop attribution: RT gate %d lines (%d reasons), other %d lines (%d reasons)" % (sum(rt.values()), len(rt), sum(other.values()), len(other)))
    for reason, n in other.most_common(20):
        a("  - %dx %s" % (n, reason))
    stages = Counter(f[0] for f in r.fails)
    if stages:
        a("  - failed pipeline stages: " + ", ".join("%s=%d" % kv for kv in stages.most_common()))
    hashes = Counter(f[2] for f in r.fails if f[2])
    if hashes:
        a("  - top failed hashes: " + ", ".join("%s(%d)" % (h, n) for h, n in hashes.most_common(10)))
    a("- stability: device-removed lines %d, slow submissions over 1 s %d, driver-build stalls %d" % (r.device_removed, sum(1 for s in r.submit_watch if s[1] > 1.0), len(r.stalls)))
    for t, s, text in sorted(r.submit_watch, key=lambda x: -x[1])[:5]:
        a("  - %.1f s slow submission at %.0f s: %s" % (s, t, text[:130]))
    for t, s, text in sorted(r.stalls, key=lambda x: -x[1])[:5]:
        a("  - %.1f s driver build stall at %.0f s: %s" % (s, t, text[:130]))
    for k, v in sorted(r.memdiag_last.items())[:6]:
        a("  - memdiag %s: %s" % (k, v[:150]))
    p = r.peaks()
    mem = []
    for k in ("memwatch_min_mb", "ramfloor_min_mb"):
        if auxd.get(k) is not None:
            mem.append("%s %d MB" % (k, auxd[k]))
    if p.get("ram_commit_peak_gb"):
        mem.append("commit peak %.1f GB" % p["ram_commit_peak_gb"])
    if p.get("vram_peak_gb"):
        mem.append("VRAM peak %.1f of %.1f GB" % (p["vram_peak_gb"], p["vram_budget_gb"]))
    if p.get("vram_local_peak_mb"):
        mem.append("video local peak %d MB, non-local %d MB" % (p["vram_local_peak_mb"], p["vram_nonlocal_peak_mb"]))
    if auxd.get("gpu_util_median") is not None:
        mem.append("GPU util median %.0f%%, p95 %.0f%%" % (auxd["gpu_util_median"], auxd["gpu_util_p95"]))
    a("- memory and GPU: " + "; ".join(mem))
    if r.caches:
        c = r.caches[-1]
        a("- shaders: translation %d from disk / %d made; DXC %d hits / %d compiled in %d thread-s; driver %d pipelines in %d thread-s; DXIL backend %d ok, %d left, %d unsupported, %d failed, %d invalid" % (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]))
    if r.mesh:
        m = r.mesh[-1]
        a("- mesh: %d native, %d two-pass pipelines, %d driver compiles; calls %d native, %d two-pass, %d skipped" % (m[0], m[1], m[2], m[3], m[4], m[5]))
    if r.twopass:
        tp = r.twopass[-1]
        a("- two-pass room: %d commands over %d thread groups, %d cut, largest ask %d, %d calls checked" % (tp[0], tp[1], tp[2], tp[3], tp[4]))
    if r.pso_create:
        slow = sorted(r.pso_create, key=lambda x: -x[4])[:10]
        a("- slowest PSO driver compiles: " + ", ".join("%s %s %d ms" % (s[0][:4], s[2][:8], s[4]) for s in slow))
    if r.waited:
        q = [w[0] for w in r.waited]
        b = [w[1] for w in r.waited]
        a("- waited-for pipelines: %d, queue ms median %.0f max %d, build ms median %.0f max %d" % (len(r.waited), statistics.median(q), max(q), statistics.median(b), max(b)))
    if r.presents:
        frac = [100.0 * pr[5] / pr[6] for pr in r.presents if pr[6]]
        if frac:
            a("- picture: %d presents sampled, not-black median %.1f%%, min %.1f%%, last %.1f%%" % (len(frac), statistics.median(frac), min(frac), frac[-1]))
    if r.untouched:
        a("- untranslated/inexact mentions: " + ", ".join("%s=%d" % kv for kv in r.untouched.most_common()))
        for phrase, ex in r.examples.items():
            for e in ex[:2]:
                a("  - %s: %s" % (phrase, e))
    a("")
    return "\n".join(L)

def table(rows):
    L = ["| run | world@ | fps med | ms p95 | draws med | RT-gated | other drops | pending peak | RAM min | VRAM peak | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, auxd in rows:
        fs = r.fps_stats()
        iw = r.inworld_census()
        w = r.world_entry()
        rt, other = r.drop_totals()
        pend = max((int(a[0]) for a in r.async_lines), default=0)
        ram = auxd.get("memwatch_min_mb")
        pk = r.peaks()
        note = "partial log" if r.partial else ""
        if r.device_removed:
            note = (note + "; device removed").strip("; ")
        L.append("| %s | %s | %s | %s | %s | %d | %d | %d | %s | %s | %s |" % (
            r.tag,
            "%.0f s" % w[0] if w else "-",
            "%.1f" % fs["fps_median"] if fs else "-",
            "%.0f" % fs["ms_p95"] if fs else "-",
            "%d" % statistics.median([c[2] for c in iw]) if iw else "-",
            sum(rt.values()), sum(other.values()), pend,
            "%d MB" % ram if ram is not None else "-",
            "%.1f/%.1f" % (pk.get("vram_peak_gb", 0), pk.get("vram_budget_gb", 0)) if pk.get("vram_peak_gb") else "-",
            note))
    return "\n".join(L)

def verdict(rows):
    """Five costs the numbers point at, aggregated across runs, each with what to read next.
    A cause appears once; the run named is the one where it is worst."""
    real = [(r, a) for r, a in rows if not r.partial]
    if not real:
        return []
    items = []
    rt = [(len(set(h for st, key, h, reason in r.fails if "ray tracing off" in reason) |
                set(h for kind, idx, pso, key, st, h, reason in r.skips if "ray tracing off" in reason)), r.tag)
          for r, a in real]
    rt.sort(reverse=True)
    if rt and rt[0][0]:
        items.append((rt[0][0] * 40, "rt",
                      "RT gate: up to %d distinct shader hashes held back by XCL_RT_GATE in %s, %d runs affected (hle/d3d12x/raytracing.rs); the probe and ray terms of the lighting resolve are unbuilt"
                      % (rt[0][0], rt[0][1], sum(1 for n, t in rt if n > 50))))
    pend = sorted(((max((int(x[0]) for x in r.async_lines), default=0), r.tag) for r, a in real), reverse=True)
    if pend and pend[0][0] > 100:
        items.append((pend[0][0] * 2, "pending",
                      "compile storm: %d pipelines pending at once in %s (admission and compile slots: real.rs admission, pso_async.rs); the frame waits on driver builds"
                      % (pend[0][0], pend[0][1])))
    stalls = sorted(((max((x[1] for x in r.stalls), default=0.0), r.tag) for r, a in real), reverse=True)
    if stalls and stalls[0][0] > 5:
        items.append((stalls[0][0] * 3, "stall",
                      "driver-build stall: %.0f s with the slot held in %s (pso_async slot policy; the submit queue drains while it builds)" % (stalls[0][0], stalls[0][1])))
    fps = []
    for r, a in real[-6:]:
        fs = r.fps_stats()
        if fs:
            fps.append((fs["fps_median"], r.tag, fs["ms_p95"]))
    if fps:
        fps.sort()
        items.append(((20 - min(fps[0][0], 19.9)) * 25, "fps",
                      "frame rate: %.1f fps median in world in %s (%.0f ms p95); read the CPU profile before changing code" % (fps[0][0], fps[0][1], fps[0][2])))
    vram = sorted((((r.peaks().get("vram_peak_gb", 0) / max(1e-9, r.peaks().get("vram_budget_gb", 0))), r.tag, r.peaks()) for r, a in real), reverse=True)
    if vram and vram[0][0] > 0.9:
        items.append(((vram[0][0] - 0.9) * 1000, "vram",
                      "VRAM: %.1f of %.1f GB in %s (texture residency and placed resources; POOL_GATE territory)" % (vram[0][2]["vram_peak_gb"], vram[0][2]["vram_budget_gb"], vram[0][1])))
    ram = sorted(((a.get("memwatch_min_mb") or 99999, r.tag) for r, a in real))
    if ram and ram[0][0] < 2000:
        items.append(((2000 - ram[0][0]) / 10.0, "ram",
                      "free RAM: down to %d MB in %s (memwatch floor 1024; unloading and admission floors are the levers)" % (ram[0][0], ram[0][1])))
    drops = sorted(((sum(r.drop_totals()[1].values()), r.tag) for r, a in real), reverse=True)
    if drops and drops[0][0]:
        items.append((drops[0][0] * 3, "drops", "non-RT drops: %d lines in %s; see its section for the reason and stage" % (drops[0][0], drops[0][1])))
    black = []
    for r, a in real[-6:]:
        fr = [100.0 * pr[5] / pr[6] for pr in r.presents if pr[6]]
        if fr:
            black.append((statistics.median(fr), r.tag))
    if black:
        black.sort()
        items.append(((80 - black[0][0]), "picture", "picture: %s shows a median of %.0f%% non-black present samples; the picture work (probes, ray resolve, shadow writes) is the gate on everything else" % (black[0][1], black[0][0])))
    items.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for score, key, text in items:
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= 5:
            break
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(description="Bottleneck analyzer over XSEmu run logs.")
    ap.add_argument("runs", nargs="*", help="run directory names under --runs (default: all)")
    ap.add_argument("--runs-dir", dest="runs_dir", default=os.path.join(os.path.expanduser("~"), ".capy", "work", "runs"))
    ap.add_argument("--out", default=None, help="write the markdown report here as well")
    ap.add_argument("--partial-note", default="a run whose log is gone is marked partial and quoted from its SUMMARY.txt")
    args = ap.parse_args(argv)
    names = args.runs
    def order(d):
        b = os.path.basename(d)
        m = re.match(r"play-([0-9]+)(.*)$", b)
        return (int(m.group(1)), m.group(2)) if m else (10 ** 9, b)
    dirs = sorted(glob.glob(os.path.join(args.runs_dir, "play-*")), key=order)
    if names:
        dirs = [d for d in dirs if os.path.basename(d) in names]
    runs, rows = [], []
    work = os.path.dirname(os.path.abspath(args.runs_dir))
    for d in dirs:
        r = Run(d)
        r.read()
        if not r.census and not r.status and not r.summary_text:
            continue
        runs.append(r)
        rows.append((r, aux(r)))
        sys.stderr.write("%s: %d lines, %d census frames, %d status samples%s\n" % (r.tag, r.log_lines, len(r.census), len(r.status), ", partial" if r.partial else ""))
    L = ["# Run analysis baseline", "",
         "Generated by tools/analyze.py over %d run directories under %s." % (len(runs), args.runs_dir),
         "A run whose xcl.log was overwritten has no full record; %s." % args.partial_note, "",
         "## Comparison", "", table(rows), "",
         "## What the numbers point at", ""]
    v = verdict(rows)
    if v:
        for i, text in enumerate(v, 1):
            L.append("%d. %s" % (i, text))
    else:
        L.append("No bottlenecks cleared the thresholds; read the per-run sections.")
    L += ["", "## Per run", ""]
    for r, auxd in rows:
        L.append(section(r, auxd, lane_env(r.tag, work)))
    text = "\n".join(L) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        sys.stderr.write("wrote %s (%d bytes)\n" % (args.out, len(text)))
    else:
        sys.stdout.write(text)
    return 0

if __name__ == "__main__":
    sys.exit(main())
