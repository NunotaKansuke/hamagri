"""Battle-royale search between the FFT scan and the final polish.

A population of particles lives in the map-frame vector of
:mod:`hamagri.pipeline.hills`.  Each FFT hill contributes a region (tempered
FFT covariance); the search then runs synchronous rounds:

1. seed     -- every hill centre plus random draws from its region are
               screened with one residual evaluation; the best draw of each
               hill becomes a particle.  With parallax, the mirror image
               (u0, alpha) -> (-u0, -alpha) of every hill is seeded as well.
2. descend  -- every live particle takes one damped Gauss-Newton step in the
               timescale chart (u0 tE, rho tE), with VBM values and a
               finite-difference Jacobian.
3. merge    -- two mature particles whose model light curves differ by less
               than ``merge_chi2`` (sum of squared residual differences) and
               whose log s, log q and log tE agree within ``same_dex`` sit in
               the same valley; the worse one is removed.  Equal light curves
               from different binaries (close / wide never merge) are the
               degeneracy and are kept.
4. cull     -- mature particles (older than ``grace`` rounds) beyond the target
               population (shrinking every ``cull_every`` rounds) are removed
               by rank, except the best on each side of s = 1 and the best of
               every hill whose particles have not yet descended
               ``region_grace`` rounds (a narrow valley starts with a poor chi2).
5. refill   -- free slots, and at least ``jumps_min`` every round, are filled
               with random jumps from surviving particles: a region Gaussian
               jump, an alpha jump, an (s, q)-block jump, or a u0 jump of one
               to two source radii (optionally with alpha).  Candidates are
               screened with one residual evaluation; the best of each slot is
               admitted.

When the best chi2 stalls the population is reseeded from the hill regions (up
to ``reseeds`` times, while a reseed still gains).

The best ``finalists`` distinct survivors by coarse chi2 are polished by LM
(a poor coarse fit can still polish to the best), and so is every further
distinct survivor within ``finalist_dchi2``, up to ``finalists_max``, plus the
best survivor of a side of s = 1 that is still missing.  Polished fits merge
into solutions by the same rule as the search.

Profiles: from each of the best ``degenerate_max`` solutions within
``degenerate_dchi2``, rho is scaled (x0.7, x0.35, x1.5) or u0 shifted by
+-rho and held fixed while LM fits the rest, then a free LM follows.  An
inflated rho smears a slightly wrong caustic geometry, and a source on the
wrong side of a caustic is a separate minimum about rho away in u0.

Free parallax: with a parallax set-up, the search holds (piEN, piEE) at the
host values.  Every distinct polished and profiled fit is then refit with piE
free; the fixed-piE fit is kept where it is better.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from .hills import Landscape, map_vector, native_parameters
from .objective import COARSE, FINE, EventModel, Objective, lm, timescale_chart

# (u0, alpha) -> (-u0, -alpha) reflects a static light curve onto itself; in
# the map-frame vector it negates u0 tE and alpha.  Parallax breaks the
# symmetry, so a hill and its mirror image are both searched.
MIRROR = np.diag([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
SEED_TE_MAX = 1.0e4  # the tE guard of the LM objective


@dataclass
class Settings:
    hills: int = 24
    seed_draws: int = 3  # region draws screened per hill besides the centre
    merge_chi2: float = 20.0
    same_dex: float = 0.02
    grace: int = 2
    region_grace: int = 10
    cull_every: int = 3
    population: tuple[int, ...] = (24, 16, 12, 10, 8)
    refill_candidates: int = 3
    jumps_min: int = 2
    jump_scale: float = 1.0
    rounds: int = 60
    stall_rounds: int = 8
    stall_chi2: float = 0.5
    reseeds: int = 3
    wall_budget: float = 420.0  # seconds of search before the polish
    finalists: int = 6
    finalist_dchi2: float = 500.0
    finalists_max: int = 12
    polish_nfev: int = 100
    polish_seconds: float = 120.0
    degenerate_dchi2: float = 100.0  # solutions within this of the best are profiled
    degenerate_max: int = 3
    profiles: tuple[tuple[str, float], ...] = (
        ("rho", 0.7), ("rho", 0.35), ("rho", 1.5), ("u0", -1.0), ("u0", 1.0))
    profile_nfev: int = 60
    profile_seconds: float = 60.0
    free_parallax: bool = True
    free_nfev: int = 400
    free_seconds: float = 600.0
    solution_dchi2: float = 25.0  # distinct fits within this of the best are reported
    timeouts: dict[str, float] = field(default_factory=lambda: {
        "evaluate": 20.0, "descend": 60.0, "polish": 300.0, "profile": 300.0, "free": 900.0})


# --------------------------------------------------------------------------
# worker side: every function takes the event model first

def _evaluate(model: EventModel, v) -> dict[str, Any]:
    """One coarse residual at map-frame vector ``v``."""
    try:
        residual = model.residual(native_parameters(v), COARSE)
        return {"ok": True, "chi2": float(residual @ residual), "residual": residual.astype(np.float32)}
    except Exception as error:  # an invalid jump is simply not admitted
        return {"ok": False, "chi2": math.inf, "error": f"{type(error).__name__}: {error}"}


def _descend(model: EventModel, v, lam: float) -> dict[str, Any]:
    """One damped Gauss-Newton step in the timescale chart from ``v``."""
    try:
        params = native_parameters(v)
        objective = Objective(model, params)
        convert, dz_dy = timescale_chart(objective)
        zero = np.zeros(objective.n)
        r0 = objective(zero)
        if objective.last_error is not None:
            raise ValueError(objective.last_error)
        chi2_0 = float(r0 @ r0)
        jacobian = objective.jacobian(zero) @ dz_dy(zero)
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ r0
        diag = np.maximum(np.diag(hessian), 1e-12 * max(np.max(np.diag(hessian)), 1e-300))
        best = {"v": list(v), "chi2": chi2_0, "residual": r0.astype(np.float32), "accepted": False}
        for _ in range(3):
            try:
                step = -np.linalg.solve(hessian + lam * np.diag(diag), gradient)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            residual = objective(convert(step))
            chi2 = float(residual @ residual)
            if objective.last_error is None and chi2 < chi2_0:
                best = {"v": map_vector(objective.unpack(convert(step))).tolist(), "chi2": chi2,
                        "residual": residual.astype(np.float32), "accepted": True}
                lam = max(lam / 3.0, 1e-7)
                break
            lam *= 4.0
        return {**best, "lambda": lam}
    except Exception as error:  # a broken step leaves the particle where it was
        return {"v": list(v), "chi2": None, "lambda": 10.0 * lam, "accepted": False,
                "error": f"{type(error).__name__}: {error}"}


def _polish(model: EventModel, parameters, nfev, seconds) -> dict[str, Any]:
    return lm(model, parameters, max_nfev=nfev, max_seconds=seconds)


def _profile(model: EventModel, parameters, kind, value, nfev, seconds, polish_nfev,
             polish_seconds) -> dict[str, Any]:
    """Shift one parameter, fit the rest with it held fixed, then a free LM."""
    started = time.perf_counter()
    base = dict(parameters)
    if kind == "rho":
        base["rho"] *= float(value)
    else:
        base["u0"] += float(value) * base["rho"]
    fixed = lm(model, base, max_nfev=nfev, max_seconds=seconds, fixed=kind)
    if fixed["status"] != "ok":
        return fixed
    result = lm(model, fixed["parameters"], max_nfev=polish_nfev, max_seconds=polish_seconds)
    result["wall_seconds"] = time.perf_counter() - started
    return result


def _free(model: EventModel, parameters, nfev, seconds) -> dict[str, Any]:
    start = {"piEN": model.piE[0], "piEE": model.piE[1], **parameters}
    return lm(model, start, max_nfev=nfev, max_seconds=seconds, free_parallax=True)


FUNCTIONS: dict[str, Callable] = {
    "evaluate": _evaluate, "descend": _descend, "polish": _polish, "profile": _profile, "free": _free}


def _worker_loop(conn, model):
    while True:
        message = conn.recv()
        if message is None:
            return
        name, args = message
        conn.send(FUNCTIONS[name](model, *args))


class Pool:
    """Worker processes holding one event model; a job over its timeout is killed.

    VBM can spend minutes on a pathological trial binary; such a job is
    dropped (result ``None``) instead of stalling a whole round.
    """

    def __init__(self, model: EventModel, workers: int, timeouts: dict[str, float]):
        self.model, self.timeouts = model, dict(timeouts)
        self.context = mp.get_context("spawn")
        self.workers = [self._spawn() for _ in range(max(int(workers), 1))]
        self.killed = 0

    def _spawn(self):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_worker_loop, args=(child, self.model), daemon=True)
        process.start()
        return [process, parent]

    def map(self, name: str, jobs: list[tuple]) -> list[Any]:
        from multiprocessing.connection import wait

        timeout = self.timeouts[name]
        results: list[Any] = [None] * len(jobs)
        pending = list(range(len(jobs)))[::-1]
        busy: dict[int, tuple[int, float]] = {}
        while pending or busy:
            for index, worker in enumerate(self.workers):
                if index not in busy and pending:
                    job = pending.pop()
                    worker[1].send((name, jobs[job]))
                    busy[index] = (job, time.monotonic() + timeout)
            ready = wait([self.workers[i][1] for i in busy], timeout=1.0)
            for index in list(busy):
                worker = self.workers[index]
                job, deadline = busy[index]
                if worker[1] in ready:
                    try:
                        results[job] = worker[1].recv()
                    except EOFError:  # the worker died
                        self.workers[index] = self._spawn()
                    del busy[index]
                elif time.monotonic() > deadline:
                    worker[0].kill()
                    worker[0].join()
                    self.workers[index] = self._spawn()
                    self.killed += 1
                    del busy[index]
        return results

    def close(self) -> None:
        for process, conn in self.workers:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# controller side

class Particle:
    __slots__ = ("pid", "v", "chi2", "residual", "lam", "age", "region", "origin", "parent", "history")

    def __init__(self, pid, v, chi2, residual, region, origin, parent=None):
        self.pid, self.v, self.chi2, self.residual = pid, np.asarray(v, dtype=np.float64), float(chi2), residual
        self.lam, self.age, self.region, self.origin, self.parent = 1e-2, 0, region, origin, parent
        self.history = [float(chi2)]

    def record(self) -> dict[str, Any]:
        return {"pid": self.pid, "chi2": self.chi2, "age": self.age, "region": self.region,
                "origin": self.origin, "parent": self.parent, "parameters": native_parameters(self.v)}


def _shape(v) -> np.ndarray:
    """(log s, log q, log tE) of a map-frame vector."""
    return np.array([v[3], v[4], v[2] / np.log(10.0)])


def _same_valley(residual_a, shape_a, residual_b, shape_b, settings: Settings) -> bool:
    if (shape_a[0] < 0.0) != (shape_b[0] < 0.0) or np.max(np.abs(shape_a - shape_b)) > settings.same_dex:
        return False
    diff = residual_a - residual_b
    return float(diff @ diff) < settings.merge_chi2


def _side(particle) -> bool:
    return particle.v[3] >= 0.0


class Battle:
    def __init__(self, pool: Pool, landscape: Landscape, settings: Settings, seed: int = 0,
                 log: Callable[[str], None] | None = None):
        self.pool, self.land, self.settings = pool, landscape, settings
        self.rng = np.random.default_rng(seed)
        self.say = log or (lambda message: None)
        self.regions: list[dict[str, Any]] = []
        self.particles: list[Particle] = []
        self.next_pid = 0
        self.log: list[dict[str, Any]] = []
        self.reseeds: list[dict[str, Any]] = []
        self.epoch_start = 0
        self.region_age: dict[int, int] = {}
        self.counts = {name: 0 for name in FUNCTIONS}

    def _map(self, name, jobs):
        self.counts[name] += len(jobs)
        return self.pool.map(name, jobs)

    def _new(self, v, result, region, origin, parent=None):
        particle = Particle(self.next_pid, v, result["chi2"], result["residual"], region, origin, parent)
        self.next_pid += 1
        return particle

    def _draw(self, region_index, center, scale):
        return center + scale * (self.regions[region_index]["chol"] @ self.rng.standard_normal(7))

    def _same(self, a: Particle, b: Particle) -> bool:
        return _same_valley(a.residual, _shape(a.v), b.residual, _shape(b.v), self.settings)

    # -- stages -----------------------------------------------------------
    def seed(self):
        candidates = []
        for hill in self.land.hills(self.settings.hills):
            center = hill["vector"].copy()
            # A hopeless PSPL fit can put tE above the objective's guard.
            center[2] = min(center[2], np.log(SEED_TE_MAX))
            region = self.land.region(center)
            region["chol"] = np.linalg.cholesky(region["cov"])
            region["center"] = center
            images = [(region, "")]
            if self.pool.model.has_parallax:
                images.append(({**region, "mean": MIRROR @ region["mean"], "cov": MIRROR @ region["cov"] @ MIRROR,
                                "chol": MIRROR @ region["chol"], "center": MIRROR @ center}, "mirror-"))
            for image, prefix in images:
                index = len(self.regions)
                self.regions.append(image)
                candidates.append((index, image["center"], f"{prefix}hill"))
                for _ in range(self.settings.seed_draws):
                    candidates.append((index, self._draw(index, image["center"], self.settings.jump_scale),
                                       f"{prefix}seed-draw"))
        self._admit_seeds(candidates)

    def reseed(self):
        self._admit_seeds([
            (index, self._draw(index, region["center"], self.settings.jump_scale), "reseed-draw")
            for index, region in enumerate(self.regions) for _ in range(self.settings.seed_draws + 1)])

    def _admit_seeds(self, candidates):
        results = self._map("evaluate", [(c[1].tolist(),) for c in candidates])
        best: dict[int, tuple] = {}
        for (index, v, origin), result in zip(candidates, results):
            if result is None or not result["ok"]:
                continue
            if index not in best or result["chi2"] < best[index][1]["chi2"]:
                best[index] = (v, result, origin)
        for index, (v, result, origin) in sorted(best.items()):
            self.particles.append(self._new(v, result, index, origin))
        self._merge()

    def descend(self):
        results = self._map("descend", [(p.v.tolist(), p.lam) for p in self.particles])
        survivors = []
        for particle, result in zip(self.particles, results):
            particle.age += 1
            self.region_age[particle.region] = max(self.region_age.get(particle.region, 0), particle.age)
            if result is None:  # killed: the particle sits on a pathological binary
                continue
            particle.lam = result["lambda"]
            if result["chi2"] is not None:
                particle.v = np.asarray(result["v"], dtype=np.float64)
                particle.chi2, particle.residual = result["chi2"], result["residual"]
            particle.history.append(particle.chi2)
            survivors.append(particle)
        self.particles = survivors

    def _merge(self):
        # Only particles that have descended for ``grace`` rounds are compared:
        # two undescended starts can look alike while lying in different valleys.
        young = [p for p in self.particles if p.age <= self.settings.grace]
        kept: list[Particle] = []
        for particle in sorted((p for p in self.particles if p.age > self.settings.grace), key=lambda p: p.chi2):
            if not any(self._same(particle, other) for other in kept):
                kept.append(particle)
        removed = len(self.particles) - len(kept) - len(young)
        self.particles = kept + young
        return removed

    def _protected(self):
        protected = set()
        for side in (False, True):
            members = [p for p in self.particles if _side(p) == side]
            if members:
                protected.add(min(members, key=lambda p: p.chi2).pid)
        for region in {p.region for p in self.particles}:
            if self.region_age.get(region, 0) < self.settings.region_grace:
                protected.add(min((p for p in self.particles if p.region == region), key=lambda p: p.chi2).pid)
        return protected

    def _target(self, round_index):
        population = self.settings.population
        return population[min((round_index - self.epoch_start) // self.settings.cull_every, len(population) - 1)]

    def cull(self, round_index):
        protected = self._protected()
        young = [p for p in self.particles if p.age <= self.settings.grace]
        mature = sorted((p for p in self.particles if p.age > self.settings.grace), key=lambda p: p.chi2)
        room = max(self._target(round_index) - len(young), 0)
        keep = mature[:room] + [p for p in mature[room:] if p.pid in protected]
        self.particles = keep + young
        return len(mature) - len(keep)

    def _jump(self, parent: Particle):
        region = self.regions[parent.region]
        kind = str(self.rng.choice(["region", "alpha", "sq", "u0", "u0-alpha"], p=[0.4, 0.2, 0.2, 0.1, 0.1]))
        v = parent.v.copy()
        sigma_alpha = np.sqrt(region["cov"][6, 6])
        if kind == "region":
            v = self._draw(parent.region, parent.v, self.settings.jump_scale * self.rng.uniform(0.3, 1.5))
        elif kind == "alpha":
            v[6] += self.rng.normal(0.0, 3.0 * sigma_alpha) if self.rng.random() < 0.7 else self.rng.uniform(-np.pi, np.pi)
        elif kind in ("u0", "u0-alpha"):
            # The source passes the caustic on the other side: u0 moves by one
            # to two source radii (v[1] = u0 tE, v[5] = ln rho tE).
            v[1] += self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.5, 2.5) * np.exp(v[5])
            if kind == "u0-alpha":
                v[6] += self.rng.normal(0.0, 3.0 * sigma_alpha)
        else:
            block = region["cov"][3:5, 3:5]
            v[3:5] += np.linalg.cholesky(block) @ self.rng.standard_normal(2) * self.rng.uniform(0.5, 2.0)
        return v, kind

    def refill(self, round_index):
        if not self.particles:
            return 0
        free = max(self._target(round_index) - len(self.particles), self.settings.jumps_min)
        ordered = sorted(self.particles, key=lambda p: p.chi2)
        weights = 1.0 / (1.0 + np.arange(len(ordered)))
        weights /= weights.sum()
        candidates = []
        for slot in range(free):
            for _ in range(self.settings.refill_candidates):
                parent = ordered[self.rng.choice(len(ordered), p=weights)]
                v, kind = self._jump(parent)
                candidates.append((slot, parent, v, kind))
        results = self._map("evaluate", [(c[2].tolist(),) for c in candidates])
        admitted = 0
        for slot in range(free):
            batch = [(c, r) for c, r in zip(candidates, results) if c[0] == slot and r is not None and r["ok"]]
            if batch:
                (_, parent, v, kind), result = min(batch, key=lambda item: item[1]["chi2"])
                # A jump into a narrow valley looks bad before it has
                # descended, so the best of each slot is always admitted.
                self.particles.append(self._new(v, result, parent.region, f"jump-{kind}", parent.pid))
                admitted += 1
        return admitted

    def search(self) -> float:
        started = time.perf_counter()
        self.seed()
        if not self.particles:
            raise RuntimeError("no hill seed gave a valid light curve")
        self.log.append({"round": 0, "alive": len(self.particles),
                         "best": min(p.chi2 for p in self.particles), "wall": time.perf_counter() - started})
        for round_index in range(1, self.settings.rounds + 1):
            self.descend()
            merged = self._merge()
            culled = self.cull(round_index)
            admitted = self.refill(round_index)
            if not self.particles:
                break
            entry = {"round": round_index, "alive": len(self.particles), "merged": merged, "culled": culled,
                     "admitted": admitted, "best": min(p.chi2 for p in self.particles),
                     "wall": time.perf_counter() - started}
            self.log.append(entry)
            self.say(f"round {round_index}: alive {entry['alive']} best {entry['best']:.1f} "
                     f"wall {entry['wall']:.0f}s")
            if entry["wall"] > self.settings.wall_budget:
                break
            stall = self.settings.stall_rounds
            if round_index - self.epoch_start > stall and \
                    self.log[-1 - stall]["best"] - entry["best"] < self.settings.stall_chi2:
                gain = (self.reseeds[-1]["best"] - entry["best"]) if self.reseeds else math.inf
                if len(self.reseeds) >= self.settings.reseeds or gain < self.settings.stall_chi2:
                    break
                self.reseeds.append({"round": round_index, "best": entry["best"]})
                self.epoch_start = round_index
                self.reseed()
                self.say(f"reseed at round {round_index}: alive {len(self.particles)}")
        return time.perf_counter() - started

    def finalists(self) -> list[Particle]:
        distinct: list[Particle] = []
        for particle in sorted(self.particles, key=lambda p: p.chi2):
            if not any(self._same(particle, other) for other in distinct):
                distinct.append(particle)
        if not distinct:
            return []
        best = distinct[0].chi2
        chosen = distinct[: self.settings.finalists] + [
            p for p in distinct[self.settings.finalists:] if p.chi2 - best < self.settings.finalist_dchi2]
        chosen = chosen[: self.settings.finalists_max]
        for side in (False, True):
            if not any(_side(p) == side for p in chosen):
                extra = next((p for p in distinct if _side(p) == side), None)
                if extra is not None:
                    chosen.append(extra)
        return chosen


# --------------------------------------------------------------------------
# solutions

def _row(result: dict[str, Any] | None, **extra) -> dict[str, Any]:
    ok = result is not None and result.get("status") == "ok"
    return {**extra, "fine_chi2": float(result["fine_chi2"]) if ok else math.inf,
            "parameters": result.get("parameters") if ok else None,
            "seconds": result.get("wall_seconds") if result else None,
            "error": None if ok else (result or {}).get("error", "timeout")}


def merge_solutions(pool: Pool, rows: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    """Polished rows grouped by the search's own merge rule, best first."""
    rows = sorted((r for r in rows if r["parameters"] and math.isfinite(r["fine_chi2"])),
                  key=lambda r: r["fine_chi2"])
    vectors = [map_vector(r["parameters"]) for r in rows]
    results = pool.map("evaluate", [(v.tolist(),) for v in vectors])
    solutions: list[dict[str, Any]] = []
    for row, v, result in zip(rows, vectors, results):
        residual = result["residual"] if result and result["ok"] else None
        twin = None
        if residual is not None:
            twin = next((s for s in solutions if s["_residual"] is not None and _same_valley(
                residual, _shape(v), s["_residual"], s["_shape"], settings)), None)
        if twin is None:
            solutions.append({**row, "members": [row["label"]], "_residual": residual, "_shape": _shape(v)})
        else:
            twin["members"].append(row["label"])
    best = solutions[0]["fine_chi2"] if solutions else math.inf
    for solution in solutions:
        solution.pop("_residual"), solution.pop("_shape")
        solution["delta_best"] = solution["fine_chi2"] - best
    return solutions


def _duplicate(a: dict[str, float], b: dict[str, float]) -> bool:
    """Starts this close would polish to the same fit."""
    return (abs(math.log(a["s"] / b["s"])) < 2e-3 and abs(math.log(a["q"] / b["q"])) < 2e-2
            and abs(math.log(a["rho"] / b["rho"])) < 0.1
            and abs(a["u0"] - b["u0"]) < 1e-3 * max(abs(b["u0"]), 0.01) and abs(a["alpha"] - b["alpha"]) < 1e-3)


def same_solution(a: dict[str, float], b: dict[str, float]) -> bool:
    """One reported solution: q > 1 mapped to (1/q, alpha + pi), then |d ln s| < 0.1,
    |d ln q| < 0.7, |d alpha| < 0.2 and the same u0 sign (piE and u0 drift along flat
    directions)."""
    def canonical(p):
        return (p["s"], 1.0 / p["q"], p["alpha"] + math.pi) if p["q"] > 1 else (p["s"], p["q"], p["alpha"])
    (sa, qa, aa), (sb, qb, ab) = canonical(a), canonical(b)
    dalpha = abs((aa - ab + math.pi) % (2 * math.pi) - math.pi)
    return (abs(math.log(sa / sb)) < 0.1 and abs(math.log(qa / qb)) < 0.7
            and (a["u0"] > 0) == (b["u0"] > 0) and dalpha < 0.2)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def solve(model: EventModel, landscape: Landscape, *, workers: int = 4, seed: int = 0,
          settings: Settings | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Battle royale, LM polish, rho/u0 profiles and (with parallax) a free-piE fit."""
    settings = settings or Settings()
    say = log or (lambda message: None)
    started = time.perf_counter()
    with Pool(model, workers, settings.timeouts) as pool:
        battle = Battle(pool, landscape, settings, seed=seed, log=say)
        search_seconds = battle.search()
        finalists = battle.finalists()
        say(f"polishing {len(finalists)} finalists")
        polished = pool.map("polish", [(native_parameters(p.v), settings.polish_nfev, settings.polish_seconds)
                                       for p in finalists])
        rows = [_row(result, label=str(p.pid), origin=p.origin, search=p.record())
                for p, result in zip(finalists, polished)]
        solutions = merge_solutions(pool, rows, settings)
        profile_rows = []
        degenerate = [s for s in solutions if s["delta_best"] < settings.degenerate_dchi2][: settings.degenerate_max]
        specs = [(s, kind, value) for s in degenerate for kind, value in settings.profiles]
        if specs:
            say(f"profiling {len(degenerate)} solutions")
            results = pool.map("profile", [
                (s["parameters"], kind, value, settings.profile_nfev, settings.profile_seconds,
                 settings.polish_nfev, settings.polish_seconds) for s, kind, value in specs])
            profile_rows = [_row(result, label=f"{s['label']}/{kind}{value:+g}", origin=f"{kind}-profile-{value:+g}",
                                 parent=s["label"]) for (s, kind, value), result in zip(specs, results)]
            solutions = merge_solutions(pool, rows + profile_rows, settings)
        free_rows = []
        if settings.free_parallax and model.has_parallax:
            starts: list[dict[str, Any]] = []
            for row in sorted((r for r in rows + profile_rows if r["parameters"] and math.isfinite(r["fine_chi2"])),
                              key=lambda r: r["fine_chi2"]):
                if not any(_duplicate(row["parameters"], other["parameters"]) for other in starts):
                    starts.append(row)
            say(f"freeing piE from {len(starts)} fits")
            results = pool.map("free", [(r["parameters"], settings.free_nfev, settings.free_seconds) for r in starts])
            for start, result in zip(starts, results):
                free = _row(result, label=f"{start['label']}/free-piE", origin="free-piE", parent=start["label"])
                if free["fine_chi2"] >= start["fine_chi2"]:  # the fixed-piE fit is itself valid
                    free.update(fine_chi2=start["fine_chi2"], kept_fixed=True,
                                parameters={"piEN": model.piE[0], "piEE": model.piE[1], **start["parameters"]})
                free_rows.append(free)
        killed = pool.killed
    candidates = [r for r in (free_rows or rows + profile_rows) if r["parameters"] and math.isfinite(r["fine_chi2"])]
    candidates.sort(key=lambda r: r["fine_chi2"])
    reported = []
    for row in candidates:
        if row["fine_chi2"] - candidates[0]["fine_chi2"] >= settings.solution_dchi2:
            break
        if not any(same_solution(row["parameters"], other["parameters"]) for other in reported):
            reported.append({"label": row["label"], "origin": row["origin"], "chi2": row["fine_chi2"],
                             "delta_best": row["fine_chi2"] - candidates[0]["fine_chi2"],
                             "parameters": row["parameters"]})
    return _jsonable({
        "best": reported[0] if reported else None,
        "solutions": reported,
        "fixed_piE_solutions": solutions,
        "finalists": rows,
        "profiles": profile_rows,
        "free_piE": free_rows,
        "log": battle.log,
        "reseeds": battle.reseeds,
        "counts": battle.counts,
        "killed_jobs": killed,
        "search_seconds": search_seconds,
        "wall_seconds": time.perf_counter() - started,
        "host_piE": list(model.piE) if model.has_parallax else None,
        "seed": seed,
        "settings": asdict(settings),
    })
