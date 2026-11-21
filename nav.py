"""Grid navigation for Mech Royale, built from the arena map files (protocol.md section 11).

What it does:
  * rasterises building/stadium/bridge footprints (plus steep slopes and the map edge) onto a 4 m grid
  * keeps a "distance field" from a goal cell (Dijkstra, computed in small time-boxed slices per tick),
    so the bot can ask "which way do I walk to get to that point without hitting a building?"
  * `steer()` bends any desired direction away from walls straight ahead

Everything here is optional: if the map can't be read, Nav() raises and the bot plays without it.
"""
import heapq
import json
import math
import os
import sys
import time
from array import array

CELL = 4.0
SQ2 = math.sqrt(2.0)
INF = float("inf")
NEIGH = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))
SOFT_COST = 3.0          # cells next to walls are allowed but cost more (a mech is 5 m wide)
SAMPLES = ((0.0, 0.0), (1.3, 1.3), (-1.3, 1.3), (1.3, -1.3), (-1.3, -1.3))


def _pip(px, py, ring):
    """Point in polygon for a flat [x0, y0, x1, y1, ...] ring (ray casting)."""
    inside = False
    n = len(ring) // 2
    j = n - 1
    for i in range(n):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class Field:
    """Distance-to-goal over the grid, filled in lazily by Dijkstra from the goal."""

    def __init__(self, nav, goal_idx):
        self.nav = nav
        self.goal = goal_idx
        self.dist = [INF] * nav.n
        self.dist[goal_idx] = 0.0
        self.heap = [(0.0, goal_idx)]
        self.stop_d = INF            # pause expanding beyond this distance (set once we know where we are)
        self.done = False

    def step(self, budget_s):
        nav = self.nav
        heap, dist = self.heap, self.dist
        blocked, soft, nx, ny = nav.blocked, nav.soft, nav.nx, nav.ny
        t_end = time.perf_counter() + budget_s
        count = 0
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if d > self.stop_d:
                heapq.heappush(heap, (d, u))
                return
            ui, uj = u % nx, u // nx
            for di, dj, w in NEIGH:
                vi, vj = ui + di, uj + dj
                if vi < 0 or vj < 0 or vi >= nx or vj >= ny:
                    continue
                v = vj * nx + vi
                if blocked[v]:
                    continue
                if di and dj and (blocked[uj * nx + vi] or blocked[vj * nx + ui]):
                    continue          # no cutting corners
                nd = d + (w * SOFT_COST if soft[v] else w)
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
            count += 1
            if (count & 63) == 0 and time.perf_counter() > t_end:
                return
        self.done = True

    def direction(self, x, y, look=3):
        """Unit vector toward lower distance-to-goal, or None if unknown / already there."""
        nav = self.nav
        idx = nav.idx(x, y)
        if idx is None:
            return None
        dist = self.dist
        if dist[idx] == INF:
            if not self.done:
                self.stop_d = INF           # we are outside what was computed: keep expanding
            return None
        if self.stop_d == INF and not self.done:
            self.stop_d = dist[idx] + 60.0  # we are settled: only expand a margin beyond our distance
        nx, ny, blocked = nav.nx, nav.ny, nav.blocked
        cur = idx
        for _ in range(look):
            ci, cj = cur % nx, cur // nx
            best, bd = cur, dist[cur]
            for di, dj, _w in NEIGH:
                vi, vj = ci + di, cj + dj
                if vi < 0 or vj < 0 or vi >= nx or vj >= ny:
                    continue
                v = vj * nx + vi
                if blocked[v]:
                    continue
                if di and dj and (blocked[cj * nx + vi] or blocked[vj * nx + ci]):
                    continue
                if dist[v] < bd:
                    best, bd = v, dist[v]
            if best == cur:
                break
            cur = best
        if cur == idx:
            return None
        tx, ty = nav.center(cur)
        dx, dy = tx - x, ty - y
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return None
        return dx / length, dy / length


class Nav:
    def __init__(self, map_dir):
        t0 = time.perf_counter()
        base = os.path.join(map_dir, "v2")
        with open(os.path.join(base, "world.json")) as f:
            world = json.load(f)
        tr = world["terrain"]
        self.tminx, self.tminy, self.tstep = float(tr["minX"]), float(tr["minY"]), float(tr["step"])
        self.tnx, self.tny = int(tr["nx"]), int(tr["ny"])
        raw = array("h")
        with open(os.path.join(base, tr["file"]), "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder == "big":
            raw.byteswap()
        if len(raw) != self.tnx * self.tny:
            raise ValueError("terrain size mismatch")
        self.terrain = raw
        b = world["bounds"]
        self.minx, self.miny = float(b["minX"]), float(b["minY"])
        self.nx = int((b["maxX"] - b["minX"]) / CELL) + 1
        self.ny = int((b["maxY"] - b["minY"]) / CELL) + 1
        self.n = self.nx * self.ny
        self.blocked = bytearray(self.n)
        self.soft = bytearray(self.n)
        self.tall = bytearray(self.n)     # blocks line of sight at eye height (kept for later use)
        rules = world.get("rules", {})
        self.step_up = float(rules.get("maxStep", 1.2))
        self.max_slope = float(rules.get("maxSlopeDeg", 50.0))
        self._rasterise(world["prisms"])
        self._slopes()
        self._dilate()
        self.fields = {}
        self.budget = 0.0
        self.build_s = time.perf_counter() - t0
        self.blocked_cells = sum(self.blocked)

    # ---------------------------------------------------------------- map
    def ground(self, x, y):
        i = int(round((x - self.tminx) / self.tstep))
        j = int(round((y - self.tminy) / self.tstep))
        i = 0 if i < 0 else (self.tnx - 1 if i >= self.tnx else i)
        j = 0 if j < 0 else (self.tny - 1 if j >= self.tny else j)
        return self.terrain[j * self.tnx + i] * 0.01

    def idx(self, x, y):
        i = int((x - self.minx) // CELL)
        j = int((y - self.miny) // CELL)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return None
        return j * self.nx + i

    def center(self, idx):
        return self.minx + (idx % self.nx + 0.5) * CELL, self.miny + (idx // self.nx + 0.5) * CELL

    def is_blocked(self, x, y):
        k = self.idx(x, y)
        return k is None or self.blocked[k] == 1

    def _rasterise(self, prisms):
        head, step_up, nx, ny = 9.0, self.step_up, self.nx, self.ny
        for p in prisms:
            ring = p["ring"]
            holes = p.get("holes") or []
            xs, ys = ring[0::2], ring[1::2]
            i0 = max(0, int((min(xs) - self.minx) // CELL))
            i1 = min(nx - 1, int((max(xs) - self.minx) // CELL))
            j0 = max(0, int((min(ys) - self.miny) // CELL))
            j1 = min(ny - 1, int((max(ys) - self.miny) // CELL))
            base, top = p["base"], p["top"]
            for j in range(j0, j1 + 1):
                cy = self.miny + (j + 0.5) * CELL
                for i in range(i0, i1 + 1):
                    cx = self.minx + (i + 0.5) * CELL
                    g = self.ground(cx, cy)
                    if base >= g + head or top <= g + step_up:
                        continue       # high enough to walk under, or low enough to step onto
                    for ox, oy in SAMPLES:
                        px, py = cx + ox, cy + oy
                        if _pip(px, py, ring) and not any(_pip(px, py, h) for h in holes):
                            k = j * nx + i
                            self.blocked[k] = 1
                            if top - g > 10.0:
                                self.tall[k] = 1
                            break

    def _slopes(self):
        lim = math.tan(math.radians(self.max_slope - 3.0))
        h = CELL / 2.0
        for j in range(self.ny):
            cy = self.miny + (j + 0.5) * CELL
            for i in range(self.nx):
                k = j * self.nx + i
                if self.blocked[k]:
                    continue
                cx = self.minx + (i + 0.5) * CELL
                gx = (self.ground(cx + h, cy) - self.ground(cx - h, cy)) / CELL
                gy = (self.ground(cx, cy + h) - self.ground(cx, cy - h)) / CELL
                if math.hypot(gx, gy) > lim:
                    self.blocked[k] = 1

    def _dilate(self):
        nx, ny = self.nx, self.ny
        for k in range(self.n):
            if self.blocked[k]:
                i, j = k % nx, k // nx
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if 0 <= a < nx and 0 <= b < ny:
                            self.soft[b * nx + a] = 1

    # ---------------------------------------------------------- planning
    def begin_tick(self, budget_s):
        self.budget = budget_s

    def _free_near(self, idx):
        """Nearest unblocked cell to idx (spiral search), or None."""
        if not self.blocked[idx]:
            return idx
        ci, cj = idx % self.nx, idx // self.nx
        for r in range(1, 12):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    a, b = ci + di, cj + dj
                    if 0 <= a < self.nx and 0 <= b < self.ny and not self.blocked[b * self.nx + a]:
                        return b * self.nx + a
        return None

    def direction_to(self, x, y, gx, gy):
        """Unit vector to walk from (x, y) toward goal (gx, gy) around buildings, or None if not known yet."""
        gi = self.idx(gx, gy)
        if gi is None:
            return None
        gi = self._free_near(gi)
        if gi is None:
            return None
        f = self.fields.get(gi)
        if f is None:
            if len(self.fields) >= 3:
                self.fields.pop(next(iter(self.fields)))     # drop the oldest
            f = self.fields[gi] = Field(self, gi)
        else:
            self.fields[gi] = self.fields.pop(gi)            # mark as most recently used
        if self.budget > 0 and not f.done:
            t0 = time.perf_counter()
            f.step(self.budget)
            self.budget = max(0.0, self.budget - (time.perf_counter() - t0))
        return f.direction(x, y)

    # ---------------------------------------------------------- steering
    def _clear(self, x, y, ux, uy, reach):
        t = 3.0
        while t <= reach + 1e-6:
            if self.is_blocked(x + ux * t, y + uy * t):
                return False
            t += 3.0
        return True

    def steer(self, x, y, ux, uy, reach=9.0):
        """Bend (ux, uy) away from a wall directly ahead; returns it unchanged if the way is clear."""
        if self._clear(x, y, ux, uy, reach):
            return ux, uy
        for a in (0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.7, -1.7, 2.3, -2.3):
            c, s = math.cos(a), math.sin(a)
            vx, vy = ux * c - uy * s, ux * s + uy * c
            if self._clear(x, y, vx, vy, reach):
                return vx, vy
        return ux, uy
