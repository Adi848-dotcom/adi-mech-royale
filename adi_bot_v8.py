"""Adi Bot v8 = v7 + HIDE (wait behind buildings instead of in the open). HIDE = False gives v7/v6 behaviour.

Adi Bot v7 = v6 + experiment switches (FLAGS below). All switches OFF = identical to v6.

Adi Bot v6: v5 + upgrade picks driven by the real upgrade ids (protocol mechfight/1).

v6 change: v5 scored upgrades by keyword, which put "reactor" (energy regen) above max-HP plating and
rated the gun/mortar "velocity" upgrades like leg servos. v6 uses an explicit id -> priority table.

v5 (kept): v4 + map-aware navigation (nav.py) for Mech Royale.

v5 adds: routes around buildings to the safe zone / loot, and bends moves away from walls.
If nav.py or the map cannot be loaded, the bot silently plays exactly like v4.

Design goals (score = placement, not kills):
  * never crash and never skip a decision (every tick is wrapped, with a safe fallback)
  * stay inside the NEXT zone circle, near its edge, standing still (silent) when nothing is happening
  * avoid fights: move away from enemies and from the sound of gunfire
  * when forced to fight: keep facing the enemy (rear armor x1.5), strafe, lead shots, use weapons sensibly
  * dodge charging lasers, break out when stuck on walls

Only fields documented in protocol.md are used. All reads use .get() so a missing field cannot crash us.
"""
import math
import random
import sys
import time
import traceback

import mechfight as mf

try:
    import nav as navmod
except Exception:                      # nav.py missing or broken: play without it
    navmod = None

# ===================== EXPERIMENT SWITCHES (change ONE per submission) =====================
LASER_FIRST = False   # True: when picking up loot prefer the laser (hitscan, 280 m range) over lightning/mortar
QUIET = True     # True: do not fire at enemies farther than QUIET_RANGE unless we were just hit (gunfire is heard 320 m away)
QUIET_RANGE = 70.0
HIDE = True           # True: wait at a spot inside the next circle that buildings screen from most directions (needs nav.py)
HIDE_STEP = 12.0      # candidate spots every 12 m
HIDE_MAX_FRAC = 0.7   # only spots within 70% of the next circle's radius (zone safety margin)
HIDE_REACH = 64.0     # how far we look for a screening building, per direction
HIDE_DIST_PENALTY = 0.02    # per metre of walking route: prefer spots close to where we are (less time in the open)
# ===========================================================================================

NAV_BUDGET_S = 0.008                   # max wall time per tick spent planning routes
TAU = math.tau
LOUD_SOUNDS = {"gun", "mortar", "laser", "laser_charge", "lightning", "explosion", "npc_gun", "death"}
UPGRADE_KEYS = [
    ("armor", 100), ("rear", 95), ("regen", 90), ("max_hp", 85), ("hp", 80), ("health", 80),
    ("speed", 70), ("turn", 65), ("energy", 60), ("boost", 55), ("jet", 50),
    ("damage", 45), ("rate", 40), ("cooldown", 40), ("sensor", 25),
]
# Priority by upgrade id (ids from hello.config.upgrades). Higher = picked first. Survival first: effective HP,
# faster healing, mobility, awareness, then arm-gun upgrades (the gun is the one weapon we always hold).
UPGRADE_PRIORITY = {
    "armor": 100, "plating": 98, "nanites": 95, "servos": 80, "sensors": 75,
    "gun_damage": 70, "gun_rate": 68, "gun_mag": 60, "gun_velocity": 55, "gyros": 50,
    "laser_damage": 52, "laser_charge": 48, "laser_pierce": 35,
    "capacitor": 40, "jets": 40, "reactor": 35,
    "mortar_reload": 32, "mortar_splash": 30, "mortar_velocity": 28,
    "lightning_efficiency": 34, "lightning_range": 30, "lightning_chain": 28,
}
WEAPON_RANK = {"lightning": 0, "laser": 1, "mortar": 2}  # lower = preferred when picking loot
if LASER_FIRST:
    WEAPON_RANK = {"laser": 0, "lightning": 1, "mortar": 2}


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def unit(dx, dy):
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, 0.0
    return dx / length, dy / length


def rot(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class HideSearch:
    """Finds a waiting spot inside the next zone circle that tall buildings screen from most directions.

    Works in small time slices per tick (never blocks an answer). If it cannot find or reach a spot,
    the bot falls back to its normal behaviour.
    """

    def __init__(self, nav):
        self.nav = nav
        self.key = None
        self.dirs = [(math.cos(TAU * k / 16.0), math.sin(TAU * k / 16.0)) for k in range(16)]
        self.cands = []
        self.pos = 0
        self.scored = []
        self.done = True
        self.bad = set()
        self.cur = None
        self.t0 = 0
        self.d0 = 0.0
        self.x0 = self.y0 = 0.0

    def _reset(self, cx, cy, nr, x, y):
        self.key = (round(cx), round(cy), round(nr))
        self.x0, self.y0 = x, y
        R = nr * HIDE_MAX_FRAC
        n = int(R // HIDE_STEP)
        cands = []
        for j in range(-n, n + 1):
            for i in range(-n, n + 1):
                px, py = cx + i * HIDE_STEP, cy + j * HIDE_STEP
                if math.hypot(px - cx, py - cy) <= R + 1e-6:
                    cands.append((px, py))
        cands.sort(key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
        self.cands, self.pos, self.scored, self.done = cands, 0, [], False
        self.bad = set()
        self.cur = None
        self.fld = None              # walking distances from where we are now (spots we cannot walk to are skipped)
        try:
            si = self.nav.idx(x, y)
            if si is not None:
                si = self.nav._free_near(si)
                if si is not None:
                    self.fld = navmod.Field(self.nav, si)
        except Exception:
            self.fld = None

    def cover(self, px, py):
        nav = self.nav
        k0 = nav.idx(px, py)
        if k0 is None or nav.blocked[k0]:
            return None
        total = 0.0
        for ux, uy in self.dirs:
            t = 4.0
            while t <= HIDE_REACH:
                k = nav.idx(px + ux * t, py + uy * t)
                if k is None:
                    break
                if nav.tall[k]:
                    total += 1.0 - t / (HIDE_REACH * 1.5)   # a nearer screening building counts more
                    break
                t += 4.0
        return total

    def _work(self, budget_s):
        if self.fld is not None and not self.fld.done:
            self.fld.step(budget_s)          # first fill in walking distances (about 0.1 s of work in total)
            return
        t_end = time.perf_counter() + budget_s
        while self.pos < len(self.cands):
            px, py = self.cands[self.pos]
            self.pos += 1
            c = self.cover(px, py)
            if c is not None:
                if self.fld is not None:
                    dpath = self.fld.dist[self.nav.idx(px, py)]
                    if dpath == float("inf"):
                        continue             # cannot walk there from here
                    pen = HIDE_DIST_PENALTY * dpath * navmod.CELL
                else:
                    pen = HIDE_DIST_PENALTY * math.hypot(px - self.x0, py - self.y0)
                self.scored.append((c - pen, px, py))
            if time.perf_counter() > t_end:
                return
        self.done = True

    def update(self, cx, cy, nr, x, y, tick, budget_s=0.003):
        key = (round(cx), round(cy), round(nr))
        if key != self.key:
            self._reset(cx, cy, nr, x, y)
        if not self.done:
            self._work(budget_s)
        if not self.scored:
            return None
        if self.cur is None or self.cur in self.bad or not self.done:
            best = None
            for sc, px, py in self.scored:
                if (px, py) in self.bad:
                    continue
                if best is None or sc > best[0]:
                    best = (sc, px, py)
            if best is None:
                return None
            new = (best[1], best[2])
            if new != self.cur:
                self.cur = new
                self.t0 = tick
                self.d0 = math.hypot(new[0] - x, new[1] - y)
        # watchdog: if we have made no progress toward the spot for 10 s, give up on it
        d = math.hypot(self.cur[0] - x, self.cur[1] - y)
        if tick - self.t0 >= 100:
            if d > 8.0 and d > self.d0 - 8.0:
                self.bad.add(self.cur)
                self.cur = None
                return None
            self.t0, self.d0 = tick, d
        return self.cur


class Bot:
    name = "adi-survivor-v8"

    def __init__(self, hello):
        cfg = hello.get("config", {}) or {}
        self.rng = random.Random(int(hello.get("seed", 0) or 0))
        loot = cfg.get("loot", {}) or {}
        self.reach = float(loot.get("pickup_radius", 4.0))
        mortar = (cfg.get("weapons", {}) or {}).get("mortar", {}) or {}
        self.mortar_g = float(mortar.get("gravity", 22.0))
        self.up_text = {}
        for u in cfg.get("upgrades", []) or []:
            try:
                paths = " ".join(str(e.get("path", "")) for e in u.get("effects", []) or [])
                self.up_text[u["id"]] = (str(u["id"]) + " " + str(u.get("name", "")) + " " + paths).lower()
            except Exception:
                pass
        # state carried between ticks
        self.errors = 0
        self.hist = []                 # (x, y, commanded_move_length)
        self.escape_until = -1
        self.escape_idx = 0
        self.escape_angle = 0.0
        self.strafe_dir = 1
        self.strafe_until = 0.0
        self.avoid = [0.0, 0.0]
        self.last_dmg_t = -999.0
        self.last_dmg_bearing = None
        self.failed_takes = set()
        self.scan_yaw = 0.0
        # navigation (optional)
        self.nav = None
        self.nav_errors = 0
        self.hide = None
        self.at_spot = False
        if navmod is not None:
            try:
                self.nav = navmod.Nav((hello.get("map") or {}).get("dir", "/arena"))
                print("v5 nav ready in %.2fs (%d blocked cells)" % (self.nav.build_s, self.nav.blocked_cells),
                      file=sys.stderr, flush=True)
            except Exception:
                print("v5 nav disabled:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
                self.nav = None
        if HIDE and self.nav is not None:
            try:
                self.hide = HideSearch(self.nav)
            except Exception:
                self.hide = None

    # ------------------------------------------------------------------ nav wrappers
    def _nav_fail(self):
        self.nav_errors += 1
        if self.nav_errors <= 2:
            print("v5 nav error:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        if self.nav_errors >= 3:
            self.nav = None            # give up on nav, keep playing like v4

    def _nav_dir(self, x, y, gx, gy):
        try:
            return self.nav.direction_to(x, y, gx, gy)
        except Exception:
            self._nav_fail()
            return None

    def _nav_steer(self, x, y, ux, uy):
        try:
            return self.nav.steer(x, y, ux, uy)
        except Exception:
            self._nav_fail()
            return ux, uy

    # ------------------------------------------------------------------ entry
    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            self.errors += 1
            if self.errors <= 3:
                print("v4 act() error:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
            try:
                me = obs["self"]
                c = obs["zone"]["next_center"]
                ux, uy = unit(c[0] - me["pos"][0], c[1] - me["pos"][1])
                return {"tick": obs["tick"], "move": [ux, uy], "boost": not me.get("in_zone", True)}
            except Exception:
                return {"tick": obs.get("tick", 0)}

    # ------------------------------------------------------------- helpers
    def _upgrade_pick(self, offer):
        choices = offer.get("choices", []) or []
        best, best_score = 0, -1
        for i, ch in enumerate(choices):
            if ch.get("id") in UPGRADE_PRIORITY:
                score = UPGRADE_PRIORITY[ch.get("id")]
            else:                       # unknown id (config changed): fall back to the keyword guess
                text = self.up_text.get(ch.get("id")) or (str(ch.get("id", "")) + " " + str(ch.get("name", ""))).lower()
                score = max((s for k, s in UPGRADE_KEYS if k in text), default=30)
            if score > best_score:
                best, best_score = i, score
        return best

    @staticmethod
    def _lead(sx, sy, sz, pos, vel, speed):
        px, py, pz = pos
        for _ in range(3):
            d = math.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2)
            t = d / max(speed, 1.0)
            px = pos[0] + vel[0] * t
            py = pos[1] + vel[1] * t
            pz = pos[2] + vel[2] * t
        return px, py, pz

    def _mortar_solution(self, sx, sy, sz, pos, vel, v, max_pitch, min_pitch):
        g = self.mortar_g
        px, py, pz = pos
        pitch = 0.0
        R = 0.0
        for _ in range(3):
            R = math.hypot(px - sx, py - sy)
            if R < 5.0:
                return None
            h = pz - sz
            disc = v ** 4 - g * (g * R * R + 2.0 * h * v * v)
            if disc < 0:
                return None
            pitch = math.atan2(v * v - math.sqrt(disc), g * R)
            tf = R / (v * max(math.cos(pitch), 0.2))
            px = pos[0] + vel[0] * tf
            py = pos[1] + vel[1] * tf
            pz = pos[2] + vel[2] * tf
        if not (min_pitch <= pitch <= max_pitch):
            return None
        return math.atan2(py - sy, px - sx), pitch, R

    # --------------------------------------------------------------- main
    def _act(self, obs):
        me = obs["self"]
        x, y, z = me["pos"]
        tick = obs["tick"]
        t = obs.get("t", tick / 20.0)
        zone = obs["zone"]
        stats = me.get("stats", {}) or {}
        act = {"tick": tick}
        hpr = me["hp"] / max(me.get("max_hp", 1), 1)
        energy_ratio = me.get("energy", 0) / max(me.get("max_energy", 1), 1)
        my_yaw, my_pitch = me.get("aim", [0.0, 0.0])
        grounded = me.get("grounded", True)
        in_zone = me.get("in_zone", True)

        # ---- navigation only helps while we walk on the ground (not on a roof or in the air)
        nav_on = False
        if self.nav is not None:
            try:
                self.nav.begin_tick(NAV_BUDGET_S)
                nav_on = bool(grounded) and abs(z - self.nav.ground(x, y)) < 3.0
            except Exception:
                self._nav_fail()

        # ---- events
        for ev in obs.get("events", []) or []:
            et = ev.get("type")
            if et == "damaged" and ev.get("weapon") != "zone" and ev.get("bearing") is not None:
                self.last_dmg_t = t
                self.last_dmg_bearing = ev["bearing"]
            elif et == "take_failed":
                self.failed_takes.add(ev.get("id"))
        recently_hit = (t - self.last_dmg_t) < 2.0

        # ---- upgrades
        if obs.get("offer"):
            act["pick"] = self._upgrade_pick(obs["offer"])

        # ---- weapons by kind
        weapons = me.get("weapons", []) or []
        held = {}
        for slot, w in enumerate(weapons):
            if w:
                held[w.get("kind")] = (slot, w)
        empty_slots = [i for i, w in enumerate(weapons) if not w]

        # ---- enemies (mechs only), sorted by distance
        enemies = []
        for m in obs.get("mechs", []) or []:
            if m.get("kind") == "mech":
                enemies.append((math.hypot(m["pos"][0] - x, m["pos"][1] - y), m))
        enemies.sort(key=lambda e: e[0])
        nearest_d, nearest = enemies[0] if enemies else (9999.0, None)
        n_near = sum(1 for d, _ in enemies if d < 80)
        fleeing = nearest is not None and (hpr < 0.35 or n_near >= 3 or (n_near >= 2 and hpr < 0.6))

        # ---- strafe direction timer
        if t >= self.strafe_until:
            self.strafe_dir = self.rng.choice((-1, 1))
            self.strafe_until = t + self.rng.uniform(0.6, 1.6)

        # ---- sound avoidance (smoothed)
        ax = ay = 0.0
        for s in obs.get("sounds", []) or []:
            if s.get("kind") in LOUD_SOUNDS and s.get("age", 0) < 1.5:
                w = {"near": 1.0, "mid": 0.5}.get(s.get("dist"), 0.15)
                b = s.get("bearing", 0.0)
                ax -= math.cos(b) * w
                ay -= math.sin(b) * w
        self.avoid = [self.avoid[0] * 0.92 + ax * 0.6, self.avoid[1] * 0.92 + ay * 0.6]
        al = math.hypot(*self.avoid)
        if al > 1.5:
            self.avoid = [self.avoid[0] * 1.5 / al, self.avoid[1] * 1.5 / al]

        # ---- zone pull toward the safe disk near the NEXT circle
        cx, cy = zone["next_center"]
        nr = zone["next_radius"]
        hold_r = max(6.0, nr * 0.55)
        edge_r = max(hold_r + 4.0, nr * 0.9)
        d_next = math.hypot(x - cx, y - cy)
        if not in_zone or d_next > edge_r:
            pull = 1.0
        else:
            pull = clamp((d_next - hold_r) / max(edge_r - hold_r, 1.0), 0.0, 1.0)
        tx, ty = cx, cy
        if self.hide is not None:
            try:
                spot = self.hide.update(cx, cy, nr, x, y, tick)
            except Exception:
                spot = None
                self.hide = None
                print("v8 hide disabled:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
            if spot is not None:
                tx, ty = spot
                ds = math.hypot(tx - x, ty - y)
                if ds < 6.0:
                    self.at_spot = True
                elif ds > 14.0:
                    self.at_spot = False
                if not in_zone:
                    pull = 1.0
                elif self.at_spot:
                    pull = 0.0
                else:
                    pull = 1.0
        zux, zuy = unit(tx - x, ty - y)
        if nav_on and pull > 0.05 and self.nav is not None:
            nd = self._nav_dir(x, y, tx, ty)
            if nd is not None:
                zux, zuy = nd           # walk around buildings instead of straight at the target
        mx, my = zux * pull, zuy * pull

        # ---- threats
        for d, m in enemies:
            if d < 100:
                ux, uy = unit(x - m["pos"][0], y - m["pos"][1])
                w = ((100.0 - d) / 100.0) * (1.6 if fleeing else 0.6)
                mx += ux * w
                my += uy * w
        if nearest is not None and nearest_d < 100:
            ex, ey = nearest["pos"][0], nearest["pos"][1]
            sx_, sy_ = unit(-(ey - y), ex - x)          # strafe = perpendicular to the enemy
            sw = 0.3 if fleeing else (0.9 if nearest_d < 70 else 0.4)
            mx += sx_ * self.strafe_dir * sw
            my += sy_ * self.strafe_dir * sw
            if nearest_d < 30 and not fleeing:
                ux, uy = unit(x - ex, y - ey)
                mx += ux * 1.0
                my += uy * 1.0
        elif nearest is None or nearest_d > 60:
            mx += self.avoid[0] * 0.8
            my += self.avoid[1] * 0.8

        # ---- ambush: hit by someone we cannot see -> face them and move off the line
        ambush_yaw = None
        if recently_hit and (nearest is None or nearest_d > 100) and self.last_dmg_bearing is not None:
            ambush_yaw = self.last_dmg_bearing
            ux, uy = -math.cos(ambush_yaw), -math.sin(ambush_yaw)
            mx += ux * 0.8 + (-uy) * self.strafe_dir * 0.6
            my += uy * 0.8 + ux * self.strafe_dir * 0.6

        # ---- laser dodge
        dodge = False
        for d, m in enemies:
            if m.get("charging") and d < 290:
                b = math.atan2(y - m["pos"][1], x - m["pos"][0])
                diff = wrap(b - m["aim"][0])
                half = math.atan2(3.2, max(d, 5.0)) + 0.12
                if abs(diff) < half:
                    side = 1 if diff > 0 else (-1 if diff < 0 else self.strafe_dir)
                    px_, py_ = -math.sin(m["aim"][0]) * side, math.cos(m["aim"][0]) * side
                    mx += px_ * 2.5
                    my += py_ * 2.5
                    dodge = True

        # ---- loot goals (only when safe)
        safe = (nearest is None or nearest_d > 55) and not fleeing and in_zone and d_next < edge_r and not recently_hit
        goal = None
        if empty_slots:
            cands = [p for p in obs.get("pickups", []) or []
                     if p.get("id") not in self.failed_takes
                     and p.get("kind") in WEAPON_RANK and p.get("kind") not in held]
            cands.sort(key=lambda p: math.hypot(p["pos"][0] - x, p["pos"][1] - y) + 25 * WEAPON_RANK[p["kind"]])
            if cands:
                p = cands[0]
                pd = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
                if pd < self.reach * 0.9 and abs(p["pos"][2] - z) < 3.5:
                    act["take"] = p["id"]
                elif safe and pd < 60:
                    goal = p["pos"]
        if goal is None and safe:
            cubes = sorted(obs.get("cubes", []) or [], key=lambda c: math.hypot(c["pos"][0] - x, c["pos"][1] - y))
            if cubes and math.hypot(cubes[0]["pos"][0] - x, cubes[0]["pos"][1] - y) < 30:
                goal = cubes[0]["pos"]
        if goal is not None:
            gx, gy = unit(goal[0] - x, goal[1] - y)
            if nav_on and self.nav is not None and math.hypot(goal[0] - x, goal[1] - y) > 8.0:
                nd = self._nav_dir(x, y, goal[0], goal[1])
                if nd is not None:
                    gx, gy = nd
            mx += gx * 1.2
            my += gy * 1.2

        # ---- finalize move vector
        L = math.hypot(mx, my)
        if L < 0.2:
            mx = my = 0.0          # stand still: silent, regenerates, no reason to move
            L = 0.0
        else:
            k = min(L, 1.0) / L
            mx, my = mx * k, my * k
            L = math.hypot(mx, my)

        # ---- bend the move away from a wall straight ahead (fleeing, strafing, anything)
        if nav_on and L > 0 and self.nav is not None:
            sx2, sy2 = self._nav_steer(x, y, mx / L, my / L)
            mx, my = sx2 * L, sy2 * L

        # ---- stuck detection / escape
        self.hist.append((x, y, L))
        if len(self.hist) > 8:
            self.hist.pop(0)
        jump = False
        escaping = tick < self.escape_until
        if not escaping and len(self.hist) >= 6 and L >= 0.5:
            recent = self.hist[-6:]
            if all(h[2] >= 0.5 for h in recent) and math.hypot(recent[-1][0] - recent[0][0], recent[-1][1] - recent[0][1]) < 1.0:
                angles = [math.pi / 2, -math.pi / 2, 2.4, -2.4, math.pi / 4, -math.pi / 4]
                self.escape_angle = angles[self.escape_idx % len(angles)]
                self.escape_idx += 1
                self.escape_until = tick + 16
                self.hist = []
                escaping = True
        if escaping and L > 0:
            mx, my = rot(mx, my, self.escape_angle)
            jump = grounded and (tick % 6 == 0)

        # ---- boost policy
        boost = False
        if not in_zone and me.get("energy", 0) > 8:
            boost = True
        elif fleeing and energy_ratio > 0.35:
            boost = True
        elif dodge and energy_ratio > 0.2:
            boost = True
        elif escaping and energy_ratio > 0.25:
            boost = True
        if dodge and grounded:
            jump = True

        act["move"] = [mx, my]
        act["boost"] = boost
        act["jump"] = jump

        # ---- aiming and firing
        fire = [False, False]
        aim = None
        if nearest is not None:
            tp, tv = nearest["pos"], (nearest.get("vel") or [0.0, 0.0, 0.0])
            s_scale = 1.0 + 0.05 * (nearest.get("cubes", 0) or 0)
            tcenter = (tp[0], tp[1], tp[2] + 4.5 * s_scale)
            muzzle = z + stats.get("muzzle_height", 7.0)
            d3 = max(nearest_d, 1.0)
            tspeed = math.hypot(tv[0], tv[1])

            gun = held.get("gun")
            lig = held.get("lightning")
            las = held.get("laser")
            mor = held.get("mortar")

            laser_charging = bool(las and (las[1].get("charge", 0) or 0) > 0)
            use_mortar = False
            mortar_sol = None
            if mor and not laser_charging and nearest_d >= 60 and tspeed < 5.0:
                msp = mor[1].get("stats", {}).get("speed", 72.0)
                mortar_sol = self._mortar_solution(
                    x, y, z + stats.get("height", 9.0) + 0.5, tcenter, tv, msp,
                    stats.get("max_pitch", 1.2), stats.get("min_pitch", -0.6))
                use_mortar = mortar_sol is not None

            if use_mortar:
                a_yaw, a_pitch, R = mortar_sol
                aim = [a_yaw, a_pitch]
                if mor[1].get("cooldown", 0) <= 0.05 and 30 < R < 230 \
                        and abs(wrap(a_yaw - my_yaw)) < 0.05 and abs(a_pitch - my_pitch) < 0.06:
                    fire[mor[0]] = True
            else:
                gspeed = 252.0
                if gun:
                    gspeed = gun[1].get("stats", {}).get("speed", 252.0)
                if laser_charging or (las and nearest_d < 200):
                    lx, ly, lz = tcenter[0] + tv[0] * 0.15, tcenter[1] + tv[1] * 0.15, tcenter[2] + tv[2] * 0.15
                else:
                    lx, ly, lz = self._lead(x, y, muzzle, tcenter, tv, gspeed)
                a_yaw = math.atan2(ly - y, lx - x)
                a_pitch = math.atan2(lz - muzzle, max(math.hypot(lx - x, ly - y), 1.0))
                aim = [a_yaw, a_pitch]
                yaw_err = abs(wrap(a_yaw - my_yaw))
                pitch_err = abs(a_pitch - my_pitch)
                tol = math.atan2(2.2 * s_scale, d3) * 0.8 + 0.02

                if lig and nearest_d < 42 and yaw_err < 0.3 and me.get("energy", 0) > 15:
                    fire[lig[0]] = True

                if las:
                    w = las[1]
                    ch = w.get("charge", 0) or 0
                    need = w.get("stats", {}).get("charge", 0.75)
                    ready = (w.get("cooldown", 0) or 0) <= 0.05
                    can_charge = nearest_d < 200 and n_near <= 1 and hpr > 0.4 and not fleeing and not dodge
                    if ch > 0:
                        if not can_charge:
                            fire[las[0]] = False          # let go now: fires early with partial damage
                        elif ch >= need - 0.05:
                            fire[las[0]] = not (yaw_err < tol * 0.9 and pitch_err < tol * 0.9 + 0.02)
                        else:
                            fire[las[0]] = True
                    elif ready and can_charge:
                        fire[las[0]] = True

                if gun and not laser_charging:
                    gw = gun[1]
                    grange = gw.get("stats", {}).get("range", 230.0)
                    if (gw.get("ammo", 1) or 0) > 0 and (gw.get("reload", 0) or 0) <= 0 \
                            and nearest_d < grange * 0.9 and yaw_err < tol + 0.02 and pitch_err < tol + 0.05:
                        fire[gun[0]] = True
            if QUIET and nearest_d >= QUIET_RANGE and not recently_hit:
                fire = [False, False]          # stay quiet: do not give our position away from afar
            if any(fire):
                act["fire"] = fire
            elif laser_charging:
                act["fire"] = fire     # explicit release
        else:
            if ambush_yaw is not None:
                aim = [ambush_yaw, 0.0]
            elif L > 0.2:
                aim = [math.atan2(my, mx), 0.0]
            else:
                self.scan_yaw = wrap(my_yaw + 0.12)    # idle: slowly scan around
                aim = [self.scan_yaw, 0.0]
            gun = held.get("gun")
            if gun and nearest_d > 120:
                gw = gun[1]
                mag = gw.get("stats", {}).get("mag", 12)
                if (gw.get("ammo", mag) or 0) < mag * 0.6 and (gw.get("reload", 0) or 0) <= 0:
                    act["reload"] = True

        if aim is not None:
            act["aim"] = aim

        mf.draw.label("v8 hp %d %s e%d" % (me["hp"], "FLEE" if fleeing else "ok", len(enemies)))
        return act


mf.run(Bot)
