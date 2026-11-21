
import math
import mechfight as mf


class Bot:
    name = "adi-survivor-v3"

    def __init__(self, hello: mf.Hello):
        config = hello["config"]
        self.reach = config["loot"]["pickup_radius"]
        self.cube_reach = config["cubes"]["pickup_radius"]

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        tick = obs["tick"]
        zone = obs["zone"]
        center = zone["next_center"]

        act: mf.Action = {"tick": tick}

        def distance(pos):
            return math.hypot(pos[0] - x, pos[1] - y)

        def unit(dx, dy):
            length = math.hypot(dx, dy)
            if length < 0.001:
                return 0.0, 0.0
            return dx / length, dy / length

        # 1. Choose upgrades intelligently.
        if obs["offer"]:
            choices = obs["offer"]["choices"]

            priorities = {
                "armor": 100,
                "regen": 90,
                "speed": 85,
                "energy": 75,
                "hp": 70,
                "shield": 65,
                "gun": 30,
                "sensors": 20,
            }

            def upgrade_score(choice):
                name = (
                    str(choice.get("id", "")) + " " +
                    str(choice.get("name", ""))
                ).lower()

                return max(
                    (score for key, score in priorities.items()
                     if key in name),
                    default=10
                )

            act["pick"] = max(
                range(len(choices)),
                key=lambda i: upgrade_score(choices[i])
            )

        # 2. Identify enemies.
        enemies = [
            m for m in obs["mechs"]
            if m["kind"] == "mech"
        ]
        enemies.sort(key=lambda m: distance(m["pos"]))

        nearest = enemies[0] if enemies else None
        enemy_dist = distance(nearest["pos"]) if nearest else 9999

        # 3. Collect weapons.
        if me["weapons"][1] is None:
            available = [
                p for p in obs["pickups"]
                if p["kind"] != "gun"
            ]

            available.sort(key=lambda p: distance(p["pos"]))

            if available and distance(available[0]["pos"]) < self.reach:
                act["take"] = available[0]["id"]

        # 4. Zone safety.
        zone_dist = distance(center)
        safe_radius = max(8.0, zone["next_radius"] * 0.65)

        goal_x, goal_y = x, y

        if not me["in_zone"] or zone_dist > safe_radius:
            goal_x, goal_y = center[0], center[1]

        else:
            # Collect nearby cubes when safe.
            cubes = sorted(
                obs["cubes"],
                key=lambda c: distance(c["pos"])
            )

            if cubes and distance(cubes[0]["pos"]) < 35:
                if enemy_dist > 35:
                    goal_x, goal_y = cubes[0]["pos"][:2]

            # Search for a second weapon.
            elif me["weapons"][1] is None:
                pickups = [
                    p for p in obs["pickups"]
                    if p["kind"] != "gun"
                ]

                pickups.sort(key=lambda p: distance(p["pos"]))

                if pickups and distance(pickups[0]["pos"]) < 55:
                    if enemy_dist > 40:
                        goal_x, goal_y = pickups[0]["pos"][:2]

        move_x, move_y = unit(goal_x - x, goal_y - y)

        # 5. Defensive movement.
        if nearest and me["in_zone"] and zone_dist < safe_radius:
            ex, ey, ez = nearest["pos"]
            dx, dy = ex - x, ey - y

            if 0.001 < enemy_dist < 55:
                away_x, away_y = unit(-dx, -dy)

                strafe_x, strafe_y = unit(-dy, dx)
                direction = 1 if (tick // 80) % 2 == 0 else -1

                if enemy_dist < 25:
                    move_x = 0.8 * away_x + 0.2 * strafe_x
                    move_y = 0.8 * away_y + 0.2 * strafe_y
                else:
                    move_x = 0.5 * away_x + 0.5 * strafe_x * direction
                    move_y = 0.5 * away_y + 0.5 * strafe_y * direction

        move_x, move_y = unit(move_x, move_y)

        # 6. Energy management.
        energy_ratio = me["energy"] / max(me["max_energy"], 1)

        boost = (
            (not me["in_zone"] and energy_ratio > 0.15)
            or
            (enemy_dist < 25 and energy_ratio > 0.45)
        )

        act.update(
            move=[move_x, move_y],
            boost=boost,
            jump=(
                me["grounded"]
                and enemy_dist < 40
                and tick % 45 == 0
            ),
            reload=False
        )

        # 7. Combat: prioritise survival.
        if nearest:
            target = nearest["pos"]
            target_vel = nearest.get("vel", [0, 0, 0])

            gun = me["weapons"][0]
            bullet_speed = 252.0

            if gun and gun["kind"] == "gun":
                bullet_speed = gun["stats"].get("speed", 252.0)

            lead_time = min(enemy_dist / bullet_speed, 0.6)

            tx = target[0] + target_vel[0] * lead_time
            ty = target[1] + target_vel[1] * lead_time
            tz = target[2] + target_vel[2] * lead_time

            aim_yaw = math.atan2(ty - y, tx - x)
            aim_pitch = math.atan2(
                tz + 4.5 - (z + 7),
                math.hypot(tx - x, ty - y)
            )

            act["aim"] = [aim_yaw, aim_pitch]

            # Fire primarily in defensive engagements.
            if enemy_dist < 90:
                fire_primary = (
                    gun is not None
                    and gun["kind"] == "gun"
                )

                second = me["weapons"][1]
                fire_secondary = False

                if second:
                    kind = second["kind"]

                    if kind == "lightning":
                        fire_secondary = enemy_dist < 35

                    elif kind == "laser":
                        charge = second.get("charge", 0)
                        fire_secondary = (
                            enemy_dist < 75
                            and charge < 0.75
                        )

                    elif kind == "mortar":
                        fire_secondary = False

                act["fire"] = [
                    fire_primary,
                    fire_secondary
                ]

        else:
            act["aim"] = [
                math.atan2(center[1] - y, center[0] - x),
                0.0
            ]

        mf.draw.label(
            f"Adi V3 | HP {me['hp']:.0f} | "
            f"Energy {me['energy']:.0f} | "
            f"Enemies {len(enemies)}"
        )

        return act


mf.run(Bot)
