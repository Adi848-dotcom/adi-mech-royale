
"""Adi Bot V1 — zone-aware movement and combat strafing."""

import math
import mechfight as mf


class Bot:
    name = "adi-survivor-v1"

    def __init__(self, hello: mf.Hello):
        self.reach = hello["config"]["loot"]["pickup_radius"]

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]

        zone = obs["zone"]
        center = zone["next_center"]

        distance_to_center = math.dist((x,y), center[:2])

        #Move toward the safe zone when far away
        #Patrol around the center when already nearby
        if distance_to_center > 20:
            goal = center
        else:
            goal = obs["tick"] * 0.025

            patrol_radius = 12.0
            goal = (
                center[0] + patrol_radius * math.cos(angle),
                center[1] + patrol_radius * math.sin(angle),
            )
        act: mf.Action = {"tick": obs["tick"]}

        # Choose an upgrade when offered.
        if obs["offer"]:
            act["pick"] = 0

        # Collect nearby equipment.
        near = [
            p for p in obs["pickups"]
            if math.dist(p["pos"][:2], (x, y)) < self.reach
        ]

        if me["weapons"][1] is None and near:
            for item in near:
                if item["kind"] != "gun":
                    act["take"] = item["id"]
                    break

        # Find visible enemies.
        enemies = sorted(
            (m for m in obs["mechs"] if m["kind"] == "mech"),
            key=lambda m: math.dist(m["pos"], me["pos"])
        )

        # Default movement: toward the next zone centre.
        dx = goal[0] - x
        dy = goal[1] - y

        distance_to_goal = math.hypot(dx, dy)

        if distance_to_goal > 0.001:
            move_x = dx / distance_to_goal
            move_y = dy / distance_to_goal
        else:
            move_x, move_y = 0.0, 0.0

        # Combat movement: strafe around nearby enemies.
        if enemies and me["in_zone"]:
            enemy = enemies[0]
            ex, ey, ez = enemy["pos"]

            enemy_dx = ex - x
            enemy_dy = ey - y
            enemy_distance = math.hypot(enemy_dx, enemy_dy)

            if 0.001 < enemy_distance < 45:
                # Perpendicular movement.
                strafe_x = -enemy_dy / enemy_distance
                strafe_y = enemy_dx / enemy_distance

                # Alternate direction periodically.
                direction = 1 if (obs["tick"] // 50) % 2 == 0 else -1

                move_x = 0.7 * strafe_x * direction + 0.3 * move_x
                move_y = 0.7 * strafe_y * direction + 0.3 * move_y

                # Retreat based on enemy distance and our health
                if me["hp"] < 35:
                    retreat_strength = 1.5
                    safe_distance = 40
                else:
                    retreat_strength = 0.8
                    safe_distance = 15

                    if enemy_distance < safe_distance:
                        move_x -= retreat_strength * enemy_dx / enemy_distance
                        move_y -= retreat_strength * enemy_dy / enemy_distance

        # Normalize movement.
        length = math.hypot(move_x, move_y)

        if length > 0.001:
            move_x /= length
            move_y /= length

        yaw = math.atan2(move_y, move_x)

        act.update(
            move=[move_x, move_y],
            aim=[yaw, 0.0],
            boost=not me["in_zone"],
            jump=obs["tick"] % 40 == 0
        )

        # Aim and fire at the nearest enemy.
        if enemies:
            target = enemies[0]["pos"]

            aim_yaw = math.atan2(target[1] - y, target[0] - x)
            horizontal_distance = math.dist(target[:2], (x, y))

            aim_pitch = math.atan2(
                target[2] - z - 2.5,
                horizontal_distance
            )

            act["aim"] = [aim_yaw, aim_pitch]
            act["fire"] = [True, True]

            mf.draw.line(
                [x, y, z + 7],
                [target[0], target[1], target[2] + 4.5],
                "#ff5040"
            )

        mf.draw.point(
            [goal[0], goal[1], z],
            "#40c0ff",
            6
        )

        mf.draw.label(
            f"Adi V1 | HP {me['hp']:.0f} | Enemies {len(enemies)}"
        )

        return act


mf.run(Bot)
