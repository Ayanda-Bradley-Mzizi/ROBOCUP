import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M

# --- CONSTANTS ---
OPPONENT_GOAL_2D = np.array([15.0, 0.0], dtype=float)
OUR_GOAL_2D      = np.array([-15.0, 0.0], dtype=float)

SHOOT_DISTANCE   = 7.0
SHOOT_DISTANCE_SQ= SHOOT_DISTANCE * SHOOT_DISTANCE

# Lanes & spacing
LANE_LATERALS    = [-1.8, 0.0, +1.8]   # y-lanes used by attackers (spread)
LEAD_AHEAD_BASE  = 3.5                  # m ahead of ball for support lanes
PERSONAL_SPACE   = 1.25                 # m – attackers keep at least this much distance
REPULSION_GAIN   = 0.7                  # strength of anti-crowding push

# Chase & locking
PASS_LOCK_MS         = 600              # how long we “stick” to the same passer
PASS_LOCK_RADIUS     = 2.8              # keep passer if still within this to ball
PASS_SWITCH_MARGIN   = 1.0              # only switch if candidate is this much closer
RECEIVER_IGNORE_R    = 2.2              # receivers won’t approach within this of passer
CHASE_ONLY_PASSER_R  = 4.0              # only the passer chases when ball within this

# Lane clearance
OPP_GAP_CLEAR    = 0.80                 # min gap from opponents along pass line

# Field clamps
FIELD_X = 15.0
FIELD_Y = 10.0


def _clamp(p):
    return np.array([np.clip(p[0], -FIELD_X, FIELD_X),
                     np.clip(p[1], -FIELD_Y, FIELD_Y)], dtype=float)

def _dist(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def _has_clear_lane(a, b, opps, gap=OPP_GAP_CLEAR):
    if not opps:
        return True
    a = np.array(a, float); b = np.array(b, float)
    ab = b - a
    ab2 = float(np.dot(ab, ab)) if np.dot(ab, ab) > 1e-8 else 1e-8
    for o in opps:
        if o is None: continue
        p = np.array(o, float)
        t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
        proj = a + t * ab
        if np.linalg.norm(p - proj) < gap:
            return False
    return True

def _nearest_opponent_ahead(ref_xy, opps, fwd_cone_deg=60.0, max_dist=12.0):
    """Nearest opponent in a +X cone from ref point."""
    if not opps: return None, None
    rx, ry = ref_xy
    best = (None, None)
    cos_th = math.cos(math.radians(fwd_cone_deg * 0.5))
    for o in opps:
        if o is None: continue
        ox, oy = o
        vx, vy = (ox - rx), (oy - ry)
        d = math.hypot(vx, vy)
        if d == 0 or d > max_dist:
            continue
        dot = vx / d  # angle to +X
        if dot >= cos_th:
            if best[1] is None or d < best[1]:
                best = (np.array([ox, oy], float), d)
    return best

def _receiver_lane_point(ball_xy, lead_ahead=LEAD_AHEAD_BASE, lane_y=0.0):
    bx, by = ball_xy
    return _clamp(np.array([min(FIELD_X-0.8, bx + lead_ahead), by + lane_y], float))

def _repel_from_points(my_target, points, min_space=PERSONAL_SPACE, gain=REPULSION_GAIN):
    """Repel from a list of points to avoid bunching (y-biased)."""
    t = np.array(my_target, float)
    for p in points:
        if p is None: continue
        d = _dist(t, p)
        if d < 1e-6:
            t[1] += min_space
            continue
        if d < min_space:
            push = (t - p).astype(float)
            push[0] *= 0.35                # keep forward bias; mostly move laterally
            n = np.linalg.norm(push) + 1e-6
            t += gain * (min_space - d) * (push / n)
    return _clamp(t)


class Strategy():
    def __init__(self, world):
        self.world = world
        self.play_mode = world.play_mode
        self.robot_model = world.robot
        self.my_head_pos_2d = self.robot_model.loc_head_position[:2].copy()
        self.player_unum = self.robot_model.unum
        self.mypos = self.my_head_pos_2d

        # You always attack +X (repo rule)
        self.side = 1
        if world.team_side_is_left:
            self.side = 0

        self.teammate_positions = [t.state_abs_pos[:2] if t.state_abs_pos is not None else None
                                   for t in world.teammates]
        self.opponent_positions = [o.state_abs_pos[:2] if o.state_abs_pos is not None else None
                                   for o in world.opponents]

        self.my_ori = self.robot_model.imu_torso_orientation
        self.ball_2d = world.ball_abs_pos[:2].copy()
        self.ball_vec = self.ball_2d - self.my_head_pos_2d
        self.ball_dir = M.vector_angle(self.ball_vec)
        self.ball_dist = float(np.linalg.norm(self.ball_vec))
        self.ball_sq_dist = self.ball_dist * self.ball_dist
        self.ball_speed = float(np.linalg.norm(world.get_ball_abs_vel(6)[:2]))

        self.goal_dir = M.target_abs_angle(self.ball_2d, OPPONENT_GOAL_2D)

        self.PM_GROUP = world.play_mode_group
        self.slow_ball_pos = world.get_predicted_ball_pos(0.5)

        self.teammates_ball_sq_dist = [
            np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)
            if p.state_last_update != 0 and (world.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
            else 1000
            for p in world.teammates
        ]
        self.opponents_ball_sq_dist = [
            np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)
            if p.state_last_update != 0 and world.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
            else 1000
            for p in world.opponents
        ]

        self.min_teammate_ball_sq_dist = min(self.teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(self.min_teammate_ball_sq_dist)
        self.min_opponent_ball_dist = math.sqrt(min(self.opponents_ball_sq_dist))

        self.active_player_unum = self.teammates_ball_sq_dist.index(self.min_teammate_ball_sq_dist) + 1

        self.my_desired_position = self.mypos.copy()
        self.my_desired_orientation = self.ball_dir

        # Attack/defense groups
        self.passing_chain = [3, 4, 5]
        self.defending_unums = [1, 2]
        self.current_passer_unum = 5  # kept for compatibility

        self._opps_list = [p for p in self.opponent_positions if p is not None]

        # Shared passer lock across cycles (per agent process)
        if not hasattr(self.world, "_attack_lock"):
            self.world._attack_lock = {"passer_unum": None, "until_ms": 0}

    def IsFormationReady(self, point_preferences):
        is_formation_ready = True
        for i in range(1, 6):
            if i == self.active_player_unum:
                continue
            if i in point_preferences:
                teammate_pos = self.teammate_positions[i-1]
                desired_pos = point_preferences[i]
                if teammate_pos is not None and desired_pos is not None:
                    distance_sq = np.sum((teammate_pos - desired_pos) ** 2)
                    if distance_sq > 0.3 * 0.3:
                        is_formation_ready = False
        return is_formation_ready

    def GetDirectionRelativeToMyPositionAndTarget(self, target):
        target_vec = target - self.my_head_pos_2d
        return M.vector_angle(target_vec)

    def DetermineGameState(self):
        ball_x = float(self.ball_2d[0])
        we_have_possession = self.min_teammate_ball_dist < self.min_opponent_ball_dist
        if we_have_possession and ball_x >= -1.0:
            return "ATTACK"
        elif not we_have_possession and ball_x >= 0.0:
            return "PRESS"
        elif not we_have_possession and ball_x < 0.0:
            return "DEFEND"
        else:
            return "REPOSITION"

    # -------------------- Passer locking & roles --------------------

    def _closest_attacker_to_ball(self):
        """Returns (unum, distance) for closest attacker to the *slow* ball point."""
        best = (None, 1e9)
        for u in self.passing_chain:
            p = self.teammate_positions[u-1]
            if p is None: continue
            d = _dist(p, self.slow_ball_pos)
            if d < best[1] or (abs(d - best[1]) < 1e-6 and (best[0] is None or u < best[0])):
                best = (u, d)
        return best

    def _stable_passer_unum(self):
        """
        Sticky passer:
        - Keep current lock if still near ball OR lock not expired.
        - Otherwise pick the closest attacker, but only switch if better by a margin.
        """
        lock = self.world._attack_lock
        now = self.world.time_local_ms

        # Current lock still valid?
        locked = lock["passer_unum"]
        if locked in self.passing_chain:
            # distance of locked to ball (if visible)
            lp = self.teammate_positions[locked-1]
            if lp is not None:
                ld = _dist(lp, self.slow_ball_pos)
            else:
                ld = 1e9
            if (now < lock["until_ms"]) or (ld <= PASS_LOCK_RADIUS):
                return locked  # keep

        # Otherwise consider switching
        closest_unum, closest_d = self._closest_attacker_to_ball()
        if closest_unum is None:
            return locked if locked in self.passing_chain else self.passing_chain[0]

        # If we had a lock, only switch when clearly better
        if locked in self.passing_chain:
            lp = self.teammate_positions[locked-1]
            locked_d = _dist(lp, self.slow_ball_pos) if lp is not None else 1e9
            if closest_d + PASS_SWITCH_MARGIN >= locked_d:
                # not a strong improvement; keep old
                return locked

        # Switch & (re)lock
        lock["passer_unum"] = closest_unum
        lock["until_ms"] = now + PASS_LOCK_MS
        return closest_unum

    def GetAttackRole(self):
        if self.player_unum not in self.passing_chain:
            return "SUPPORT"

        passer_unum = self._stable_passer_unum()
        if self.player_unum == passer_unum and self.min_opponent_ball_dist > 0.8:
            return "PASSER"

        # Receiver = next in chain from locked passer
        chain = self.passing_chain
        try:
            idx = chain.index(passer_unum)
            receiver_unum = chain[(idx + 1) % len(chain)]
        except ValueError:
            receiver_unum = chain[0]

        if self.player_unum == receiver_unum:
            return "RECEIVER"
        return "ADVANCE"

    # -------------------- Attacking targets --------------------

    def _away_from_pressure_forward_target(self, passer_pos, raw_target, min_ahead=1.6, sidestep=1.1):
        """Ensure target is forward (+X) from passer, and sidestep away from nearest opponent in front."""
        passer_pos = np.array(passer_pos, float)
        tx, ty = float(raw_target[0]), float(raw_target[1])

        # forward bias
        if (tx - passer_pos[0]) < min_ahead:
            tx = passer_pos[0] + min_ahead

        opp, d = _nearest_opponent_ahead(passer_pos, self._opps_list, fwd_cone_deg=40, max_dist=6.0)
        if opp is not None:
            oy = float(opp[1])
            ty = ty + sidestep if (oy - passer_pos[1]) < 0 else ty - sidestep

        tx = min(tx, FIELD_X - 0.6)
        return _clamp((tx, ty))

    def GetPassTargetAndPosition(self, role, passer_pos_2d):
        passer_pos_2d = np.array(passer_pos_2d, float)
        passer_unum = self._stable_passer_unum()
        passer_pos = self.teammate_positions[passer_unum-1] if passer_unum in self.passing_chain else passer_pos_2d

        # --- PASSER: shoot if clear & in range; else forward-away pass
        if role == "PASSER":
            # SHOOT at exactly (15,0)
            goal_target = OPPONENT_GOAL_2D.copy()
            if np.sum((passer_pos_2d - goal_target) ** 2) < SHOOT_DISTANCE_SQ:
                if _has_clear_lane(passer_pos_2d, goal_target, self._opps_list, gap=0.9):
                    return goal_target, 0

            # Receiver is next in chain from *locked passer*
            chain = self.passing_chain
            try:
                idx = chain.index(passer_unum)
                receiver_unum = chain[(idx + 1) % len(chain)]
            except ValueError:
                receiver_unum = chain[0]

            recv_pos = self.teammate_positions[receiver_unum - 1]
            if recv_pos is None:
                recv_pos = passer_pos_2d + np.array([3.0, 0.0])

            fwd_target = self._away_from_pressure_forward_target(passer_pos_2d, recv_pos, min_ahead=1.6, sidestep=1.1)

            if not _has_clear_lane(passer_pos_2d, fwd_target, self._opps_list, gap=OPP_GAP_CLEAR):
                alts = [
                    fwd_target + np.array([0.0, +0.9]),
                    fwd_target + np.array([0.0, -0.9]),
                    passer_pos_2d + np.array([1.6, +0.9]),
                    passer_pos_2d + np.array([1.6, -0.9]),
                ]
                for cand in alts:
                    cand = _clamp(cand)
                    if _has_clear_lane(passer_pos_2d, cand, self._opps_list, gap=OPP_GAP_CLEAR):
                        return cand, receiver_unum
            return fwd_target, receiver_unum

        # --- RECEIVER: lane ahead; DO NOT crowd the passer
        elif role == "RECEIVER":
            lane_index = self.passing_chain.index(self.player_unum) % len(LANE_LATERALS)
            lane_y = LANE_LATERALS[lane_index]
            lead = _receiver_lane_point(self.ball_2d, lead_ahead=LEAD_AHEAD_BASE, lane_y=lane_y)

            # stay out of passer’s bubble; repel from passer & other attackers
            repel_points = [self.teammate_positions[u-1] for u in self.passing_chain if u != self.player_unum]
            if passer_pos is not None and _dist(lead, passer_pos) < RECEIVER_IGNORE_R:
                # nudge laterally away from passer
                sign = -1.0 if (lead[1] >= passer_pos[1]) else 1.0
                lead[1] += sign * (RECEIVER_IGNORE_R - _dist(lead, passer_pos) + 0.3)
            lead = _repel_from_points(lead, repel_points)
            return lead, None

        # --- ADVANCE: wider lane; also respect passer bubble
        elif role == "ADVANCE":
            lane_index = self.passing_chain.index(self.player_unum) % len(LANE_LATERALS)
            lane_y = LANE_LATERALS[lane_index] * 1.3
            lead = _receiver_lane_point(self.ball_2d, lead_ahead=LEAD_AHEAD_BASE + 1.0, lane_y=lane_y)

            repel_points = [self.teammate_positions[u-1] for u in self.passing_chain if u != self.player_unum]
            if passer_pos is not None and _dist(lead, passer_pos) < RECEIVER_IGNORE_R:
                sign = -1.0 if (lead[1] >= passer_pos[1]) else 1.0
                lead[1] += sign * (RECEIVER_IGNORE_R - _dist(lead, passer_pos) + 0.3)
            lead = _repel_from_points(lead, repel_points)
            return lead, None

        # SUPPORT/DEFENSE fallback – keep assigned formation
        else:
            return self.my_desired_position, None

    # -------------------- Defense (unchanged) --------------------

    def GetDefenseTarget(self):
        min_opp_sq_dist = min(self.opponents_ball_sq_dist)
        opponent_passer_unum = self.opponents_ball_sq_dist.index(min_opp_sq_dist) + 1

        if opponent_passer_unum == 5:
            intercept_target_pos = OUR_GOAL_2D + np.array([2.0, 0.0])
        else:
            opponent_receiver_unum = opponent_passer_unum + 1
            opponent_receiver_pos = self.opponent_positions[opponent_receiver_unum - 1]
            if opponent_receiver_pos is not None:
                intercept_target_pos = np.array(opponent_receiver_pos, float)
            else:
                intercept_target_pos = np.array([-5.0, 0.0], float)

        if self.player_unum == self.defending_unums[0] and self.min_teammate_ball_dist > 1.5:
            intercept_target_pos, _ = self.world.get_intersection_point_with_ball(1.5)
            intercept_target_pos = np.array(intercept_target_pos, float)

        if intercept_target_pos[0] > -1.0:
            intercept_target_pos[0] = -1.0
        return _clamp(intercept_target_pos)
