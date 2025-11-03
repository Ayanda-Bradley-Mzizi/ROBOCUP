import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M

# --- CONSTANTS ---
OPPONENT_GOAL_2D = np.array([15.0, 0.0], dtype=float)
OUR_GOAL_2D      = np.array([-15.0, 0.0], dtype=float)

GOAL_POST_LEFT   = np.array([15.0,  1.05], dtype=float)
GOAL_POST_RIGHT  = np.array([15.0, -1.05], dtype=float)

SHOOT_DISTANCE    = 7.0
SHOOT_DISTANCE_SQ = SHOOT_DISTANCE * SHOOT_DISTANCE

# Lanes & spacing
LANE_LATERALS   = [-1.8, 0.0, 1.8]
LEAD_AHEAD_BASE = 3.5
PERSONAL_SPACE  = 1.25
REPULSION_GAIN  = 0.7

# Passer locking
PASS_LOCK_MS        = 800
PASS_LOCK_RADIUS    = 3.2
PASS_SWITCH_MARGIN  = 1.2
RECEIVER_IGNORE_R   = 2.2
CHASE_ONLY_PASSER_R = 4.0

# Lane clearance
OPP_GAP_CLEAR = 0.80

# Field clamps
FIELD_X = 15.0
FIELD_Y = 10.0


def _clamp(p):
    return np.array([np.clip(p[0], -FIELD_X, FIELD_X),
                     np.clip(p[1], -FIELD_Y, FIELD_Y)], dtype=float)


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _has_clear_lane(a, b, opps, gap=OPP_GAP_CLEAR):
    """Returns True if segment a->b has no opponent within 'gap' distance."""
    if not opps:
        return True
    a = np.array(a, float); b = np.array(b, float)
    ab = b - a
    ab2 = float(np.dot(ab, ab)) if np.dot(ab, ab) > 1e-8 else 1e-8
    for o in opps:
        if o is None:
            continue
        p = np.array(o, float)
        t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
        proj = a + t * ab
        if np.linalg.norm(p - proj) < gap:
            return False
    return True


def _nearest_opponent_ahead(ref_xy, opps, fwd_cone_deg=60.0, max_dist=12.0):
    """Nearest opponent in a +X cone from ref point (returns (pos, dist) or (None, None))."""
    if not opps:
        return None, None
    rx, ry = ref_xy
    best = (None, None)
    cos_th = math.cos(math.radians(fwd_cone_deg * 0.5))
    for o in opps:
        if o is None:
            continue
        ox, oy = o
        vx, vy = (ox - rx), (oy - ry)
        d = math.hypot(vx, vy)
        if d == 0 or d > max_dist:
            continue
        dot = vx / d
        if dot >= cos_th:
            if best[1] is None or d < best[1]:
                best = (np.array([ox, oy], float), d)
    return best


def _receiver_lane_point(ball_xy, lead_ahead=LEAD_AHEAD_BASE, lane_y=0.0):
    bx, by = ball_xy
    return _clamp(np.array([min(FIELD_X - 0.8, bx + lead_ahead), by + lane_y], float))


def _repel_from_points(my_target, points, min_space=PERSONAL_SPACE, gain=REPULSION_GAIN):
    """Push target away from nearby points to avoid bunching."""
    t = np.array(my_target, float)
    for p in points:
        if p is None:
            continue
        d = _dist(t, p)
        if d < 1e-6:
            t[1] += min_space
            continue
        if d < min_space:
            push = (t - p).astype(float)
            push[0] *= 0.35
            n = np.linalg.norm(push) + 1e-6
            t += gain * (min_space - d) * (push / n)
    return _clamp(t)


def _closest_point_on_segment(A, B, P):
    """Closest point on segment AB to P."""
    A = np.array(A, float); B = np.array(B, float); P = np.array(P, float)
    AB = B - A
    AP = P - A
    mag_sq_AB = np.dot(AB, AB)
    if mag_sq_AB == 0.0:
        return A
    t = np.clip(np.dot(AP, AB) / mag_sq_AB, 0.0, 1.0)
    return A + t * AB


def _calculate_smart_shot_target(shooter_pos, opps):
    """Keeper-aware shot: prefer corners; avoid nearby opponents in goal area."""
    sy = shooter_pos[1]
    target_left  = np.array([15.0,  0.3], float)
    target_right = np.array([15.0, -0.3], float)

    keeper_y = None
    if opps:
        for opp in opps:
            if opp is None:
                continue
            ox, oy = opp[0], opp[1]
            if ox > 13.5:
                keeper_y = oy
                break

    if keeper_y is not None:
        return target_right if keeper_y > 0 else target_left

    if sy > 0.3:
        return target_right
    if sy < -0.3:
        return target_left

    left_safe, right_safe = True, True
    if opps:
        for opp in opps:
            if opp is None:
                continue
            ox, oy = opp[0], opp[1]
            if ox > 12.0:
                if _dist(target_left,  (ox, oy)) < 2.0:
                    left_safe = False
                if _dist(target_right, (ox, oy)) < 2.0:
                    right_safe = False
    if left_safe and not right_safe:
        return target_left
    if right_safe and not left_safe:
        return target_right
    return target_left


class Strategy:
    def __init__(self, world):
        self.world = world
        self.play_mode = world.play_mode
        self.robot_model = world.robot
        self.my_head_pos_2d = self.robot_model.loc_head_position[:2].copy()
        self.player_unum = self.robot_model.unum
        self.mypos = self.my_head_pos_2d

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

        self.passing_chain = [3, 4, 5]
        self.defending_unums = [1, 2]
        self.current_passer_unum = 5

        self._opps_list = [p for p in self.opponent_positions if p is not None]

        if not hasattr(self.world, "_attack_lock"):
            self.world._attack_lock = {"passer_unum": None, "until_ms": 0}

    def IsFormationReady(self, point_preferences):
        """Returns True if non-active players are near their desired points."""
        is_ready = True
        for i in range(1, 6):
            if i == self.active_player_unum:
                continue
            if i in point_preferences:
                teammate_pos = self.teammate_positions[i - 1]
                desired_pos = point_preferences[i]
                if teammate_pos is not None and desired_pos is not None:
                    if np.sum((teammate_pos - desired_pos) ** 2) > 0.3 * 0.3:
                        is_ready = False
        return is_ready

    def IsOurKickoff(self):
        return self.play_mode == self.world.M_OUR_KICKOFF

    def GetDirectionRelativeToMyPositionAndTarget(self, target):
        return M.vector_angle(target - self.my_head_pos_2d)

    def DetermineGameState(self):
        """ATTACK / PRESS / DEFEND / REPOSITION based on ball x and possession."""
        ball_x = float(self.ball_2d[0])
        we_have_possession = self.min_teammate_ball_dist < self.min_opponent_ball_dist
        if we_have_possession and ball_x >= -1.0:
            return "ATTACK"
        if not we_have_possession and ball_x >= 0.0:
            return "PRESS"
        if not we_have_possession and ball_x < 0.0:
            return "DEFEND"
        return "REPOSITION"

    # ---- Passer locking & roles ----

    def _closest_attacker_to_ball(self):
        """(unum, dist) for closest attacker to slow ball point."""
        best = (None, 1e9)
        for u in self.passing_chain:
            p = self.teammate_positions[u - 1]
            if p is None:
                continue
            d = _dist(p, self.slow_ball_pos)
            if d < best[1] or (abs(d - best[1]) < 0.05 and (best[0] is None or u < best[0])):
                best = (u, d)
        return best

    def _stable_passer_unum(self):
        """Sticky passer with conservative switching."""
        lock = self.world._attack_lock
        now = self.world.time_local_ms

        locked = lock["passer_unum"]
        if locked in self.passing_chain:
            lp = self.teammate_positions[locked - 1]
            ld = _dist(lp, self.slow_ball_pos) if lp is not None else 1e9
            if (now < lock["until_ms"]) or (ld <= PASS_LOCK_RADIUS):
                return locked

        closest_unum, closest_d = self._closest_attacker_to_ball()
        if closest_unum is None:
            return locked if locked in self.passing_chain else self.passing_chain[0]

        if locked in self.passing_chain:
            lp = self.teammate_positions[locked - 1]
            locked_d = _dist(lp, self.slow_ball_pos) if lp is not None else 1e9
            if closest_d + PASS_SWITCH_MARGIN >= locked_d:
                lock["until_ms"] = now + PASS_LOCK_MS
                return locked

        lock["passer_unum"] = closest_unum
        lock["until_ms"] = now + PASS_LOCK_MS
        return closest_unum

    def GetAttackRole(self):
        """Return PASSER / RECEIVER / ADVANCE / SUPPORT."""
        if self.player_unum not in self.passing_chain:
            return "SUPPORT"

        passer_unum = self._stable_passer_unum()
        if self.player_unum == passer_unum and self.min_opponent_ball_dist > 0.8:
            return "PASSER"

        chain = self.passing_chain
        try:
            idx = chain.index(passer_unum)
            receiver_unum = chain[(idx + 1) % len(chain)]
        except ValueError:
            receiver_unum = chain[0]

        if self.player_unum == receiver_unum:
            return "RECEIVER"
        return "ADVANCE"

    # ---- Attacking targets ----

    def _away_from_pressure_forward_target(self, passer_pos, raw_target, min_ahead=1.6, sidestep=0.5):
        """Forward bias + sidestep away from nearest front pressure."""
        passer_pos = np.array(passer_pos, float)
        tx, ty = float(raw_target[0]), float(raw_target[1])
        if (tx - passer_pos[0]) < min_ahead:
            tx = passer_pos[0] + min_ahead
        opp, _ = _nearest_opponent_ahead(passer_pos, self._opps_list, fwd_cone_deg=40, max_dist=6.0)
        if opp is not None:
            oy = float(opp[1])
            ty = ty + sidestep if (oy - passer_pos[1]) < 0 else ty - sidestep
        tx = min(tx, FIELD_X - 0.6)
        return _clamp((tx, ty))

    def GetPassTargetAndPosition(self, role, passer_pos_2d):
        """Return (target_position, receiver_unum) or (shoot_target, 0)."""
        passer_pos_2d = np.array(passer_pos_2d, float)
        passer_unum = self._stable_passer_unum()
        passer_pos = self.teammate_positions[passer_unum - 1] if passer_unum in self.passing_chain else passer_pos_2d

        # Shooting option for all roles
        dist_to_goal_sq = np.sum((passer_pos_2d - OPPONENT_GOAL_2D) ** 2)
        if dist_to_goal_sq < SHOOT_DISTANCE_SQ:
            shoot_target = _calculate_smart_shot_target(passer_pos_2d, self._opps_list)
            if _has_clear_lane(passer_pos_2d, shoot_target, self._opps_list, gap=0.75):
                return shoot_target, 0

        # PASSER
        if role == "PASSER":
            chain = self.passing_chain
            try:
                idx = chain.index(passer_unum)
                receiver_unum = chain[(idx + 1) % len(chain)]
            except ValueError:
                receiver_unum = chain[0]

            recv_pos = self.teammate_positions[receiver_unum - 1]
            if recv_pos is None:
                recv_pos = passer_pos_2d + np.array([3.0, 0.0])
            fwd_target = self._away_from_pressure_forward_target(passer_pos_2d, recv_pos, min_ahead=1.6, sidestep=0.5)

            if not _has_clear_lane(passer_pos_2d, fwd_target, self._opps_list, gap=OPP_GAP_CLEAR):
                alts = [
                    fwd_target + np.array([0.0,  0.9]),
                    fwd_target + np.array([0.0, -0.9]),
                    passer_pos_2d + np.array([1.6,  0.9]),
                    passer_pos_2d + np.array([1.6, -0.9]),
                ]
                for cand in alts:
                    cand = _clamp(cand)
                    if _has_clear_lane(passer_pos_2d, cand, self._opps_list, gap=OPP_GAP_CLEAR):
                        return cand, receiver_unum
            return fwd_target, receiver_unum

        # RECEIVER
        elif role == "RECEIVER":
            lane_index = self.passing_chain.index(self.player_unum) % len(LANE_LATERALS)
            lane_y = LANE_LATERALS[lane_index]
            lead = _receiver_lane_point(self.ball_2d, lead_ahead=LEAD_AHEAD_BASE, lane_y=lane_y)

            repel_points = [self.teammate_positions[u - 1] for u in self.passing_chain if u != self.player_unum]

            if self.ball_2d[0] > 8.0:
                P = np.array(lead)
                A = self.ball_2d
                closest_pt1 = _closest_point_on_segment(A, np.array([15.0,  0.3]), P)
                closest_pt2 = _closest_point_on_segment(A, np.array([15.0, -0.3]), P)
                repel_points.extend([closest_pt1, closest_pt2])

            passer_pos = passer_pos if (passer_pos := passer_pos) is not None else None
            if passer_pos is not None and _dist(lead, passer_pos) < RECEIVER_IGNORE_R:
                sign = -1.0 if (lead[1] >= passer_pos[1]) else 1.0
                lead[1] += sign * (RECEIVER_IGNORE_R - _dist(lead, passer_pos) + 0.3)

            lead = _repel_from_points(lead, repel_points)
            return lead, None

        # ADVANCE
        elif role == "ADVANCE":
            lane_index = self.passing_chain.index(self.player_unum) % len(LANE_LATERALS)
            lane_y = LANE_LATERALS[lane_index] * 1.3
            lead = _receiver_lane_point(self.ball_2d, lead_ahead=LEAD_AHEAD_BASE + 1.0, lane_y=lane_y)

            repel_points = [self.teammate_positions[u - 1] for u in self.passing_chain if u != self.player_unum]

            if self.ball_2d[0] > 8.0:
                P = np.array(lead)
                A = self.ball_2d
                closest_pt1 = _closest_point_on_segment(A, np.array([15.0,  0.3]), P)
                closest_pt2 = _closest_point_on_segment(A, np.array([15.0, -0.3]), P)
                repel_points.extend([closest_pt1, closest_pt2])

            passer_pos = passer_pos if (passer_pos := passer_pos) is not None else None
            if passer_pos is not None and _dist(lead, passer_pos) < RECEIVER_IGNORE_R:
                sign = -1.0 if (lead[1] >= passer_pos[1]) else 1.0
                lead[1] += sign * (RECEIVER_IGNORE_R - _dist(lead, passer_pos) + 0.3)

            lead = _repel_from_points(lead, repel_points)
            return lead, None

        # SUPPORT/DEFENSE fallback
        else:
            return self.my_desired_position, None

    # ---- Anti-baseline defense ----

    def GetOpponentWithBall(self):
        """Opponent unum closest to ball, or None."""
        if not self.opponents_ball_sq_dist:
            return None
        min_dist = min(self.opponents_ball_sq_dist)
        if min_dist > 100:
            return None
        return self.opponents_ball_sq_dist.index(min_dist) + 1

    def GetBaselineNextReceiver(self, current_holder_unum):
        """Sequential next receiver 1→2→3→4→5 (wraps to 1)."""
        if current_holder_unum is None:
            return None
        if current_holder_unum >= 5:
            return 1
        return current_holder_unum + 1

    def GetPassLaneInterceptPoint(self, passer_unum, receiver_unum):
        """Mid-lane interception point, slightly biased toward receiver."""
        if passer_unum is None or receiver_unum is None:
            return None
        passer_pos = self.opponent_positions[passer_unum - 1]
        receiver_pos = self.opponent_positions[receiver_unum - 1]
        if passer_pos is None or receiver_pos is None:
            return None
        passer_pos = np.array(passer_pos, float)
        receiver_pos = np.array(receiver_pos, float)
        intercept = passer_pos + 0.6 * (receiver_pos - passer_pos)
        intercept[0] = min(intercept[0], 5.0)
        return _clamp(intercept)

    def GetPlayer5MarkingPosition(self):
        """Mark opponent player 5; bias between them and ball/goal as needed."""
        opp5_pos = self.opponent_positions[4]
        if opp5_pos is None:
            return None
        opp5_pos = np.array(opp5_pos, float)
        ball_pos = self.ball_2d

        mark_vec = opp5_pos - ball_pos
        mark_dist = np.linalg.norm(mark_vec)
        if mark_dist < 0.1:
            return _clamp(opp5_pos + np.array([-0.8, 0.0]))

        mark_pos = ball_pos + 0.7 * mark_vec
        if _dist(opp5_pos, OUR_GOAL_2D) < 8.0:
            to_goal = OUR_GOAL_2D - opp5_pos
            to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-6)
            mark_pos = opp5_pos + 0.9 * to_goal_norm
        return _clamp(mark_pos)

    def GetAntiBaselineDefenseTarget(self):
        """Coordinator for anti-baseline defense roles."""
        my_unum = self.player_unum
        ball_holder = self.GetOpponentWithBall()
        next_receiver = self.GetBaselineNextReceiver(ball_holder)
        ball_x = self.ball_2d[0]

        if my_unum == 1:
            goal_y_offset = np.clip(self.ball_2d[1] * 0.4, -2.0, 2.0)
            return _clamp(np.array([-14.0, goal_y_offset]))

        elif my_unum == 2:
            if ball_x < 2.0:
                if ball_holder and self.opponent_positions[ball_holder - 1] is not None:
                    return _clamp(self.opponent_positions[ball_holder - 1])
                return _clamp(self.ball_2d)
            return _clamp(np.array([-6.0, np.clip(self.ball_2d[1] * 0.5, -4.0, 4.0)]))

        elif my_unum == 3:
            if ball_holder and next_receiver:
                intercept_pt = self.GetPassLaneInterceptPoint(ball_holder, next_receiver)
                if intercept_pt is not None:
                    return intercept_pt
            return _clamp(np.array([-3.0, np.clip(self.ball_2d[1] * 0.6, -5.0, 5.0)]))

        elif my_unum == 4:
            mark_pos = self.GetPlayer5MarkingPosition()
            if mark_pos is not None:
                return mark_pos
            return _clamp(np.array([-1.0, np.clip(self.ball_2d[1], -6.0, 6.0)]))

        elif my_unum == 5:
            opp5_pos = self.opponent_positions[4]
            if opp5_pos is not None and opp5_pos[0] < -5.0:
                return _clamp(np.array([-8.0, np.clip(self.ball_2d[1] * 0.5, -4.0, 4.0)]))
            return _clamp(np.array([1.0, np.clip(self.ball_2d[1] * 0.7, -6.0, 6.0)]))

        return _clamp(np.array([-5.0, 0.0]))

    def GetDefenseTarget(self):
        """Top-level defense target."""
        return self.GetAntiBaselineDefenseTarget()
