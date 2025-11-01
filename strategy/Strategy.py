import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M


# --- CONSTANTS ---
OPPONENT_GOAL_2D = np.array([15.0, 0.0], dtype=float)
OUR_GOAL_2D      = np.array([-15.0, 0.0], dtype=float)

# Goal post positions (based on goal width ~2.1m in simulation)
GOAL_POST_LEFT = np.array([15.0, 1.05], dtype=float)   # left post from attacker's view
GOAL_POST_RIGHT = np.array([15.0, -1.05], dtype=float)  # right post from attacker's view

SHOOT_DISTANCE   = 7.0
SHOOT_DISTANCE_SQ= SHOOT_DISTANCE * SHOOT_DISTANCE


# Lanes & spacing
LANE_LATERALS    = [-1.8, 0.0, +1.8]   # y-lanes used by attackers (spread)
LEAD_AHEAD_BASE  = 3.5                  # m ahead of ball for support lanes
PERSONAL_SPACE   = 1.25                 # m – attackers keep at least this much distance
REPULSION_GAIN   = 0.7                  # strength of anti-crowding push


# Chase & locking
PASS_LOCK_MS         = 800              # how long we "stick" to the same passer (increased from 600)
PASS_LOCK_RADIUS     = 3.2              # keep passer if still within this to ball (increased from 2.8)
PASS_SWITCH_MARGIN   = 1.2              # only switch if candidate is this much closer (increased from 1.0)
RECEIVER_IGNORE_R    = 2.2              # receivers won't approach within this of passer
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
import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M


# --- CONSTANTS ---
OPPONENT_GOAL_2D = np.array([15.0, 0.0], dtype=float)
OUR_GOAL_2D      = np.array([-15.0, 0.0], dtype=float)

# Goal post positions (based on goal width ~2.1m in simulation)
GOAL_POST_LEFT = np.array([15.0, 1.05], dtype=float)   # left post from attacker's view
GOAL_POST_RIGHT = np.array([15.0, -1.05], dtype=float)  # right post from attacker's view

SHOOT_DISTANCE   = 7.0
SHOOT_DISTANCE_SQ= SHOOT_DISTANCE * SHOOT_DISTANCE


# Lanes & spacing
LANE_LATERALS    = [-1.8, 0.0, +1.8]   # y-lanes used by attackers (spread)
LEAD_AHEAD_BASE  = 3.5                  # m ahead of ball for support lanes
PERSONAL_SPACE   = 1.25                 # m – attackers keep at least this much distance
REPULSION_GAIN   = 0.7                  # strength of anti-crowding push


# Chase & locking
PASS_LOCK_MS         = 800              # how long we "stick" to the same passer (increased from 600)
PASS_LOCK_RADIUS     = 3.2              # keep passer if still within this to ball (increased from 2.8)
PASS_SWITCH_MARGIN   = 1.2              # only switch if candidate is this much closer (increased from 1.0)
RECEIVER_IGNORE_R    = 2.2              # receivers won't approach within this of passer
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


def _calculate_smart_shot_target(shooter_pos, opps):
    """
    Calculate where to shoot to avoid the keeper and maximize scoring chance.
    Strategy:
    - Aim for corners/sides of goal (low success rate in center)
    - Prefer bottom corners (statistics show 62% of goals scored low)
    - Avoid shooting directly at visible opponents in goal area
    
    Returns: (target_x, target_y) numpy array
    """
    sx, sy = shooter_pos[0], shooter_pos[1]
    
    # Default targets: aim for goal posts with small inward offset (0.3m from post)
    # This keeps shots on target while avoiding center
    target_left = np.array([15.0, 0.75], float)   # aim inside left post
    target_right = np.array([15.0, -0.75], float) # aim inside right post
    
    # Check if there are opponents near goal (likely goalkeeper)
    keeper_y = None
    if opps:
        for opp in opps:
            if opp is None:
                continue
            ox, oy = opp[0], opp[1]
            # Check if opponent is in goal area (x > 13.5, near goal line)
            if ox > 13.5:
                keeper_y = oy
                break
    
    # If we detected a keeper, shoot away from them
    if keeper_y is not None:
        if keeper_y > 0:
            # Keeper on left (positive y), shoot right
            return target_right
        else:
            # Keeper on right (negative y), shoot left
            return target_left
    
    # No keeper detected: shoot to the side opposite to where shooter is positioned
    # This creates a diagonal shot which is harder to save
    if sy > 0.3:
        # Shooter on left side of field, shoot to right post
        return target_right
    elif sy < -0.3:
        # Shooter on right side of field, shoot to left post
        return target_left
    else:
        # Shooter is central: alternate or pick based on slight bias
        # Pick the side that's farther from any nearby opponent
        left_safe = True
        right_safe = True
        if opps:
            for opp in opps:
                if opp is None:
                    continue
                ox, oy = opp[0], opp[1]
                if ox > 12.0:  # opponent in attacking third
                    if _dist(target_left, (ox, oy)) < 2.0:
                        left_safe = False
                    if _dist(target_right, (ox, oy)) < 2.0:
                        right_safe = False
        
        if left_safe and not right_safe:
            return target_left
        elif right_safe and not left_safe:
            return target_right
        else:
            # Both safe or both blocked: default to left
            return target_left


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
            if d < best[1]:
                best = (u, d)
            elif abs(d - best[1]) < 0.05 and (best[0] is None or u < best[0]):
                # Tie-breaker: prefer lower unum for consistency
                best = (u, d)
        return best


    def _stable_passer_unum(self):
        """
        Sticky passer with improved switching logic:
        - Keep current lock if still reasonably close to ball OR lock not expired.
        - Only switch if new candidate is significantly closer (margin).
        """
        lock = self.world._attack_lock
        now = self.world.time_local_ms


        # Current lock still valid?
        locked = lock["passer_unum"]
        if locked in self.passing_chain:
            # distance of locked player to ball
            lp = self.teammate_positions[locked-1]
            if lp is not None:
                ld = _dist(lp, self.slow_ball_pos)
            else:
                ld = 1e9
            
            # Keep lock if: time not expired OR still close to ball
            if (now < lock["until_ms"]) or (ld <= PASS_LOCK_RADIUS):
                return locked  # keep


        # Consider switching
        closest_unum, closest_d = self._closest_attacker_to_ball()
        if closest_unum is None:
            return locked if locked in self.passing_chain else self.passing_chain[0]


        # If we had a lock, only switch when clearly better
        if locked in self.passing_chain:
            lp = self.teammate_positions[locked-1]
            locked_d = _dist(lp, self.slow_ball_pos) if lp is not None else 1e9
            # Require new candidate to be at least PASS_SWITCH_MARGIN closer
            if closest_d + PASS_SWITCH_MARGIN >= locked_d:
                # Not a strong enough improvement; keep old
                lock["until_ms"] = now + PASS_LOCK_MS  # refresh timer
                return locked


        # Switch & (re)lock
        lock["passer_unum"] = closest_unum
        lock["until_ms"] = now + PASS_LOCK_MS
        return closest_unum


    def GetAttackRole(self):
        """
        Determine role for attacking players.
        Returns: "PASSER", "RECEIVER", "ADVANCE", or "SUPPORT"
        """
        if self.player_unum not in self.passing_chain:
            return "SUPPORT"


        passer_unum = self._stable_passer_unum()
        
        # PASSER: the locked closest player, but only if opponent not too close
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
        """
        Get the target position and pass recipient for the given role.
        Now supports shooting for ALL roles when in range!
        
        Returns: (target_position, receiver_unum or 0 for shoot)
        """
        passer_pos_2d = np.array(passer_pos_2d, float)
        passer_unum = self._stable_passer_unum()
        passer_pos = self.teammate_positions[passer_unum-1] if passer_unum in self.passing_chain else passer_pos_2d


        # ========== SHOOTING LOGIC (for ALL roles) ==========
        # Check if this player is in shooting range
        dist_to_goal_sq = np.sum((passer_pos_2d - OPPONENT_GOAL_2D) ** 2)
        in_shooting_range = dist_to_goal_sq < SHOOT_DISTANCE_SQ
        
        if in_shooting_range:
            # Calculate smart shooting target (away from keeper, toward corners)
            shoot_target = _calculate_smart_shot_target(passer_pos_2d, self._opps_list)
            
            # Check if we have a clear lane to shoot
            if _has_clear_lane(passer_pos_2d, shoot_target, self._opps_list, gap=0.85):
                return shoot_target, 0  # 0 indicates shooting, not passing
        
        
        # ========== ROLE-SPECIFIC BEHAVIOR (if not shooting) ==========
        
        # --- PASSER: forward-away pass to receiver
        if role == "PASSER":
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


            # stay out of passer's bubble; repel from passer & other attackers
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


    # ==================== ANTI-BASELINE DEFENSE ====================
    
    def GetOpponentWithBall(self):
        """
        Determine which opponent (unum 1-5) is closest to the ball.
        Returns opponent unum (1-5) or None
        """
        if not self.opponents_ball_sq_dist:
            return None
        min_dist = min(self.opponents_ball_sq_dist)
        if min_dist > 100:  # no visible opponents
            return None
        return self.opponents_ball_sq_dist.index(min_dist) + 1
    
    
    def GetBaselineNextReceiver(self, current_holder_unum):
        """
        Baseline passes sequentially: 1→2→3→4→5
        Returns the next receiver unum, or None if current holder is unknown
        """
        if current_holder_unum is None:
            return None
        # Baseline cycles: 1→2→3→4→5→(back to 1 after goal kick, etc.)
        if current_holder_unum >= 5:
            return 1  # wrap around (though usually 5 shoots)
        return current_holder_unum + 1
    
    
    def GetPassLaneInterceptPoint(self, passer_unum, receiver_unum):
        """
        Calculate optimal interception point between passer and receiver.
        Returns: interception point (x, y) or None if positions unknown
        """
        if passer_unum is None or receiver_unum is None:
            return None
        
        passer_pos = self.opponent_positions[passer_unum - 1]
        receiver_pos = self.opponent_positions[receiver_unum - 1]
        
        if passer_pos is None or receiver_pos is None:
            return None
        
        # Intercept at midpoint of passing lane, slightly closer to receiver
        # (gives us time to close down after they receive)
        passer_pos = np.array(passer_pos, float)
        receiver_pos = np.array(receiver_pos, float)
        
        # 60% toward receiver from passer
        intercept = passer_pos + 0.6 * (receiver_pos - passer_pos)
        
        # Ensure we're in our defensive half mostly
        intercept[0] = min(intercept[0], 5.0)  # don't chase too far forward
        
        return _clamp(intercept)
    
    
    def GetPlayer5MarkingPosition(self):
        """
        Get position to tightly mark opponent player 5 (the only shooter).
        Returns: marking position (x, y) or None
        """
        opp5_pos = self.opponent_positions[4]  # player 5 is index 4
        if opp5_pos is None:
            return None
        
        opp5_pos = np.array(opp5_pos, float)
        ball_pos = self.ball_2d
        
        # Position between ball and player 5, closer to player 5
        # This blocks passing lane to them
        mark_vec = opp5_pos - ball_pos
        mark_dist = np.linalg.norm(mark_vec)
        
        if mark_dist < 0.1:
            # They're at the ball, mark tightly
            return _clamp(opp5_pos + np.array([-0.8, 0.0]))
        
        # Position 70% of the way from ball to player 5
        mark_pos = ball_pos + 0.7 * mark_vec
        
        # If player 5 is in shooting range, mark even tighter
        dist_to_our_goal = _dist(opp5_pos, OUR_GOAL_2D)
        if dist_to_our_goal < 8.0:
            # Tight marking: get between them and our goal
            to_goal = OUR_GOAL_2D - opp5_pos
            to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-6)
            mark_pos = opp5_pos + 0.9 * to_goal_norm
        
        return _clamp(mark_pos)


    def GetAntiBaselineDefenseTarget(self):
        """
        MAIN ANTI-BASELINE DEFENSE COORDINATOR
        
        Strategy:
        1. Identify opponent ball holder
        2. Predict next receiver (sequential passing)
        3. Position defenders to:
           - Press ball holder (Player 1)
           - Block passing lane to next receiver (Player 2)
           - Tightly mark player 5 (Player 4)
           - Sweep behind (Player 3)
        
        Returns: defensive target position for current player
        """
        my_unum = self.player_unum
        ball_holder = self.GetOpponentWithBall()
        next_receiver = self.GetBaselineNextReceiver(ball_holder)
        
        ball_x = self.ball_2d[0]
        
        # ===== ROLE: GOALKEEPER (Player 1) =====
        if my_unum == 1:
            # Stay near goal, move laterally with ball
            goal_y_offset = np.clip(self.ball_2d[1] * 0.4, -2.0, 2.0)
            return _clamp(np.array([-14.0, goal_y_offset]))
        
        # ===== ROLE: AGGRESSIVE PRESSER (Player 2) =====
        elif my_unum == 2:
            # If ball is in our half: PRESS the ball holder aggressively
            if ball_x < 2.0:
                # Chase ball holder directly
                if ball_holder and self.opponent_positions[ball_holder-1] is not None:
                    return _clamp(self.opponent_positions[ball_holder-1])
                # If we don't see holder, press toward ball
                return _clamp(self.ball_2d)
            else:
                # Ball in their half: hold defensive midfield position
                return _clamp(np.array([-6.0, np.clip(self.ball_2d[1] * 0.5, -4.0, 4.0)]))
        
        # ===== ROLE: PASSING LANE INTERCEPTOR (Player 3) =====
        elif my_unum == 3:
            # Block the passing lane between current holder and next receiver
            if ball_holder and next_receiver:
                intercept_pt = self.GetPassLaneInterceptPoint(ball_holder, next_receiver)
                if intercept_pt is not None:
                    return intercept_pt
            
            # Fallback: central defensive midfielder
            return _clamp(np.array([-3.0, np.clip(self.ball_2d[1] * 0.6, -5.0, 5.0)]))
        
        # ===== ROLE: PLAYER 5 MARKER (Player 4) =====
        elif my_unum == 4:
            # Tightly mark opponent player 5 (the only shooter!)
            mark_pos = self.GetPlayer5MarkingPosition()
            if mark_pos is not None:
                return mark_pos
            
            # Fallback: shadow position
            return _clamp(np.array([-1.0, np.clip(self.ball_2d[1], -6.0, 6.0)]))
        
        # ===== ROLE: DEFENSIVE SWEEPER (Player 5) =====
        elif my_unum == 5:
            # Sweep behind the defense, cover gaps
            # Position based on ball location and opponent threats
            
            # If opponent player 5 is advancing, help mark them
            opp5_pos = self.opponent_positions[4]
            if opp5_pos is not None:
                opp5_x = opp5_pos[0]
                # If they're deep in our half, drop back
                if opp5_x < -5.0:
                    return _clamp(np.array([-8.0, np.clip(self.ball_2d[1] * 0.5, -4.0, 4.0)]))
            
            # Otherwise: high defensive midfielder position to intercept
            return _clamp(np.array([1.0, np.clip(self.ball_2d[1] * 0.7, -6.0, 6.0)]))
        
        # Fallback: return to own half
        return _clamp(np.array([-5.0, 0.0]))


    # -------------------- Defense (UPDATED) --------------------


    def GetDefenseTarget(self):
        """
        Updated defense target that uses anti-baseline strategy
        """
        # Use anti-baseline defense strategy
        return self.GetAntiBaselineDefenseTarget()
