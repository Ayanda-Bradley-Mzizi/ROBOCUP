from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np


from strategy.Assignment import role_assignment
from strategy.Strategy import Strategy


from formation.Formation import GetFormationForSituation  # changed thsi from formation.Formation import GenerateBasicFormation



class Agent(Base_Agent):
    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:


        # define robot type
        robot_type = (0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4)[unum - 1]


        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)


        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)  # filtered walk parameters for fat proxy


        self.init_pos = ([-14, 0], [-9, -5], [-9, 0], [-9, 5], [-5, -5], [-5, 0], [-5, 5], [-1, -6], [-1, -2.5], [-1, 2.5], [-1, 6])[unum - 1]  # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:]  # copy position list
        self.state = 0


        # Avoid center circle by moving the player back
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3


        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0], -pos[1])))  # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None:  # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:  # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)  # reset fat proxy walk


    def move(self, target_2d=(0, 0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position
        '''
        r = self.world.robot


        if self.fat_proxy_cmd is not None:  # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute)  # ignore obstacles
            return


        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])


        # Args: target, is_target_abs, ori, is_ori_abs, distance
        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)


    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick
        '''
        return self.behavior.execute("Dribble", None, None)


        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()


        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance


        if self.fat_proxy_cmd is None:  # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)  # Basic_Kick has no kick distance control
        else:  # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0, 0), target_2d=(0, 0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick toward a specific target
        '''
        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)


        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)


        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])


        # Convert direction to degrees
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()


        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance


        if self.fat_proxy_cmd is None:  # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)  # Basic_Kick has no kick distance control
        else:  # fat proxy behavior
            return self.fat_proxy_kick()


    def think_and_send(self):


        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw


        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True)  # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        # --------------------------------------- 3. Broadcast
        self.radio.broadcast()


        # --------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None:  # normal behavior
            self.scom.commit_and_send(strategyData.robot_model.get_command())
        else:  # fat proxy behavior
            self.scom.commit_and_send(self.fat_proxy_cmd.encode())
            self.fat_proxy_cmd = ""


    def select_skill(self, strategyData):
        # --------------------------------------- 2. Decide action
        drawer = self.world.draw
        flag_first_time = True


        # 1) Getting up guard
        if self.state == 1 or self.behavior.is_ready("Get_Up"):
            self.state = 0 if self.behavior.execute("Get_Up") else 1
            return


        # 2) Role assignment + formation
        current_game_state = strategyData.DetermineGameState()
        formation_positions = GetFormationForSituation(current_game_state)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)


        strategyData.my_desired_position = point_preferences.get(
            strategyData.player_unum, strategyData.mypos
        )
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )


        # If formation not ready and we're NOT in the passing chain, go take your spot.
        if not strategyData.IsFormationReady(point_preferences) and flag_first_time:
            flag_first_time = False
            if strategyData.robot_model.unum in strategyData.passing_chain:
                pass  # attackers keep playing even if shape isn't perfect
            else:
                return self.move(
                    strategyData.my_desired_position,
                    orientation=strategyData.my_desired_orientation
                )


        # ------------------------------------------------------
        # STRATEGY EXECUTION (ATTACK vs. DEFEND)
        # ------------------------------------------------------
        my_unum = strategyData.robot_model.unum
        drawer.clear_all()


        # ===== A) ATTACK (unums in passing_chain: typically 3,4,5) =====
        if my_unum in strategyData.passing_chain:


            attack_role = strategyData.GetAttackRole()
            target_pos, pass_target_unum = strategyData.GetPassTargetAndPosition(
                attack_role, strategyData.mypos
            )
            drawer.annotation((0, 10.5), f"ATTACK: Role={attack_role}", drawer.Color.green, "status")


            # World state
            if hasattr(strategyData.world, "ball_abs_pos"):
                ball_xy = tuple(strategyData.world.ball_abs_pos[:2])
            else:
                ball_xy = tuple(strategyData.ball_abs_pos[:2])
            my_xy = tuple(strategyData.mypos)
            goal_x = self._goal_x()  # fixed +X goal = 15.0


            # ========== ONLY CLOSEST ATTACKER CHASES BALL ==========
            # Determine who is closest to ball among attackers
            closest_attacker_unum = None
            closest_dist = 1e9
            for u in strategyData.passing_chain:
                tpos = strategyData.teammate_positions[u-1]
                if tpos is None:
                    continue
                d = np.linalg.norm(tpos - ball_xy)
                if d < closest_dist:
                    closest_dist = d
                    closest_attacker_unum = u
            
            # Check if I have the ball (possession)
            i_have_ball = self._i_have_ball(strategyData)
            
            # If I DON'T have ball AND I'm NOT the closest: move to support position
            if not i_have_ball and my_unum != closest_attacker_unum:
                # Move to role-based support position instead of chasing
                if attack_role == "RECEIVER":
                    lateral = 0.6 if (my_unum % 2 == 0) else -0.6
                    lead_pt = self._lead_run_point(strategyData, ball_xy, my_xy, lead_ahead=3.5, lateral=lateral)
                    drawer.line(my_xy, lead_pt, 2, drawer.Color.cyan, "support_run")
                    return self.move(lead_pt, orientation=strategyData.ball_dir, is_aggressive=False)
                
                elif attack_role == "ADVANCE":
                    lateral = 1.0 if (my_unum % 2 == 0) else -1.0
                    lead_pt = self._lead_run_point(strategyData, ball_xy, my_xy, lead_ahead=4.0, lateral=lateral)
                    drawer.line(my_xy, lead_pt, 2, drawer.Color.cyan, "advance_run")
                    return self.move(lead_pt, orientation=strategyData.ball_dir, is_aggressive=False)
                
                else:  # PASSER role but not closest
                    # Stay in formation position
                    drawer.annotation((0, 9), "PASSER: Holding Position", drawer.Color.yellow, "status2")
                    return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)


            # ========== BALL POSSESSION: SHOOT OR PASS ==========
            if i_have_ball:
                # Check if target is a shooting target (pass_target_unum == 0 means shoot)
                if pass_target_unum == 0:
                    # SHOOT at target_pos (calculated to avoid keeper)
                    drawer.line(my_xy, tuple(target_pos), 3, drawer.Color.red, "shoot_line")
                    drawer.annotation((0, 9), f"SHOOTING! Target: {target_pos}", drawer.Color.orange, "shoot_status")
                    return self.kickTarget(strategyData, my_xy, tuple(target_pos))
                
                # Otherwise PASS
                if attack_role == "PASSER":
                    # Check if we have a forward passing lane
                    if self._has_forward_lane(strategyData, my_xy, tuple(target_pos), max_opponent_gap=0.75):
                        drawer.line(my_xy, tuple(target_pos), 2, drawer.Color.red, "pass_line")
                        drawer.annotation((0, 9), f"PASSING to {pass_target_unum}", drawer.Color.yellow, "pass_status")
                        return self.kickTarget(strategyData, my_xy, tuple(target_pos))
                    else:
                        # No clear lane: dodge/carry forward
                        carry = self._dodge_forward_target(strategyData, ball_xy, step_forward=1.5, sidestep=1.0)
                        drawer.line(my_xy, carry, 2, drawer.Color.orange, "carry_dodge")
                        return self.move(carry, orientation=strategyData.ball_dir, is_aggressive=True)
                
                # RECEIVER or ADVANCE with ball: try to advance or shoot
                else:
                    # Move forward with ball
                    fwd_pt = self._clamp_to_field((my_xy[0] + 1.5, my_xy[1]), xlim=15.0, ylim=10.05)
                    drawer.line(my_xy, fwd_pt, 2, drawer.Color.green, "advance_with_ball")
                    return self.move(fwd_pt, orientation=strategyData.ball_dir, is_aggressive=True)


            # ========== CHASE BALL (only closest attacker reaches here) ==========
            y_bias = 0.15 if (my_unum % 2 == 0) else -0.15
            chase_pt = self._intercept_or_collect_point(strategyData, y_bias=y_bias)
            drawer.line(my_xy, chase_pt, 2, drawer.Color.yellow, "chase_ball")
            drawer.annotation((0, 9), f"CHASING (closest: {closest_attacker_unum})", drawer.Color.white, "chase_status")
            return self.move(chase_pt, orientation=strategyData.ball_dir, is_aggressive=True)


        # ===== B) DEFEND (e.g., unums 1,2) =====
        elif my_unum in strategyData.defending_unums:


            if my_unum == strategyData.active_player_unum:
                drawer.annotation((0, 10.5), "DEFEND: Ball Collector", drawer.Color.blue, "status")
                # Clear to midfield (0,0) fast
                return self.kickTarget(strategyData, strategyData.mypos, (0, 0))
            else:
                target_pos = strategyData.GetDefenseTarget()
                role_name = "Interceptor" if my_unum == 4 else "Goal Defender"
                drawer.annotation((0, 10.5), f"DEFEND: {role_name}", drawer.Color.blue, "status")
                drawer.line(strategyData.mypos, target_pos, 2, drawer.Color.orange, "interception_line")
                return self.move(target_pos, orientation=strategyData.ball_dir, is_aggressive=True)


        # ===== C) FALLBACK =====
        else:
            return self.move(
                strategyData.my_desired_position,
                orientation=strategyData.my_desired_orientation
            )


    # ---------------------- Attack helpers (fixed +X to goal at x=15.0) ----------------------


    def _goal_x(self):
        return 15.0


    def _ensure_forward_target(self, strategyData, ball_xy, target_xy, min_ahead=1.8, sidestep=1.2, goal_buffer=0.8):
        """
        Ensure the target is forward (+X) by at least min_ahead.
        If an opponent is in front 'blocking', nudge laterally away.
        Clamp near goal line with a small buffer.
        """
        bx, by = ball_xy
        tx, ty = target_xy


        # At least min_ahead forward
        if (tx - bx) < min_ahead:
            tx = bx + min_ahead


        # If someone is ahead in our lane, sidestep away
        opp, d = self._nearest_opponent_ahead(strategyData, ref_xy=ball_xy, fwd_cone_deg=35, max_dist=5.0)
        if opp is not None:
            ox, oy = opp
            # pick the side that increases distance to the opponent
            ty = ty + sidestep if (oy - by) < 0 else ty - sidestep


        # Soft cap near goal line
        tx = min(tx, 15.0 - goal_buffer)


        return self._clamp_to_field((tx, ty), xlim=15.0, ylim=10.05)


    def _nearest_opponent_ahead(self, strategyData, ref_xy, fwd_cone_deg=60.0, max_dist=12.0):
        """
        Return (opp_xy, dist) for the nearest opponent in a forward (+X) cone.
        None if not found.
        """
        try:
            opps = [o[:2] for o in strategyData.opponent_positions if o is not None]
        except:
            opps = []
        if not opps:
            return None, None


        rx, ry = ref_xy
        best = (None, None)
        cos_th = math.cos(math.radians(fwd_cone_deg * 0.5))
        for o in opps:
            ox, oy = o
            vx, vy = (ox - rx), (oy - ry)
            d = math.hypot(vx, vy)
            if d == 0 or d > max_dist:
                continue
            # forward is +X, check angle to +X
            dot = vx / d  # (vx,vy)·(1,0) / |v| = vx/|v|
            if dot >= cos_th:  # inside cone
                if best[1] is None or d < best[1]:
                    best = (o, d)
        return best


    def _lead_run_point(self, strategyData, ball_xy, my_xy, lead_ahead=3.5, lateral=0.0):
        """
        Adaptive lead point:
        - Always in +X
        - Cap ahead distance by nearest opponent ahead (leave ~1.5m buffer)
        - Lateral dodge away from that opponent
        """
        bx, by = ball_xy


        # Look for pressure in front of the ball
        opp, d = self._nearest_opponent_ahead(strategyData, ref_xy=ball_xy, fwd_cone_deg=70, max_dist=10.0)


        # Base forward amount
        ahead = lead_ahead


        # If there's an opponent ahead, don't outrun by too much; leave buffer
        if d is not None:
            ahead = max(2.0, min(lead_ahead, d - 1.5))  # keep ~1.5m short of the opp wall
            # Lateral push away from opponent
            ox, oy = opp
            y_dir = -1.0 if (oy - by) > 0 else 1.0
            lateral = max(abs(lateral), 1.0) * y_dir


        # also avoid overrunning the goal line
        ahead = min(ahead, 15.0 - 0.8 - bx)


        lead_x = bx + ahead
        lead_y = by + lateral
        return self._clamp_to_field((lead_x, lead_y), xlim=15.0, ylim=10.05)


    def _clamp_to_field(self, p, xlim=15.0, ylim=10.05):
        # Keep point inside pitch rectangle.
        return (max(-xlim, min(xlim, p[0])), max(-ylim, min(ylim, p[1])))


    def _has_forward_lane(self, strategyData, from_xy, to_xy, max_opponent_gap=0.8):
        # Simple lane check along the pass segment
        try:
            opps = [o[:2] for o in strategyData.opponent_positions if o is not None]
        except:
            opps = []
        if not opps:
            return True
        a = np.array(from_xy)
        b = np.array(to_xy)
        ab = b - a
        ab2 = float(np.dot(ab, ab)) if np.dot(ab, ab) > 1e-6 else 1e-6
        for o in opps:
            p = np.array(o)
            t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
            proj = a + t * ab
            if np.linalg.norm(p - proj) < max_opponent_gap:
                return False
        return True


    def _dodge_forward_target(self, strategyData, ball_xy, step_forward=1.5, sidestep=1.0):
        """
        Small forward + lateral move to open a lane.
        Chooses the lateral direction that increases separation from nearest opponent ahead.
        """
        bx, by = ball_xy
        opp, d = self._nearest_opponent_ahead(strategyData, ref_xy=ball_xy, fwd_cone_deg=70, max_dist=8.0)
        ny = by
        if opp is not None:
            ox, oy = opp
            # choose away from the opponent's y
            ny = by + (sidestep if (oy - by) < 0 else -sidestep)
        nx = min(15.0 - 0.8, bx + step_forward)
        return self._clamp_to_field((nx, ny), xlim=15.0, ylim=10.05)


    # -------- Ball/possession & chase helpers (use World's predictors) --------


    _POSSESSION_RADIUS = 0.35          # meters: close enough to control
    _POSSESSION_BALL_SPD_MAX = 1.0     # m/s: treat as "controllable" if ball is slower than this
    _PLAYER_CHASE_SPEED = 1.6          # m/s: avg speed to reach the ball (tune to your Walk)
    _APPROACH_BEHIND = 0.22            # m: stand a bit behind the ball for forward first touch


    def _ball_xy(self, strategyData):
        """Get absolute 2D ball position from World."""
        w = strategyData.world
        return tuple(w.ball_abs_pos[:2])


    def _ball_speed(self, strategyData, hist_steps=3):
        """Current absolute ball speed magnitude (m/s). Uses World.get_ball_abs_vel."""
        w = strategyData.world
        v = w.get_ball_abs_vel(hist_steps)
        return float(np.linalg.norm(v[:2]))


    def _i_have_ball(self, strategyData):
        """
        Possession heuristic:
        - close to ball
        - AND ball not moving too fast
        """
        bx, by = self._ball_xy(strategyData)
        mx, my = tuple(strategyData.mypos)
        close = np.hypot(bx - mx, by - my) <= self._POSSESSION_RADIUS
        slow_enough = self._ball_speed(strategyData) <= self._POSSESSION_BALL_SPD_MAX
        return close and slow_enough


    def _intercept_or_collect_point(self, strategyData, y_bias=0.0):
        """
        If ball is moving fast → intercept at predicted meeting point.
        Else → collect slightly behind the ball along +X so first touch is forward.
        """
        w = strategyData.world
        bx, by = self._ball_xy(strategyData)


        # Decide: intercept vs static collect
        if self._ball_speed(strategyData) > 0.5 and len(w.ball_2d_pred_pos) > 0:
            # Use World's intersection with average player chase speed
            ip, _dist = w.get_intersection_point_with_ball(self._PLAYER_CHASE_SPEED)
            ip = np.array(ip, dtype=float)


            # Approach slightly "behind" the ball relative to its velocity if we have it; else behind along +X
            v = w.get_ball_abs_vel(3)
            v2 = v[:2]
            if np.linalg.norm(v2) > 1e-3:
                dir_back = -v2 / np.linalg.norm(v2)  # behind relative to motion
            else:
                dir_back = np.array([-1.0, 0.0])     # fallback: behind along +X direction
            approach = ip + self._APPROACH_BEHIND * dir_back
            approach[1] += y_bias
            return self._clamp_to_field(tuple(approach), xlim=15.0, ylim=10.05)


        # Ball slow → simple collect point: a bit behind along +X
        collect = (bx - self._APPROACH_BEHIND, by + y_bias)
        return self._clamp_to_field(collect, xlim=15.0, ylim=10.05)


    # --------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]


        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg(self.kick_direction - r.imu_torso_orientation):.2f} 20)"
            self.fat_proxy_walk = np.zeros(3)  # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d - (-0.1, 0), None, True)  # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot


        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)


        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return


        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg(orientation - r.imu_torso_orientation)
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")
