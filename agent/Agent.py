from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment
from strategy.Strategy import Strategy
from formation.Formation import GetFormationForSituation

BALL_PROXIMITY_BUFFER = 0.8

class Agent(Base_Agent):
    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:

        robot_type = (0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4)[unum - 1]

        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name,
                         enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0
        self.kick_direction = 0
        self.kick_distance = 0

        # Fat proxy fields (enabled when is_fat_proxy is True)
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)

        # Initial formation position
        self.init_pos = ([-14, 0], [-9, -5], [-9, 0], [-9, 5], [-5, -5], [-5, 0], [-5, 5],
                         [-1, -6], [-1, -2.5], [-1, 2.5], [-1, 6])[unum - 1]

    def _get_opp_kickoff_pos(self, unum):
        """Defensive positions for opponent kickoffs."""
        if unum == 5:
            return (-2.0, 0.0)
        elif unum == 3:
            return (-0.5, -3.0)
        elif unum == 4:
            return (-0.5, 3.0)
        else:
            init_pos_list = [
                (-14, 0), (-9, -5), (-9, 0), (-9, 5), (-5, -5),
                (-5, 0), (-5, 5), (-1, -6), (-1, -2.5), (-1, 2.5), (-1, 6)
            ]
            return init_pos_list[unum - 1]

    def beam(self, avoid_center_circle=False, is_opp_kickoff=False):
        """Set initial pose using server beam command."""
        r = self.world.robot

        if is_opp_kickoff:
            pos = self._get_opp_kickoff_pos(self.unum)
        else:
            pos = self.init_pos[:]

        self.state = 0

        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0], -pos[1])))
        else:
            if self.fat_proxy_cmd is None:
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)

    def move(self, target_2d=(0, 0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        """Walk to target position."""
        r = self.world.robot

        if self.fat_proxy_cmd is not None:
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute)
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)

    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        """Kick or dribble based on behavior implementation."""
        # Dribble used for carry-forward
        return self.behavior.execute("Dribble", None, None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
            return self.fat_proxy_kick()

    def kickTarget(self, strategyData, mypos_2d=(0, 0), target_2d=(0, 0), abort=False, enable_pass_command=False):
        """Kick toward a specific target point."""
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        kick_distance = np.linalg.norm(vector_to_target)
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        kick_direction = np.degrees(direction_radians)

        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
            return self.fat_proxy_kick()

    def think_and_send(self):
        """Main per-cycle control."""
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True)
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass

        self.radio.broadcast()

        if self.fat_proxy_cmd is None:
            self.scom.commit_and_send(strategyData.robot_model.get_command())
        else:
            self.scom.commit_and_send(self.fat_proxy_cmd.encode())
            self.fat_proxy_cmd = ""

    def select_skill(self, strategyData):
        """Select high-level action."""
        drawer = self.world.draw
        flag_first_time = True

        if self.state == 1 or self.behavior.is_ready("Get_Up"):
            self.state = 0 if self.behavior.execute("Get_Up") else 1
            return

        if strategyData.play_mode == self.world.M_THEIR_KICKOFF:
            if not strategyData.IsOurKickoff():
                target_pos = self._get_opp_kickoff_pos(strategyData.robot_model.unum)
                target_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(target_pos)
                if np.linalg.norm(np.array(strategyData.mypos) - np.array(target_pos)) > 0.1:
                    drawer.annotation((0, 10.5), "KICKOFF DEFENSE: Moving to Position", drawer.Color.orange, "status")
                    return self.move(target_pos, orientation=target_ori)
                else:
                    drawer.annotation((0, 10.5), "KICKOFF DEFENSE: Holding Position", drawer.Color.orange, "status")
                    return self.behavior.execute("Zero_Bent_Knees_Auto_Head")

        if strategyData.play_mode == self.world.M_THEIR_GOAL_KICK:
            my_unum = strategyData.robot_model.unum
            if my_unum == 5:
                target_pos = (13.0, 1.0)
                target_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget((0.0, 0.0))
                if np.linalg.norm(np.array(strategyData.mypos) - np.array(target_pos)) > 0.1:
                    drawer.annotation((0, 10.5), "GOAL KICK PRESS: Moving to (13, 1)", drawer.Color.orange, "status")
                    return self.move(target_pos, orientation=target_ori)
                else:
                    return self.move(target_pos, orientation=target_ori)

        current_game_state = strategyData.DetermineGameState()
        formation_positions = GetFormationForSituation(current_game_state)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)

        strategyData.my_desired_position = point_preferences.get(
            strategyData.player_unum, strategyData.mypos
        )
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )

        if not strategyData.IsFormationReady(point_preferences) and flag_first_time:
            flag_first_time = False
            if strategyData.robot_model.unum in strategyData.passing_chain:
                pass
            else:
                return self.move(
                    strategyData.my_desired_position,
                    orientation=strategyData.my_desired_orientation
                )

        my_unum = strategyData.robot_model.unum
        drawer.clear_all()

        if current_game_state == "M_OUR_FREE_KICK" or current_game_state == "M_OUR_CORNER_KICK":
            drawer.annotation((0, 10.5), "SET PLAY: Hold Position", drawer.Color.purple, "status")
            if (my_xy := strategyData.mypos) is not None and np.linalg.norm(my_xy - strategyData.world.ball_abs_pos[:2]) > 0.7:
                drawer.line(my_xy, strategyData.world.ball_abs_pos[:2], 2, drawer.Color.purple, "setplay_line")
                return self.move(
                    strategyData.world.ball_abs_pos[:2],
                    orientation=strategyData.my_desired_orientation
                )
            else:
                return self.kickTarget(strategyData, strategyData.mypos, (self._goal_x(), 0))

        if my_unum in strategyData.passing_chain:
            attack_role = strategyData.GetAttackRole()
            target_pos, pass_target_unum = strategyData.GetPassTargetAndPosition(
                attack_role, strategyData.mypos
            )
            drawer.annotation((0, 10.5), f"ATTACK: Role={attack_role}", drawer.Color.green, "status")

            if hasattr(strategyData.world, "ball_abs_pos"):
                ball_xy = tuple(strategyData.world.ball_abs_pos[:2])
            else:
                ball_xy = tuple(strategyData.ball_abs_pos[:2])
            my_xy = tuple(strategyData.mypos)

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

            i_have_ball = self._i_have_ball(strategyData)

            if not i_have_ball and my_unum != closest_attacker_unum:
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
                else:
                    drawer.annotation((0, 9), "PASSER: Holding Position", drawer.Color.yellow, "status2")
                    return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)

            ball_dist_to_me = np.linalg.norm(np.array(my_xy) - np.array(ball_xy))

            if my_unum == closest_attacker_unum or i_have_ball:
                if ball_dist_to_me < BALL_PROXIMITY_BUFFER:
                    if pass_target_unum == 0:
                        drawer.line(my_xy, tuple(target_pos), 3, drawer.Color.red, "shoot_line")
                        return self.kickTarget(strategyData, my_xy, tuple(target_pos))

                    if attack_role == "PASSER":
                        if self._has_forward_lane(strategyData, my_xy, tuple(target_pos), max_opponent_gap=0.3):
                            drawer.line(my_xy, tuple(target_pos), 2, drawer.Color.red, "pass_line")
                            return self.kickTarget(strategyData, my_xy, tuple(target_pos))
                        else:
                            carry = self._dodge_forward_target(strategyData, ball_xy, step_forward=1.5, sidestep=1.0)
                            drawer.line(my_xy, carry, 2, drawer.Color.orange, "carry_dodge")
                            return self.kickTarget(strategyData, my_xy, carry)
                    else:
                        fwd_pt = self._clamp_to_field((my_xy[0] + 1.5, my_xy[1]), xlim=15.0, ylim=10.05)
                        drawer.line(my_xy, fwd_pt, 2, drawer.Color.green, "advance_with_ball")
                        return self.kickTarget(strategyData, my_xy, fwd_pt)
                else:
                    if my_unum == closest_attacker_unum and not i_have_ball:
                        y_bias = 0.15 if (my_unum % 2 == 0) else -0.15
                        chase_pt = self._intercept_or_collect_point(strategyData, y_bias=y_bias)
                        drawer.line(my_xy, chase_pt, 2, drawer.Color.yellow, "chase_ball")
                        return self.move(chase_pt, orientation=strategyData.ball_dir, is_aggressive=True)
                    elif i_have_ball:
                        fwd_pt = self._clamp_to_field((my_xy[0] + 1.5, my_xy[1]), xlim=15.0, ylim=10.05)
                        drawer.line(my_xy, fwd_pt, 2, drawer.Color.green, "advance_with_ball_walk")
                        return self.move(fwd_pt, orientation=strategyData.ball_dir, is_aggressive=True)

            if my_unum == closest_attacker_unum and not i_have_ball:
                y_bias = 0.15 if (my_unum % 2 == 0) else -0.15
                chase_pt = self._intercept_or_collect_point(strategyData, y_bias=y_bias)
                drawer.line(my_xy, chase_pt, 2, drawer.Color.yellow, "chase_ball_fallback")
                return self.move(chase_pt, orientation=strategyData.ball_dir, is_aggressive=True)

        elif my_unum in strategyData.defending_unums:
            if my_unum == strategyData.active_player_unum:
                drawer.annotation((0, 10.5), "DEFEND: Ball Collector", drawer.Color.blue, "status")
                return self.kickTarget(strategyData, strategyData.mypos, (0, 0))
            else:
                target_pos = strategyData.GetDefenseTarget()
                role_name = "Interceptor" if my_unum == 4 else "Goal Defender"
                drawer.annotation((0, 10.5), f"DEFEND: {role_name}", drawer.Color.blue, "status")
                drawer.line(strategyData.mypos, target_pos, 2, drawer.Color.orange, "interception_line")
                return self.move(target_pos, orientation=strategyData.ball_dir, is_aggressive=True)
        else:
            return self.move(
                strategyData.my_desired_position,
                orientation=strategyData.my_desired_orientation
            )

    def _goal_x(self):
        return 15.0

    def _ensure_forward_target(self, strategyData, ball_xy, target_xy, min_ahead=1.8, sidestep=1.2, goal_buffer=0.8):
        """Bias target forward; sidestep away from nearest opponent ahead; clamp near goal line."""
        bx, by = ball_xy
        tx, ty = target_xy

        if (tx - bx) < min_ahead:
            tx = bx + min_ahead

        opp, d = self._nearest_opponent_ahead(strategyData, ref_xy=ball_xy, fwd_cone_deg=35, max_dist=5.0)
        if opp is not None:
            ox, oy = opp
            ty = ty + sidestep if (oy - by) < 0 else ty - sidestep

        tx = min(tx, 15.0 - goal_buffer)
        return self._clamp_to_field((tx, ty), xlim=15.0, ylim=10.05)

    def _nearest_opponent_ahead(self, strategyData, ref_xy, fwd_cone_deg=60.0, max_dist=12.0):
        """Nearest opponent in a forward (+X) cone."""
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
            dot = vx / d
            if dot >= cos_th:
                if best[1] is None or d < best[1]:
                    best = (o, d)
        return best

    def _lead_run_point(self, strategyData, ball_xy, my_xy, lead_ahead=3.5, lateral=0.0):
        """Compute an ahead-and-lateral lead point with forward bias and goal clamp."""
        bx, by = ball_xy
        opp, d = self._nearest_opponent_ahead(strategyData, ref_xy=ball_xy, fwd_cone_deg=70, max_dist=10.0)
        ahead = lead_ahead
        if d is not None:
            ahead = max(2.0, min(lead_ahead, d - 1.5))
            ox, oy = opp
            y_dir = -1.0 if (oy - by) > 0 else 1.0
            lateral = max(abs(lateral), 1.0) * y_dir
        ahead = min(ahead, 15.0 - 0.8 - bx)
        lead_x = bx + ahead
        lead_y = by + lateral
        return self._clamp_to_field((lead_x, lead_y), xlim=15.0, ylim=10.05)

    def _clamp_to_field(self, p, xlim=15.0, ylim=10.05):
        """Clamp a point to the field rectangle."""
        return (max(-xlim, min(xlim, p[0])), max(-ylim, min(ylim, p[1])))

    def _has_forward_lane(self, strategyData, from_xy, to_xy, max_opponent_gap=0.5):
        """Check if the pass line keeps a safe gap from opponents."""
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
        """Small forward and lateral adjustment to open a passing lane."""
        bx, by = ball_xy
        opp, d = self._nearest_opponent_ahead(strategyData, ref_xy=ball_xy, fwd_cone_deg=70, max_dist=8.0)
        ny = by
        if opp is not None:
            ox, oy = opp
            ny = by + (sidestep if (oy - by) < 0 else -sidestep)
        nx = min(15.0 - 0.8, bx + step_forward)
        return self._clamp_to_field((nx, ny), xlim=15.0, ylim=10.05)

    _POSSESSION_RADIUS = 0.35
    _POSSESSION_BALL_SPD_MAX = 1.0
    _PLAYER_CHASE_SPEED = 1.6
    _APPROACH_BEHIND = 0.22

    def _ball_xy(self, strategyData):
        """Absolute 2D ball position."""
        w = strategyData.world
        return tuple(w.ball_abs_pos[:2])

    def _ball_speed(self, strategyData, hist_steps=3):
        """Ball speed magnitude."""
        w = strategyData.world
        v = w.get_ball_abs_vel(hist_steps)
        return float(np.linalg.norm(v[:2]))

    def _i_have_ball(self, strategyData):
        """Possession test using distance and ball speed."""
        bx, by = self._ball_xy(strategyData)
        mx, my = tuple(strategyData.mypos)
        close = np.hypot(bx - mx, by - my) <= self._POSSESSION_RADIUS
        slow_enough = self._ball_speed(strategyData) <= self._POSSESSION_BALL_SPD_MAX
        return close and slow_enough

    def _intercept_or_collect_point(self, strategyData, y_bias=0.0):
        """Interception point for fast ball or collect-behind point for slow ball."""
        w = strategyData.world
        bx, by = self._ball_xy(strategyData)

        if self._ball_speed(strategyData) > 0.5 and len(w.ball_2d_pred_pos) > 0:
            ip, _dist = w.get_intersection_point_with_ball(self._PLAYER_CHASE_SPEED)
            ip = np.array(ip, dtype=float)
            v = w.get_ball_abs_vel(3)
            v2 = v[:2]
            if np.linalg.norm(v2) > 1e-3:
                dir_back = -v2 / np.linalg.norm(v2)
            else:
                dir_back = np.array([-1.0, 0.0])
            approach = ip + self._APPROACH_BEHIND * dir_back
            approach[1] += y_bias
            return self._clamp_to_field(tuple(approach), xlim=15.0, ylim=10.05)

        collect = (bx - self._APPROACH_BEHIND, by + y_bias)
        return self._clamp_to_field(collect, xlim=15.0, ylim=10.05)

    # -------- Fat proxy methods --------

    def fat_proxy_kick(self):
        """Issue proxy kick when close enough to the ball."""
        w = self.world
        r = self.world.robot
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg(self.kick_direction - r.imu_torso_orientation):.2f} 20)"
            self.fat_proxy_walk = np.zeros(3)
            return True
        else:
            self.fat_proxy_move(ball_2d - (-0.1, 0), None, True)
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        """Proxy dash control to move and orient."""
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
