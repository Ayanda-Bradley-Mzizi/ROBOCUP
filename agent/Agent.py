from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GetFormationForSituation #changed thsi from formation.Formation import GenerateBasicFormation 


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
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
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""



        




    def select_skill(self, strategyData):
        #--------------------------------------- 2. Decide action
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

            # --- RECEIVER/ADVANCE: sprint to through-ball rendezvous point beyond the ball (+X) ---
            if attack_role in ("RECEIVER", "ADVANCE"):
                # Small lateral spread to avoid bunching; slightly wider for ADVANCE
                if attack_role == "RECEIVER":
                    lateral = 0.8 if (my_unum % 2 == 0) else -0.8
                    lead_ahead = 5.0
                else:  # ADVANCE
                    lateral = 1.2 if (my_unum % 2 == 0) else -1.2
                    lead_ahead = 5.0

                lead_pt = self._lead_run_point(ball_xy, my_xy, lead_ahead=lead_ahead, lateral=lateral)
                drawer.line(my_xy, lead_pt, 2, drawer.Color.cyan, "run_lane")
                return self.move(lead_pt, orientation=strategyData.ball_dir, is_aggressive=True)

            # --- PASSER: only pass forward (+X); if lane blocked, carry forward first ---
            if attack_role == "PASSER":
                # If the suggested target isn't far enough ahead, push it forward
                fwd_target = self._ensure_forward_target(ball_xy, target_pos, min_ahead=2.0)

                if self._has_forward_lane(strategyData, my_xy, fwd_target, max_opponent_gap=0.8):
                    drawer.line(my_xy, fwd_target, 2, drawer.Color.red, "pass_line")
                    return self.kickTarget(strategyData, my_xy, fwd_target)
                else:
                    # No clean lane → carry (dribble) forward a bit to open angles
                    carry = (min(goal_x, ball_xy[0] + 2.5), ball_xy[1])  # always push toward +X
                    carry = self._clamp_to_field(carry, xlim=15.0, ylim=10.05)
                    drawer.line(my_xy, carry, 2, drawer.Color.magenta, "carry_forward")
                    return self.move(carry, orientation=strategyData.ball_dir, is_aggressive=True)

            # Fallback for attackers: gentle forward nudge to avoid stalling
            nudge = self._clamp_to_field((my_xy[0] + 1.5, my_xy[1]), xlim=15.0, ylim=10.05)
            return self.move(nudge, orientation=strategyData.ball_dir, is_aggressive=True)

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












    # attack helpers

    def _goal_x(self):
        return 15.0

    def _ensure_forward_target(self, ball_xy, target_xy, min_ahead=2.0):
        #Ensure the pass target is at least 'min_ahead' meters AHEAD of the ball along +X.

        bx, by = ball_xy
        tx, ty = target_xy
        # force forward direction (+X)
        if (tx - bx) < min_ahead:
            tx = bx + max(min_ahead, abs(tx - bx))
        # clamp inside field if needed
        return self._clamp_to_field((tx, ty), xlim=15.0, ylim=10.05)

    def _lead_run_point(self, ball_xy, my_xy, lead_ahead=5.0, lateral=0.0):

        #Lead run ALWAYS beyond the ball in +X.

        bx, by = ball_xy
        lead_x = bx + lead_ahead
        lead_y = by + lateral
        return self._clamp_to_field((lead_x, lead_y), xlim=15.0, ylim=10.05)

    def _clamp_to_field(self, p, xlim=15.0, ylim=10.05):
        # Keep point inside pitch rectangle.
        return (max(-xlim, min(xlim, p[0])), max(-ylim, min(ylim, p[1])))

    def _has_forward_lane(self, strategyData, from_xy, to_xy, max_opponent_gap=0.8):
   
        # Same simple lane check as before.
      
        try:
            opps = [o[:2] for o in strategyData.opponent_positions if o is not None]
        except:
            opps = []
        if not opps:
            return True
        a = np.array(from_xy); b = np.array(to_xy)
        ab = b - a
        ab2 = float(np.dot(ab, ab)) if np.dot(ab, ab) > 1e-6 else 1e-6
        for o in opps:
            p = np.array(o)
            t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
            proj = a + t * ab
            if np.linalg.norm(p - proj) < max_opponent_gap:
                return False
        return True
















    

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
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
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")