import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M

# --- CONSTANTS ---
# Our goal is at -15 on the X-axis (if we are on the left side)
# Their goal is at +15 on the X-axis
OPPONENT_GOAL_2D = np.array([15.05, 0])
OUR_GOAL_2D = np.array([-15.05, 0])
SHOOT_DISTANCE_SQ = 7.0 * 7.0 # Shoot when within 7 meters of goal (7*7 = 49)

class Strategy():
    def __init__(self, world):
        self.world = world # Keep a reference to the world object
        self.play_mode = world.play_mode
        self.robot_model = world.robot 
        self.my_head_pos_2d = self.robot_model.loc_head_position[:2]
        self.player_unum = self.robot_model.unum
        # Get my current position from the teammates list for consistency (or use my_head_pos_2d)
        self.mypos = self.my_head_pos_2d
        
        # Side is generally 1, but we can assume we are attacking the +X goal (right side)
        self.side = 1 
        if world.team_side_is_left:
            self.side = 0

        # Teammate positions (2D)
        self.teammate_positions = [teammate.state_abs_pos[:2] if teammate.state_abs_pos is not None 
                                   else None
                                   for teammate in world.teammates
                                   ]
        
        # Opponent positions (2D)
        self.opponent_positions = [opponent.state_abs_pos[:2] if opponent.state_abs_pos is not None 
                                   else None
                                   for opponent in world.opponents
                                   ]

        self.my_ori = self.robot_model.imu_torso_orientation
        self.ball_2d = world.ball_abs_pos[:2]
        self.ball_vec = self.ball_2d - self.my_head_pos_2d
        self.ball_dir = M.vector_angle(self.ball_vec)
        self.ball_dist = np.linalg.norm(self.ball_vec)
        self.ball_sq_dist = self.ball_dist * self.ball_dist
        self.ball_speed = np.linalg.norm(world.get_ball_abs_vel(6)[:2])
        
        self.goal_dir = M.target_abs_angle(self.ball_2d, OPPONENT_GOAL_2D)

        self.PM_GROUP = world.play_mode_group

        self.slow_ball_pos = world.get_predicted_ball_pos(0.5)

        # list of squared distances between teammates (including self) and slow ball (sq distance is set to 1000 in some conditions)
        self.teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2) 
                                     if p.state_last_update != 0 and (world.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
                                     else 1000
                                     for p in world.teammates ]

        # list of squared distances between opponents and slow ball (sq distance is set to 1000 in some conditions)
        self.opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2) 
                                     if p.state_last_update != 0 and world.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
                                     else 1000
                                     for p in world.opponents ]

        self.min_teammate_ball_sq_dist = min(self.teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(self.min_teammate_ball_sq_dist)
        self.min_opponent_ball_dist = math.sqrt(min(self.opponents_ball_sq_dist))

        self.active_player_unum = self.teammates_ball_sq_dist.index(self.min_teammate_ball_sq_dist) + 1

        self.my_desired_position = self.mypos
        self.my_desired_orientation = self.ball_dir
        
        # New: Tracking the passing chain to maintain the Attacking vision
        self.passing_chain = [3, 4 ,5] # Use unums 1, 2, 3 for attack
        self.defending_unums = [1, 2]  # Use unums 4, 5 for defense
        self.current_passer_unum = 5 # Start with robot 1 as the passer

    def IsFormationReady(self, point_preferences):
        is_formation_ready = True
        
        # Iterate through all possible UNUMs (1 through 5)
        for i in range(1, 6):
            # Skip the active player, as they need to go to the ball
            if i == self.active_player_unum: 
                continue
                
            # 1. CHECK IF PLAYER WAS ASSIGNED A ROLE (KeyError Fix)
            # This check prevents the KeyError if the assignment function excluded player 'i'.
            if i in point_preferences:
                
                # The position list is 0-indexed, so we use i-1
                teammate_pos = self.teammate_positions[i-1] 
                desired_pos = point_preferences[i]

                # 2. Check for None values (Robustness Check)
                if teammate_pos is not None and desired_pos is not None:
                    # Calculate squared distance
                    distance_sq = np.sum((teammate_pos - desired_pos) ** 2)
                    
                    # Check distance against tolerance (0.3m squared is 0.09)
                    if distance_sq > 0.3 * 0.3:
                        is_formation_ready = False
                        
            # NOTE: If player 'i' is NOT in point_preferences (e.g., they were invisible), 
            # we treat them as irrelevant to the formation check for now and skip to the next player.
                
            return is_formation_ready

    def GetDirectionRelativeToMyPositionAndTarget(self,target):
        target_vec = target - self.my_head_pos_2d
        target_dir = M.vector_angle(target_vec)
        return target_dir
    
    def DetermineGameState(self):
        # Uses simplified logic for offense/defense based on who is closer to the ball
        ball_x = self.ball_2d[0]
        we_have_possession = self.min_teammate_ball_dist < self.min_opponent_ball_dist
        
        # Adjust for field side if necessary, but assume attacking +X side (right)
        if self.world.team_side_is_left is False:
             ball_x *= -1 # Treat as if we are on the left side
        
        if we_have_possession and ball_x >= -1.0: # Ball is safely in the middle or their half
            return "ATTACK"
        elif not we_have_possession and ball_x >= 0.0: # Opponent has ball in their half
            return "PRESS"
        elif not we_have_possession and ball_x < 0.0: # Opponent has ball in our half
            return "DEFEND"
        else:
             return "REPOSITION" # For kickoffs, corner kicks, etc. (use current formation)

    # ----------------------------------------------------------------------
    # | ATTACKING LOGIC (Role 1, 2, 3: Passer, Receiver, Advance)
    # ----------------------------------------------------------------------
    
    def GetAttackRole(self):
        """Determines the current role (Passer, Receiver, Advance) for an attacking robot."""
        
        if self.player_unum not in self.passing_chain:
            return "SUPPORT" # Defender or unassigned player stays back
            
        if self.player_unum == self.active_player_unum and self.min_opponent_ball_dist > 1.0:
            return "PASSER"
            
        # The next robot in the chain is the designated receiver
        try:
            current_passer_index = self.passing_chain.index(self.active_player_unum)
            receiver_index = (current_passer_index + 1) % len(self.passing_chain)
            receiver_unum = self.passing_chain[receiver_index]
        except ValueError:
            # Active player is not in the attack chain, likely a defender has the ball.
            # Treat the lowest unum attacker as the receiver to restart the chain.
            receiver_unum = self.passing_chain[0]

        if self.player_unum == receiver_unum:
            return "RECEIVER"
        else:
            # If not the Passer or the designated Receiver, they are Advancing
            return "ADVANCE"


    def GetPassTargetAndPosition(self, role, passer_pos_2d):
        """
        Calculates the target position based on the current attack role.
        Returns: (Target_Position_2D, Target_Unum_for_Pass)
        """
        
        if role == "PASSER":
            # 1. Check if we should shoot
            if np.sum((passer_pos_2d - OPPONENT_GOAL_2D) ** 2) < SHOOT_DISTANCE_SQ:
                return OPPONENT_GOAL_2D, 0 # Target Unum 0 signals 'Shoot'

            # 2. Get the next receiver in the chain
            try:
                current_passer_index = self.passing_chain.index(self.player_unum)
                receiver_index = (current_passer_index + 1) % len(self.passing_chain)
                receiver_unum = self.passing_chain[receiver_index]
            except ValueError:
                # Fallback to passing to the first player in the chain (unum 1)
                receiver_unum = self.passing_chain[0] 

            # 3. Target is the receiver's current or an advanced position (if assigned formation is not ready)
            target_pos = self.teammate_positions[receiver_unum - 1]
            if target_pos is None:
                # If receiver is unknown, pass to a safe advanced position (e.g., halfway up the field)
                target_pos = np.array([5.0, 0.0]) 

            return target_pos, receiver_unum
        
        elif role == "RECEIVER":
            # Target for the Receiver is simply their assigned formation position
            return self.my_desired_position, None
            
        elif role == "ADVANCE":
            # Target for the Advance player is their assigned formation position
            return self.my_desired_position, None
            
        else: # SUPPORT
            # Target for defenders/support is their assigned formation position
            return self.my_desired_position, None


    # ----------------------------------------------------------------------
    # | DEFENDING LOGIC (Exploiting Opponent's UNUM pass order)
    # ----------------------------------------------------------------------

    def GetDefenseTarget(self):
        """
        Calculates a target position to intercept the opponent's predictable pass (1->2->3->4->5->Goal).
        Returns: Target_Position_2D
        """
        
        # 1. Find the opponent who currently has the ball (closest to ball)
        min_opp_sq_dist = min(self.opponents_ball_sq_dist)
        opponent_passer_unum = self.opponents_ball_sq_dist.index(min_opp_sq_dist) + 1
        opponent_passer_pos = self.opponent_positions[opponent_passer_unum - 1]
        
        # Check if opponent 5 is the passer
        if opponent_passer_unum == 5:
            # Opponent 5 passes to the goal (Target is the area near our goal)
            intercept_target_pos = OUR_GOAL_2D + np.array([2.0, 0.0]) # 2m in front of our goal
            
        else:
            # Opponent passes to the next unum (e.g., 1 -> 2, 3 -> 4)
            opponent_receiver_unum = opponent_passer_unum + 1
            
            # 2. Get the opponent receiver's last known position
            # This is where the pass is directed
            opponent_receiver_pos = self.opponent_positions[opponent_receiver_unum - 1]
            
            if opponent_receiver_pos is not None:
                 # 3. Intercept the line between passer and receiver
                 # For simplicity, move to the expected pass recipient's position
                 intercept_target_pos = opponent_receiver_pos
            else:
                 # If we can't see the receiver, default to a central defensive position
                 intercept_target_pos = np.array([-5.0, 0.0])
        
        # For a more robust interception, calculate the intersection point with the ball's path.
        # However, for the simple pass interception, moving to the receiver's spot is a good first step.
        
        # If the robot is U4 (aggressive defender), use intersection point to intercept the ball.
        # Target is the ball's predicted interception point to try and steal it mid-pass.
        if self.player_unum == self.defending_unums[0] and self.min_teammate_ball_dist > 1.5: # if not too close to the ball already
             # Use a relatively fast speed for interception (e.g., 1.5 m/s)
            intercept_target_pos, _ = self.world.get_intersection_point_with_ball(1.5)
            
        # Ensure the defender does not go too far forward
        if intercept_target_pos[0] > -1.0:
             intercept_target_pos[0] = -1.0
             
        return intercept_target_pos