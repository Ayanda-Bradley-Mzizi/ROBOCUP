import numpy as np

# Use a high number for distance when a player's position is None, 
# ensuring they are ranked last for all roles.
INVISIBLE_DISTANCE = float('inf') 

def role_assignment(teammate_positions, formation_positions): 

    # Input : Locations of all teammate locations and positions (5 players, 0-indexed)
    # Output : Map from unum (1-5) -> assigned position
    #-----------------------------------------------------------#
    
    num_players = len(teammate_positions) 
    num_roles = len(formation_positions)
    
    if num_players == 0 or num_roles == 0:
        return {}
    
    # We proceed with min(num_players, num_roles) for the algorithm loop range.
    n = min(num_players, num_roles)
    
    # Initialize preference dictionaries
    # Maps player_index (0-4) -> list of preferred role_indices
    players_preferences = {} 
    # Maps role_index (0-4) -> list of preferred player_indices (0-4)
    roles_preferences = {}   

    # Helper function to calculate distance or assign penalty
    def get_distance_sq(pos1, pos2):
        if pos1 is None:
            # Player is not visible, assign max penalty distance
            return INVISIBLE_DISTANCE
        
        # We assume pos2 (formation position) is never None
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

    # 1. PLAYER PREFERENCE (ranking roles per player by distance)
    for i in range(num_players):
        player_pos = teammate_positions[i]

        # Instead of dict (which overwrote on equal distances), use sortable tuples
        ranked = []
        for j in range(num_roles):
            role_pos = formation_positions[j]
            dist = get_distance_sq(player_pos, role_pos)
            ranked.append((dist, j, j))   # (distance, tie-breaker, role_index)

        ranked.sort(key=lambda x: (x[0], x[1]))  # sort by distance, then index
        players_preferences[i] = [role_index for _, __, role_index in ranked]

    # 2. ROLE PREFERENCE (ranking players per role by distance)
    for i in range(num_roles):
        role_pos = formation_positions[i]

        ranked = []
        for j in range(num_players):
            player_pos = teammate_positions[j]
            dist = get_distance_sq(player_pos, role_pos)
            ranked.append((dist, j, j))   # (distance, tie-breaker, player_index)

        ranked.sort(key=lambda x: (x[0], x[1]))
        roles_preferences[i] = [player_index for _, __, player_index in ranked]
        
    # 3. GALE-SHAPLEY ALGORITHM
    unmatched_players = list(range(n)) 
    current_matches = {} # role_index -> player_index mapping (0-4)
    
    next_proposal_index = [0] * n  # track who each player proposes to next

    while unmatched_players:
        player = unmatched_players[0] 
        
        if next_proposal_index[player] >= len(players_preferences[player]):
            unmatched_players.pop(0)
            continue
            
        role = players_preferences[player][next_proposal_index[player]]
        next_proposal_index[player] += 1
        
        if role not in current_matches:
            current_matches[role] = player
            unmatched_players.pop(0)
        else:
            current_player = current_matches[role]

            # Compare preference (lower index = preferred)
            if roles_preferences[role].index(player) < roles_preferences[role].index(current_player):
                current_matches[role] = player
                unmatched_players.pop(0)
                unmatched_players.append(current_player)
            # else role rejects new player

    # 4. Convert to UNUM -> position mapping
    point_preferences = {}
    for role_index, player_index in current_matches.items():
        player_unum = player_index + 1
        point_preferences[player_unum] = np.array(formation_positions[role_index])
    
    return point_preferences
