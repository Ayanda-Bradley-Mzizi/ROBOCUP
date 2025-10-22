import numpy as np

def role_assignment(teammate_positions, formation_positions): 

    # Input : Locations of all teammate locations and positions
    # Output : Map from unum -> positions
    #-----------------------------------------------------------#

    #Here is where the preferences are calculated (Ayanda)
    players_preferences={}
    roles_preferences={}

    #THIS IS FOR PLAYER PREFERNCE (Making a list of preferences for each plyr ranked by distance)
    length=len(teammate_positions)
    for i in range(length):
        currentX=teammate_positions[i][0]
        currentY=teammate_positions[i][1]
        t_distances={}
        t_current_preference=[]
        for j in range(len(formation_positions)):
            formationX=formation_positions[j][0]
            formationY=formation_positions[j][1]
            distance=np.sqrt((formationX-currentX)**2+(formationY-currentY)**2)
            t_distances[distance]=j
        sorted_distances = dict(sorted(t_distances.items()))
        players_preferences[i] = list(sorted_distances.values())

    #THIS IS FOR ROLE PREFERNCE (making a list of preferences for each role ranked by distance)
    for i in range(len(formation_positions)):
        currentX=formation_positions[i][0]
        currentY=formation_positions[i][1]
        f_distances={}
        f_current_preference=[]
        for j in range(length):
            teammateX=teammate_positions[j][0]
            teammateY=teammate_positions[j][1]
            distance=np.sqrt((teammateX-currentX)**2+(teammateY-currentY)**2)
            f_distances[distance]=j
        sorted_distances = dict(sorted(f_distances.items()))
        roles_preferences[i] = list(sorted_distances.values())
        
    #initialize roles and players as free 
    n = len(teammate_positions)  # Get  nmbr of players
    unmatched_players = list(range(n))  #set all players as unmatched first by making a list of all players
    current_matches = {}  # role to player mapping (will change 2 player to role later )
    next_proposal_index = [0] * n  #  (i dont really understand what this line does , i got it from copilot but i know it tracks the next role each player will propose to)

    #gyale shapley algo (Cameron)
    while unmatched_players: #keep looping until all players are matched (unmatched_players is empty)
        player = unmatched_players[0]
        # Get the next role this player should propose to
        role = players_preferences[player][next_proposal_index[player]]
        next_proposal_index[player] += 1
        if role not in current_matches:
            # Role is free , accept proposal
            current_matches[role] = player
            unmatched_players.pop(0)
        else:
            # Role compares current match v new proposal
            current_player = current_matches[role]
            if roles_preferences[role].index(player) < roles_preferences[role].index(current_player):
                #  new player is preferred
                current_matches[role] = player
                unmatched_players.pop(0)
                unmatched_players.append(current_player)

    #printing current_matches would give final matches of players to roles
    point_preferences = {}
    # Convert role to player mapping to player to position mapping
    for role, player in current_matches.items():
        point_preferences[player + 1] = formation_positions[role]
    
    return point_preferences
