import numpy as np

def GetDefensiveFormation():
    
    #When opponent has ball in our half  
        
    formation = [
        np.array([-13,0]),    # Goalkeeper 
        np.array([-10,-3]),   #L defender 
        np.array([-10,3]),    # R defender 
        np.array([-8,-1]),    #def  mid
        np.array([-8,1])      #def Mid
    ]
    return formation

def GetBuildUpFormation():
    
    #When we have ball in our half 
    
    formation = [
        np.array([-13,0]),    #goalkeeper 
        np.array([-11,-4]),   #left back - wide 
        np.array([-11,4]),    #right back - wide 
        np.array([-9,0]),     #centre mid - pivot player
        np.array([-7,0])      #forward 
    ]
    return formation

def GetPressingFormation():
    
   # when opponent has ball in their half 
  
    formation = [
        np.array([-13,0]),    #goalkeeper 
        np.array([-5,-3]),    # high def
        np.array([-5,3]),     # high Def
        np.array([-3,-1]),    #pressing mid
        np.array([-3,1])      # pressing mid
    ]
    return formation

def GetAttackingFormation():
    
    #When we have ball in their half 
    
    formation = [
        np.array([-13,0]),    #goalkeeper 
        np.array([-7,-5]),    #attacking fullback 
        np.array([-7,5]),     #attacking fullback 
        np.array([-4,-2]),    #attacking mid 
        np.array([-2,0])      #striker 
    ]
    return formation

def GetFormationForSituation(game_state):
    #chooses formation basd on game state
    if game_state == "defensive":
        return GetDefensiveFormation()
    elif game_state == "build_up":
        return GetBuildUpFormation()
    elif game_state == "pressing":
        return GetPressingFormation()
    elif game_state == "attacking":
        return GetAttackingFormation()
    else:
        # default to defensive if cant pick
        return GetDefensiveFormation()

"""
def GenerateBasicFormation():


    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-7, -2]),  # Left Defender
        np.array([-0, 3]),   # Right Defender
        np.array([7, 1]),    # Forward Left
        np.array([12, 0])      # Forward Right
    ]



    # formation = [
    #     np.array([-13, 0]),    # Goalkeeper
    #     np.array([-10, -2]),  # Left Defender
    #     np.array([-11, 3]),   # Center Back Left
    #     np.array([-8, 0]),    # Center Back Right
    #     np.array([-3, 0]),   # Right Defender
    #     np.array([0, 1]),    # Left Midfielder
    #     np.array([2, 0]),    # Center Midfielder Left
    #     np.array([3, 3]),     # Center Midfielder Right
    #     np.array([8, 0]),     # Right Midfielder
    #     np.array([9, 1]),    # Forward Left
    #     np.array([12, 0])      # Forward Right
    # ]

    return formation"""


