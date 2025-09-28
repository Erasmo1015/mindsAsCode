# Using 'A' for the agent, '#' for walls, and numbers for blocks.
# Note: The numbers correspond to the block_colors array in environment_jax.py
# 1 = green, 2 = blue, 3 = purple, 4 = pink, 5 = cyan

# --------------------------------------------------------------------------
# Task 1: block_cycle (Requires 3 blocks)
# --------------------------------------------------------------------------
BLOCK_CYCLE_GRID_1 = '''
#######
#A    #
#     #
# 1 2 #
#  3  #
#     #
#######
'''
BLOCK_CYCLE_GRID_2 = '''
#######
#1    #
#  A  #
#     #
#    2#
#3    #
#######
'''
BLOCK_CYCLE_GRID_3 = '''
#######
#     #
#  1  #
# A 2 #
#  3  #
#     #
#######
'''
BLOCK_CYCLE_GRID_4 = '''
#######
#     #
#1  A2#
#     #
#3    #
#     #
#######
'''

# --------------------------------------------------------------------------
# Task 2: clockwise_patrol (Requires walls to patrol)
# --------------------------------------------------------------------------
PATROL_GRID_1 = '''
#######
#A    #
# ### #
# # # #
# # 1 #
# 1   #
#######
'''
PATROL_GRID_2 = '''
#######
#    A#
# ### #
# #   #
#1###2#
#     #
#######
'''
PATROL_GRID_3 = '''
#######
#     #
# ### #
# #1A #
# ### #
#     #
#######
'''
PATROL_GRID_4 = '''
#######
#A    #
#     #
# ##  #
#  #  #
#   1 #
#######
'''

# --------------------------------------------------------------------------
# Task 3: counter_patrol
# --------------------------------------------------------------------------
COUNTER_PATROL_GRID_1 = '''
#######
#A    #
# ### #
# # # #
# # 1 #
# 1   #
#######
'''
COUNTER_PATROL_GRID_2 = '''
#######
#    A#
# ### #
# #   #
#1###2#
#     #
#######
'''
COUNTER_PATROL_GRID_3 = '''
#######
#     #
# ### #
# #1A #
# ### #
#     #
#######
'''
COUNTER_PATROL_GRID_4 = '''
#######
#A    #
#     #
# ##  #
#  #  #
#   1 #
#######
'''

# --------------------------------------------------------------------------
# Task 4: left_right
# --------------------------------------------------------------------------
LEFT_RIGHT_GRID_1 = '''
#######
#     #
#A    #
#  2  #
##    #
#     #
#######
'''
LEFT_RIGHT_GRID_2 = '''
#######
#     #
# #  A#
#  #  #
#    1#
#     #
#######
'''
LEFT_RIGHT_GRID_3 = '''
#######
# 3A  #
#   # #
#     #
#  1  #
#     #
#######
'''
LEFT_RIGHT_GRID_4 = '''
#######
# 2   #
#  A  #
#  1  #
#     #
#     #
#######
'''

# --------------------------------------------------------------------------
# Task 5: pair_blue (Requires at least 2 blue blocks)
# --------------------------------------------------------------------------
PAIR_BLUE_GRID_1 = '''
#######
#A 2  #
## ## #
#  3# #
# ##21#
#     #
#######
'''
PAIR_BLUE_GRID_2 = '''
#######
#   2 #
#3### #
#     #
#    ##
#  2 A#
#######
'''
PAIR_BLUE_GRID_3 = '''
#######
#   2 #
# ###1#
#     #
### # #
#  23A#
#######
'''
PAIR_BLUE_GRID_4 = '''
#######
#    A#
## #1 #
#  1  #
# 1# ##
# 2# 2#
#######
'''

# --------------------------------------------------------------------------
# Task 6: patrol_with_a_star (More complex walls)
# --------------------------------------------------------------------------
PATROL_A_STAR_GRID_1 = '''
#######
#A#12 #
# #   #
#    3#
#     #
# 44  #
#######
'''
PATROL_A_STAR_GRID_2 = '''
#######
# 44  #
#A   2#
#3   2#
#3   1#
#   1 #
#######
'''
PATROL_A_STAR_GRID_3 = '''
#######
#4A  3#
#2# #1#
#2# #1#
#2# #1#
#     #
#######
'''
PATROL_A_STAR_GRID_4 = '''
#######
# 111 #
#A    #
# #   #
# #   #
#     #
#######
'''

# --------------------------------------------------------------------------
# Task 7: pattern_l (Open space is best)
# --------------------------------------------------------------------------
PATTERN_L_GRID_1 = '''
#######
#A #1 #
#     #
#     #
#     #
#     #
#######
'''
PATTERN_L_GRID_2 = '''
#######
# A   #
#     #
#    1#
#     #
#     #
#######
'''
PATTERN_L_GRID_3 = '''
#######
#  #  #
# A   #
#1    #
#     #
#   #3#
#######
'''
PATTERN_L_GRID_4 = '''
#######
#     #
#A 1  #
#    2#
#   3 #
#    ##
#######
'''

# --------------------------------------------------------------------------
# Task 8: pickup_green_a_star (Requires 2 green blocks & obstacles)
# --------------------------------------------------------------------------
PICKUP_GREEN_GRID_1 = '''
#######
#     #
#   # #
# A #1#
#   # #
# 1 # #
#######
'''
PICKUP_GREEN_GRID_2 = '''
#######
#    1#
#  #  #
#  #  #
#A #  #
#  # 1#
#######
'''
PICKUP_GREEN_GRID_3 = '''
#######
#  # 1#
#  #  #
#  #  #
#A # 1#
#     #
#######
'''
PICKUP_GREEN_GRID_4 = '''
#######
#    1#
#  #  #
#     #
#A # 1#
#  #  #
#######
'''

# --------------------------------------------------------------------------
# Task 9: snake (Requires some obstacles to snake around)
# --------------------------------------------------------------------------
SNAKE_GRID_1 = '''
#######
#A    #
#     #
#     #
#     #
#    1#
#######
'''
SNAKE_GRID_2 = '''
#######
#A    #
#     #
##### #
#     #
#1    #
#######
'''
SNAKE_GRID_3 = '''
#######
#A    #
#     #
#     #
# # # #
# #1# #
#######
'''
SNAKE_GRID_4 = '''
#######
#A #1 #
#  # 2#
#  #  #
#  #3 #
#     #
#######
'''

# --------------------------------------------------------------------------
# Task 10: up_down 
# --------------------------------------------------------------------------
UP_DOWN_GRID_1 = '''
#######
#     #
#A    #
#  2  #
##    #
#     #
#######
'''
UP_DOWN_GRID_2 = '''
#######
#     #
#    A#
# ##  #
#    1#
#     #
#######
'''
UP_DOWN_GRID_3 = '''
#######
#  A  #
#   # #
#  3  #
#  1  #
#     #
#######
'''
UP_DOWN_GRID_4 = '''
#######
#     #
#  2  #
# # # #
#  A  #
#   # #
#######
'''

# ==========================================================================
# Main dictionary mapping task names to their list of grid layouts
# ==========================================================================
custom_grid_layouts = {
    "block_cycle": [BLOCK_CYCLE_GRID_1, BLOCK_CYCLE_GRID_2, BLOCK_CYCLE_GRID_3, BLOCK_CYCLE_GRID_4],
    "clockwise_patrol": [PATROL_GRID_1, PATROL_GRID_2, PATROL_GRID_3, PATROL_GRID_4],
    "counter_patrol": [COUNTER_PATROL_GRID_1, COUNTER_PATROL_GRID_2, COUNTER_PATROL_GRID_3, COUNTER_PATROL_GRID_4], # Can reuse patrol grids
    "left_right": [LEFT_RIGHT_GRID_1, LEFT_RIGHT_GRID_2, LEFT_RIGHT_GRID_3, LEFT_RIGHT_GRID_4],
    "pair_blue": [PAIR_BLUE_GRID_1, PAIR_BLUE_GRID_2, PAIR_BLUE_GRID_3, PAIR_BLUE_GRID_4],
    "patrol_with_a_star": [PATROL_A_STAR_GRID_1, PATROL_A_STAR_GRID_2, PATROL_A_STAR_GRID_3, PATROL_A_STAR_GRID_4],
    "pattern_l": [PATTERN_L_GRID_1, PATTERN_L_GRID_2, PATTERN_L_GRID_3, PATTERN_L_GRID_4],
    "pickup_green_a_star": [PICKUP_GREEN_GRID_1, PICKUP_GREEN_GRID_2, PICKUP_GREEN_GRID_3, PICKUP_GREEN_GRID_4],
    "snake": [SNAKE_GRID_1, SNAKE_GRID_2, SNAKE_GRID_3, SNAKE_GRID_4],
    "up_down": [UP_DOWN_GRID_1, UP_DOWN_GRID_2, UP_DOWN_GRID_3, UP_DOWN_GRID_4],
}

# The task list should correspond to the keys in the dictionary above
task_list = list(custom_grid_layouts.keys())