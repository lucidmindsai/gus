from mesa import Model
# Why are latitude and longitude unhelpful for small distances?

# Well, here's a table of how many degrees of latitude and longitude
# correspond to a given distance, at the equator:
# Decimals	Degrees	        Distance
# 0         1.0             111 km
# 1	        0.1             11.1 km
# 2	        0.01            1.11 km
# 3	        0.001      	    111 m
# 4	        0.0001  	    11.1 m
# 5	        0.00001	        1.11 m
# 6	        0.000001	    111 mm
# 7	        0.0000001	    11.1 mm
# 8	        0.00000001	    1.11 mm

# For miyawaki method, we'll need to operate on the millimetre level

class Miyawaki(Model):
    #TODO
    pass

# pseudo-code for tree regeneration,
# inspired by descriptions of the model PICUS v1.2 (Lexer & Hönninger, 2001)
# We should decide which of these factors are most important and start there.
# We could also include:
# - seed dispersal distance
# - frost and spring times (for germination)


# for each tree in population:
#     seed_score = base_seed_score(size, age)
#     seed_score += vigour_score(vigour) + canopy_score(canopy_position) # proxy variable?: light exposure + (size + health)
#     seed_score += genetic_score(genetics) # can be a species specific score, normalised to a default
#     seed_score += mast_year_score(mast_year) # mast years of extra seed production (also could be species specific)

#     if species_meets_conditions(species_characteristics):
#         seed_score += species_bonus

#     seeds_produced = convert_score_to_seeds(seed_score)

#     seeds_disp = disperse_seeds(seeds_produced, dispersal_mechanism)
#     viable_seeds = seeds_disp - seed_loss(predation, loss_factors)
    
#     for each viable_seed:
#         if suitable_germination_conditions(location, soil, moisture, temp) and has_space(location):
#             sapling_chance = calculate_sapling_chance(viable_seed)
#             if sapling_chance > threshold:
#                 spawn_sapling(location)