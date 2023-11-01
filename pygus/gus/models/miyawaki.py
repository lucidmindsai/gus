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
    pass