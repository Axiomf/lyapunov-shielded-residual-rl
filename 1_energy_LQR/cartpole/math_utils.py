import math


def wrap_angle(angle):
    """Map an angle to [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip(value, lower, upper):
    return min(max(value, lower), upper)
