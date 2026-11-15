"""Roll-compensated image-plane face measurements, not expression recognition."""
import math


def measure_face(points, width, height):
    def value(index, key):
        point = points[index]
        return point[key] if isinstance(point, dict) else getattr(point, key)

    ex = (value(263,'x')-value(33,'x'))*width
    ey = (value(263,'y')-value(33,'y'))*height
    distance = math.hypot(ex,ey)
    if not math.isfinite(distance) or distance < 1 or width <= 0 or height <= 0:
        return None
    # Preserve the historical normalized-coordinate scale for a level face.
    # Projection onto the eye-line normal removes camera-plane head roll.
    nx,ny = -ey/distance,ex/distance
    def gap(a,b):
        dx=(value(a,'x')-value(b,'x'))*width
        dy=(value(a,'y')-value(b,'y'))*height
        return abs(dx*nx+dy*ny)/distance*width/height
    result={'mouth_open':gap(13,14),'brow_left':gap(105,159),'brow_right':gap(334,386)}
    return result if all(math.isfinite(v) for v in result.values()) else None
