from dataclasses import dataclass

Point2D = list[float] # [float, float]
Line2D = list[Point2D] # [Point2D, Point2D]
Polygon2D = list[Point2D]
Point3D = list[float] # [float, float, float]
Line3D = list[Point3D] # [Point3D, Point3D]
Polygon3D = list[Point3D]
RotationMatrix3D = list[list[float]]

@dataclass
class WorldParams:
    width: float
    height: float
    depth: float

@dataclass
class CameraParams:
    center: Point3D
    target: Point3D
    focalLength: float
    width: float
    height: float