from typing import cast
from matplotlib import patches, pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from types_ import Line3D, Point2D, Point3D, Polygon2D, Polygon3D, CameraParams, RotationMatrix3D


class Camera:
    _t: Point3D = [0., 0., 0.]
    _it: Point3D = [0., 0., 0.]
    _R: RotationMatrix3D = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    _iR: RotationMatrix3D = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    _f: float = .1
    _fov: float = 60.
    _width_px: int
    _height_px: int
    _width: float = 1.
    _height: float = 1.

    def __init__(
            self,
            center: Point3D,
            target: Point3D,
            focalLength: float,
            fieldOfView: float,
            widthPX: int,
            heightPX: int
        ) -> None:
        self.move(center)
        self.look(target)
        self._f = focalLength
        self._fov = fieldOfView
        self._width_px = widthPX
        self._height_px = heightPX
        self.initWidth()
        self.initHeight()

    def initWidth(self) -> None:
        self._width = 2 * self._f * np.tan(np.radians(self._fov) / 2)

    def initHeight(self) -> None:
        self._height = self._width * (self._height_px / self._width_px)

    def getFocalLength(self) -> float:
        return self._f

    def getWidth(self) -> float:
        return self._width

    def getHeight(self) -> float:
        return self._height

    def getWidthPX(self) -> int:
        return self._width_px

    def getHeightPX(self) -> int:
        return self._height_px

    def getCenter(self) -> Point3D:
        return self._t

    def getPXSize(self) -> float:
        return self._width / self._width_px

    def getRayDirection(self, x: float, y: float) -> Point3D:
        P = np.array([x, y, -self._f])
        return list(P / np.linalg.norm(P))

    def move(self, center: Point3D) -> None:
        self.update_t(center)
        self.update_it()

    def update_t(self, center: Point3D) -> None:
        self._t = center

    def update_it(self) -> None:
        iR = np.array(self._iR)
        t = np.array(self._t)
        it = -t @ iR
        self._it = list(it)

    def look(self, target: Point3D) -> None:
        self.update_R(target)
        self.update_iR()
        self.update_it()

    def update_R(self, target: Point3D):
        C = np.array(self._t)
        T = np.array(target)
        nz = T - C
        z = -nz / np.linalg.norm(nz)
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        y = np.cross(z, x)
        R = np.row_stack([x, y, z])
        self._R = list(R)

    def update_iR(self) -> None:
        R = np.array(self._R)
        iR = R.T
        self.iR = list(iR)

    def localize(self, points: list[Point3D]) -> list[Point3D]:
        P = np.array(points)
        R = np.array(self._iR)
        t = np.array(self._it)
        Q = P @ R + t
        return list(Q)

    def globalize(self, points: list[Point3D]) -> list[Point3D]:
        P = np.array(points)
        R = np.array(self._R)
        t = np.array(self._t)
        Q = P @ R + t
        return list(Q)

    def project(self, point3Ds: list[Point3D]) -> list[Point2D]:
        P = np.array(point3Ds)
        Q = []
        f = self._f
        for p in P:
            z = abs(p[2])
            p = p / z * f
            if p[2] < 0:
                Q.append([p[0], p[1]])
        return Q

def createCamera(cameraParams: CameraParams) -> Camera:
    camera = Camera(
        center=cameraParams.center,
        target=cameraParams.target,
        focalLength=cameraParams.focalLength,
        fieldOfView=cameraParams.fieldOfView,
        widthPX=cameraParams.widthPX,
        heightPX=cameraParams.heightPX
    )
    return camera

def plotCamera(ax: Axes3D, camera: Camera) -> None:
    # plot camera center
    center = camera.getCenter()
    plotPoint3Ds(ax, [center], ['black'])
    # plot camera axes
    axes = camera.globalize([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    axis_x = axes[0]
    axis_y = axes[1]
    axis_z = axes[2]
    axes_colors = ['red', 'green', 'blue']
    plotLine3Ds(ax, [[center, axis_x], [center, axis_y], [center, axis_z]], axes_colors)
    # plot camera image plane
    f = camera.getFocalLength()
    w = camera.getWidth()
    # half width
    hw = w / 2
    h = camera.getHeight()
    # half height
    hh = h / 2
    image_plane = [
        [-hw, -hh, -f],
        [-hw,  hh, -f],
        [ hw,  hh, -f],
        [ hw, -hh, -f]
    ]
    image_plane = camera.globalize(image_plane)
    plotPolygon3Ds(ax, [image_plane], ['blue'], 
        alpha=.3,
        linewidths=1,
        edgecolors='k'
    )

def plotPoint3Ds(ax: Axes3D, points: list[Point3D], colors: list[str]) -> None:
    P = np.array(points)
    X = P[:, 0]
    Y = P[:, 1]
    Z = P[:, 2]
    ax.scatter(X, Y, Z, color=colors) # type: ignore

def plotLine3Ds(ax: Axes3D, lines: list[Line3D], colors: list[str]) -> None:
    for line, color in zip(lines, colors):
        l = np.array(line)
        lx = l[:, 0]
        ly = l[:, 1]
        lz = l[:, 2]
        ax.plot(lx, ly, lz, color=color)

def plotPolygon3Ds(ax: Axes3D, polygons: list[Polygon3D], colors: list[str], **kwargs) -> None:
    collection = Poly3DCollection(
        polygons,
        facecolors=colors,
        **kwargs
    )
    ax.add_collection3d(collection)

def render(ax: Axes, camera: Camera, polygon3Ds: list[Polygon3D], colors: list[str]) -> None:
    # localize polygons
    polygon3Ds = [camera.localize(polygon3D) for polygon3D in polygon3Ds]
    # get depth for each polygon
    depths = np.array([getPolygonDepth(polygon3D) for polygon3D in polygon3Ds])
    # indirect sort based on depth
    I = np.argsort(depths)
    # project polygons
    polygon2Ds = [camera.project(polygon3D) for polygon3D in polygon3Ds]
    # plot each 2D polygon
    for i in I:
        plotPolygon2D(ax, polygon2Ds[i], colors[i])
        # plt.show()

def getPolygonDepth(polygon: Polygon3D) -> float:
    P = np.array(polygon)
    Pz = P[:, 2]
    # z = np.min(Pz)
    z = np.mean(Pz)
    # z = np.max(Pz)
    z = cast(float, z)
    return z

def plotPolygon2D(ax: Axes, points: Polygon2D, color: str) -> None:
    polygon = patches.Polygon(
        points,
        closed=True, 
        facecolor=color, 
        edgecolor='black', 
        linewidth=1
    )
    ax.add_patch(polygon)

def renderRayTracing(ax: Axes, camera: Camera, polygon3Ds: list[Polygon3D], colors: list[str]) -> None:
    # localize polygons
    polygon3Ds = [camera.localize(polygon3D) for polygon3D in polygon3Ds]
    # plot each each pixel
    pxSize = camera.getPXSize()
    pxSizeHalf = pxSize / 2
    # TODO: add `camera.getPixels()`
    hw = camera.getWidth() / 2
    hh = camera.getHeight() / 2
    for i in range(camera.getWidthPX()):
        for j in range(camera.getHeightPX()):
            x = i * pxSize + pxSizeHalf - hw
            y = j * pxSize + pxSizeHalf - hh
            D = np.array(camera.getRayDirection(x, y))
            color = shade(D, polygon3Ds, colors)
            if color:
                plotPixel(ax, x, y, pxSize, color)

def shade(D: np.ndarray, polygons: list[Polygon3D], colors: list[str]) -> str:
    color = ''
    depth = -float('inf')

    for polygon, c in zip(polygons, colors):
        d = rayPolygonIntersection(D, polygon)
        if d > depth:
            depth = d
            color = c
    
    return color

def rayPolygonIntersection(D: np.ndarray, polygon: Polygon3D) -> float:
    # Calculate normal of the polygon's plane
    N = calculateNormal(polygon)

    # Ray-plane intersection
    t = np.dot(np.array(polygon[0]), N) / np.dot(D, N)
    if t < 0:
        return -float('inf')  # No intersection, the polygon is behind the ray

    # Calculate the intersection point
    P = t * D

    # Check if the intersection point is inside the polygon
    if not isPointInPolygon3D(P, polygon):
        return -float('inf')

    return P[2]

def isPointInPolygon3D(P: np.ndarray, polygon: Polygon3D):
    # Ensure the polygon has more than 2 points
    if len(polygon) < 3:
        return False

    # Calculate normal of the polygon's plane
    N = calculateNormal(polygon)

    # Project the polygon and the point onto a 2D plane
    polygon2D = projectTo2D(np.array(polygon), N)
    p = projectTo2D(np.array([P]), N)[0]

    # Perform point-in-polygon test in 2D
    return isPointInPolygon2D(p, polygon2D)

def projectTo2D(polygon: np.ndarray, normal: np.ndarray) -> np.ndarray:
    # Determine the dominant axis of the normal
    dominantAxis = np.argmax(np.abs(normal))

    # Project the polygon onto the plane that drops the dominant axis
    return np.delete(polygon, dominantAxis, axis=1)

def calculateNormal(polygon: Polygon3D) -> np.ndarray:
    A = np.array(polygon[0])
    B = np.array(polygon[1])
    C = np.array(polygon[2])
    BA = A - B
    BC = C - B
    N = np.cross(BA, BC)
    N /= np.linalg.norm(N)
    return N

def isPointInPolygon2D(P: np.ndarray, polygon: np.ndarray) -> bool:
    n = len(polygon)
    # If the polygon has less than 3 vertices, it's not valid
    if n < 3:
        return False

    # Initialize the side based on the first edge
    first_side = isLeftOrOnEdge(P, polygon[0], polygon[1])

    for i in range(1, n):
        A, B = polygon[i], polygon[(i + 1) % n]
        if isLeftOrOnEdge(P, A, B) != first_side:
            return False

    # Special case: if the point is on the first edge, it is considered inside
    if crossProduct(np.array(P) - np.array(polygon[0]), np.array(polygon[1]) - np.array(polygon[0])) == 0:
        return True

    return True

def isLeftOrOnEdge(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> bool:
    # Calculate the cross product of vectors AP and AB
    AP = np.array([P[0] - A[0], P[1] - A[1]])
    AB = np.array([B[0] - A[0], B[1] - A[1]])
    cp = crossProduct(AP, AB)
    return cp > 0 or cp == 0  # True if P is left of AB or on the edge

def crossProduct(A: np.ndarray, B: np.ndarray) -> float:
    return A[0] * B[1] - A[1] * B[0]

def plotPixel(ax: Axes, x: float, y: float, s: float, color: str) -> None:
    hs = s / 2
    polygon = patches.Rectangle(
        (x - hs, y - hs),
        width=s,
        height=s,
        facecolor=color, 
    )
    ax.add_patch(polygon)