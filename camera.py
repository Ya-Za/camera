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

    def getCenter(self) -> Point3D:
        return self._t

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