
from typing import cast
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from types_ import Point3D, Polygon3D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def createAxes(wWidth: float, wHeight: float, wDepth: float, cWidth: float, cHeight: float) -> tuple[Axes3D, Axes]:
    fig = plt.figure()

    ax3D = fig.add_subplot(121, projection='3d')
    ax3D = cast(Axes3D, ax3D)

    ax3D.set_aspect('equal')

    # Set plot limits
    ax3D.set_xlim(-wWidth / 2, wWidth / 2)
    ax3D.set_ylim(-wHeight / 2, wHeight / 2)
    ax3D.set_zlim(-wDepth / 2, wDepth / 2)

    # Set plot labels
    ax3D.set_xlabel('x')
    ax3D.set_ylabel('y')
    ax3D.set_zlabel('z')

    ax2D = fig.add_subplot(122)
    ax2D.set_aspect('equal')

    # Set plot limits
    ax2D.set_xlim(-cWidth / 2, cWidth / 2)
    ax2D.set_ylim(-cHeight / 2, cHeight / 2)

    # Set plot labels
    ax2D.set_xlabel('x')
    ax2D.set_ylabel('y')

    return ax3D, ax2D # type: ignore

def createBox() -> tuple[list[Polygon3D], list[str]]:
    # Define vertices of a unit cube
    vertices: list[Point3D] = [
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 1.],
        [1., 1., 1.],
        [0., 1., 1.]
    ]
    vertices = list(
        np.array(vertices) -
        np.array([.5, .5, .5])
    )

    # Generate the list of sides' vertices
    faces = [[vertices[idx] for idx in face] for face in [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 3, 7, 4],  # Front face
        [1, 2, 6, 5],  # Back face
        [0, 1, 5, 4],  # Left face
        [3, 2, 6, 7]   # Right face
    ]]

    # Define the colors for each face
    colors = ['blue', 'green', 'red', 'yellow', 'orange', 'purple']

    return faces, colors

def plotShape3D(ax: Axes3D, polygons: list[Polygon3D], colors: list[str]) -> None:
    # Create a Poly3DCollection object for the faces
    face_collection = Poly3DCollection(polygons, facecolors=colors, linewidths=1, edgecolors='k')

    # Add the collection to the plot
    ax.add_collection3d(face_collection)


