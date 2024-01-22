from matplotlib import pyplot as plt
from shapes_3d import createAxes, createBox, plotShape3D
from camera import createCamera, plotCamera, render
from types_ import WorldParams, CameraParams


def main() -> None:
    worldParams = WorldParams(
        width=10.0,
        height=10.0,
        depth=10.0
    )
    cameraParams = CameraParams(
        center=[1.0, 0.0, 2.0],
        target=[0.0, 0.0, 0.0],
        focalLength=0.5,
        width=1.0,
        height=1.0
    )

    ax3D, ax2D = createAxes(
        worldParams.width,
        worldParams.height,
        worldParams.depth,
        cameraParams.width,
        cameraParams.height
    )
    polygons, colors = createBox()
    plotShape3D(ax3D, polygons, colors)
    camera = createCamera(cameraParams)
    plotCamera(ax3D, camera)
    render(ax2D, camera, polygons, colors)
    # rednerRayTracing(ax2D, camera, polygons, colors)
    plt.show()

if __name__ == '__main__':
    main()
