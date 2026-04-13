import math
import pyproj
import shapely
from shapely.ops import transform


def reproject_polygon(
    poly,
    crs_from=pyproj.CRS("EPSG:4326"),
    crs_to=pyproj.CRS("EPSG:32633"),
):
    project = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True).transform
    return transform(project, poly)


def validate_crs(crs):
    if not isinstance(crs, pyproj.CRS):
        crs = pyproj.CRS(crs)

    if crs.is_geographic:
        raise TypeError("CRS is geographic, must be projected")

    if crs.axis_info[0].unit_name.lower() not in ("metre", "meter"):
        raise TypeError("CRS is projected but not in meters")


def expand_bbox_to_resolution(bbox: shapely.Polygon, crs, resolution_x, resolution_y):
    validate_crs(crs)

    if check_bbox_fit(bbox, crs, resolution_x, resolution_y):
        return bbox

    x_min, y_min, x_max, y_max = bbox.bounds
    x_max = x_min + resolution_x * math.ceil((x_max - x_min) / resolution_x)
    y_max = y_min + resolution_y * math.ceil((y_max - y_min) / resolution_y)
    return shapely.box(x_min, y_min, x_max, y_max)


def check_bbox_fit(bbox: shapely.Polygon, crs, resolution_x, resolution_y):
    validate_crs(crs)

    x_min, y_min, x_max, y_max = bbox.bounds
    check_x = (x_max - x_min) % resolution_x == 0
    check_y = (y_max - y_min) % resolution_y == 0
    return check_x and check_y


if __name__ == "__main__":
    box1 = shapely.box(10, 10, 30, 55)
    box2 = shapely.box(10, 10, 32, 56)
    crs = "EPSG:32633"  # dummy crs
    resx = 10
    resy = 5

    print(box1.bounds)
    print(check_bbox_fit(box1, crs, resx, resy))
    print(expand_bbox_to_resolution(box1, crs, resx, resy).bounds)

    print(box2.bounds)
    print(check_bbox_fit(box2, crs, resx, resy))
    print(expand_bbox_to_resolution(box2, crs, resx, resy).bounds)
