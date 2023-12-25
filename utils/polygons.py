from shapely.geometry import Polygon
import numpy as np
import cv2


def sampling_points(polygon, num_points=20):
    exterior = polygon.exterior
    points = []
    for distance in np.arange(0, exterior.length, exterior.length / num_points):
        point = exterior.interpolate(distance)
        points.append(point)

    return points


def sampling_points_from_mask(mask, num_points=20):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        poly = Polygon(np.squeeze(contour))
        points.append(sampling_points(poly, num_points=num_points))

    return points
