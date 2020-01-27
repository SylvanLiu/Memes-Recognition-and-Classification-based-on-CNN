#!/usr/bin/env python

import cv2.cv2 as cv2
import numpy as np
DRAW_IMAGE_BUTTON = True
IF_PRINT = True


def PoseEstimation(path_for_read, image_points):
    # Read Image
    im = cv2.imread(path_for_read)
    size = im.shape

    # Default 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    print('Camera Matrix: ' + '\n' + format(camera_matrix))
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    print('Rotation Vector: '+'\n' + format(rotation_vector))
    print('Translation Vector: '+'\n' + format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    nose_end_point2D = cv2.projectPoints(np.array(
        [(0.0, 0.0, float(focal_length))]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0]
    print('nose_end_point2D: ' + str(nose_end_point2D))
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    p12_vector = ((int(p1[0])-int(p2[0])), (int(p1[1])-int(p2[1])))
    p2p1_distance = np.sqrt(
        np.square(int(p1[0])-int(p2[0])) + np.square(int(p1[1])-int(p2[1])))
    p2p1x_tangent = (int(p1[1])-int(p2[1]))/(int(p1[0])-int(p2[0]))
    print('p12_vector: ' + str(p12_vector))
    print('p2p1x_tangent' + str(p2p1x_tangent))
    print('p2p1_distance: ' + str(p2p1_distance/focal_length))

    cv2.line(im, p1, p2, (255, 0, 0), 2)
    # Display image
    cv2.imshow("Output", im)
    cv2.waitKey(0)
