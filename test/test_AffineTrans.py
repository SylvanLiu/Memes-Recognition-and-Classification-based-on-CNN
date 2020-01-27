# coding=utf-8

import cv2.cv2 as cv2
import os
import copy
import numpy

imgSize = [96, 96]
coord5point = [[10.7, 16.3],
               [40.5, 17.6],
               [25.8, 15.6],
               [11.9, 39.2],
               [32.7, 39.4]]

face_landmarks = [[112, 109],
                  [138, 110],
                  [119, 128],
                  [121, 148],
                  [145, 147]]

face_landmarks68 = [[109, 111], [108, 119], [109, 128], [111, 137], [115, 146], [120, 154], [125, 161], [129, 168], [136, 171], [146, 170], [158, 166], [170, 160], [180, 152], [188, 142], [191, 129], [190, 115], [190, 102], [106, 100], [108, 98], [111, 100], [115, 102], [119, 105], [126, 104], [134, 101], [142, 98], [151, 98], [159, 102], [124, 111], [122, 117], [121, 122], [119, 128], [118, 134], [120, 136], [124, 137], [128, 135], [
    133, 134], [112, 109], [114, 107], [117, 108], [121, 110], [117, 111], [114, 111], [138, 110], [140, 107], [145, 107], [151, 108], [146, 110], [141, 110], [121, 148], [121, 145], [123, 143], [126, 144], [129, 143], [136, 144], [145, 147], [142, 147], [129, 147], [126, 148], [124, 147], [123, 148], [145, 147], [138, 151], [131, 153], [128, 154], [125, 153], [123, 151], [121, 148], [123, 148], [124, 148], [127, 148], [130, 148], [142, 147]]


def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])


def warp_im(img_im, orgi_landmarks, tar_landmarks):
    pts1 = numpy.float64(numpy.matrix(
        [[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix(
        [[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    print(M[:2])
    return dst, M[:2]


def draw_landmark(img_im, land):

    img = copy.deepcopy(img_im)
    n = numpy.array(land)
    for i in range(len(land)):
        cv2.circle(img, (int(n[i][0]), int(n[i][1])), 2, (0, 0, 255), -1)
    cv2.imshow('aaa', img)
    # cv2.imwrite('land5.jpg',img)
    # cv2.waitKey(0)


def draw_landmark_warpAffine(img, land, M):

    new_n = []
    for i in range(len(land)):
        pts = []
        pts.append(numpy.squeeze(numpy.array(M[0]))[
                   0]*land[i][0]+numpy.squeeze(numpy.array(M[0]))[1]*land[i][1]+numpy.squeeze(numpy.array(M[0]))[2])
        pts.append(numpy.squeeze(numpy.array(M[1]))[
                   0]*land[i][0]+numpy.squeeze(numpy.array(M[1]))[1]*land[i][1]+numpy.squeeze(numpy.array(M[1]))[2])
        new_n.append(pts)
    n = numpy.array(new_n)
    for i in range(len(land)):
        cv2.circle(img, (int(n[i][0]), int(n[i][1])), 2, (0, 0, 255), -1)
    cv2.imshow('bbb', img)
    # cv2.imwrite('land68.jpg',img)
    # cv2.waitKey(0)


def main():
    pic_path = '2.jpg'
    img_im = cv2.imread(pic_path)
    draw_landmark(img_im, face_landmarks)
    #cv2.imshow('affine_img_im', img_im)
    dst, M = warp_im(img_im, face_landmarks, coord5point)
    cv2.imshow('affine', dst)
    # cv2.imwrite('affine.jpg',dst)
    draw_landmark_warpAffine(dst, face_landmarks68, M)
    crop_im = dst[0:imgSize[0], 0:imgSize[1]]
    cv2.imshow('affine_crop_im', crop_im)
    # cv2.imwrite('affine_crop_im.jpg',crop_im)


if __name__ == '__main__':
    main()
    cv2.waitKey()
