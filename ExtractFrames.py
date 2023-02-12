#!/usr/bin/env python
# coding: utf-8


import cv2

# Path to the video file
vidObj = cv2.VideoCapture("/.../Movie.mp4")

count = 0
success = 1

while success:
    success,image = vidObj.read()
    cv2.imwrite("frame%d.jpg" % count, image)
    count += 1


