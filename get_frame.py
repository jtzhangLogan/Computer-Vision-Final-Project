import cv2.cv2 as cv2
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

frames = []
count = 0

files = [f for f in os.listdir('.') if os.path.isfile(f)]

path = './captures'
os.mkdir(path)

for f in files:
    print("Start processing file {}".format(f))
    if f == "get_frame.py":
        continue
    f_name = os.path.splitext(f)[0]
    vidcap = cv2.VideoCapture(f)
    success, image = vidcap.read()
    while success:
        frames.append(image)
        success, image = vidcap.read()
        count += 1

    idx = np.round(np.linspace(0, len(frames) - 1, 150)).astype(int)

    for i in range(150):
        print("{}_frame_{}.jpg".format(f_name, i))
        cv2.imwrite(os.path.join(path, "{}_frame_{}.jpg".format(f_name, i)), frames[idx[i]])

    frames.clear()
    count = 0
