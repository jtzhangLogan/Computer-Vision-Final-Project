import os
import cv2

img_array = []
for filename in natsort.natsorted(os.listdir('/results/rs6_60_nlayers'),
                                  reverse=False):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('rs6_60_nlayers_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()