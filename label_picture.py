import cv2
import os
import numpy as np

init_line_pos = None
end_line_pos = None
lines = []


def click(event, x, y, flags, parameters):
    global init_line_pos, end_line_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        if init_line_pos is None:
            init_line_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if init_line_pos is not None:
            end_line_pos = (x, y)
            print(f"New line from {init_line_pos} to {end_line_pos}!")
            lines.append([init_line_pos, end_line_pos])
            init_line_pos = None
            end_line_pos = None


cv2.namedWindow('Image')
cv2.setMouseCallback('Image', click)

image_path = './images/daisy-1003447_960_720.jpg'
image = cv2.imread(image_path)

while True:
    for line in lines:
        cv2.line(image, line[0], line[1], (50, 255, 5), 2, cv2.LINE_AA)
        cv2.circle(image, line[0], 10, (100, 0, 255), -1, cv2.LINE_AA)

    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cv2.imwrite(f"./mod_images/{image_path.split(os.sep)[-1].split('.')[0]}.png", image)
lines = np.array(lines)
np.save(f"./1DData/{image_path.split(os.sep)[-1].split('.')[0]}.npy", lines)
