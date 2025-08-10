#Nenad Beric - SV23/2021
#Popravljam kolokvijum 1

import matplotlib.pyplot as plt
import cv2
import sys
import csv

min_area = 2000
max_area = 50000

def has_valid_child(i, contours, hierarchy):
    child_idx = hierarchy[i][2]
    while child_idx != -1:
        child_area = cv2.contourArea(contours[child_idx])
        if min_area < child_area < max_area:
            return True
        child_idx = hierarchy[child_idx][0]
    return False

def detect_counts(image):
    img = cv2.imread(image)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_clean = cv2.dilate(image_bin, kernel, iterations=1)
    edges_clean = cv2.erode(edges_clean, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    filled = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    x_contours = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        parent_idx = hierarchy[i][3]

        if has_valid_child(i, contours, hierarchy):
            continue

        if parent_idx != -1:
            continue

        x_contours.append(cnt)

    # image_contours = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(image_contours, x_contours, -1, (0, 0, 255), 2)
    # cv2.imshow('Binary image', image_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return len(x_contours)


if __name__=="__main__":
    base_directory = sys.argv[1]

    images = []
    for i in range(1, 11):
        images.append(f"{base_directory}game{i}.png") 

    detected_counts = [0] * len(images)
    for i, image in enumerate(images):
        detected_counts[i] = detect_counts(images[i])

    actual_counts = []
    with open(base_directory + "results.csv", 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            actual_counts.append(int(row['num_x']))


    absolute_errors = [abs(detected_counts[i] - actual_counts[i]) for i in range(len(detected_counts))]
    mae = sum(absolute_errors) / len(detected_counts)

    print(mae)