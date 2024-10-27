import cv2
import os
import numpy as np

# Path to the folder containing the images
image_folder = r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\static\output"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]

# Load images and compute histograms
histograms = []
for path in image_paths:
    image = cv2.imread(path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # Normalize the histogram
    histograms.append((path, hist))

# Define a similarity threshold (you may need to experiment with this value)
similarity_threshold = 0.7

# Grouping images
groups = []
visited = set()

for i in range(len(histograms)):
    if i in visited:
        continue
    current_group = [histograms[i][0]]
    visited.add(i)
    for j in range(i + 1, len(histograms)):
        if j in visited:
            continue
        # Compare histograms using correlation
        similarity = cv2.compareHist(histograms[i][1], histograms[j][1], cv2.HISTCMP_CORREL)
        if similarity >= similarity_threshold:
            current_group.append(histograms[j][0])
            visited.add(j)
    groups.append(current_group)

# Output the groups
for idx, group in enumerate(groups):
    print(f"Group {idx + 1}:")
    for image_path in group:
        print(f" - {image_path}")
    print()
