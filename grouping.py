import cv2
import pandas as pd

def group_similar_images(image, bounding_boxes):
    histograms = []
    print("Bounding boxes values accepting")
    for index, box in bounding_boxes.items():
        print(box)
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        cropped_image = image[ymin:ymax, xmin:xmax]

        # Calculate histogram for the cropped region
        image_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append((box, hist))
    print("Created histograms")

    # Group images based on histogram similarity
    similarity_threshold = 0.5
    groups = {}
    visited = set()
    group_id = 1

    for i in range(len(histograms)):
        if i in visited:
            continue
        current_group = [histograms[i][0]]
        visited.add(i)
        for j in range(i + 1, len(histograms)):
            if j in visited:
                continue
            similarity = cv2.compareHist(histograms[i][1], histograms[j][1], cv2.HISTCMP_CORREL)
            if similarity >= similarity_threshold:
                current_group.append(histograms[j][0])
                visited.add(j)
        groups[group_id] = current_group
        group_id+=1
    print("Created different groups")
    print(groups)

    return groups
