import cv2
import random

def create_labeled_image(img, grouped_boxes):
    # Define a color for each label
    colors = {}
    for label in grouped_boxes.keys():
        # Assign a random color for each label if not already assigned
        colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Draw bounding boxes with labels
    for label, boxes in grouped_boxes.items():
        color = colors[label]
        for box in boxes:
            # Extract coordinates
            xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            
            # Draw the rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Put the label text
            cv2.putText(img, f'Label {label}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save or display the resulting image
    # cv2.imwrite('labeled_image.jpg', img)
    return img