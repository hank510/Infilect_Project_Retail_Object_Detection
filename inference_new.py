import torch
import os
import cv2
import pandas as pd

def process_image_inference(image):
    # Your object detection code here
    model = torch.hub.load(r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\yolov5", 'custom', path=r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\yolov5\best.pt", source='local') 
    # Image
    img = image
    # Inference
    results = model(img)
    print("Inference done")    
    # save_dir = r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\static\output"
    # os.makedirs(save_dir, exist_ok=True)

    data = results.pandas().xyxy[0]
    # print(data)

    # Results, change the flowing to: results.show()
    # results.show()  # or .show(), .save(), .crop(), .pandas(), etc
    # results.save()

    df = pd.DataFrame(data)

# Criteria to filter bounding boxes
    def is_non_squarish(xmin, ymin, xmax, ymax):
        width = xmax - xmin
        height = ymax - ymin
        aspect_ratio = width / height
        return not (0.5 <= aspect_ratio <= 1.5)

    # Process bounding boxes and save images
    bounding_boxes = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax'])

    for index, row in df.iterrows():
        if row['confidence'] > 0.7 and is_non_squarish(row['xmin'], row['ymin'], row['xmax'], row['ymax']):
            # Crop image based on bounding box coordinates
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            bounding_boxes.loc[len(bounding_boxes)] = [xmin, ymin, xmax, ymax]
    
    print("Bounding boxes created")
    print(bounding_boxes)    
    return bounding_boxes
