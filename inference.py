import torch
import os
import cv2
import pandas as pd


model = torch.hub.load(r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\yolov5", 'custom', path=r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\yolov5\best.pt", source='local') 
# Image
img = r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\static\images\Good Quality-1993_8.jpg"
image = cv2.imread(img)
# Inference
results = model(img)

save_dir = r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\static\output"
os.makedirs(save_dir, exist_ok=True)

data = results.pandas().xyxy[0]
# print(results.pandas().xyxy[0])

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
cropped_count = 0
for index, row in df.iterrows():
    if row['confidence'] > 0.7 and is_non_squarish(row['xmin'], row['ymin'], row['xmax'], row['ymax']):
        # Crop image based on bounding box coordinates
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cropped_image = image[ymin:ymax, xmin:xmax]
        
        # Save the cropped image
        output = r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\static\output"
        output_path = os.path.join(output, f'cropped_image_{index}.jpg')
        cv2.imwrite(output_path, cropped_image)
        cropped_count += 1

print(f"Total cropped images saved: {cropped_count}")
