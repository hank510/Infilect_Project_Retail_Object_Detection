import torch
import os
model = torch.hub.load(r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\yolov5", 'custom', path=r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\yolov5\best.pt", source='local') 
# Image
img = r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\static\images\2024_01_16_1705384430514.jpg"
# Inference
results = model(img)

save_dir = r"C:\Users\vrush\Documents\Coding\Projects\Infilect_Project_Retail_Object_Detection\static\output"
os.makedirs(save_dir, exist_ok=True)
# Results, change the flowing to: results.show()
results.show()  # or .show(), .save(), .crop(), .pandas(), etc
results.save()