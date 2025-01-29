import cv2
import numpy as np
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import cv2
import requests
from matplotlib import rcParams
import matplotlib.pyplot as plt
# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")


# target image
target_image = Image.open('reference3.png').convert("RGB")
target_sizes = torch.Tensor([target_image.size[::-1]])

# Query image
ref_image = Image.open('sample1.png').convert("RGB")

# Process input and query image
inputs = processor(images=target_image, query_images=ref_image, return_tensors="pt").to(device)


# Get predictions
with torch.no_grad():
  outputs = model.image_guided_detection(**inputs)


img = cv2.cvtColor(np.array(target_image), cv2.COLOR_BGR2RGB)
outputs.logits = outputs.logits.cpu()
outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()

results = processor.post_process_image_guided_detection(outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes)
boxes, scores = results[0]["boxes"], results[0]["scores"]

# Draw predicted bounding boxes
for box, score in zip(boxes, scores):
    box = [int(i) for i in box.tolist()]

    img = cv2.rectangle(img, box[:2], box[2:], (255,0,0), 5)
    if box[3] + 25 > 768:
        y = box[3] - 10
    else:
        y = box[3] + 25

plt.imshow(img[:,:,::-1])