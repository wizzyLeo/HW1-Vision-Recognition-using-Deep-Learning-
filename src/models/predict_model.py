import torch
import json
import pandas as pd 
import os

device = (
    torch.device("mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu")
)

def predict_model(model, test_loader, save_path="prediction.csv"):
  model.eval()
  model.to(device)

  with open("/Users/wizzy/Documents/school/vision/project-1/src/models/class_to_idx.json") as f:
    class_to_idx = json.load(f)
  idx_to_class = {v: k for k, v in class_to_idx.items()}

  results = []
  image_ids = []

  with torch.no_grad():
    for images, filenames in test_loader:
      images = images.to(device, non_blocking=True) 
      outputs = model(images)
      _, preds = torch.max(outputs, 1)
      preds_labels = [idx_to_class[p] for p in preds.cpu().tolist()]
      results.extend(preds_labels)
      image_ids.extend([os.path.splitext(fname)[0] for fname in filenames])
  df = pd.DataFrame({
      'image_name': image_ids,
      'pred_label': results
  })
  df.to_csv(save_path, index=False)