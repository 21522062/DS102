import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def extract_feature(model, train_dataloader, device):
  
  # Get features shape
  model.to(device)
  imgs, _ = next(iter(train_dataloader))

  model.eval()
  with torch.inference_mode():
    outputs = model.avgpool(model.features(imgs[0].unsqueeze(0).to(device)))

  # Save features
  X_t = torch.zeros(size=(len(train_dataloader.dataset), outputs.shape[1]))
  y_t = torch.zeros(len(train_dataloader.dataset), dtype=torch.int64)
  i = 0

  # Extract feature
  model.eval()
  with torch.inference_mode():

    for X, y in tqdm(train_dataloader):

      # Send X, y to device
      X, y = X.to(device), y.to(device)

      # Pass through model
      outputs = model.features(X)
      outputs = model.avgpool(outputs)

      # Save to dataset
      n = len(X)

      X_t[i: i+n] = outputs.squeeze()
      y_t[i: i+n] = y

      i += n

  # Bring to TensorDataset
  data_t = TensorDataset(X_t, y_t)

  return data_t