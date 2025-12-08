import json
import torch
from torch.utils.data import Dataset

from dataset.CONSTANTS import CANVAS

class PlacementDataset(Dataset):
    def __init__(self, json_path, max_items=5):
        self.max_items = max_items

        # Load list of samples from JSON file
        with open(json_path, "r") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract box dimensions (input) and coordinates (target)
        boxes = sample["input"]      # [[l,w], ...]
        coords = sample["output"]    # [[x,y], ...]

        canvas_tensor = torch.tensor(CANVAS, dtype=torch.float32)

        flat_boxes = torch.tensor(
            [v for pair in boxes for v in pair],
            dtype=torch.float32
        )

        flat_boxes = torch.cat([canvas_tensor, flat_boxes], dim=0)

        flat_coords = torch.tensor(
            [v for pair in coords for v in pair],
            dtype=torch.float32
        )

        # Your training loop expects three values: inputs, dims, targets
        dims = flat_boxes.clone()

        return flat_boxes, dims, flat_coords
