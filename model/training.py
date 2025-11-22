import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import PlacementModel
from model.loss import compute_loss
from dataset.dataset import PlacementDataset
from dataset.CONSTANTS import CANVAS

def train(
    dataset: PlacementDataset,
    num_epochs: int = 500,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
):
    model = PlacementModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs, dims, targets = batch  # unpack batch
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = compute_loss(CANVAS, dims, predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "../saves/placement_model.pth")

if __name__ == "__main__":
    dataset = PlacementDataset("training_data.json")
    train(dataset)