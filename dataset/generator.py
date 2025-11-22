import json
import random

GRID_SIZE = 400
NUM_BOXES = 5
NUM_SAMPLES = 50000   # change this to generate more/less samples


def place_boxes():
    """
    Place 5 non-overlapping boxes inside the GRID_SIZE.
    Returns:
        input_boxes: [(length, width), ...]
        coords: [(x, y), ...]
    """
    # Randomly sample box dimensions
    boxes = [(random.randint(30, 100), random.randint(20, 80)) for _ in range(NUM_BOXES)]
    placed = []

    for (l, w) in boxes:
        success = False
        for _ in range(500):  # try up to 500 random placements per box
            x = random.randint(0, GRID_SIZE - l)
            y = random.randint(0, GRID_SIZE - w)

            # Check overlap
            collision = False
            for (ox, oy, (ol, ow)) in placed:
                if not (x + l <= ox or ox + ol <= x or y + w <= oy or oy + ow <= y):
                    collision = True
                    break

            if not collision:
                placed.append((x, y, (l, w)))
                success = True
                break

        if not success:
            # If a placement fails, retry entire sample
            return place_boxes()

    # Format data
    input_boxes = boxes
    output_coords = [(x, y) for (x, y, _) in placed]

    return {
        "input": input_boxes,
        "output": output_coords
    }


def generate_dataset(n=NUM_SAMPLES):
    dataset = [place_boxes() for _ in range(n)]
    return dataset


if __name__ == "__main__":
    dataset = generate_dataset(NUM_SAMPLES)
    with open("training_data.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {NUM_SAMPLES} samples → training_data.json")
