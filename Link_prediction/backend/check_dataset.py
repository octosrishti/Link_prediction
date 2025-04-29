import torch

DATA_PATH = "C:/Users/lenovo/MProject/data/facebook_edges_processed.pt"

# Load the raw file
loaded_data = torch.load(DATA_PATH, weights_only=False)

print("🔍 Raw Loaded Data Type:", type(loaded_data))
print(f" Tuple Length: {len(loaded_data)}")

# Print keys inside each dictionary
for i, part in enumerate(loaded_data):
    print(f"\n🔹 Part {i}: Type = {type(part)}")
    if isinstance(part, dict):
        print("   Keys:", list(part.keys()))
    else:
        print("  Unexpected format!")
