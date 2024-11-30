import pickle
import os

def check_file_structure(filepath):
    print(f"\nChecking structure of {filepath}:")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print("First key type:", type(next(iter(data.keys()))))
        print("First value type:", type(next(iter(data.values()))))
        print("Sample of first few items:")
        for i, (k, v) in enumerate(data.items()):
            if i < 3:
                print(f"Key: {k}, Value: {v}")
            else:
                break
    elif isinstance(data, list):
        print("First item type:", type(data[0]))
        print("Sample of first few items:")
        for item in data[:3]:
            print(item)

for filename in ["explain.pkl", "review.pkl"]:
    filepath = os.path.join("original/beauty/user_item_texts", filename)
    if os.path.exists(filepath):
        check_file_structure(filepath)
