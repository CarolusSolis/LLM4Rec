import numpy as np
import scipy.sparse as sp
import os
import pickle
import re

FILE_NAME = "beauty_100_users_500_items"

def extract_ids_from_text(text):
    """Extract user_id and item_id from formatted text strings."""
    parts = text.split()
    try:
        user_id = int(parts[0].replace('user_', ''))
        item_id = int(parts[-1].replace('item_', '').rstrip(':'))
        return user_id, item_id
    except:
        return None, None

def load_and_check_matrices(path):
    print(f"\nChecking matrices in {path}:")
    matrices = {}
    for name in ["train_matrix.npz", "val_matrix.npz", "test_matrix.npz"]:
        matrix_path = os.path.join(path, name)
        if os.path.exists(matrix_path):
            matrix = sp.load_npz(matrix_path)
            matrices[name] = matrix
            print(f"\n{name}:")
            print(f"Shape: {matrix.shape}")
            print(f"Number of non-zero elements: {matrix.nnz}")
            print(f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")
            
            # Check user interactions
            user_interactions = np.array(matrix.sum(axis=1)).flatten()
            users_without_interactions = np.where(user_interactions == 0)[0]
            if len(users_without_interactions) > 0:
                print(f"WARNING: Found {len(users_without_interactions)} users with no interactions!")
                print(f"User indices: {users_without_interactions}")
            else:
                print("All users have at least one interaction ")
            
            # Print unique items in each matrix
            unique_items = set(matrix.nonzero()[1])
            print(f"Number of unique items: {len(unique_items)}")
            print(f"Item ID range: [{min(unique_items)}, {max(unique_items)}]")
            
            # Verify sequential item IDs
            if max(unique_items) >= len(unique_items):
                print(f"WARNING: Item IDs are not sequential! Max ID ({max(unique_items)}) >= number of items ({len(unique_items)})")
            
            # Verify sequential user IDs
            unique_users = set(matrix.nonzero()[0])
            if max(unique_users) >= matrix.shape[0]:
                print(f"WARNING: User IDs are not sequential! Max ID ({max(unique_users)}) >= number of users ({matrix.shape[0]})")
    
    # Check for item consistency across splits
    if len(matrices) > 1:
        train_items = set(matrices["train_matrix.npz"].nonzero()[1])
        print(f"\nDetailed item overlap analysis:")
        print(f"Training set unique items: {len(train_items)}")
        
        for name, matrix in matrices.items():
            if name != "train_matrix.npz":
                split_items = set(matrix.nonzero()[1])
                items_not_in_train = split_items - train_items
                if items_not_in_train:
                    print(f"\n{name}:")
                    print(f"- Total unique items: {len(split_items)}")
                    print(f"- Items not in training set: {len(items_not_in_train)}")
                    print(f"- Missing item IDs: {sorted(items_not_in_train)}")
                else:
                    print(f"\n{name}:")
                    print(f"- Total unique items: {len(split_items)}")
                    print(f"- All items present in training set ")
    
    return matrices

def check_item_texts(path):
    print(f"\nChecking item_texts in {path}:")
    item_text_files = ["brand.pkl", "categories.pkl", "description.pkl", "title.pkl"]
    max_item_id = -1
    for filename in item_text_files:
        filepath = os.path.join(path, "item_texts", filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    print(f"{filename}: {len(data)} items")
                    if data:
                        max_id = max(data.keys())
                        print(f"- ID range: [0, {max_id}]")
                        if max_id >= len(data):
                            print(f"WARNING: IDs not sequential! Max ID ({max_id}) >= number of items ({len(data)})")
                elif isinstance(data, list):
                    print(f"{filename}: {len(data)} items")
                    max_item_id = max(max_item_id, len(data)-1)
    return max_item_id

def check_user_item_texts(path, num_users, num_items):
    print(f"\nChecking user_item_texts in {path}:")
    text_files = ["explain.pkl", "review.pkl"]
    for filename in text_files:
        filepath = os.path.join(path, "user_item_texts", filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    print(f"\n{filename}:")
                    print(f"Total interactions: {len(data)}")
                    
                    # Extract and verify user/item IDs
                    user_ids = set()
                    item_ids = set()
                    for item in data:
                        if len(item) >= 2:
                            user_id, item_id = extract_ids_from_text(item[0])
                            if user_id is not None:
                                user_ids.add(user_id)
                            if item_id is not None:
                                item_ids.add(item_id)
                    
                    print(f"Unique users: {len(user_ids)}")
                    print(f"User ID range: [0, {max(user_ids)}]")
                    if max(user_ids) >= num_users:
                        print(f"WARNING: User IDs not sequential! Max ID ({max(user_ids)}) >= number of users ({num_users})")
                    
                    print(f"Unique items: {len(item_ids)}")
                    print(f"Item ID range: [0, {max(item_ids)}]")
                    if max(item_ids) >= num_items:
                        print(f"WARNING: Item IDs not sequential! Max ID ({max(item_ids)}) >= number of items ({num_items})")

def check_meta_file(path):
    print(f"\nChecking meta.pkl in {path}:")
    meta_path = os.path.join(path, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            print("Meta file contents:")
            for key, value in meta.items():
                if isinstance(value, dict):
                    print(f"- {key}: {len(value)} mappings")
                else:
                    print(f"- {key}: {value}")
            
            # Verify mappings if they exist
            if 'user_mapping' in meta and 'item_mapping' in meta:
                user_map = meta['user_mapping']
                item_map = meta['item_mapping']
                
                # Check sequential new IDs
                new_user_ids = set(user_map.values())
                new_item_ids = set(item_map.values())
                
                print("\nChecking ID mappings:")
                print(f"User mapping: {len(user_map)} users")
                print(f"- New user IDs range: [0, {max(new_user_ids)}]")
                if max(new_user_ids) >= len(new_user_ids):
                    print(f"WARNING: New user IDs not sequential!")
                
                print(f"Item mapping: {len(item_map)} items")
                print(f"- New item IDs range: [0, {max(new_item_ids)}]")
                if max(new_item_ids) >= len(new_item_ids):
                    print(f"WARNING: New item IDs not sequential!")
            return meta
    else:
        print("Meta file not found!")
        return None

def main():
    # Check original dataset
    print("=== Original Beauty Dataset ===")
    orig_matrices = load_and_check_matrices("original/beauty")
    max_item_id = check_item_texts("original/beauty")
    check_user_item_texts("original/beauty", float('inf'), max_item_id)

    # Check reduced dataset
    print("\n=== Reduced Beauty Dataset ===")
    meta = check_meta_file(FILE_NAME)
    reduced_matrices = load_and_check_matrices(FILE_NAME)
    check_item_texts(FILE_NAME)
    if meta:
        check_user_item_texts(FILE_NAME, meta['num_users'], meta['num_items'])
    else:
        check_user_item_texts(FILE_NAME, float('inf'), float('inf'))

if __name__ == "__main__":
    main()
