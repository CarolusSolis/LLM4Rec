"""
Create a reduced version of the Beauty dataset with N most active users and M most interacted items.

This script creates a smaller version of the Beauty dataset while maintaining its structure
and relationships. It selects the most active users and their most interacted items,
preserving all associated metadata and interaction texts.

The script maintains the following structure and files:
1. Interaction Matrices:
   - train_matrix.npz: Training interactions
   - val_matrix.npz: Validation interactions
   - test_matrix.npz: Test interactions

2. Item Metadata (in item_texts/):
   - brand.pkl: Brand information
   - categories.pkl: Category information
   - description.pkl: Item descriptions
   - title.pkl: Item titles

3. User-Item Interaction Texts (in user_item_texts/):
   - explain.pkl: User explanations for purchases
   - review.pkl: User reviews

Usage:
    python create_small_beauty.py

The script will create a new directory 'beauty_N_users_M_items' containing the reduced dataset.
Current settings: N=100 users, M=100 items.

Note:
    - Users are selected based on their interaction count in the training set
    - Items are selected based on their interaction count with the selected users
    - All matrices and metadata files are filtered to maintain consistency
"""

import numpy as np
import scipy.sparse as sp
import os
import shutil
import pickle

# Constants
N_USERS = 100
N_ITEMS = 100
ORIGINAL_PATH = "original/beauty"
TARGET_PATH = "beauty_100_users_100_items"

def load_sparse_matrix(filepath):
    return sp.load_npz(filepath)

def get_most_active_users(train_matrix, n_users):
    user_interactions = np.array(train_matrix.sum(axis=1)).flatten()
    top_users = np.argsort(user_interactions)[-n_users:]
    return top_users

def get_most_interacted_items(train_matrix, user_indices, n_items):
    item_interactions = np.array(train_matrix[user_indices].sum(axis=0)).flatten()
    top_items = np.argsort(item_interactions)[-n_items:]
    return top_items

def filter_matrix(matrix, user_indices, item_indices):
    return matrix[user_indices][:, item_indices]

def create_directory_structure():
    if os.path.exists(TARGET_PATH):
        shutil.rmtree(TARGET_PATH)
    os.makedirs(TARGET_PATH)
    os.makedirs(os.path.join(TARGET_PATH, "item_texts"))
    os.makedirs(os.path.join(TARGET_PATH, "user_item_texts"))

def filter_pickle_file(input_path, output_path, indices, is_user=True):
    if not os.path.exists(input_path):
        return
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        filtered_data = {k: v for k, v in data.items() if k in indices}
    elif isinstance(data, list):
        filtered_data = [data[i] for i in indices]
    else:
        print(f"Unsupported data type for {input_path}")
        return
    
    with open(output_path, 'wb') as f:
        pickle.dump(filtered_data, f)

def extract_ids_from_text(text):
    """Extract user_id and item_id from formatted text strings like 'user_X explains...' or 'user_X wrote...'"""
    parts = text.split()
    try:
        user_id = int(parts[0].replace('user_', ''))
        item_id = int(parts[-1].replace('item_', '').rstrip(':'))
        return user_id, item_id
    except:
        return None, None

def filter_user_item_texts(data, top_users, top_items):
    filtered_data = []
    for item in data:
        if len(item) >= 2:  # Each item should be [formatted_text, content]
            user_id, item_id = extract_ids_from_text(item[0])
            if user_id is not None and item_id is not None:
                if user_id in top_users and item_id in top_items:
                    filtered_data.append(item)
    return filtered_data

def main():
    # Create new directory structure
    create_directory_structure()
    
    # Load matrices
    train_matrix = load_sparse_matrix(os.path.join(ORIGINAL_PATH, "train_matrix.npz"))
    val_matrix = load_sparse_matrix(os.path.join(ORIGINAL_PATH, "val_matrix.npz"))
    test_matrix = load_sparse_matrix(os.path.join(ORIGINAL_PATH, "test_matrix.npz"))
    
    # Get top users and items
    top_users = get_most_active_users(train_matrix, N_USERS)
    top_items = get_most_interacted_items(train_matrix, top_users, N_ITEMS)
    
    # Convert to sets for faster lookup
    top_users_set = set(top_users)
    top_items_set = set(top_items)
    
    # Filter and save matrices
    for name, matrix in [("train_matrix", train_matrix), 
                        ("val_matrix", val_matrix), 
                        ("test_matrix", test_matrix)]:
        filtered = filter_matrix(matrix, top_users, top_items)
        sp.save_npz(os.path.join(TARGET_PATH, name + ".npz"), filtered)
    
    # Filter item texts
    item_text_files = ["brand.pkl", "categories.pkl", "description.pkl", "title.pkl"]
    for filename in item_text_files:
        input_path = os.path.join(ORIGINAL_PATH, "item_texts", filename)
        output_path = os.path.join(TARGET_PATH, "item_texts", filename)
        filter_pickle_file(input_path, output_path, top_items_set, is_user=False)
    
    # Filter user-item texts
    user_item_text_files = ["explain.pkl", "review.pkl"]
    for filename in user_item_text_files:
        input_path = os.path.join(ORIGINAL_PATH, "user_item_texts", filename)
        output_path = os.path.join(TARGET_PATH, "user_item_texts", filename)
        if os.path.exists(input_path):
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            filtered_data = filter_user_item_texts(data, top_users_set, top_items_set)
            with open(output_path, 'wb') as f:
                pickle.dump(filtered_data, f)

if __name__ == "__main__":
    main()
