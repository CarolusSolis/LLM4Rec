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
N_ITEMS = 500
ORIGINAL_PATH = "original/beauty"
TARGET_PATH = f"beauty_{N_USERS}_users_{N_ITEMS}_items"

def load_sparse_matrix(filepath):
    return sp.load_npz(filepath)

def get_most_active_users(train_matrix, n_users):
    user_interactions = np.array(train_matrix.sum(axis=1)).flatten()
    top_users = np.argsort(user_interactions)[-n_users:]
    return top_users

def get_valid_item_indices(train_matrix, val_matrix, test_matrix, user_indices, n_items, text_files_path):
    """Get most interacted items across all splits that exist in all text files."""
    # Get interactions from all matrices for selected users
    train_interactions = np.array(train_matrix[user_indices].sum(axis=0)).flatten()
    val_interactions = np.array(val_matrix[user_indices].sum(axis=0)).flatten()
    test_interactions = np.array(test_matrix[user_indices].sum(axis=0)).flatten()
    
    # Get total interactions for ranking
    total_interactions = train_interactions + val_interactions + test_interactions
    
    # Get the length of each text file
    text_files = ["brand.pkl", "categories.pkl", "description.pkl", "title.pkl"]
    min_length = float('inf')
    for filename in text_files:
        path = os.path.join(text_files_path, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                min_length = min(min_length, len(data))
    
    # Get all valid items (within bounds and has training interaction)
    all_items = np.arange(min(min_length, train_matrix.shape[1]))
    valid_mask = train_interactions[all_items] > 0
    valid_items = all_items[valid_mask]
    
    # Sort by total interactions
    sorted_indices = np.argsort(total_interactions[valid_items])[::-1]
    selected_items = valid_items[sorted_indices][:n_items]
    
    # Print statistics about item selection
    print("\nItem selection statistics:")
    print(f"Items with train interactions: {np.sum(train_interactions[selected_items] > 0)}")
    print(f"Items with val interactions: {np.sum(val_interactions[selected_items] > 0)}")
    print(f"Items with test interactions: {np.sum(test_interactions[selected_items] > 0)}")
    print(f"Items with interactions in all splits: {np.sum((train_interactions[selected_items] > 0) & (val_interactions[selected_items] > 0) & (test_interactions[selected_items] > 0))}")
    
    # Ensure all selected items have training interactions
    assert np.all(train_interactions[selected_items] > 0), "Some selected items have no training interactions!"
    
    return selected_items

def get_active_users(train_matrix, val_matrix, test_matrix, user_indices):
    """Filter out users who have no interactions in any of the matrices."""
    # Get interactions for each matrix
    train_interactions = np.array(train_matrix.sum(axis=1)).flatten()
    val_interactions = np.array(val_matrix.sum(axis=1)).flatten()
    test_interactions = np.array(test_matrix.sum(axis=1)).flatten()
    
    # User must have at least one interaction in each matrix
    active_mask = (train_interactions > 0) & (val_interactions > 0) & (test_interactions > 0)
    active_users = user_indices[active_mask]
    
    # Print detailed statistics
    print(f"\nUser interaction statistics:")
    print(f"Users with no training interactions: {np.sum(train_interactions == 0)}")
    print(f"Users with no validation interactions: {np.sum(val_interactions == 0)}")
    print(f"Users with no test interactions: {np.sum(test_interactions == 0)}")
    print(f"Users with interactions in all splits: {np.sum(active_mask)}")
    
    return active_users, active_mask

def filter_matrix(matrix, user_indices, item_indices):
    """Filter matrix to only include specified users and items."""
    filtered = matrix[user_indices][:, item_indices]
    # Zero out any items that aren't in the training set for val/test matrices
    return filtered

def create_id_mapping(ids):
    """Create a mapping from original IDs to new sequential IDs (0 to N-1)."""
    return {old_id: new_id for new_id, old_id in enumerate(sorted(ids))}

def filter_pickle_file(input_path, output_path, indices, id_mapping=None, is_user=True):
    """Filter pickle file and optionally remap IDs."""
    if not os.path.exists(input_path):
        return
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Processing {input_path}")
    print(f"Data type: {type(data)}")
    print(f"Data length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
    
    if isinstance(data, dict):
        if id_mapping:
            # Remap keys using id_mapping
            filtered_data = {id_mapping[k]: v for k, v in data.items() if k in indices}
        else:
            filtered_data = {k: v for k, v in data.items() if k in indices}
    elif isinstance(data, list):
        # Add bounds checking
        valid_indices = [i for i in indices if i < len(data)]
        if len(valid_indices) < len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} indices were out of range")
        filtered_data = [data[i] for i in valid_indices]
    else:
        print(f"Unsupported data type for {input_path}")
        return
    
    with open(output_path, 'wb') as f:
        pickle.dump(filtered_data, f)

def filter_item_text_file(input_path, output_path, top_items, item_mapping):
    """Filter item text files that have format ['The X of item_Y is/are:', ' content']."""
    if not os.path.exists(input_path):
        return
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Processing {input_path}")
    filtered_data = []
    
    for entry in data:
        # Extract item ID from text like 'The brand of item_X is:' or 'The categories of item_X are:'
        desc_text, content = entry
        try:
            # Split by 'item_' first
            after_item = desc_text.split('item_')[1]
            # Find the item ID by splitting at either 'is:' or 'are:'
            if ' is:' in after_item:
                item_str = after_item.split(' is:')[0]
                suffix = ' is:'
            elif ' are:' in after_item:
                item_str = after_item.split(' are:')[0]
                suffix = ' are:'
            else:
                raise ValueError("Neither 'is:' nor 'are:' found in text")
                
            item_id = int(item_str)
            
            if item_id in top_items:
                # Get the prefix (e.g., 'The brand of')
                prefix = desc_text.split('item_')[0]
                # Create new description with remapped ID
                new_desc = f"{prefix}item_{item_mapping[item_id]}{suffix}"
                filtered_data.append([new_desc, content])
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not process entry {entry}: {e}")
            continue
    
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

def filter_user_item_texts(data, top_users, top_items, user_mapping, item_mapping):
    """Filter user-item texts and remap IDs to sequential indices."""
    filtered_data = []
    for item in data:
        if len(item) >= 2:  # Each item should be [formatted_text, content]
            user_id, item_id = extract_ids_from_text(item[0])
            if user_id is not None and item_id is not None:
                if user_id in top_users and item_id in top_items:
                    # Create new formatted text with remapped IDs
                    new_text = f"user_{user_mapping[user_id]} wrote about item_{item_mapping[item_id]}:"
                    filtered_data.append([new_text, item[1]])
    return filtered_data

def create_meta_file(n_users, n_items, output_path, user_mapping, item_mapping):
    """Create meta file with ID mappings."""
    meta = {
        'num_users': n_users,
        'num_items': n_items,
        'dataset': 'beauty',
        'version': f'small_{n_users}users_{n_items}items',
        'user_mapping': user_mapping,  # Maps original ID -> new ID
        'item_mapping': item_mapping,  # Maps original ID -> new ID
        'reverse_user_mapping': {v: k for k, v in user_mapping.items()},  # Maps new ID -> original ID
        'reverse_item_mapping': {v: k for k, v in item_mapping.items()}   # Maps new ID -> original ID
    }
    with open(os.path.join(output_path, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

def create_directory_structure():
    if os.path.exists(TARGET_PATH):
        shutil.rmtree(TARGET_PATH)
    os.makedirs(TARGET_PATH)
    os.makedirs(os.path.join(TARGET_PATH, "item_texts"))
    os.makedirs(os.path.join(TARGET_PATH, "user_item_texts"))

def main():
    # Create new directory structure
    create_directory_structure()
    
    # Load matrices
    train_matrix = load_sparse_matrix(os.path.join(ORIGINAL_PATH, "train_matrix.npz"))
    val_matrix = load_sparse_matrix(os.path.join(ORIGINAL_PATH, "val_matrix.npz"))
    test_matrix = load_sparse_matrix(os.path.join(ORIGINAL_PATH, "test_matrix.npz"))
    
    # Get initial top users and items
    initial_top_users = get_most_active_users(train_matrix, N_USERS)
    top_items = get_valid_item_indices(train_matrix, val_matrix, test_matrix, 
                                     initial_top_users, N_ITEMS, 
                                     os.path.join(ORIGINAL_PATH, "item_texts"))
    
    # Filter matrices first
    filtered_train = filter_matrix(train_matrix, initial_top_users, top_items)
    filtered_val = filter_matrix(val_matrix, initial_top_users, top_items)
    filtered_test = filter_matrix(test_matrix, initial_top_users, top_items)
    
    # Get active users after filtering (must have interactions in all splits)
    top_users, active_mask = get_active_users(filtered_train, filtered_val, filtered_test, initial_top_users)
    print(f"\nFinal dataset statistics:")
    print(f"Active users: {len(top_users)} out of {N_USERS} initial users")
    print(f"Items: {len(top_items)}")
    
    # Create ID mappings
    user_mapping = create_id_mapping(top_users)
    item_mapping = create_id_mapping(top_items)
    
    # Further filter matrices to only include active users
    filtered_train = filtered_train[active_mask]
    filtered_val = filtered_val[active_mask]
    filtered_test = filtered_test[active_mask]
    
    # Save filtered matrices
    sp.save_npz(os.path.join(TARGET_PATH, "train_matrix.npz"), filtered_train)
    sp.save_npz(os.path.join(TARGET_PATH, "val_matrix.npz"), filtered_val)
    sp.save_npz(os.path.join(TARGET_PATH, "test_matrix.npz"), filtered_test)
    
    # Convert to sets for faster lookup
    top_users_set = set(top_users)
    top_items_set = set(top_items)
    
    # Filter item texts
    item_text_files = ["brand.pkl", "categories.pkl", "description.pkl", "title.pkl"]
    for filename in item_text_files:
        input_path = os.path.join(ORIGINAL_PATH, "item_texts", filename)
        output_path = os.path.join(TARGET_PATH, "item_texts", filename)
        filter_item_text_file(input_path, output_path, top_items_set, item_mapping)

    # Filter user-item texts
    user_item_text_files = ["explain.pkl", "review.pkl"]
    for filename in user_item_text_files:
        input_path = os.path.join(ORIGINAL_PATH, "user_item_texts", filename)
        output_path = os.path.join(TARGET_PATH, "user_item_texts", filename)
        if os.path.exists(input_path):
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            filtered_data = filter_user_item_texts(data, top_users_set, top_items_set, user_mapping, item_mapping)
            with open(output_path, 'wb') as f:
                pickle.dump(filtered_data, f)
    
    # Create meta.pkl
    create_meta_file(len(top_users), N_ITEMS, TARGET_PATH, user_mapping, item_mapping)
    
    print("Dataset creation completed successfully!")

if __name__ == "__main__":
    main()
