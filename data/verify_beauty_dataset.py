import numpy as np
import scipy.sparse as sp
import os
import pickle

FILE_NAME = "beauty_100_users_500_items"

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
    for filename in item_text_files:
        filepath = os.path.join(path, "item_texts", filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    print(f"{filename}: {len(data)} items")
                elif isinstance(data, list):
                    print(f"{filename}: {len(data)} items")

def check_user_item_texts(path):
    print(f"\nChecking user_item_texts in {path}:")
    text_files = ["explain.pkl", "review.pkl"]
    for filename in text_files:
        filepath = os.path.join(path, "user_item_texts", filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    print(f"{filename}: {len(data)} interactions")
                elif isinstance(data, list):
                    print(f"{filename}: {len(data)} interactions")

def main():
    # Check original dataset
    print("=== Original Beauty Dataset ===")
    orig_matrices = load_and_check_matrices("original/beauty")
    check_item_texts("original/beauty")
    check_user_item_texts("original/beauty")

    # Check reduced dataset
    print("\n=== Reduced Beauty Dataset ===")
    reduced_matrices = load_and_check_matrices(FILE_NAME)
    check_item_texts(FILE_NAME)
    check_user_item_texts(FILE_NAME)

    # Verify user activity
    if 'train_matrix.npz' in orig_matrices and reduced_matrices:
        orig_train = orig_matrices['train_matrix.npz']
        reduced_train = reduced_matrices['train_matrix.npz']
        
        print("\nUser Activity Verification:")
        orig_user_activity = np.array(orig_train.sum(axis=1)).flatten()
        reduced_user_activity = np.array(reduced_train.sum(axis=1)).flatten()
        
        print(f"Original dataset - Avg interactions per user: {orig_user_activity.mean():.2f}")
        print(f"Reduced dataset - Avg interactions per user: {reduced_user_activity.mean():.2f}")
        print(f"Original dataset - Max interactions per user: {orig_user_activity.max():.2f}")
        print(f"Reduced dataset - Max interactions per user: {reduced_user_activity.max():.2f}")

if __name__ == "__main__":
    main()
