import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import pickle
import os

# Set style for better visualizations
plt.style.use('ggplot')  # Using a built-in style instead
sns.set_theme()  # This is the preferred way to set seaborn style

# Load matrices
data_path = 'original/beauty/'
train_matrix = sparse.load_npz(os.path.join(data_path, 'train_matrix.npz'))
val_matrix = sparse.load_npz(os.path.join(data_path, 'val_matrix.npz'))
test_matrix = sparse.load_npz(os.path.join(data_path, 'test_matrix.npz'))

print(f'Train matrix shape: {train_matrix.shape}')
print(f'Validation matrix shape: {val_matrix.shape}')
print(f'Test matrix shape: {test_matrix.shape}')

# Analyze sparsity and basic statistics
def analyze_matrix(matrix, name):
    n_users, n_items = matrix.shape
    n_interactions = matrix.nnz
    sparsity = 100 * (1 - n_interactions / (n_users * n_items))
    
    print(f'\n{name} Matrix Statistics:')
    print(f'Number of users: {n_users}')
    print(f'Number of items: {n_items}')
    print(f'Number of interactions: {n_interactions}')
    print(f'Sparsity: {sparsity:.2f}%')
    
    # User interaction statistics
    user_interactions = np.array(matrix.sum(axis=1)).flatten()
    print(f'Average interactions per user: {user_interactions.mean():.2f}')
    print(f'Median interactions per user: {np.median(user_interactions):.2f}')
    
    return user_interactions

train_interactions = analyze_matrix(train_matrix, 'Train')
val_interactions = analyze_matrix(val_matrix, 'Validation')
test_interactions = analyze_matrix(test_matrix, 'Test')

# Visualize distribution of interactions per user
plt.figure(figsize=(12, 6))
plt.hist(train_interactions, bins=50, alpha=0.5, label='Train')
plt.hist(val_interactions, bins=50, alpha=0.5, label='Validation')
plt.hist(test_interactions, bins=50, alpha=0.5, label='Test')
plt.xlabel('Number of interactions')
plt.ylabel('Number of users')
plt.title('Distribution of interactions per user')
plt.legend()
plt.show()

# List item text files
item_texts_path = os.path.join(data_path, 'item_texts')
print('\nItem text files:')
for file in os.listdir(item_texts_path):
    if not file.startswith('.'):
        print(f'- {file}')

# List user-item text files
user_item_texts_path = os.path.join(data_path, 'user_item_texts')
print('\nUser-item text files:')
for file in os.listdir(user_item_texts_path):
    if not file.startswith('.'):
        print(f'- {file}')
