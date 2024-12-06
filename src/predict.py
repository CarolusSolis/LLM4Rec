'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''

import re
import os
import sys
import pickle
import fsspec
import random
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from accelerate import Accelerator

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config
from transformers import GPT2Tokenizer

sys.path.append("libs")
from libs.tokenizer import TokenizerWithUserItemIDTokensBatch

from libs.data import UserItemContentGPTDatasetBatch
from libs.data import RecommendationGPTTrainGeneratorBatch
from libs.data import RecommendationGPTTestGeneratorBatch

from libs.model import GPT4RecommendationBaseModel
from libs.model import ContentGPTForUserItemWithLMHeadBatch
from libs.model import CollaborativeGPTwithItemRecommendHead
from libs.util import Recall_at_k, NDCG_at_k
    
def save_local(remote_path, local_path, remote_mode, local_mode):
    '''
        Save the remote file in remote_path
        to the local_path...
    '''
    with fsspec.open(remote_path, remote_mode) as f:
        content = f.read()
    with fsspec.open(local_path, local_mode) as f:
        f.write(content)


def save_remote(local_path, remote_path, local_mode, remote_mode):
    '''
        Save the local file in local_path
        to the remote_path...
    '''
    with fsspec.open(local_path, local_mode) as f:
        content = f.read()
    with fsspec.open(remote_path, remote_mode) as f:
        f.write(content)


# Change from HDFS to local paths
server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Points to LLM4Rec root
local_root = os.path.join(server_root, "tmp")
if not os.path.exists(local_root):
    os.makedirs(local_root, exist_ok=True)

_config = {
    "activation_function": "gelu_new",
    "architectures": [
    "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
    "text-generation": {
        "do_sample": True,
        "max_length": 50
    }
    },
    "vocab_size": 50257
}

def main():
    # Define the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str,
        help="specify the dataset for experiment")
    args = parser.parse_args()
    
    dataset = args.dataset
    
    accelerator.print("-----Current Setting-----")
    accelerator.print(f"dataset: {dataset}")
    accelerator.print(f"lambda_V: {args.lambda_V}")
    
    '''
        Get the basic information of the dataset
    '''
    accelerator.print("-----Begin Obtaining Dataset Info-----")
    data_root = os.path.join(server_root, "data", dataset)
    meta_path = os.path.join(data_root, "meta.pkl")

    with fsspec.open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
        
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    accelerator.print(f"num_users: {num_users}")
    accelerator.print(f"num_items: {num_items}")
    accelerator.print("-----End Obtaining Dataset Info-----\n")


    '''
        Obtain the tokenizer with user/item tokens
    '''
    accelerator.print("-----Begin Obtaining the Tokenizer-----")
    tokenizer_root = os.path.join(server_root, "model", "pretrained", "tokenizer")
    accelerator.print(f"Loading pretrained tokenizer from {tokenizer_root}...")
    remote_vocab_file = os.path.join(tokenizer_root, "vocab_file.json")
    remote_merges_file = os.path.join(tokenizer_root, "merges.txt")
    vocab_file = os.path.join(local_root, "vocab_file.json")
    merges_file = os.path.join(local_root, "merges.txt")

    save_local(remote_vocab_file, vocab_file, "r", "w")
    save_local(remote_merges_file, merges_file, "r", "w")
        
    tokenizer = TokenizerWithUserItemIDTokensBatch(vocab_file, 
                                                   merges_file,
                                                   num_users,
                                                   num_items)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")


    '''
        Obtain the testing data generator
    '''
    accelerator.print("-----Begin Obtaining the Collaborative Data Generator-----")
    remote_train_mat_path = os.path.join(data_root, "train_matrix.npz")
    local_train_mat_path = os.path.join(local_root, "train_matrix.npz")
    accelerator.print(f"Loading data from {remote_train_mat_path}...")
    save_local(remote_train_mat_path, local_train_mat_path, "rb", "wb")
    
    remote_test_mat_path = os.path.join(data_root, "test_matrix.npz")
    local_test_mat_path = os.path.join(local_root, "test_matrix.npz")
    save_local(remote_test_mat_path, local_test_mat_path, "rb", "wb")
    
    # Get the testing data generator
    train_mat = load_npz(local_train_mat_path)
    test_mat = load_npz(local_test_mat_path)
    test_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, test_mat, test_query_dataset="XueyingJia/amazon-search-test")


    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Collaborative Data Generator-----\n")


    '''
        Extend the config of the original GPT model
    '''
    accelerator.print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items
    accelerator.print("Success!")
    accelerator.print("-----End Setting Up the Config-----\n")


    '''
        Instantiate the pretrained GPT2 model
    '''
    accelerator.print("-----Begin Instantiating the Pretrained GPT Model-----")
    gpt2model = GPT2Model(config)
    pretrained_root = os.path.join(server_root, "model", "pretrained")
    accelerator.print(f"Loading pretrained weights from {pretrained_root}...")
    remote_pretrained_weights_path = os.path.join(pretrained_root, "gpt2", "pytorch_model.bin")
    local_pretrained_weights_path = os.path.join(local_root, "gpt2", "pytorch_model.bin")
    save_local(remote_pretrained_weights_path, local_pretrained_weights_path, "rb", "wb")
    gpt2model.load_state_dict(torch.load(local_pretrained_weights_path), strict=False)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pretrained GPT Model-----\n")


    '''
        Instantiate the GPT for recommendation content model
    '''
    accelerator.print("-----Begin Instantiating the Content GPT Model-----")
    base_model = GPT4RecommendationBaseModel(config, gpt2model)

    pretrained_root = os.path.join(server_root, "model", dataset, "rec")
    remote_pretrained_user_emb_path = os.path.join(pretrained_root, f"user_embeddings_{args.lambda_V}.pt") 
    remote_pretrained_item_emb_path = os.path.join(pretrained_root, f"item_embeddings_{args.lambda_V}.pt") 
    local_pretrained_user_emb_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}.pt")
    local_pretrained_item_emb_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}.pt")
    
    save_local(remote_pretrained_user_emb_path, local_pretrained_user_emb_path, "rb", "wb")
    save_local(remote_pretrained_item_emb_path, local_pretrained_item_emb_path, "rb", "wb")    

    base_model.user_embeddings.load_state_dict(
        torch.load(local_pretrained_user_emb_path, map_location=device))
    accelerator.print("Load pretrained user embeddings: Success!")
    base_model.item_embeddings.load_state_dict(
        torch.load(local_pretrained_item_emb_path, map_location=device))
    accelerator.print("Load pretrained item embeddings: Success!")

    rec_model = CollaborativeGPTwithItemRecommendHead(config, base_model)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Content GPT Model-----\n")

    
    '''
        Create a data sampler for distributed training
    '''
    accelerator.print("-----Begin Creating the DataLoader-----")

    # Create the testing data loader
    # Note that we only do the testing in the main process!
    batch_size = 256
    test_data_loader = DataLoader(test_data_gen, 
                                  batch_size=batch_size, 
                                  collate_fn=test_data_gen.collate_fn)
    accelerator.print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    rec_model.to(device)
    
    # Set the model to evaluation mode
    rec_model.eval()  
    
    # Initialize metrics
    total_queries = 0
    hits_20 = 0
    hits_40 = 0
    ndcg_sum = 0

    # IDCG@40 for a single relevant item at position 1
    IDCG = 1.0 / np.log2(2)  # log2(1 + 1) since ranking is 1-based
    
    # Create log file
    log_path = os.path.join(pretrained_root, f"predict_logs_{args.lambda_V}.txt")
    with fsspec.open(log_path, "w") as f:
        f.write("Input Prompt\tTop-10 Predicted Items\tTarget Item\tRank of Target\n")

    with torch.no_grad():
        for batch_idx, (input_ids, train_mat, target_mat, attention_mask) in enumerate(test_data_loader):
            # Decode input prompts for logging
            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            # Move tensors to the correct device
            input_ids = input_ids.to(device)
            train_mat = train_mat.to(device)
            target_mat = target_mat.to(device)
            attention_mask = attention_mask.to(device)

            # Get item scores and rank them
            rec_loss, item_scores = rec_model(input_ids, 
                                            target_mat, 
                                            attention_mask)
            
            # Set score of interacted items to the lowest
            item_scores[train_mat > 0] = -float("inf")  

            # Move to CPU for metric calculation
            target_mat = target_mat.cpu().numpy()
            item_scores = item_scores.cpu().numpy()
            
            # Update batch size for proper averaging
            batch_size = target_mat.shape[0]
            total_queries += batch_size
            
            # Calculate top-k indices
            topk_20_idxes = np.argpartition(-item_scores, 20, axis=1)[:, :20]
            topk_40_idxes = np.argpartition(-item_scores, 40, axis=1)[:, :40]
            
            # Get top 10 for logging
            topk_10_idxes = np.argpartition(-item_scores, 10, axis=1)[:, :10]
            # Sort the top 10 by score
            for i in range(batch_size):
                topk_10_scores = item_scores[i, topk_10_idxes[i]]
                sorted_indices = np.argsort(-topk_10_scores)
                topk_10_idxes[i] = topk_10_idxes[i][sorted_indices]
            
            with fsspec.open(log_path, "a") as f:
                for i in range(batch_size):
                    target_idx = np.where(target_mat[i] > 0)[0][0]  # Get the single target item index
                    hits_20 += int(target_idx in topk_20_idxes[i])
                    hits_40 += int(target_idx in topk_40_idxes[i])
                    
                    # Calculate NDCG
                    rank = -1  # Default if not in top 40
                    if target_idx in topk_40_idxes[i]:
                        # Get the rank (1-based)
                        rank = np.where(topk_40_idxes[i] == target_idx)[0][0] + 1
                        # Calculate DCG (rel=1 since we have binary relevance)
                        dcg = 1.0 / np.log2(rank + 1)
                        # Normalize by IDCG
                        ndcg_sum += dcg / IDCG
                    
                    # Log the results
                    top_10_items = [f"item_{idx}" for idx in topk_10_idxes[i]]
                    f.write(f"{prompts[i]}\t{' '.join(top_10_items)}\titem_{target_idx}\t{rank}\n")
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                accelerator.print(f"Processed {batch_idx + 1} batches...")

    # Calculate final metrics
    recall_20 = hits_20 / total_queries
    recall_40 = hits_40 / total_queries
    ndcg_40 = ndcg_sum / total_queries  # This is now properly normalized

    accelerator.print(f"Recall@20: {recall_20:.4f}")
    accelerator.print(f"Recall@40: {recall_40:.4f}")
    accelerator.print(f"NDCG@40: {ndcg_40:.4f}")
    
    results_path = os.path.join(pretrained_root, f"results_{args.lambda_V}.txt")
    with fsspec.open(results_path, "w") as f:
        f.write("Recall@20,Recall@40,NDCG@40\n")
        f.write(f"{recall_20:.4f},{recall_40:.4f},{ndcg_40:.4f}")


if __name__ == "__main__":
    main()
