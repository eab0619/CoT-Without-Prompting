import torch
from typing import Tuple

def calculate_confidence(logits: Tuple[torch.Tensor, ...]) -> float:
    """
    Calculate confidence score based on the difference between top two probabilities.
    
    Args:
        logits: Tuple where each element is of size (1, d_vocab)
        
    Returns:
        float: Average delta between top two probabilities across all tokens
    """
    deltas = 0
    for i, score in enumerate(logits):
        logit = torch.softmax(score[0], dim=-1)  # score[0] because of the batch dimension
        max_val, max_arg = torch.topk(logit, k=2)
        
        deltas += max_val[0] - max_val[1]
    
    deltas = deltas.item() / len(logits)
    return deltas