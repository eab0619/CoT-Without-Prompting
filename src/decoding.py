from typing import List, Tuple, Dict
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .utils import calculate_confidence


def CoT_Decoding(
    messages: List[Dict[str, str]],
    k: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_new_tokens: int = 512,
    agg_path: bool = False
) -> Tuple[str, float]:
    """
    Chain of Thought Decoding for language models.
    
    Args:
        messages: List of conversation messages
        k: Number of top tokens to consider
        model: Language model
        tokenizer: Tokenizer for the model
        max_new_tokens: Maximum new tokens to generate (default: 512)
        device: Device to run model on
        agg_path: Whether to aggregate paths (default: False)
    
    Returns:
        Tuple[str, float]: (best_answer, confidence_score)
    """
    model = model.to(device)
    
    tokenized_chat: torch.Tensor =  tokenizer.apply_chat_template(messages,tokenize=True, add_generation_prompt=True, return_tensors="pt")
    tokenized_chat = tokenized_chat.to(device)  # shape (1, seq_len)
    
    # Create attention mask
    attention_mask: torch.Tensor = torch.ones_like(tokenized_chat, dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = model(input_ids=tokenized_chat, attention_mask=attention_mask) 
        logits: torch.Tensor = out.logits  # shape (1, seq_len, d_vocab)

    _, idx = torch.topk(logits[0,-1,:], k =k) # get top k tokens
    paths: List[Tuple[str, float]] = []
    
    # Greedy generation
    for i in idx:  # i contains dimension 0 tensor
        new_tokenized_chat = torch.cat([tokenized_chat, i.unsqueeze(0).unsqueeze(0)], dim=-1)
        new_attention_mask = torch.cat([attention_mask, torch.ones((1,1), dtype=torch.long, device=device)], dim=-1)
        
        out_gen = model.generate(
            input_ids=new_tokenized_chat, 
            attention_mask=new_attention_mask,
            do_sample=False,
            output_scores=True, 
            return_dict_in_generate=True,
            temperature= None,
            top_p = None,
            top_k = None,
            max_new_tokens=max_new_tokens
        )
        
        scores = out_gen.scores  # score is a tuple, where each element is of shape (1, d_vocab)
        out_sequences_squeezed = out_gen.sequences[0]  # out_gen.sequences is of shape (1, total_seq_len)
        
        # answ_sequences includes the generated token from top k (first step in CoT Greedy)
        ans_sequences = out_sequences_squeezed[len(tokenized_chat[0]):]  # ans_sequences is now of shape (new_seq_len+1,)
        assert len(scores) == ans_sequences.shape[0] -1, "The length of scores and dimension of output sequence must be the same!"
        
        ans_text: str = tokenizer.decode(ans_sequences, skip_special_tokens=True)
        # Calculate delta
        delta_score: float = calculate_confidence(scores)
        paths.append((ans_text, delta_score))

    if agg_path:
        ans_dict: Dict[str, float] = {}
        for ans_text, delta_score in paths:
            ans_dict[ans_text] = ans_dict.get(ans_text, 0) + delta_score
        max_answer = max(ans_dict, key=ans_dict.get)
        max_score = ans_dict[max_answer]
        return max_answer, max_score
    else:
        max_answer, max_score = max(paths, key=lambda x: x[1])
        return max_answer, max_score