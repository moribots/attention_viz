"""
This module provides helper functions to compute additional metrics for ablation analysis,
such as KL divergence, entropy differences, and rank changes.
"""

import torch

def compute_extra_metrics(baseline_probs, ablated_probs, token_clicked, tokenizer, epsilon=1e-10):
	"""
	Compute additional metrics including:
	  - KL divergence between baseline and ablated distributions
	  - Change in top token probability
	  - Entropy differences before and after ablation
	  - Rank change for the clicked token
	
	:param baseline_probs: Torch tensor with baseline probabilities.
	:param ablated_probs: Torch tensor with probabilities after ablation.
	:param token_clicked: The token (string) that was clicked.
	:param tokenizer: The tokenizer to convert token to its ID.
	:param epsilon: Small constant to avoid log(0).
	:return: Dictionary with computed metrics.
	"""
	# Calculate KL divergence
	kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
	# Calculate the drop in probability for the top token
	delta_top_prob = baseline_probs.max().item() - ablated_probs.max().item()
	# Calculate entropy for baseline and ablated distributions
	baseline_entropy = -torch.sum(baseline_probs * torch.log(baseline_probs + epsilon)).item()
	ablated_entropy = -torch.sum(ablated_probs * torch.log(ablated_probs + epsilon)).item()
	entropy_diff = ablated_entropy - baseline_entropy
	# Determine rank of the clicked token in both distributions
	clicked_token_id = tokenizer.convert_tokens_to_ids(token_clicked)
	baseline_sorted = torch.argsort(baseline_probs, descending=True)
	ablated_sorted = torch.argsort(ablated_probs, descending=True)
	baseline_rank = (baseline_sorted == clicked_token_id).nonzero(as_tuple=True)[0].item() + 1
	ablated_rank = (ablated_sorted == clicked_token_id).nonzero(as_tuple=True)[0].item() + 1
	rank_change = ablated_rank - baseline_rank
	# Inline comment: Return all the metrics in a neat dictionary
	return {
		 "KL Divergence": kl_div,
		 "Delta Top Token Probability": delta_top_prob,
		 "Baseline Entropy": baseline_entropy,
		 "Ablated Entropy": ablated_entropy,
		 "Entropy Increase": entropy_diff,
		 "Baseline Rank": baseline_rank,
		 "Ablated Rank": ablated_rank,
		 "Rank Change": rank_change
	}
