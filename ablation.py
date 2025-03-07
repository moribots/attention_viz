"""
Module containing functions to evaluate the impact of ablating attention heads
and to find the best combination of heads to ablate.

The evaluation is based on changes in the probability distribution for a selected token.
This module leverages the ablation hooks defined in ablation_hooks.py.
"""

import torch
from ablation_hooks import make_ablate_hook, make_permutation_hook, make_sparsification_hook
from config import DEFAULT_SCALE, DEFAULT_SPARSITY_THRESHOLD, EPSILON, INITIAL_PROGRESS, FINAL_PROGRESS, PRESELECTION_FRACTION, DEFAULT_MAX_HEAD_LAYER_PAIRS

def evaluate_candidate(truncated_ids, baseline_probs, ablation_set, token_index=-1, scale=DEFAULT_SCALE, ablation_method='standard',
					   sparsity_threshold=DEFAULT_SPARSITY_THRESHOLD, lm_model=None, epsilon=EPSILON, target_token_id=None):
	"""
	Evaluate the impact of ablating a given set of attention heads for a selected token.

	The function computes a score for the token at the specified token_index.
	If target_token_id is None, the score is based on the KL divergence between the baseline 
	and ablated probability distributions plus the change in the top token's probability.
	If target_token_id is provided, the score is simply the probability of that token.

	:param truncated_ids: Torch tensor of input token IDs up to (and including) the token of interest.
	:param baseline_probs: Baseline next-token probability distribution for the token at token_index.
	:param ablation_set: List of (layer, head) tuples representing the heads to ablate.
	:param token_index: The index of the token to evaluate (default: last token, -1).
	:param scale: Scale factor for standard ablation (default: DEFAULT_SCALE).
	:param ablation_method: Method of ablation ('standard', 'permute', or 'sparsify').
	:param sparsity_threshold: Threshold for zeroing small activations (default: DEFAULT_SPARSITY_THRESHOLD).
	:param lm_model: The language model instance.
	:param epsilon: Small constant to avoid division by zero (default: EPSILON).
	:param target_token_id: Optional token ID for which to maximize the probability.
	:return: Ablation score as a float. Higher score indicates a larger impact.
	"""
	hook_handles = []
	# Attach the appropriate hook for each head in the ablation set.
	for (layer, head) in ablation_set:
		if ablation_method == 'standard':
			hook = make_ablate_hook(head, scale=scale, lm_model=lm_model)
		elif ablation_method == 'permute':
			hook = make_permutation_hook(head, lm_model=lm_model)
		elif ablation_method == 'sparsify':
			hook = make_sparsification_hook(head, sparsity_threshold, lm_model=lm_model)
		else:
			hook = make_ablate_hook(head, scale=scale, lm_model=lm_model)
		hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(hook)
		hook_handles.append(hook_handle)
	
	# Compute the logits for the selected token.
	with torch.no_grad():
		ablated_logits = lm_model(truncated_ids).logits[0, token_index, :]
	
	# Remove hooks to reset the model state.
	for handle in hook_handles:
		handle.remove()
	
	ablated_probs = torch.softmax(ablated_logits, dim=-1)
	
	# Return probability for target token if specified.
	if target_token_id is not None:
		return ablated_probs[target_token_id].item()
	
	# Compute score based on KL divergence and change in top token probability.
	kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
	delta_top_prob = baseline_probs.max().item() - ablated_probs.max().item()
	combined_score = 1.0 * kl_div + 1.0 * delta_top_prob
	return combined_score


def find_best_ablation_combo(truncated_ids, baseline_probs, token_index=-1, max_head_layer_pairs=DEFAULT_MAX_HEAD_LAYER_PAIRS, scale=DEFAULT_SCALE, ablation_method='standard',
							  sparsity_threshold=DEFAULT_SPARSITY_THRESHOLD, lm_model=None, progress_callback=None, search_strategy='greedy', 
							  target_token_id=None):
	"""
	Find the best combination of attention heads to ablate.

	Depending on whether a target token is specified, the function either:
	 - Maximizes the overall change in distribution.
	 - Maximizes the probability of the target token.
	 
	It stops early if the target token becomes the top prediction.
	The search proceeds by evaluating individual candidates, preselecting top ones,
	optionally performing a pair search (for 'iterative' strategy), and then greedily expanding the set.

	:param truncated_ids: Torch tensor of input token IDs up to the token of interest.
	:param baseline_probs: Baseline probability distribution for the token at token_index.
	:param token_index: Index of the token to evaluate.
	:param max_head_layer_pairs: Maximum number of heads to ablate (default: DEFAULT_MAX_HEAD_LAYER_PAIRS).
	:param scale: Ablation scale factor.
	:param ablation_method: Method of ablation ('standard', 'permute', 'sparsify').
	:param sparsity_threshold: Threshold for sparsification.
	:param lm_model: The language model instance.
	:param progress_callback: Callback function to report progress.
	:param search_strategy: 'greedy' or 'iterative' search strategy.
	:param target_token_id: Optional token ID to maximize probability for.
	:return: Tuple (best_set, best_score) where best_set is a list of (layer, head) tuples.
	"""
	# Create a list of all possible (layer, head) combinations.
	candidate_list = [(layer, head) for layer in range(lm_model.config.n_layer)
					  for head in range(lm_model.config.n_head)]
	candidate_scores = []
	total_candidates = len(candidate_list)
	
	# Evaluate each candidate head.
	for idx, candidate in enumerate(candidate_list):
		score = evaluate_candidate(
			truncated_ids, baseline_probs, [candidate],
			token_index=token_index, scale=scale, ablation_method=ablation_method,
			sparsity_threshold=sparsity_threshold, lm_model=lm_model,
			target_token_id=target_token_id
		)
		candidate_scores.append((candidate, score))
		if progress_callback is not None:
			progress_callback(int((idx + 1) / total_candidates * INITIAL_PROGRESS))
	
	# Sort candidates by score (higher is better).
	candidate_scores.sort(key=lambda x: x[1], reverse=True)
	
	# Pre-select the top candidates based on the preselection fraction.
	preselected_count = max(1, int(len(candidate_scores) * PRESELECTION_FRACTION))
	preselected = [cand for cand, _ in candidate_scores[:preselected_count]]

	# Initialize the best set and score using an empty ablation set.
	best_set = []
	best_score = evaluate_candidate(
		truncated_ids, baseline_probs, best_set,
		token_index=token_index, scale=scale, ablation_method=ablation_method,
		sparsity_threshold=sparsity_threshold, lm_model=lm_model,
		target_token_id=target_token_id
	)
	
	# Optional pair search for iterative strategy.
	if search_strategy == 'iterative' and max_head_layer_pairs >= 2 and len(preselected) >= 2:
		best_pair = None
		best_pair_score = best_score  # Start with the empty set score.
		
		for i in range(len(preselected)):
			for j in range(i + 1, len(preselected)):
				pair = [preselected[i], preselected[j]]
				score = evaluate_candidate(
					truncated_ids, baseline_probs, pair,
					token_index=token_index, scale=scale, ablation_method=ablation_method,
					sparsity_threshold=sparsity_threshold, lm_model=lm_model,
					target_token_id=target_token_id
				)
				# For both objectives, higher scores are better.
				if score > best_pair_score:
					best_pair_score = score
					best_pair = pair
					
		if best_pair is not None:
			best_set = best_pair
			best_score = best_pair_score

	# Greedy iterative expansion.
	improved = True
	iteration_count = 0
	total_iterations = len(preselected) + 1
	
	while improved and len(best_set) < max_head_layer_pairs:
		improved = False
		best_candidate = None
		candidate_score = best_score
		
		for candidate in preselected:
			if candidate in best_set:
				continue
				
			test_set = best_set + [candidate]
			score = evaluate_candidate(
				truncated_ids, baseline_probs, test_set,
				token_index=token_index, scale=scale, ablation_method=ablation_method,
				sparsity_threshold=sparsity_threshold, lm_model=lm_model,
				target_token_id=target_token_id
			)
			# For both objectives, higher scores are better.
			if score > candidate_score:
				candidate_score = score
				best_candidate = candidate
				improved = True
				
		if best_candidate is not None:
			best_set.append(best_candidate)
			best_score = candidate_score
			
			# If targeting a specific token, check if it has become the top prediction.
			if target_token_id is not None:
				hook_handles = []
				for (layer, head) in best_set:
					if ablation_method == 'standard':
						hook = make_ablate_hook(head, scale=scale, lm_model=lm_model)
					elif ablation_method == 'permute':
						hook = make_permutation_hook(head, lm_model=lm_model)
					elif ablation_method == 'sparsify':
						hook = make_sparsification_hook(head, sparsity_threshold, lm_model=lm_model)
					else:
						hook = make_ablate_hook(head, scale=scale, lm_model=lm_model)
					hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(hook)
					hook_handles.append(hook_handle)
				
				with torch.no_grad():
					ablated_logits = lm_model(truncated_ids).logits[0, token_index, :]
				
				for handle in hook_handles:
					handle.remove()
					
				ablated_probs = torch.softmax(ablated_logits, dim=-1)
				# If the target token is now the top prediction, stop early.
				top_token_id = torch.argmax(ablated_probs).item()
				if top_token_id == target_token_id:
					print(f"Target token reached rank 1! Stopping ablation search early after adding {len(best_set)} heads.")
					break
		
		iteration_count += 1
		if progress_callback is not None:
			progress_callback(int(INITIAL_PROGRESS + (iteration_count / total_iterations * FINAL_PROGRESS)))
			
	return best_set, best_score


def evaluate_all_heads(truncated_ids, baseline_probs, lm_model, token_index=-1, scale=DEFAULT_SCALE, ablation_method='standard', sparsity_threshold=DEFAULT_SPARSITY_THRESHOLD):
	"""
	Evaluate the impact of ablating each attention head individually for the token at token_index.

	Iterates over all (layer, head) pairs in the model, computing an ablation score using evaluate_candidate.
	The resulting dictionary maps each (layer, head) tuple to its corresponding ablation score.

	:param truncated_ids: Input token IDs up to the token of interest.
	:param baseline_probs: Baseline probability distribution for the token at token_index.
	:param lm_model: The language model instance.
	:param token_index: Index of the token to evaluate.
	:param scale: Scale factor for standard ablation (default: DEFAULT_SCALE).
	:param ablation_method: Method of ablation ('standard', 'permute', or 'sparsify').
	:param sparsity_threshold: Threshold for sparsification (default: DEFAULT_SPARSITY_THRESHOLD).
	:return: Dictionary mapping (layer, head) tuples to their ablation score.
	"""
	head_scores = {}
	for layer in range(lm_model.config.n_layer):
		for head in range(lm_model.config.n_head):
			score = evaluate_candidate(
				truncated_ids, baseline_probs, [(layer, head)],
				token_index=token_index, scale=scale, ablation_method=ablation_method,
				sparsity_threshold=sparsity_threshold, lm_model=lm_model
			)
			head_scores[(layer, head)] = score
	return head_scores
