"""
This module contains functions to create various ablation hooks (standard, permutation,
and sparsification) and functions to evaluate the impact of ablating a set of attention heads.
"""

import torch

def make_ablate_hook(selected_head, scale=0.0, lm_model=None):
	"""
	Standard Ablation Hook.
	
	This hook scales the output for the selected head by the given factor.
	A scale of 0.0 effectively removes the head's contribution.
	
	:param selected_head: Index of the head to ablate.
	:param scale: Scale factor (0.0 means complete removal).
	:param lm_model: The language model instance to access configuration.
	:return: A forward hook function.
	"""
	def hook(module, input, output):
		# Calculate head dimension based on model configuration
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		# Determine the slice of the output corresponding to the head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			# Clone output to avoid in-place modifications
			attn_output_clone = attn_output.clone()
			# Inline comment: Scale out the head’s contribution
			attn_output_clone[:, :, start:end] *= scale
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			output_clone[:, :, start:end] *= scale
			return output_clone
	return hook

def make_permutation_hook(selected_head, lm_model=None):
	"""
	Permutation Ablation Hook.
	
	This hook randomly permutes the outputs of the selected head along the token dimension.
	I use this to check if the exact ordering is critical for performance.
	
	:param selected_head: Index of the head to permute.
	:param lm_model: The language model instance to access configuration.
	:return: A forward hook function.
	"""
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			# Loop over batch elements
			for i in range(attn_output_clone.size(0)):
				# Generate a random permutation of token positions
				perm = torch.randperm(attn_output_clone.size(1))
				# Inline comment: Apply the permutation only to the selected head slice
				attn_output_clone[i, :, start:end] = attn_output_clone[i, perm, start:end]
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			for i in range(output_clone.size(0)):
				perm = torch.randperm(output_clone.size(1))
				output_clone[i, :, start:end] = output_clone[i, perm, start:end]
			return output_clone
	return hook

def make_sparsification_hook(selected_head, sparsity_threshold, lm_model=None):
	"""
	Structured Sparsification Hook.
	
	This hook zeros out small values in the selected head's output that fall below the given threshold.
	
	:param selected_head: Index of the head.
	:param sparsity_threshold: Values below this threshold are zeroed.
	:param lm_model: The language model instance to access configuration.
	:return: A forward hook function.
	"""
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			# Create a mask where absolute values are below the threshold
			mask = torch.abs(attn_output_clone[:, :, start:end]) < sparsity_threshold
			# Inline comment: Zero out the low-importance activations
			attn_output_clone[:, :, start:end] = attn_output_clone[:, :, start:end].masked_fill(mask, 0)
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			mask = torch.abs(output_clone[:, :, start:end]) < sparsity_threshold
			output_clone[:, :, start:end] = output_clone[:, :, start:end].masked_fill(mask, 0)
			return output_clone
	return hook

def evaluate_candidate(truncated_ids, baseline_probs, ablation_set, token_index=-1, scale=0.0, ablation_method='standard',
					   sparsity_threshold=0.1, lm_model=None, epsilon=1e-10):
	"""
	Evaluate the impact of ablating a given set of attention heads for a selected token.

	This function computes a combined ablation score for the token at the specified token_index.
	The score is based on the KL divergence between the baseline and ablated probability distributions,
	plus the change in the top token's probability.

	:param truncated_ids: Torch tensor of input token IDs up to (and including) the token of interest.
	:param baseline_probs: Baseline next-token probability distribution for the token at token_index.
	:param ablation_set: List of (layer, head) tuples representing the heads to ablate.
	:param token_index: The index of the token to evaluate (default is -1, i.e. the last token).
	:param scale: Scale factor for standard ablation (0.0 indicates full ablation).
	:param ablation_method: Method of ablation ('standard', 'permute', or 'sparsify').
	:param sparsity_threshold: Threshold for zeroing small activations (used in sparsification).
	:param lm_model: The language model instance.
	:param epsilon: Small constant to avoid division by zero.
	:return: Combined ablation score as a float.
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
	
	# Compute the logits for the selected token using the provided token_index.
	with torch.no_grad():
		ablated_logits = lm_model(truncated_ids).logits[0, token_index, :]
	
	# Remove hooks to reset the model.
	for handle in hook_handles:
		handle.remove()
	
	ablated_probs = torch.softmax(ablated_logits, dim=-1)
	# Calculate KL divergence and top token probability change.
	kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
	delta_top_prob = baseline_probs.max().item() - ablated_probs.max().item()
	combined_score = 1.0 * kl_div + 1.0 * delta_top_prob
	return combined_score


def find_best_ablation_combo(truncated_ids, baseline_probs, token_index=-1, max_head_layer_pairs=10, scale=0.0, ablation_method='standard',
							  sparsity_threshold=0.1, lm_model=None, progress_callback=None, search_strategy='greedy'):
	"""
	Find the best combination of attention heads to ablate for maximal effect on the token at token_index.

	The function evaluates individual heads, pre-selects top candidates, and then uses an iterative pair
	search followed by greedy expansion to identify a combination of heads whose ablation maximally changes
	the prediction for the specified token.

	:param truncated_ids: Torch tensor of input token IDs up to the token of interest.
	:param baseline_probs: Baseline probability distribution for the token at token_index.
	:param token_index: The index of the token to evaluate.
	:param max_head_layer_pairs: Maximum number of heads to ablate.
	:param scale: Ablation scale factor.
	:param ablation_method: Ablation method ('standard', 'permute', or 'sparsify').
	:param sparsity_threshold: Threshold for sparsification.
	:param lm_model: The language model instance.
	:param progress_callback: Callback function to report progress.
	:param search_strategy: 'greedy' or 'iterative' search strategy.
	:return: Tuple (best_set, best_score) where best_set is a list of (layer, head) tuples.
	"""
	# Create a list of all possible (layer, head) combinations.
	candidate_list = [(layer, head) for layer in range(lm_model.config.n_layer)
					  for head in range(lm_model.config.n_head)]
	candidate_scores = []
	total_candidates = len(candidate_list)
	# Evaluate each candidate head.
	for idx, candidate in enumerate(candidate_list):
		score = evaluate_candidate(truncated_ids, baseline_probs, [candidate],
								   token_index=token_index, scale=scale, ablation_method=ablation_method,
								   sparsity_threshold=sparsity_threshold, lm_model=lm_model)
		candidate_scores.append((candidate, score))
		if progress_callback is not None:
			progress_callback(int((idx + 1) / total_candidates * 40))
	# Sort candidates by score in descending order.
	candidate_scores.sort(key=lambda x: x[1], reverse=True)
	# Pre-select the top 20% of candidates.
	preselected = [cand for cand, _ in candidate_scores[:max(1, len(candidate_scores) // 5)]]

	# Initialize best_set and best_score with an empty set.
	best_set = []
	best_score = evaluate_candidate(truncated_ids, baseline_probs, best_set,
									token_index=token_index, scale=scale, ablation_method=ablation_method,
									sparsity_threshold=sparsity_threshold, lm_model=lm_model)
	# Optional pair search if enabled.
	if search_strategy == 'iterative' and max_head_layer_pairs >= 2 and len(preselected) >= 2:
		best_pair = None
		best_pair_score = best_score  # start with the empty set score
		for i in range(len(preselected)):
			for j in range(i + 1, len(preselected)):
				pair = [preselected[i], preselected[j]]
				score = evaluate_candidate(truncated_ids, baseline_probs, pair,
										   token_index=token_index, scale=scale, ablation_method=ablation_method,
										   sparsity_threshold=sparsity_threshold, lm_model=lm_model)
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
			score = evaluate_candidate(truncated_ids, baseline_probs, test_set,
									   token_index=token_index, scale=scale, ablation_method=ablation_method,
									   sparsity_threshold=sparsity_threshold, lm_model=lm_model)
			if score > candidate_score:
				candidate_score = score
				best_candidate = candidate
				improved = True
		if best_candidate is not None:
			best_set.append(best_candidate)
			best_score = candidate_score
		iteration_count += 1
		if progress_callback is not None:
			progress_callback(int(iteration_count / total_iterations * 100))
	return best_set, best_score


def evaluate_all_heads(truncated_ids, baseline_probs, lm_model, token_index=-1, scale=0.0, ablation_method='standard', sparsity_threshold=0.1):
	"""
	Evaluate the impact of ablating each attention head individually for the token at token_index.

	This function iterates over all (layer, head) pairs in the model, computing an ablation score using evaluate_candidate.
	The score reflects the impact on the prediction for the token at the specified index.

	:param truncated_ids: Input token IDs up to the token of interest.
	:param baseline_probs: Baseline probability distribution for the token at token_index.
	:param lm_model: The language model instance.
	:param token_index: Index of the token to evaluate.
	:param scale: Scale factor for standard ablation (0.0 means full ablation).
	:param ablation_method: Method of ablation ('standard', 'permute', 'sparsify').
	:param sparsity_threshold: Threshold for sparsification.
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


