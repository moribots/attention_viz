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

def evaluate_candidate(truncated_ids, baseline_probs, ablation_set, scale=0.0, ablation_method='standard',
					   sparsity_threshold=0.1, lm_model=None, epsilon=1e-10):
	"""
	Evaluate the impact of ablating a given set of heads.

	I compute a combined score based on KL divergence and the drop in the top token's probability.
	
	:param truncated_ids: Input token IDs up to the token of interest.
	:param baseline_probs: Baseline next-token probability distribution.
	:param ablation_set: List of (layer, head) tuples to ablate.
	:param scale: Scale factor for standard ablation.
	:param ablation_method: Method to use ('standard', 'permute', 'sparsify').
	:param sparsity_threshold: Threshold for sparsification.
	:param lm_model: The language model instance.
	:param epsilon: Small constant to avoid division by zero.
	:return: Combined score as a float.
	"""
	hook_handles = []
	# Attach the appropriate hook for each head in the ablation set
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
	# Get the logits after applying the hooks (i.e., after ablation)
	with torch.no_grad():
		ablated_logits = lm_model(truncated_ids).logits[0, -1, :]
	# Remove all hooks to reset the model state
	for handle in hook_handles:
		handle.remove()
	ablated_probs = torch.softmax(ablated_logits, dim=-1)
	# Calculate KL divergence and top token probability difference
	kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
	delta_top_prob = baseline_probs.max().item() - ablated_probs.max().item()
	# I combine these metrics with equal weights (alpha and beta are both 1.0)
	combined_score = 1.0 * kl_div + 1.0 * delta_top_prob
	return combined_score

def find_best_ablation_combo(truncated_ids, baseline_probs, max_heads=10, scale=0.0, ablation_method='standard',
							  sparsity_threshold=0.1, lm_model=None, progress_callback=None, search_strategy='greedy'):
	"""
	Find the best combination of heads to ablate for maximal effect.

	This function first evaluates each head individually, then pre-selects the top 20%.
	It optionally performs an iterative pair search and a greedy expansion to refine the selection.
	It greedily add heads from that preselected pool until no improvement or max_heads is reached.
	
	:param truncated_ids: Input token IDs.
	:param baseline_probs: Baseline probability distribution.
	:param max_heads: Maximum number of heads to ablate.
	:param scale: Ablation scale factor.
	:param ablation_method: Method to use ('standard', 'permute', 'sparsify').
	:param sparsity_threshold: Threshold for sparsification.
	:param lm_model: The language model instance.
	:param progress_callback: Callback to report progress.
	:param search_strategy: 'greedy' or 'iterative'
	:return: Tuple (best_set, best_score)
	"""
	# Create a list of all possible (layer, head) combinations
	candidate_list = [(layer, head) for layer in range(lm_model.config.n_layer)
					  for head in range(lm_model.config.n_head)]
	candidate_scores = []
	total_candidates = len(candidate_list)
	# Evaluate each candidate head and report progress
	for idx, candidate in enumerate(candidate_list):
		score = evaluate_candidate(truncated_ids, baseline_probs, [candidate],
								   scale=scale, ablation_method=ablation_method,
								   sparsity_threshold=sparsity_threshold, lm_model=lm_model)
		candidate_scores.append((candidate, score))
		if progress_callback is not None:
			progress_callback(int((idx + 1) / total_candidates * 40))
	# Sort candidates by score in descending order
	candidate_scores.sort(key=lambda x: x[1], reverse=True)
	# Pre-select the top 20% of candidates
	preselected = [cand for cand, _ in candidate_scores[:max(1, len(candidate_scores) // 5)]]

	# Initialize best_set and best_score using an iterative pair search if enabled
	best_set = []
	best_score = evaluate_candidate(truncated_ids, baseline_probs, best_set,
									scale=scale, ablation_method=ablation_method,
									sparsity_threshold=sparsity_threshold, lm_model=lm_model)
	if search_strategy == 'iterative' and max_heads >= 2 and len(preselected) >= 2:
		best_pair = None
		best_pair_score = best_score  # start with the score of the empty set
		# Check all pairs among preselected heads
		for i in range(len(preselected)):
			for j in range(i + 1, len(preselected)):
				pair = [preselected[i], preselected[j]]
				score = evaluate_candidate(truncated_ids, baseline_probs, pair,
										   scale=scale, ablation_method=ablation_method,
										   sparsity_threshold=sparsity_threshold, lm_model=lm_model)
				# Inline comment: Update best_pair if we find a higher combined score
				if score > best_pair_score:
					best_pair_score = score
					best_pair = pair
		if best_pair is not None:
			best_set = best_pair
			best_score = best_pair_score

	# Greedy iterative expansion: try adding one head at a time
	improved = True
	iteration_count = 0
	total_iterations = len(preselected) + 1  # heuristic for progress update
	while improved and len(best_set) < max_heads:
		improved = False
		best_candidate = None
		candidate_score = best_score
		# Evaluate adding each preselected candidate not already in best_set
		for candidate in preselected:
			if candidate in best_set:
				continue
			test_set = best_set + [candidate]
			score = evaluate_candidate(truncated_ids, baseline_probs, test_set,
									   scale=scale, ablation_method=ablation_method,
									   sparsity_threshold=sparsity_threshold, lm_model=lm_model)
			# Inline comment: If this candidate improves the score, choose it
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

def evaluate_all_heads(truncated_ids, baseline_probs, lm_model, scale=0.0, ablation_method='standard', sparsity_threshold=0.1):
	"""
	Evaluate the impact of ablating each attention head individually.

	This function iterates over all (layer, head) pairs in the model, computes the ablation
	score using evaluate_candidate, and returns a dictionary mapping each (layer, head) tuple to its score.
	
	The ablation score is computed as a combination of the KL divergence between the baseline and ablated
	probability distributions and the change in the top token probability. A higher score indicates a 
	more critical head.
	
	:param truncated_ids: Input token IDs up to the token of interest.
	:param baseline_probs: Baseline next-token probability distribution.
	:param lm_model: The language model instance.
	:param scale: Scale factor for standard ablation (0.0 means full ablation).
	:param ablation_method: Method to use ('standard', 'permute', 'sparsify').
	:param sparsity_threshold: Threshold for structured sparsification.
	:return: Dictionary mapping (layer, head) tuples to their ablation score.
	"""
	head_scores = {}
	for layer in range(lm_model.config.n_layer):
		for head in range(lm_model.config.n_head):
			score = evaluate_candidate(
				truncated_ids, baseline_probs, [(layer, head)],
				scale=scale, ablation_method=ablation_method,
				sparsity_threshold=sparsity_threshold, lm_model=lm_model
			)
			head_scores[(layer, head)] = score
	return head_scores

