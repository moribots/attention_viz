"""
Module containing functions to create various ablation hooks for modifying
the outputs of attention heads in a language model. The hooks provided include:
 - Standard ablation: scales the head's output.
 - Permutation ablation: randomly permutes the head's output along the token dimension.
 - Sparsification: zeroes out activations that fall below a given threshold.
"""

import torch
from config import DEFAULT_SCALE, DEFAULT_SPARSITY_THRESHOLD

def make_ablate_hook(selected_head, scale=DEFAULT_SCALE, lm_model=None):
	"""
	Create a standard ablation hook that scales the output for a specific attention head.

	The hook computes the head's slice based on the model configuration and scales it.
	A scale factor of 0.0 removes the head's contribution.

	:param selected_head: Index of the head to ablate.
	:param scale: Scale factor (default is DEFAULT_SCALE, 0.0 means complete removal).
	:param lm_model: The language model instance to access configuration.
	:return: A forward hook function to be registered on the attention module.
	"""
	def hook(module, input, output):
		# Calculate head dimension using the model's hidden size and number of heads.
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		# Determine the slice corresponding to the selected head.
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			# Clone output to avoid in-place modifications.
			attn_output_clone = attn_output.clone()
			# Scale out the head’s contribution.
			attn_output_clone[:, :, start:end] *= scale
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			output_clone[:, :, start:end] *= scale
			return output_clone
	return hook

def make_permutation_hook(selected_head, lm_model=None):
	"""
	Create a permutation ablation hook that randomly permutes the outputs of a selected head.

	This hook is used to determine if the exact token ordering of the head's output is critical.
	
	:param selected_head: Index of the head to permute.
	:param lm_model: The language model instance to access configuration.
	:return: A forward hook function to be registered on the attention module.
	"""
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			# Loop over each batch element.
			for i in range(attn_output_clone.size(0)):
				# Generate a random permutation of token positions.
				perm = torch.randperm(attn_output_clone.size(1))
				# Apply the permutation only to the selected head slice.
				attn_output_clone[i, :, start:end] = attn_output_clone[i, perm, start:end]
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			for i in range(output_clone.size(0)):
				perm = torch.randperm(output_clone.size(1))
				output_clone[i, :, start:end] = output_clone[i, perm, start:end]
			return output_clone
	return hook

def make_sparsification_hook(selected_head, sparsity_threshold=DEFAULT_SPARSITY_THRESHOLD, lm_model=None):
	"""
	Create a sparsification hook that zeroes out small values in the selected head's output.

	Values with an absolute value below the specified threshold are set to zero.
	
	:param selected_head: Index of the head.
	:param sparsity_threshold: Threshold below which activations are zeroed.
	:param lm_model: The language model instance to access configuration.
	:return: A forward hook function to be registered on the attention module.
	"""
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			# Create a mask identifying activations below the threshold.
			mask = torch.abs(attn_output_clone[:, :, start:end]) < sparsity_threshold
			# Zero out the low-importance activations.
			attn_output_clone[:, :, start:end] = attn_output_clone[:, :, start:end].masked_fill(mask, 0)
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			mask = torch.abs(output_clone[:, :, start:end]) < sparsity_threshold
			output_clone[:, :, start:end] = output_clone[:, :, start:end].masked_fill(mask, 0)
			return output_clone
	return hook
