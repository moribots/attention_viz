"""
This module identifies and analyzes attention circuits in transformer models.
It implements algorithms for discovering functional subgraphs (circuits) based on
ablation studies and path patching experiments.
"""

import time
from functools import lru_cache
from typing import Union, List, Optional, Dict, Tuple

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import (
	MAX_PATH_PATCHING_TIME,
	HEAD_CLASSIFICATION_LATE_LAYER_THRESHOLD,
	HEAD_CLASSIFICATION_MID_LAYER_THRESHOLD,
	NAME_MOVER_PROB_THRESHOLD,
	INDUCTION_PROB_LOW,
	INDUCTION_PROB_HIGH,
	PROGRESS_STAGE_1,
	PROGRESS_STAGE_2,
	PROGRESS_STAGE_3,
	PROGRESS_STAGE_4,
	PROGRESS_STAGE_5,
	ATTENTION_TO_TOKEN_DUMMY,
	HEAD_INFLUENCE_DUMMY,
	PATH_PATCHING_DUMMY_EFFECT,
	PATH_PATCHING_PROGRESS_START,
	PATH_PATCHING_PROGRESS_RANGE,
)


class CircuitFinder:
	"""
	CircuitFinder identifies attention circuits in transformer models by analyzing
	the flow of information through attention heads and their connections.
	"""

	def __init__(self, transformer_model, tokenizer):
		"""
		Initialize the CircuitFinder with a transformer model and its tokenizer.

		:param transformer_model: The transformer model to analyze.
		:param tokenizer: The tokenizer associated with the model.
		"""
		self.model = transformer_model
		self.tokenizer = tokenizer
		self.num_layers = transformer_model.config.n_layer
		self.num_heads = transformer_model.config.n_head
		# Graph representation of the circuit.
		self.circuit_graph = None
		# Cache for activations.
		self.activation_cache = {}
		# Maximum allowed time for path patching experiments (in seconds).
		self.max_path_patching_time = MAX_PATH_PATCHING_TIME

	# =========================================================================
	# High-Level Circuit Analysis Methods
	# =========================================================================

	def classify_head(self, layer: int, head: int, pred_info: dict, input_text: str) -> str:
		"""
		Classify an attention head into a role based on prediction information and input token statistics.
		The classification is heuristic-based and uses layer position and probability thresholds.

		:param layer: Index of the layer containing the head.
		:param head: Index of the attention head.
		:param pred_info: Dictionary with keys 'token' and 'prob' for head's prediction.
		:param input_text: The original input text.
		:return: A string representing the head's role.
		"""
		token = pred_info.get('token', '')
		prob = pred_info.get('prob', 0.0)
		# Tokenize the input text and count occurrences of the predicted token.
		tokens = self.tokenizer.tokenize(input_text)
		token_count = tokens.count(token)
		# Heuristics based on layer position.
		if layer >= int(self.num_layers * HEAD_CLASSIFICATION_LATE_LAYER_THRESHOLD):
			if token_count == 1 and prob > NAME_MOVER_PROB_THRESHOLD:
				return "Name Mover"
			elif token_count > 1:
				return "Duplicate Token"
			else:
				return "S-Inhibition"
		if int(self.num_layers * HEAD_CLASSIFICATION_MID_LAYER_THRESHOLD) <= layer < int(self.num_layers * HEAD_CLASSIFICATION_LATE_LAYER_THRESHOLD):
			if token_count > 1 and INDUCTION_PROB_LOW < prob < INDUCTION_PROB_HIGH:
				return "Induction"
			else:
				return "Previous Token"
		if layer < int(self.num_layers * HEAD_CLASSIFICATION_MID_LAYER_THRESHOLD):
			return "Backup Name Mover"
		return "Unknown"

	def compute_logit_contribution(
		self, layer: int, head: int, input_text: str, target_token_id: Optional[int]
	) -> float:
		"""
		Compute the logit-lens contribution of a given head to the target token's logit.
		The method extracts the head's activation for the final token, projects it using the head's
		c_proj weight slice, applies the final layer norm, and computes the dot product with the target
		token's unembedding vector.

		:param layer: Layer index.
		:param head: Head index.
		:param input_text: Input text.
		:param target_token_id: The target token's ID; if None, returns 0.0.
		:return: Contribution as a float.
		"""
		if target_token_id is None:
			return 0.0
		# Encode input and collect activations.
		input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
		input_ids_tuple = tuple(input_ids[0].tolist())
		activations = self._collect_activations(input_ids_tuple)
		hidden_size = self.model.config.hidden_size
		head_dim = hidden_size // self.num_heads
		# Use the final layer norm.
		ln_f = self.model.transformer.ln_f
		# Get unembedding matrix (assumed tied with token embeddings via lm_head).
		lm_head_weight = self.model.lm_head.weight  # shape [vocab_size, hidden_size]
		# Retrieve activation for the specified head.
		head_act = activations.get(layer, {}).get(head)
		if head_act is None:
			return 0.0
		# Use activation from the final token.
		head_act_last = head_act[:, -1, :]  # shape [1, head_dim]
		# Extract head projection matrix from c_proj weight.
		c_proj_weight = self.model.transformer.h[layer].attn.c_proj.weight  # shape [hidden_size, hidden_size]
		head_proj = c_proj_weight[:, head * head_dim:(head + 1) * head_dim]  # shape [hidden_size, head_dim]
		# Project activation into full hidden space.
		contrib_vector = head_act_last @ head_proj.T  # shape [1, hidden_size]
		# Apply final layer norm.
		contrib_ln = ln_f(contrib_vector)  # shape [1, hidden_size]
		# Get target token's unembedding vector.
		target_emb = lm_head_weight[target_token_id]  # shape [hidden_size]
		# Compute dot product as contribution.
		contribution = (contrib_ln * target_emb).sum().item()
		return contribution

	def build_circuit_from_ablation(
		self,
		important_heads: List[Tuple[int, int]],
		input_text: str,
		target_token_id: Optional[Union[int, str]] = None,
		path_threshold: float = 0.05,
		progress_callback: Optional[callable] = None,
	) -> nx.DiGraph:
		"""
		Build a circuit graph from ablation study results and path patching experiments.
		The resulting graph includes attention head nodes (with role and logit contribution annotations),
		token nodes, and optionally an output node. Edges are added based on direct layer-to-layer connections
		and significant path patching effects.

		:param important_heads: List of (layer, head) tuples deemed important.
		:param input_text: Input text for analysis.
		:param target_token_id: Optional token ID to focus on (for output node).
		:param path_threshold: Threshold for including patching paths in the circuit.
		:param progress_callback: Optional callback for reporting progress (expects integer progress values).
		:return: A NetworkX directed graph representing the circuit.
		"""
		if progress_callback:
			progress_callback(PROGRESS_STAGE_1)
		G = nx.DiGraph()
		input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
		input_ids_single = input_ids[0]
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids_single)

		# Retrieve head predictions (assumes _get_head_predictions returns a dict mapping (layer, head) to prediction info)
		head_predictions = self._get_head_predictions(input_text, important_heads)

		# Add nodes for each important head.
		for layer, head in important_heads:
			node_id = f"L{layer}H{head}"
			pred_info = head_predictions.get((layer, head), {})
			pred_token = pred_info.get('token', '??')
			pred_prob = pred_info.get('prob', 0.0)
			role = self.classify_head(layer, head, pred_info, input_text)
			logit_contrib = self.compute_logit_contribution(layer, head, input_text, target_token_id)
			G.add_node(
				node_id,
				layer=layer,
				head=head,
				type="attention_head",
				pred_token=pred_token,
				pred_prob=pred_prob,
				role=role,
				logit_contrib=logit_contrib,
			)

		# Add token nodes.
		for i, token in enumerate(tokens):
			node_id = f"T{i}"
			G.add_node(node_id, token=token, position=i, type="token")

		# Add output node if a target token is specified.
		if target_token_id is not None:
			try:
				if isinstance(target_token_id, str):
					target_token_id = int(target_token_id)
				elif isinstance(target_token_id, torch.Tensor):
					target_token_id = target_token_id.item()
				else:
					target_token_id = int(target_token_id)
				target_token = self.tokenizer.decode([target_token_id]).strip()
				G.add_node("OUTPUT", token=target_token, type="output")
			except (ValueError, TypeError) as e:
				print(f"Error processing target_token_id: {e}")
				target_token_id = None

		if progress_callback:
			progress_callback(PROGRESS_STAGE_2)

		# Add edges between heads based on direct connections (adjacent layers).
		connect_heads = []
		layer_nums = sorted({layer for layer, _ in important_heads})
		for i in range(len(layer_nums) - 1):
			layer1 = layer_nums[i]
			layer2 = layer_nums[i + 1]
			heads1 = [head for layer, head in important_heads if layer == layer1]
			heads2 = [head for layer, head in important_heads if layer == layer2]
			for head1 in heads1:
				for head2 in heads2:
					connect_heads.append(((layer1, head1), (layer2, head2)))
		if connect_heads:
			for (src_layer, src_head), (dst_layer, dst_head) in connect_heads:
				src_id = f"L{src_layer}H{src_head}"
				dst_id = f"L{dst_layer}H{dst_head}"
				# Closer layers receive higher weight.
				weight = 1.0 / max(1, dst_layer - src_layer)
				G.add_edge(src_id, dst_id, weight=weight, effect=weight)

		if progress_callback:
			progress_callback(PROGRESS_STAGE_3)

		# Run path patching if the number of important heads is modest.
		if 1 < len(important_heads) <= 8:
			path_effects = self.run_path_patching(
				input_text,
				start_pos=0,
				end_pos=len(tokens) - 1,
				significant_threshold=path_threshold,
				progress_callback=progress_callback,
				max_pairs=100,
			)
			for (src_layer, src_head), (dst_layer, dst_head) in path_effects:
				src_id = f"L{src_layer}H{src_head}"
				dst_id = f"L{dst_layer}H{dst_head}"
				if src_id in G.nodes and dst_id in G.nodes:
					effect = path_effects[((src_layer, src_head), (dst_layer, dst_head))]
					if not G.has_edge(src_id, dst_id) or G.edges[src_id, dst_id]['weight'] < abs(effect):
						G.add_edge(src_id, dst_id, weight=abs(effect), effect=effect)

		if progress_callback:
			progress_callback(PROGRESS_STAGE_4)

		# Add edges from tokens to first-layer heads.
		important_token_indices = [0]
		if len(tokens) > 1:
			important_token_indices.append(len(tokens) - 1)
		if len(tokens) > 2:
			important_token_indices.append(len(tokens) // 2)
		min_layer = min(layer for layer, _ in important_heads)
		for i in important_token_indices:
			token_id = f"T{i}"
			for layer, head in important_heads:
				if layer == min_layer:
					head_id = f"L{layer}H{head}"
					weight = self._get_attention_to_token(input_text, layer, head, i)
					if weight > path_threshold:
						G.add_edge(token_id, head_id, weight=weight)

		# Add edges from last-layer heads to the output node (if target token provided).
		if target_token_id is not None:
			last_layer = max(layer for layer, _ in important_heads)
			for layer, head in important_heads:
				if layer == last_layer:
					head_id = f"L{layer}H{head}"
					influence = self._get_head_influence_on_token(input_text, layer, head, int(target_token_id))
					if influence > path_threshold:
						G.add_edge(head_id, "OUTPUT", weight=influence)

		if progress_callback:
			progress_callback(PROGRESS_STAGE_5)

		self.circuit_graph = G
		return G

	def visualize_circuit(self, save_path: Optional[str] = None) -> None:
		"""
		Visualize the identified circuit as a directed graph.
		Node labels include IDs, roles, and prediction info. Nodes are color-coded by type and role.

		:param save_path: Optional file path to save the visualization.
		:return: None (the graph is displayed or saved).
		"""
		if self.circuit_graph is None:
			raise ValueError("No circuit graph available. Run build_circuit_from_ablation first.")

		G = self.circuit_graph
		plt.figure(figsize=(16, 10))
		pos = {}

		# Position token nodes on the far left.
		token_nodes = [(n, d.get('position', 0)) for n, d in G.nodes(data=True) if d.get('type') == "token"]
		token_nodes.sort(key=lambda x: x[1])
		for i, (node, _) in enumerate(token_nodes):
			pos[node] = (-2, -(i - len(token_nodes) / 2))

		# Position attention head nodes by layer.
		max_layer = max(d.get('layer', 0) for _, d in G.nodes(data=True) if d.get('type') == "attention_head")
		for layer in range(max_layer + 1):
			heads_in_layer = [n for n, d in G.nodes(data=True) if d.get('type') == "attention_head" and d.get('layer') == layer]
			for i, node in enumerate(sorted(heads_in_layer)):
				pos[node] = (layer * 2, i - len(heads_in_layer) / 2)

		# Position the output node on the far right.
		if "OUTPUT" in G.nodes:
			pos["OUTPUT"] = (max_layer * 2 + 2, 0)

		# Define color mapping for head roles.
		role_colors = {
			"Name Mover": "lightgreen",
			"Duplicate Token": "gold",
			"S-Inhibition": "orange",
			"Induction": "violet",
			"Previous Token": "cyan",
			"Backup Name Mover": "lightblue",
			"Unknown": "gray"
		}

		# Draw token nodes.
		token_node_list = [n for n, d in G.nodes(data=True) if d.get('type') == "token"]
		nx.draw_networkx_nodes(G, pos, nodelist=token_node_list, node_color='skyblue',
							   node_size=500, alpha=0.8, node_shape='s')
		# Draw attention head nodes by role.
		head_node_list = [n for n, d in G.nodes(data=True) if d.get('type') == "attention_head"]
		for role, color in role_colors.items():
			nodes_with_role = [n for n in head_node_list if G.nodes[n].get("role", "Unknown") == role]
			if nodes_with_role:
				nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_role, node_color=color,
									   node_size=700, alpha=0.8)
		# Draw output node.
		output_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == "output"]
		nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='salmon',
							   node_size=800, alpha=0.8)

		# Create labels for nodes.
		labels = {}
		for n, d in G.nodes(data=True):
			if d.get('type') == "token":
				labels[n] = f"{d.get('position')}:{d.get('token')}"
			elif d.get('type') == "attention_head":
				role = d.get("role", "Unknown")
				pred_token = d.get("pred_token", "??")
				labels[n] = f"L{d.get('layer')}H{d.get('head')}\n[{role}]\n→ {pred_token}"
			elif d.get('type') == "output":
				labels[n] = f"OUTPUT\n({d.get('token','')})"
		nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_family="monospace")

		# Draw edges with widths proportional to weight.
		edges = G.edges(data=True)
		edge_weights = [d.get('weight', 1) * 3 for _, _, d in edges]
		nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray',
							   arrows=True, connectionstyle='arc3,rad=0.1')

		# Add legend with explanations.
		legend_elements = [
			plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='skyblue',
					   markersize=15, label='Input Token'),
			plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon',
					   markersize=15, label='Output Token')
		]
		role_explanations = {
			"Name Mover": "Propagates specific token information",
			"Duplicate Token": "Processes repeated tokens",
			"S-Inhibition": "Suppresses certain activations",
			"Induction": "Completes sequences based on context",
			"Previous Token": "Propagates previous token context",
			"Backup Name Mover": "Early-layer support for propagation",
			"Unknown": "Unclassified function"
		}
		for role, color in role_colors.items():
			if any(G.nodes[n].get("role", "Unknown") == role for n in head_node_list):
				legend_elements.append(
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
							   markersize=15, label=f'{role}: {role_explanations[role]}')
				)
		plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
				   fancybox=True, shadow=True, ncol=2, fontsize=9)

		plt.title('Transformer Circuit - Attention Flow Graph')
		plt.axis('off')
		plt.tight_layout()
		if save_path:
			plt.savefig(save_path)
		plt.show()

	def run_path_patching(
		self,
		input_text: str,
		start_pos: int,
		end_pos: int,
		source_input: Optional[str] = None,
		significant_threshold: float = 0.1,
		progress_callback: Optional[callable] = None,
		max_pairs: int = 100,
	) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
		"""
		Run path patching experiments to identify causal effects between heads.
		Path patching replaces activations from one run with activations from a corrupted run
		to measure the effect of specific paths.

		:param input_text: The original text to analyze.
		:param start_pos: Starting token position for analysis.
		:param end_pos: Ending token position for analysis.
		:param source_input: Optional alternative input; if None, a corrupted input is generated.
		:param significant_threshold: Threshold above which an effect is considered significant.
		:param progress_callback: Optional callback to report progress.
		:param max_pairs: Maximum number of head pairs to test.
		:return: Dictionary mapping ((src_layer, src_head), (dst_layer, dst_head)) to effect size.
		"""
		if source_input is None:
			source_input = self._create_corrupted_input(input_text)
		input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
		source_ids = self.tokenizer.encode(source_input, return_tensors="pt")
		cache_key = str(input_ids.detach().cpu().numpy().tobytes())
		if cache_key in self.activation_cache:
			clean_activations = self.activation_cache[cache_key]
		else:
			clean_activations = self._collect_activations(tuple(input_ids[0].tolist()))
			self.activation_cache[cache_key] = clean_activations
		source_cache_key = str(source_ids.detach().cpu().numpy().tobytes())
		if source_cache_key in self.activation_cache:
			source_activations = self.activation_cache[source_cache_key]
		else:
			source_activations = self._collect_activations(tuple(source_ids[0].tolist()))
			self.activation_cache[source_cache_key] = source_activations
		path_effects = {}
		start_time = time.time()
		pairs_to_test = []
		# Generate candidate pairs for adjacent layers.
		for src_layer in range(self.num_layers - 1):
			dst_layer = src_layer + 1
			for src_head in range(self.num_heads):
				for dst_head in range(self.num_heads):
					pairs_to_test.append((src_layer, src_head, dst_layer, dst_head))
		if len(pairs_to_test) > max_pairs:
			np.random.shuffle(pairs_to_test)
			pairs_to_test = pairs_to_test[:max_pairs]
		total_pairs = len(pairs_to_test)
		for i, (src_layer, src_head, dst_layer, dst_head) in enumerate(pairs_to_test):
			elapsed = time.time() - start_time
			if elapsed > self.max_path_patching_time:
				print(f"Path patching timeout after {elapsed:.1f}s. Processed {i}/{total_pairs} pairs.")
				break
			if progress_callback and i % 5 == 0:
				progress_callback(PATH_PATCHING_PROGRESS_START + int((i / total_pairs) * PATH_PATCHING_PROGRESS_RANGE))
			effect = self._measure_path_effect(
				input_ids,
				clean_activations,
				source_activations,
				src_layer,
				src_head,
				dst_layer,
				dst_head,
				start_pos,
				end_pos,
			)
			if abs(effect) > significant_threshold:
				path_effects[((src_layer, src_head), (dst_layer, dst_head))] = effect
		return path_effects

	def find_heads_by_detection(
		self,
		seq: str,
		detection_pattern: Union[torch.Tensor, str],
		threshold: float = 0.5,
		exclude_bos: bool = False,
		exclude_current_token: bool = False,
		error_measure: str = "abs",
		heads: Optional[Union[list, dict]] = None,
		cache: Optional[dict] = None,
	) -> List[Tuple[int, int]]:
		"""
		Detect and return a list of attention head indices that match the specified detection pattern.
		A similarity score is computed for each head's attention pattern against the provided pattern,
		and heads with scores above the threshold are returned.

		:param seq: The input sequence as a string.
		:param detection_pattern: Either a tensor of shape (seq_len, seq_len) or a string specifying a supported pattern.
		:param threshold: Minimum similarity score for a head to be selected.
		:param exclude_bos: Whether to exclude attention to the beginning-of-sequence token.
		:param exclude_current_token: Whether to exclude attention to the current token (diagonal).
		:param error_measure: The error measure to use ("abs" or "mul").
		:param heads: Optional list or dict of heads to restrict the search.
		:param cache: Optional activation cache; if not provided, model.run_with_cache is used.
		:return: List of (layer, head) tuples with scores above the threshold.
		"""
		cfg = self.model.config
		tokens = self.tokenizer.encode(seq, add_special_tokens=False)
		seq_len = len(tokens)
		# If detection_pattern is a string, obtain the corresponding tensor.
		if isinstance(detection_pattern, str):
			detection_pattern = self.get_detection_pattern(seq, detection_pattern)
		# Move detection pattern to the same device as the model.
		detection_pattern = detection_pattern.to(self.model.cfg.device)
		# Get activation cache if not provided.
		if cache is None:
			tokens_tensor = self.tokenizer.encode(seq, return_tensors="pt")
			_, cache = self.model.run_with_cache(tokens_tensor, remove_batch_dim=True)
		# Prepare heads to consider.
		if heads is None:
			layer2heads = {layer: list(range(cfg.n_head)) for layer in range(cfg.n_layer)}
		elif isinstance(heads, list):
			from collections import defaultdict
			layer2heads = defaultdict(list)
			for layer, head in heads:
				layer2heads[layer].append(head)
		else:
			layer2heads = heads

		# Initialize scores for all heads.
		scores = torch.full((cfg.n_layer, cfg.n_head), -1.0, device=self.model.cfg.device)
		for layer, head_list in layer2heads.items():
			if ("pattern", layer, "attn") in cache:
				layer_attn = cache[("pattern", layer, "attn")]
			else:
				continue
			for head in head_list:
				head_attn = layer_attn[head]  # shape (seq_len, seq_len)
				score = self.compute_head_attention_similarity_score(
					head_attn, detection_pattern, exclude_bos, exclude_current_token, error_measure
				)
				scores[layer, head] = score
		# Select heads with scores above the threshold.
		selected = []
		for layer in range(cfg.n_layer):
			for head in range(cfg.n_head):
				if scores[layer, head] >= threshold:
					selected.append((layer, head))
		return selected

	# =========================================================================
	# Helper Methods (Internal Use)
	# =========================================================================

	@lru_cache(maxsize=8)
	def _collect_activations(self, input_ids_tuple: tuple) -> Dict[int, Dict[int, torch.Tensor]]:
		"""
		Collect activations from the model for a given input.
		Uses an LRU cache to avoid redundant computations.

		:param input_ids_tuple: Input token IDs as a tuple (for caching).
		:return: Dictionary of activations organized by layer and head.
		"""
		try:
			device = next(self.model.parameters()).device
			input_ids = torch.tensor([input_ids_tuple], device=device)
			activations = {}

			def get_activation_hook(layer_idx, head_idx):
				def hook(module, input, output):
					if layer_idx not in activations:
						activations[layer_idx] = {}
					if isinstance(output, tuple):
						activations[layer_idx][head_idx] = output[0].detach().clone()
					else:
						activations[layer_idx][head_idx] = output.detach().clone()
					return output
				return hook

			hooks = []
			for layer in range(self.num_layers):
				# Register a hook per layer (using head index 0 since activations are combined).
				hook = get_activation_hook(layer, 0)
				h = self.model.transformer.h[layer].attn.register_forward_hook(hook)
				hooks.append(h)
			with torch.no_grad():
				self.model(input_ids)
			for h in hooks:
				h.remove()
			return activations
		except Exception as e:
			print(f"Error in _collect_activations: {e}")
			return {i: {} for i in range(self.num_layers)}

	def _measure_path_effect(
		self,
		input_ids: torch.Tensor,
		clean_activations: Dict,
		source_activations: Dict,
		src_layer: int,
		src_head: int,
		dst_layer: int,
		dst_head: int,
		start_pos: int,
		end_pos: int,
	) -> float:
		"""
		Measure the effect of patching a specific path between two heads.
		This simulates replacing activations from the source input and measuring the impact.
		(Dummy implementation provided; replace with real logic if available.)

		:param input_ids: Tensor of input IDs.
		:param clean_activations: Activations from the original input.
		:param source_activations: Activations from the corrupted input.
		:param src_layer: Source layer index.
		:param src_head: Source head index.
		:param dst_layer: Destination layer index.
		:param dst_head: Destination head index.
		:param start_pos: Starting token position for analysis.
		:param end_pos: Ending token position for analysis.
		:return: Measured effect as a float.
		"""
		# Define a hook that replaces source activation.
		def src_hook(module, input, output):
			if isinstance(output, tuple):
				patched_output = list(output)
				patched_output[0] = patched_output[0].clone()
				source_act = source_activations.get(src_layer, {}).get(src_head)
				if source_act is not None:
					patched_output[0][:, :, :] = source_act
				return tuple(patched_output)
			else:
				output_clone = output.clone()
				source_act = source_activations.get(src_layer, {}).get(src_head)
				if source_act is not None:
					output_clone[:, :, :] = source_act
				return output_clone

		# For demonstration, return a dummy effect value.
		return PATH_PATCHING_DUMMY_EFFECT

	def _get_attention_to_token(self, input_text: str, layer: int, head: int, token_index: int) -> float:
		"""
		Compute the attention weight from a specific head to a token.
		(Dummy implementation; in a full implementation, this would extract values from the attention matrix.)

		:param input_text: The input text.
		:param layer: Layer index.
		:param head: Head index.
		:param token_index: Token index.
		:return: Attention weight (dummy value provided).
		"""
		return ATTENTION_TO_TOKEN_DUMMY

	def _get_head_influence_on_token(self, input_text: str, layer: int, head: int, target_token_id: int) -> float:
		"""
		Compute the influence of a head on the target token using its output and unembedding projection.
		(Dummy implementation; replace with full computation if available.)

		:param input_text: The input text.
		:param layer: Layer index.
		:param head: Head index.
		:param target_token_id: Target token ID.
		:return: Influence score (dummy value provided).
		"""
		return HEAD_INFLUENCE_DUMMY

	def _get_head_predictions(self, input_text: str, important_heads: List[Tuple[int, int]]) -> Dict[Tuple[int, int], dict]:
		"""
		Obtain each head's next-token prediction information by running the model.
		For each head, extracts the token it is "copying" and the associated probability.

		:param input_text: The input text.
		:param important_heads: List of (layer, head) tuples to consider.
		:return: Dictionary mapping (layer, head) to a dict with keys 'token' and 'prob'.
		"""
		input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
		input_ids_tuple = tuple(input_ids[0].tolist())
		activations = self._collect_activations(input_ids_tuple)
		predictions = {}
		hidden_size = self.model.config.hidden_size
		head_dim = hidden_size // self.num_heads
		ln_f = self.model.transformer.ln_f
		lm_head_weight = self.model.lm_head.weight  # shape [vocab_size, hidden_size]

		for layer, head in important_heads:
			# Get the activation for the layer (using combined activation from head index 0).
			head_act = activations.get(layer, {}).get(0)
			if head_act is None:
				continue

			# Reshape to separate heads.
			batch_size, seq_len, _ = head_act.shape
			reshaped_act = head_act.view(batch_size, seq_len, self.num_heads, head_dim)
			# Get the activation for the specific head (last token only).
			head_act_last = reshaped_act[:, -1, head, :]  # shape [batch_size, head_dim]
			c_proj_weight = self.model.transformer.h[layer].attn.c_proj.weight  # shape [hidden_size, hidden_size]
			head_proj = c_proj_weight[:, head * head_dim:(head + 1) * head_dim]  # shape [hidden_size, head_dim]
			contrib_vector = head_act_last @ head_proj.T  # shape [batch_size, hidden_size]
			contrib_ln = ln_f(contrib_vector)  # shape [batch_size, hidden_size]
			logits = contrib_ln @ lm_head_weight.T  # shape [batch_size, vocab_size]
			probs = torch.softmax(logits, dim=-1)[0]
			top_prob, top_index = torch.max(probs, dim=-1)
			pred_token = self.tokenizer.decode([top_index.item()]).strip()
			predictions[(layer, head)] = {"token": pred_token, "prob": top_prob.item()}

		return predictions

	def _create_corrupted_input(self, input_text: str) -> str:
		"""
		Create a corrupted version of the input text for control experiments.
		For example, by shuffling the tokens.

		:param input_text: The original input text.
		:return: A corrupted version of the input text.
		"""
		tokens = self.tokenizer.tokenize(input_text)
		corrupted = " ".join(np.random.permutation(tokens))
		return corrupted

	# =========================================================================
	# Detection Pattern Utility Methods
	# =========================================================================

	def get_previous_token_head_detection_pattern(self, tokens: torch.Tensor) -> torch.Tensor:
		"""
		Generate a detection pattern for previous-token heads.
		The pattern is a lower triangular matrix with ones shifted one position right.

		:param tokens: Tensor of token IDs with shape (seq_len,).
		:return: Detection pattern tensor of shape (seq_len, seq_len).
		"""
		seq_len = tokens.shape[0]
		detection_pattern = torch.zeros(seq_len, seq_len)
		if seq_len > 1:
			detection_pattern[1:, :-1] = torch.eye(seq_len - 1)
		return torch.tril(detection_pattern)

	def get_duplicate_token_head_detection_pattern(self, tokens: torch.Tensor) -> torch.Tensor:
		"""
		Generate a detection pattern for duplicate-token heads.
		Marks positions where tokens are duplicates (diagonal is zeroed out).

		:param tokens: Tensor of token IDs with shape (seq_len,).
		:return: Detection pattern tensor of shape (seq_len, seq_len).
		"""
		seq_len = tokens.shape[0]
		token_pattern = tokens.repeat(seq_len, 1)  # shape (seq_len, seq_len)
		eq_mask = (token_pattern == token_pattern.T).int()
		for i in range(seq_len):
			eq_mask[i, i] = 0
		return torch.tril(eq_mask.float())

	def get_induction_head_detection_pattern(self, tokens: torch.Tensor) -> torch.Tensor:
		"""
		Generate a detection pattern for induction heads by shifting the duplicate-token pattern.
		
		:param tokens: Tensor of token IDs with shape (seq_len,).
		:return: Detection pattern tensor of shape (seq_len, seq_len).
		"""
		duplicate_pattern = self.get_duplicate_token_head_detection_pattern(tokens)
		shifted_tensor = torch.roll(duplicate_pattern, shifts=1, dims=1)
		shifted_tensor[:, 0] = 0
		return torch.tril(shifted_tensor)

	def get_detection_pattern(self, seq: str, pattern_type: str) -> torch.Tensor:
		"""
		Given an input sequence and a pattern type, return the corresponding detection pattern tensor.
		Supported types: "previous_token_head", "duplicate_token_head", "induction_head".

		:param seq: Input sequence as a string.
		:param pattern_type: Type of detection pattern.
		:return: Detection pattern tensor of shape (seq_len, seq_len).
		:raises ValueError: If an unsupported pattern type is provided.
		"""
		tokens_str = self.tokenizer.encode(seq, add_special_tokens=False)
		tokens_tensor = torch.tensor(tokens_str)
		if pattern_type == "previous_token_head":
			return self.get_previous_token_head_detection_pattern(tokens_tensor)
		elif pattern_type == "duplicate_token_head":
			return self.get_duplicate_token_head_detection_pattern(tokens_tensor)
		elif pattern_type == "induction_head":
			return self.get_induction_head_detection_pattern(tokens_tensor)
		else:
			raise ValueError(f"Unsupported pattern type: {pattern_type}")

	@staticmethod
	def compute_head_attention_similarity_score(
		attn_pattern: torch.Tensor,
		detection_pattern: torch.Tensor,
		exclude_bos: bool,
		exclude_current_token: bool,
		error_measure: str,
	) -> float:
		"""
		Compute the similarity between an attention pattern and a detection pattern.
		Two error measures are supported:
		 - "mul": Element-wise multiplication; returns the sum of the product divided by the attention sum.
		 - "abs": 1 minus (mean absolute difference multiplied by sequence length), mapping lower error to higher score.

		:param attn_pattern: Attention pattern tensor of shape (seq_len, seq_len).
		:param detection_pattern: Expected detection pattern tensor.
		:param exclude_bos: If True, set the first column of the attention pattern to 0.
		:param exclude_current_token: If True, zero out the diagonal.
		:param error_measure: Error measure to use ("mul" or "abs").
		:return: Similarity score as a float.
		:raises ValueError: If an unsupported error measure is provided.
		"""
		attn = attn_pattern.clone()
		if exclude_bos:
			attn[:, 0] = 0
		if exclude_current_token:
			attn.fill_diagonal_(0)
		if error_measure == "mul":
			score = (attn * detection_pattern).sum() / attn.sum()
			return score.item()
		elif error_measure == "abs":
			abs_diff = (attn - detection_pattern).abs()
			seq_len = attn.shape[0]
			mean_error = abs_diff.mean() * seq_len
			return 1 - round(mean_error.item(), 3)
		else:
			raise ValueError(f"Unsupported error measure: {error_measure}")
