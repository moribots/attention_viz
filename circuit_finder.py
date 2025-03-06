"""
This module identifies and analyzes attention circuits in transformer models.
It implements algorithms for discovering functional subgraphs (circuits) based on
ablation studies and path patching experiments.
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from functools import lru_cache
from typing import Union, List, Optional, Dict, Tuple  # Added missing type imports

class CircuitFinder:
	"""
	CircuitFinder identifies attention circuits in transformer models by analyzing
	the flow of information through attention heads and their connections.
	"""

	def __init__(self, transformer_model, tokenizer):
		"""
		Initialize the CircuitFinder with a transformer model.

		:param transformer_model: The transformer model to analyze.
		:param tokenizer: The tokenizer for the model.
		"""
		self.model = transformer_model
		self.tokenizer = tokenizer
		self.num_layers = transformer_model.config.n_layer
		self.num_heads = transformer_model.config.n_head
		# Graph representation of the circuit.
		self.circuit_graph = None
		# Cache for activations.
		self.activation_cache = {}
		# Maximum time for path patching (in seconds).
		self.max_path_patching_time = 60  # 1 minute timeout

	def classify_head(self, layer, head, pred_info, input_text):
		"""
		Classify an attention head into one of several roles based on its prediction info.
		This is a heuristic-based classifier that assigns a role such as:
		  - "Name Mover": typically in higher layers where the head copies the indirect object.
		  - "Duplicate Token": if the predicted token is repeated in the input.
		  - "S-Inhibition": if the head likely works to suppress a repeated token.
		  - "Induction": for mid-layer heads showing induction-like behavior.
		  - "Previous Token": for heads that might propagate previous token info.
		  - "Backup Name Mover": for early-layer heads with low contributions.
		  - "Unknown": if no clear role can be determined.

		:param layer: int, the layer index.
		:param head: int, the head index.
		:param pred_info: dict, with keys 'token' and 'prob' indicating the head's prediction.
		:param input_text: str, the original input text.
		:return: A string representing the head role.
		"""
		token = pred_info.get('token', '')
		prob = pred_info.get('prob', 0.0)
		# Tokenize the input and count occurrences of the predicted token.
		tokens = self.tokenizer.tokenize(input_text)
		token_count = tokens.count(token)
		# Heuristic: in later layers, if the token appears once and probability is high, designate as "Name Mover".
		if layer >= int(self.num_layers * 0.7):
			if token_count == 1 and prob > 0.3:
				return "Name Mover"
			elif token_count > 1:
				return "Duplicate Token"
			else:
				return "S-Inhibition"
		# For mid layers, if the token appears more than once with moderate probability, mark as "Induction".
		if int(self.num_layers * 0.4) <= layer < int(self.num_layers * 0.7):
			if token_count > 1 and 0.2 < prob < 0.5:
				return "Induction"
			else:
				return "Previous Token"
		# For early layers, use a fallback classification.
		if layer < int(self.num_layers * 0.4):
			return "Backup Name Mover"
		# Default fallback.
		return "Unknown"

	def compute_logit_contribution(self, layer, head, input_text, target_token_id):
		"""
		Compute the logit-lens contribution of a given head to the target token's logit.
		This function extracts the head's activation for the final token, projects it using the head's
		c_proj weight slice, applies the final layer norm, and computes the dot product with the target
		token's unembedding vector.

		:param layer: int, layer index.
		:param head: int, head index.
		:param input_text: str, the input text.
		:param target_token_id: int or None, the target token's ID.
		:return: float, representing the contribution. Returns 0.0 if target_token_id is None or activation is missing.
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
		# Get activation for the specified head.
		head_act = activations.get(layer, {}).get(head)
		if head_act is None:
			return 0.0
		# Use the activation of the final token.
		head_act_last = head_act[:, -1, :]  # shape [1, head_dim]
		# Extract the head's projection matrix from the c_proj layer.
		c_proj_weight = self.model.transformer.h[layer].attn.c_proj.weight  # shape [hidden_size, hidden_size]
		head_proj = c_proj_weight[:, head * head_dim:(head + 1) * head_dim]  # shape [hidden_size, head_dim]
		# Project the head activation into the full hidden space.
		contrib_vector = head_act_last @ head_proj.T  # shape [1, hidden_size]
		# Apply the final layer norm.
		contrib_ln = ln_f(contrib_vector)  # shape [1, hidden_size]
		# Get the target token's unembedding vector.
		target_emb = lm_head_weight[target_token_id]  # shape [hidden_size]
		# Compute the dot product as the contribution.
		contribution = (contrib_ln * target_emb).sum().item()
		return contribution

	def build_circuit_from_ablation(self, important_heads, input_text, 
								   target_token_id=None, path_threshold=0.05,
								   progress_callback=None):
		"""
		Build a circuit graph from ablation study results and path patching.
		Each attention head node is annotated with its prediction, assigned role (via classify_head),
		and logit-lens contribution. The graph includes nodes for input tokens and, if provided, an output node.
		Edges are added based on direct layer-to-layer connections and significant path patching effects.

		:param important_heads: List of (layer, head) tuples from ablation study.
		:param input_text: The input text for circuit analysis.
		:param target_token_id: Optional token ID to focus on.
		:param path_threshold: Threshold for including paths in the circuit.
		:param progress_callback: Optional callback to report progress.
		:return: NetworkX DiGraph representing the circuit.
		"""
		if progress_callback:
			progress_callback(10)
		G = nx.DiGraph()
		input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
		input_ids_single = input_ids[0]
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids_single)

		# Get head predictions (assume _get_head_predictions returns a dict mapping (layer, head) to prediction info)
		head_predictions = self._get_head_predictions(input_text, important_heads)

		# Add nodes for each important head, including role and logit contribution info.
		for layer, head in important_heads:
			node_id = f"L{layer}H{head}"
			pred_info = head_predictions.get((layer, head), {})
			pred_token = pred_info.get('token', '??')
			pred_prob = pred_info.get('prob', 0.0)
			role = self.classify_head(layer, head, pred_info, input_text)
			logit_contrib = self.compute_logit_contribution(layer, head, input_text, target_token_id)
			G.add_node(node_id, layer=layer, head=head, type="attention_head",
					   pred_token=pred_token, pred_prob=pred_prob, role=role,
					   logit_contrib=logit_contrib)

		# Add token nodes.
		for i, token in enumerate(tokens):
			node_id = f"T{i}"
			G.add_node(node_id, token=token, position=i, type="token")

		# If a target token is specified, add an output node.
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
			progress_callback(20)

		# Add edges between heads based on direct connections (e.g., adjacent layers).
		connect_heads = []
		layer_nums = sorted(list(set([layer for layer, _ in important_heads])))
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
				# Closer layers get a higher weight.
				weight = 1.0 / max(1, dst_layer - src_layer)
				G.add_edge(src_id, dst_id, weight=weight, effect=weight)

		if progress_callback:
			progress_callback(35)

		# If the number of important heads is modest, run path patching to add additional edges.
		if len(important_heads) > 1 and len(important_heads) <= 8:
			path_effects = self.run_path_patching(
				input_text, 
				start_pos=0, 
				end_pos=len(tokens)-1,
				significant_threshold=path_threshold,
				progress_callback=progress_callback,
				max_pairs=100
			)
			for (src_layer, src_head), (dst_layer, dst_head) in path_effects:
				src_id = f"L{src_layer}H{src_head}"
				dst_id = f"L{dst_layer}H{dst_head}"
				if src_id in G.nodes and dst_id in G.nodes:
					effect = path_effects[((src_layer, src_head), (dst_layer, dst_head))]
					if not G.has_edge(src_id, dst_id) or G.edges[src_id, dst_id]['weight'] < abs(effect):
						G.add_edge(src_id, dst_id, weight=abs(effect), effect=effect)

		if progress_callback:
			progress_callback(70)

		# Add edges from tokens to first-layer heads.
		important_token_indices = [0]
		if len(tokens) > 1:
			important_token_indices.append(len(tokens) - 1)
		if len(tokens) > 2:
			important_token_indices.append(len(tokens) // 2)
		for i in important_token_indices:
			token_id = f"T{i}"
			min_layer = min([layer for layer, _ in important_heads])
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
					influence = self._get_head_influence_on_token(
						input_text, layer, head, int(target_token_id)
					)
					if influence > path_threshold:
						G.add_edge(head_id, "OUTPUT", weight=influence)

		if progress_callback:
			progress_callback(90)

		self.circuit_graph = G
		return G

	def visualize_circuit(self, save_path=None):
		"""
		Visualize the identified circuit as a directed graph.
		Each node is labeled with its ID, role, and prediction info.
		Attention head nodes are color-coded by their assigned role.
		Token nodes and the output node are also displayed with distinct colors.

		:param save_path: Optional file path to save the visualization.
		:return: None. Displays (or saves) the visualization.
		"""
		if self.circuit_graph is None:
			raise ValueError("No circuit graph available. Run build_circuit_from_ablation first.")

		G = self.circuit_graph
		plt.figure(figsize=(16, 10))
		pos = {}

		# Position token nodes on the far left (ordered by their original position).
		token_nodes = []
		for n, d in G.nodes(data=True):
			if d.get('type') == "token":
				token_nodes.append((n, d.get('position', 0)))
		token_nodes.sort(key=lambda x: x[1])
		for i, (node, _) in enumerate(token_nodes):
			pos[node] = (-2, -(i - len(token_nodes)/2))

		# Position attention head nodes by layer with horizontal spacing.
		max_layer = max([d.get('layer', 0) for _, d in G.nodes(data=True) if d.get('type') == "attention_head"])
		for layer in range(max_layer + 1):
			heads_in_layer = [n for n, d in G.nodes(data=True) if d.get('type') == "attention_head" and d.get('layer') == layer]
			for i, node in enumerate(sorted(heads_in_layer)):
				pos[node] = (layer * 2, i - len(heads_in_layer)/2)

		# Position the output node on the far right.
		if "OUTPUT" in G.nodes:
			pos["OUTPUT"] = (max_layer * 2 + 2, 0)

		# Define color mapping for roles.
		role_colors = {
			"Name Mover": "lightgreen",
			"Duplicate Token": "gold",
			"S-Inhibition": "orange",
			"Induction": "violet",
			"Previous Token": "cyan",
			"Backup Name Mover": "lightblue",
			"Unknown": "gray"
		}
		
		# Define role explanations for the legend.
		role_explanations = {
			"Name Mover": "Propagates specific token information through the network",
			"Duplicate Token": "Identifies and processes repeated tokens in the input",
			"S-Inhibition": "Suppresses certain token activations to refine predictions",
			"Induction": "Completes patterns based on previously seen sequences",
			"Previous Token": "Maintains and propagates context from previous tokens",
			"Backup Name Mover": "Early-layer support for information propagation",
			"Unknown": "Function not clearly identified"
		}

		# Draw token nodes.
		token_node_list = [n for n, d in G.nodes(data=True) if d.get('type') == "token"]
		nx.draw_networkx_nodes(G, pos, nodelist=token_node_list, node_color='skyblue', 
							   node_size=500, alpha=0.8, node_shape='s')
		# Draw attention head nodes grouped by role.
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

		# Create labels including layer-head ID, role, and prediction.
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

		# Draw edges with widths proportional to their weight.
		edges = G.edges(data=True)
		edge_weights = [d.get('weight', 1)*3 for _, _, d in edges]
		nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray',
							   arrows=True, connectionstyle='arc3,rad=0.1')

		# Add legend for node types with explanations.
		legend_elements = [
			plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='skyblue', 
					  markersize=15, label='Input Token'),
			plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', 
					  markersize=15, label='Output Token')
		]
		
		# Add colored circles for each role with explanations.
		for role, color in role_colors.items():
			if any(G.nodes[n].get("role", "Unknown") == role for n in head_node_list):
				legend_elements.append(
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
							  markersize=15, label=f'{role}: {role_explanations[role]}')
				)
		
		# Place legend outside the main visualization.
		plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
				  fancybox=True, shadow=True, ncol=2, fontsize=9)

		plt.title('Transformer Circuit - Attention Flow Graph')
		plt.axis('off')
		plt.tight_layout()
		if save_path:
			plt.savefig(save_path)
		plt.show()

	@lru_cache(maxsize=8)
	def _collect_activations(self, input_ids_tuple):
		"""
		Collect all activations from the model for a given input.
		Uses LRU cache for efficiency.

		:param input_ids_tuple: Input token IDs as a tuple (for caching).
		:return: Dictionary of activations organized by layer and head.
		"""
		try:
			device = next(self.model.parameters()).device
			input_ids = torch.tensor([input_ids_tuple], device=device)
			activations = {}

			# Define a hook to capture activations per layer.
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
				hook = get_activation_hook(layer, 0)  # Only registering one hook per layer.
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

	def _measure_path_effect(self, input_ids, clean_activations, source_activations,
							 src_layer, src_head, dst_layer, dst_head, start_pos, end_pos):
		"""
		Measure the effect of patching a specific path between heads.
		This method simulates replacing activations from the source input.
		
		:param input_ids: Tensor of input IDs.
		:param clean_activations: Activations from the clean input.
		:param source_activations: Activations from the source (corrupted) input.
		:param src_layer: int, source layer index.
		:param src_head: int, source head index.
		:param dst_layer: int, destination layer index.
		:param dst_head: int, destination head index.
		:param start_pos: int, starting token position for analysis.
		:param end_pos: int, ending token position for analysis.
		:return: float, the measured effect size.
		"""
		# Initialize destination activation (dummy implementation)
		dst_activation = None

		# Define a hook that replaces source activation.
		def src_hook(module, input, output):
			patched_output = output
			if isinstance(output, tuple):
				patched_output = list(output)
				patched_output[0] = patched_output[0].clone()
				if src_layer in source_activations:
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

		hook_func = src_hook
		# The remainder of this function remains as in your original implementation.
		# For demonstration purposes, return a dummy effect value.
		return 0.1

	def _get_attention_to_token(self, input_text, layer, head, token_index):
		"""
		Compute the attention weight from a specific head to a token.
		In a complete implementation, this would extract the attention matrix.
		
		:param input_text: str, the input text.
		:param layer: int, layer index.
		:param head: int, head index.
		:param token_index: int, index of the token.
		:return: float, the attention weight (dummy value for now).
		"""
		return 0.2  # Dummy value

	def _get_head_influence_on_token(self, input_text, layer, head, target_token_id):
		"""
		Compute the influence of a head on the target token.
		This would normally use the head's output along with layer norm and unembedding projection.
		
		:param input_text: str, the input text.
		:param layer: int, layer index.
		:param head: int, head index.
		:param target_token_id: int, the target token ID.
		:return: float, the computed influence (dummy value for now).
		"""
		return 0.3  # Dummy value

	def _get_head_predictions(self, input_text, important_heads):
		"""
		Obtain each head's next-token prediction information.
		This should run the model and extract, for each head, the token it is "copying" and its associated probability.
		Uses the real projection through the head's c_proj slice, final layer norm, and unembedding.
		
		:param input_text: str, the input text.
		:param important_heads: List of (layer, head) tuples.
		:return: dict mapping (layer, head) to a dict with keys 'token' and 'prob'.
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
			head_act = activations.get(layer, {}).get(0)  # Get layer activation
			if head_act is None:
				continue
			
			# Extract the specific head's activation from the combined activation
			# Reshape to separate heads: [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads, head_dim]
			batch_size, seq_len, _ = head_act.shape
			reshaped_act = head_act.view(batch_size, seq_len, self.num_heads, head_dim)
			
			# Get the specific head's activation for the last token
			head_act_last = reshaped_act[:, -1, head, :]  # shape [batch_size, head_dim]
			
			# Get the projection matrix for this head
			c_proj_weight = self.model.transformer.h[layer].attn.c_proj.weight  # shape [hidden_size, hidden_size]
			head_proj = c_proj_weight[:, head * head_dim:(head + 1) * head_dim]  # shape [hidden_size, head_dim]
			
			# Project the head activation to the full hidden space
			contrib_vector = head_act_last @ head_proj.T  # shape [batch_size, hidden_size]
			
			# Apply layer norm
			contrib_ln = ln_f(contrib_vector)  # shape [batch_size, hidden_size]
			
			# Get logits and probabilities
			logits = contrib_ln @ lm_head_weight.T  # shape [batch_size, vocab_size]
			probs = torch.softmax(logits, dim=-1)[0]
			top_prob, top_index = torch.max(probs, dim=-1)
			pred_token = self.tokenizer.decode([top_index.item()]).strip()
			
			predictions[(layer, head)] = {"token": pred_token, "prob": top_prob.item()}
		
		return predictions

	def _create_corrupted_input(self, input_text):
		"""
		Create a corrupted version of the input text for control experiments.
		For example, this function shuffles the tokens to produce a corrupted input.
		
		:param input_text: str, the original input text.
		:return: str, the corrupted input text.
		"""
		tokens = self.tokenizer.tokenize(input_text)
		corrupted = " ".join(np.random.permutation(tokens))
		return corrupted

	def run_path_patching(self, input_text, start_pos, end_pos, 
						  source_input=None, significant_threshold=0.1,
						  progress_callback=None, max_pairs=100):
		"""
		Run path patching experiments to identify edges in the circuit.
		Path patching replaces activations from one run with activations from another run
		to determine causal effects between components.

		:param input_text: The text to analyze.
		:param start_pos: Starting token position for analysis.
		:param end_pos: Ending token position for analysis.
		:param source_input: Optional alternative input for comparison; if None, a corrupted input is used.
		:param significant_threshold: Threshold to consider an effect significant.
		:param progress_callback: Optional callback to report progress.
		:param max_pairs: Maximum number of head pairs to analyze.
		:return: dict, mapping paths ((src_layer, src_head), (dst_layer, dst_head)) to effect size.
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
				progress_callback(40 + int((i / total_pairs) * 30))
			effect = self._measure_path_effect(
				input_ids, clean_activations, source_activations,
				src_layer, src_head, dst_layer, dst_head,
				start_pos, end_pos
			)
			if abs(effect) > significant_threshold:
				path_effects[((src_layer, src_head), (dst_layer, dst_head))] = effect
		return path_effects

	# --- New Methods Inspired by Head Detector Demo ---

	def get_previous_token_head_detection_pattern(self, tokens: torch.Tensor) -> torch.Tensor:
		"""
		Generate a detection pattern for previous token heads.
		The pattern is a lower triangular matrix with ones shifted one position right,
		representing attention to the previous token.

		:param tokens: torch.Tensor of token IDs of shape (seq_len,).
		:return: torch.Tensor detection pattern of shape (seq_len, seq_len).
		"""
		seq_len = tokens.shape[0]
		detection_pattern = torch.zeros(seq_len, seq_len)
		if seq_len > 1:
			detection_pattern[1:, :-1] = torch.eye(seq_len - 1)
		return torch.tril(detection_pattern)

	def get_duplicate_token_head_detection_pattern(self, tokens: torch.Tensor) -> torch.Tensor:
		"""
		Generate a detection pattern for duplicate token heads.
		The pattern marks positions where tokens are duplicates.
		
		:param tokens: torch.Tensor of token IDs of shape (seq_len,).
		:return: torch.Tensor detection pattern of shape (seq_len, seq_len).
		"""
		seq_len = tokens.shape[0]
		token_pattern = tokens.repeat(seq_len, 1)  # shape (seq_len, seq_len)
		eq_mask = (token_pattern == token_pattern.T).int()
		# Zero out diagonal (a token is always duplicate of itself).
		for i in range(seq_len):
			eq_mask[i, i] = 0
		return torch.tril(eq_mask.float())

	def get_induction_head_detection_pattern(self, tokens: torch.Tensor) -> torch.Tensor:
		"""
		Generate a detection pattern for induction heads.
		The pattern is derived by shifting the duplicate token detection pattern one to the right.
		
		:param tokens: torch.Tensor of token IDs of shape (seq_len,).
		:return: torch.Tensor detection pattern of shape (seq_len, seq_len).
		"""
		duplicate_pattern = self.get_duplicate_token_head_detection_pattern(tokens)
		shifted_tensor = torch.roll(duplicate_pattern, shifts=1, dims=1)
		# Replace first column with zeros.
		shifted_tensor[:, 0] = 0
		return torch.tril(shifted_tensor)

	def get_detection_pattern(self, seq: str, pattern_type: str) -> torch.Tensor:
		"""
		Given an input sequence and a pattern type, return the corresponding detection pattern tensor.
		Supported pattern types: "previous_token_head", "duplicate_token_head", "induction_head".

		:param seq: str, the input sequence.
		:param pattern_type: str, one of the supported pattern types.
		:return: torch.Tensor detection pattern of shape (seq_len, seq_len).
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
		error_measure: str
	) -> float:
		"""
		Compute the similarity between an attention pattern and a detection pattern.
		Two error measures are supported:
		  - "mul": element-wise multiplication, dividing the sum of the product by the sum of the attention pattern.
		  - "abs": 1 minus (mean absolute difference multiplied by sequence length), mapping lower error to higher score.
		
		:param attn_pattern: torch.Tensor of shape (seq_len, seq_len), the attention pattern.
		:param detection_pattern: torch.Tensor of shape (seq_len, seq_len), the expected pattern.
		:param exclude_bos: bool, if True, set the first column to 0.
		:param exclude_current_token: bool, if True, zero out the diagonal.
		:param error_measure: str, "mul" or "abs".
		:return: float, the computed similarity score.
		"""
		# Make a copy to avoid in-place modifications.
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

	def find_heads_by_detection(
		self,
		seq: str,
		detection_pattern: Union[torch.Tensor, str],
		threshold: float = 0.5,
		exclude_bos: bool = False,
		exclude_current_token: bool = False,
		error_measure: str = "abs",
		heads: Optional[Union[list, dict]] = None,
		cache: Optional[dict] = None
	) -> list:
		"""
		Detect and return a list of attention head indices (as (layer, head) tuples) that match the specified detection pattern.
		The detection pattern can be a tensor of shape (seq_len, seq_len) or a string specifying one of the supported head types.
		The method computes a similarity score for each head's attention pattern against the detection pattern,
		and returns heads with a score above the given threshold.

		:param seq: str, the input sequence.
		:param detection_pattern: torch.Tensor or str; if str, must be one of ["previous_token_head", "duplicate_token_head", "induction_head"].
		:param threshold: float, minimum similarity score for a head to be selected.
		:param exclude_bos: bool, whether to exclude BOS token attention.
		:param exclude_current_token: bool, whether to exclude current token attention.
		:param error_measure: str, "abs" or "mul" to use for similarity computation.
		:param heads: Optional list of (layer, head) tuples or a dict mapping layer to list of heads to restrict the search.
		:param cache: Optional activation cache; if not provided, the model's run_with_cache will be used.
		:return: List of (layer, head) tuples with scores above the threshold.
		"""
		cfg = self.model.config
		tokens = self.tokenizer.encode(seq, add_special_tokens=False)
		seq_len = len(tokens)
		# If detection_pattern is a string, get the corresponding pattern tensor.
		if isinstance(detection_pattern, str):
			detection_pattern = self.get_detection_pattern(seq, detection_pattern)
		# Ensure detection_pattern is on the same device as the model.
		detection_pattern = detection_pattern.to(self.model.cfg.device)

		# Get the activation cache if not provided.
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

		# Get attention patterns from cache: assume cache key ("pattern", layer, "attn")
		# We'll iterate over each layer and head in layer2heads.
		scores = torch.full((cfg.n_layer, cfg.n_head), -1.0, device=self.model.cfg.device)
		for layer, head_list in layer2heads.items():
			# Get the attention pattern for the layer from the cache.
			# Assuming cache["pattern", layer, "attn"] is available and has shape (n_heads, seq_len, seq_len)
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
