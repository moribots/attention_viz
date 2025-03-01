import multiprocessing as mp
try:
	mp.set_start_method('fork', force=True)
except RuntimeError:
	pass

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from transformers import GPT2Tokenizer, GPT2Model
from transformers import GPT2LMHeadModel
import numpy as np
import dash_bootstrap_components as dbc
from tqdm import tqdm

import diskcache
from dash.long_callback import DiskcacheLongCallbackManager

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

external_stylesheets = [dbc.themes.BOOTSTRAP]

NUM_COMBOS = 4

# -------------------------------
# Load the model and tokenizer
# -------------------------------
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
model.eval()
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
lm_model.eval()
NUM_LAYERS = model.config.n_layer
NUM_HEADS = model.config.n_head

# -------------------------------
# Utility functions
# -------------------------------
def get_attention_data(input_text, layer, head, threshold=0.0):
	input_ids = tokenizer.encode(input_text, return_tensors='pt')
	tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
	with torch.no_grad():
		outputs = model(input_ids)
	attentions = outputs.attentions
	attn = attentions[layer][0, head, :, :]
	attn_data = attn.cpu().numpy()
	filtered_attn_data = np.where(attn_data >= threshold, attn_data, 0)
	return filtered_attn_data, tokens

def make_ablate_hook(selected_head, scale=0.0):
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			attn_output_clone[:, :, start:end] *= scale
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			output_clone[:, :, start:end] *= scale
			return output_clone
	return hook

# -------------------------------
# New Ablation Techniques
# -------------------------------
def make_permutation_hook(selected_head):
	"""
	Permutation Ablation: Shuffle the outputs of the selected head along the token dimension.
	"""
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			for i in range(attn_output_clone.size(0)):
				perm = torch.randperm(attn_output_clone.size(1))
				attn_output_clone[i, :, start:end] = attn_output_clone[i, perm, start:end]
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			for i in range(output_clone.size(0)):
				perm = torch.randperm(output_clone.size(1))
				output_clone[i, :, start:end] = output_clone[i, perm, start:end]
			return output_clone
	return hook

def make_sparsification_hook(selected_head, sparsity_threshold):
	"""
	Structured Sparsification: Zero out values in the selected head's output that are below the sparsity_threshold.
	"""
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			mask = torch.abs(attn_output_clone[:, :, start:end]) < sparsity_threshold
			attn_output_clone[:, :, start:end] = attn_output_clone[:, :, start:end].masked_fill(mask, 0)
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			mask = torch.abs(output_clone[:, :, start:end]) < sparsity_threshold
			output_clone[:, :, start:end] = output_clone[:, :, start:end].masked_fill(mask, 0)
			return output_clone
	return hook

# -------------------------------
# New Helper Function for Extra Metrics
# -------------------------------
def compute_extra_metrics(baseline_probs, ablated_probs, token_clicked, epsilon=1e-10):
	"""
	Compute additional metrics including entropy and rank change for the clicked token.
	"""
	kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
	delta_top_prob = baseline_probs.max().item() - ablated_probs.max().item()
	baseline_entropy = -torch.sum(baseline_probs * torch.log(baseline_probs + epsilon)).item()
	ablated_entropy = -torch.sum(ablated_probs * torch.log(ablated_probs + epsilon)).item()
	entropy_diff = ablated_entropy - baseline_entropy
	clicked_token_id = tokenizer.convert_tokens_to_ids(token_clicked)
	baseline_sorted = torch.argsort(baseline_probs, descending=True)
	ablated_sorted = torch.argsort(ablated_probs, descending=True)
	baseline_rank = (baseline_sorted == clicked_token_id).nonzero(as_tuple=True)[0].item() + 1
	ablated_rank = (ablated_sorted == clicked_token_id).nonzero(as_tuple=True)[0].item() + 1
	rank_change = ablated_rank - baseline_rank
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

def evaluate_candidate(truncated_ids, baseline_probs, ablation_set, scale=0.0, ablation_method='standard', sparsity_threshold=0.1, epsilon=1e-10):
	hook_handles = []
	for (layer, head) in ablation_set:
		if ablation_method == 'standard':
			hook = make_ablate_hook(head, scale=scale)
		elif ablation_method == 'permute':
			hook = make_permutation_hook(head)
		elif ablation_method == 'sparsify':
			hook = make_sparsification_hook(head, sparsity_threshold)
		else:
			hook = make_ablate_hook(head, scale=scale)
		hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(hook)
		hook_handles.append(hook_handle)
	with torch.no_grad():
		ablated_logits = lm_model(truncated_ids).logits[0, -1, :]
	for handle in hook_handles:
		handle.remove()
	ablated_probs = torch.softmax(ablated_logits, dim=-1)
	kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
	delta_top_prob = baseline_probs.max().item() - ablated_probs.max().item()
	alpha, beta = 1.0, 1.0
	combined_score = alpha * kl_div + beta * delta_top_prob
	return combined_score

def find_best_ablation_combo(truncated_ids, baseline_probs, max_heads=10, scale=0.0, ablation_method='standard', sparsity_threshold=0.1, progress_callback=None, search_strategy='greedy'):
	candidate_list = [(layer, head) for layer in range(lm_model.config.n_layer) for head in range(lm_model.config.n_head)]
	candidate_scores = []
	total_candidates = len(candidate_list)
	# Evaluate each candidate and update progress (scaled to 0-40%)
	for idx, candidate in enumerate(candidate_list):
		score = evaluate_candidate(truncated_ids, baseline_probs, [candidate], scale=scale, ablation_method=ablation_method, sparsity_threshold=sparsity_threshold)
		candidate_scores.append((candidate, score))
		if progress_callback is not None:
			progress_callback(int((idx + 1) / total_candidates * 40))
	candidate_scores.sort(key=lambda x: x[1], reverse=True)
	# Pre-select top 20%
	preselected = [cand for cand, _ in candidate_scores[:max(1, len(candidate_scores)//5)]
	
	]
	# -------------------------------
	# New iterative search over pairs (heuristic) to refine search strategy
	# -------------------------------
	best_set = []
	best_score = evaluate_candidate(truncated_ids, baseline_probs, best_set, scale=scale, ablation_method=ablation_method, sparsity_threshold=sparsity_threshold)
	if search_strategy == 'iterative' and max_heads >= 2 and len(preselected) >= 2:
		best_pair = None
		best_pair_score = best_score  # starting from empty set score
		for i in range(len(preselected)):
			for j in range(i+1, len(preselected)):
				pair = [preselected[i], preselected[j]]
				score = evaluate_candidate(truncated_ids, baseline_probs, pair, scale=scale, ablation_method=ablation_method, sparsity_threshold=sparsity_threshold)
				if score > best_pair_score:
					best_pair_score = score
					best_pair = pair
		if best_pair is not None:
			best_set = best_pair
			best_score = best_pair_score

	improved = True
	iteration_count = 0
	total_iterations = len(preselected) + 1  # heuristic for scaling progress in the while loop
	while improved and len(best_set) < max_heads:
		improved = False
		best_candidate = None
		candidate_score = best_score
		for candidate in preselected:
			if candidate in best_set:
				continue
			test_set = best_set + [candidate]
			score = evaluate_candidate(truncated_ids, baseline_probs, test_set, scale=scale, ablation_method=ablation_method, sparsity_threshold=sparsity_threshold)
			if score > candidate_score:
				candidate_score = score
				best_candidate = candidate
				improved = True
		if best_candidate is not None:
			best_set.append(best_candidate)
			best_score = candidate_score
		iteration_count += 1
		if progress_callback is not None:
			# Update progress from 40% to 90%
			progress_callback(int(iteration_count / (total_iterations) * 100))
	return best_set, best_score

# -------------------------------
# Build the Dash app
# -------------------------------
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
				long_callback_manager=long_callback_manager)

layer_and_head_options = [
	{'label': f"Layer {layer}, Head {head}", 'value': f"{layer}-{head}"}
	for layer in range(NUM_LAYERS)
	for head in range(NUM_HEADS)
]

sidebar = dbc.Card(
	[
		dbc.CardHeader("Instructions"),
		dbc.CardBody([
			html.P("- You can modify the input text on the right.", className="card-text"),
			html.P("- Select one or more layer-head pairs to view their attention heatmaps.", className="card-text"),
			html.P("- Use the slider to set a threshold and filter weak attention links.", className="card-text"),
			html.P("- Click a cell in any heatmap to view details about that token.", className="card-text"),
			html.P("- Ablation (causal tracing) lets you see how head removal changes predictions.", className="card-text")
		])
	],
	style={"width": "100%", "marginBottom": "20px"}
)

main_content = dbc.Card(
	dbc.CardBody([
		html.Div([
			html.Label("Input Text:"),
			dcc.Input(id="input-text", type="text",
					  value="The quick brown fox jumps over the lazy dog.",
					  style={'width': '100%'})
		], style={'marginBottom': '20px'}),

		html.Div([
			html.Label("Select Layer–Head Combos:"),
			dcc.Dropdown(id="combo-dropdown",
						 options=layer_and_head_options,
						 value=["0-0"],
						 multi=True, clearable=False)
		], style={'width': '90%', 'marginBottom': '20px'}),

		html.Div([
			html.Label("Attention Threshold (0.0 - 1.0):"),
			dcc.Slider(id="threshold-slider", min=0.0, max=1.0, step=0.01, value=0.0,
					   marks={i/10: f"{i/10}" for i in range(0, 11)})
		], style={'marginTop': '20px', 'marginBottom': '20px'}),

		html.Div([
			html.Label("Ablation Scale Factor (0.0 = full ablation, 1.0 = no ablation):"),
			dcc.Slider(id="ablation-scale-slider", min=0.0, max=1.0, step=0.01, value=0.0,
					   marks={i/10: f"{i/10}" for i in range(0, 11)})
		], style={'marginTop': '20px', 'marginBottom': '20px'}),

		html.Div([
			html.Label("Sparsity Threshold (for Structured Sparsification):"),
			dcc.Slider(id="sparsity-threshold-slider", min=0.0, max=1.0, step=0.01, value=0.1,
					   marks={i/10: f"{i/10}" for i in range(0, 11)})
		], style={'marginTop': '20px', 'marginBottom': '20px'}),

		html.Div([
			dcc.RadioItems(
				id="causal-intervention",
				options=[
					{'label': 'None', 'value': 'none'},
					{'label': 'Standard Ablation', 'value': 'standard'},
					{'label': 'Permutation Ablation', 'value': 'permute'},
					{'label': 'Structured Sparsification', 'value': 'sparsify'}
				],
				value='none'
			)
		], style={'marginTop': '20px', 'marginBottom': '20px'}),

		# Add Prev/Next buttons and a Store to track the current page of combos
		html.Div([
			html.Button("Previous Page", id="prev-page-btn", n_clicks=0,
						style={'marginRight': '10px'}),
			html.Button("Next Page", id="next-page-btn", n_clicks=0),
			dcc.Store(id="page-store", data=0)  # Store the current "page" of combos
		], style={'marginBottom': '20px'}),

		dcc.Loading(
			id="loading-graph",
			type="circle",
			children=dcc.Graph(
				id="attention-heatmap",
				style={
					"width": "100%",       # let it expand horizontally
					"minHeight": "700px",  # decent vertical space
					# "overflowX": "auto"    # scroll horizontally if wide
				}
			)
		),
		

		html.Div(id="token-info", style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'}),

		html.Button("Run Ablation Study", id="run-ablation-study", n_clicks=0),
		dbc.Progress(id="ablation-progress", striped=True, animated=True,
					 style={'marginTop': '20px', 'height': '20px'}),
		html.Div(id="ablation-result", style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'})
	]),
	style={"width": "100%"}
)

app.layout = dbc.Container([
	dbc.Row(dbc.Col(html.H1("Interactive Attention Visualization"), width=12),
			style={"marginTop": "20px", "marginBottom": "20px"}),
	dbc.Row([dbc.Col(sidebar, width=3), dbc.Col(main_content, width=9)])
], fluid=True)

# -------------------------------
# Callback to change page when Prev/Next are clicked
# -------------------------------
@app.callback(
	Output("page-store", "data"),
	[Input("prev-page-btn", "n_clicks"),
	 Input("next-page-btn", "n_clicks")],
	[State("combo-dropdown", "value"),
	 State("page-store", "data")]
)
def update_page(prev_clicks, next_clicks, combos, current_page):
	"""
	Adjust the page index based on which button is clicked.
	We show up to 10 combos per page.
	"""
	if not combos:
		return 0

	ctx = dash.callback_context
	if not ctx.triggered:
		return current_page
	button_id = ctx.triggered[0]["prop_id"].split(".")[0]

	import math
	page_size = NUM_COMBOS
	total_pages = max(1, math.ceil(len(combos) / page_size))

	new_page = current_page
	if button_id == "next-page-btn":
		new_page = current_page + 1
		if new_page >= total_pages:
			new_page = total_pages - 1
	elif button_id == "prev-page-btn":
		new_page = current_page - 1
		if new_page < 0:
			new_page = 0

	return new_page

# -------------------------------
# Callback to update the attention heatmap
# -------------------------------
@app.callback(
	Output("attention-heatmap", "figure"),
	[Input("input-text", "value"),
	 Input("combo-dropdown", "value"),
	 Input("threshold-slider", "value"),
	 Input("page-store", "data")]  # Use current page to pick which combos to show
)
def update_heatmap(input_text, selected_combos, threshold, current_page):
	import math

	if not isinstance(selected_combos, list):
		selected_combos = [selected_combos]
	combos = []
	for combo in selected_combos:
		try:
			layer_str, head_str = combo.split("-")
			combos.append((int(layer_str), int(head_str)))
		except Exception:
			continue

	page_size = NUM_COMBOS
	start_idx = current_page * page_size
	end_idx = start_idx + page_size
	combos_on_this_page = combos[start_idx:end_idx]

	n = len(combos_on_this_page)
	if n == 0:
		# If there are no combos, just display a simple message
		fig = go.Figure()
		fig.add_annotation(text="No combos selected.", showarrow=False)
		fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
		return fig

	# Decide rows/cols to handle up to 10 combos with minimal whitespace
	# (Feel free to adjust this logic if you prefer a fixed layout)
	if n <= 2:
		rows, cols = 1, n
	elif n <= 4:
		rows, cols = 2, 2

	# Create subplot titles
	subplot_titles = [f"Layer {layer}, Head {head}" for (layer, head) in combos_on_this_page]
	total_subplots = rows * cols
	# Pad titles if fewer than rows*cols
	while len(subplot_titles) < total_subplots:
		subplot_titles.append("")

	# Create subplots with minimal spacing
	fig = make_subplots(
		rows=rows,
		cols=cols,
		subplot_titles=subplot_titles,
		horizontal_spacing=0.2,  # reduce horizontal gap
		vertical_spacing=0.05     # reduce vertical gap
	)

	# Add each heatmap
	for i, (layer, head) in enumerate(combos_on_this_page):
		attn_data, tokens = get_attention_data(input_text, layer, head, threshold)
		heatmap = go.Heatmap(
			z=attn_data,
			x=tokens,
			y=tokens,
			colorscale='Viridis',
			colorbar=dict(title="Attention Weight"),
			hovertemplate="From Token: %{y}<br>To Token: %{x}<br>Attention: %{z:.4f}<extra></extra>"
		)
		row = (i // cols) + 1
		col = (i % cols) + 1
		fig.add_trace(heatmap, row=row, col=col)

	# Force each subplot to have a square aspect ratio
	for subplot_index in range(1, total_subplots + 1):
		xaxis_key = "xaxis" if subplot_index == 1 else f"xaxis{subplot_index}"
		yaxis_key = "yaxis" if subplot_index == 1 else f"yaxis{subplot_index}"
		if xaxis_key in fig.layout and yaxis_key in fig.layout:
			scaleanchor_val = "x" if subplot_index == 1 else f"x{subplot_index}"
			fig.layout[yaxis_key].scaleanchor = scaleanchor_val
			fig.layout[yaxis_key].scaleratio = 1

	# -----------------------------------------------------------------
	# 1) We'll let the figure fill 100% of the container's width.
	# 2) We compute the figure's height from the grid aspect ratio:
	#    - If each subplot is a square, total aspect ratio = (cols / rows).
	#    - We set width = 100% in dcc.Graph, but we need a "placeholder" numeric
	#      width in fig for the ratio calc. We'll guess ~ 1000px for the base.
	#    - Then height = (width * rows / cols).
	# 3) If the container is bigger or smaller, the figure will scale up/down,
	#    preserving squares. Some leftover space is possible if container
	#    ratio doesn't match the grid ratio, but each subplot stays square.
	# -----------------------------------------------------------------
	base_width = 1300
	fig_width = base_width
	# Keep squares => total figure ratio = (cols / rows)
	# So height = width * (rows / cols)
	if rows < 2 :
		fig_height = fig_width / (3.0 + 1.0 / 3.0)
	else:
		fig_height = fig_width * (rows / cols)

	fig.update_layout(
		autosize=True,
		width=fig_width,
		height=fig_height,
		margin=dict(l=0, r=0, t=50, b=0),
		paper_bgcolor="white"
	)
	return fig

# -------------------------------
# Callback to display token details on cell click
# -------------------------------
@app.callback(
	Output("token-info", "children"),
	[Input("attention-heatmap", "clickData"),
	 Input("input-text", "value"),
	 Input("causal-intervention", "value"),
	 Input("combo-dropdown", "value"),
	 Input("ablation-scale-slider", "value"),
	 Input("sparsity-threshold-slider", "value")]
)
def update_token_info(clickData, input_text, causal_intervention, combo_dropdown, ablation_scale, sparsity_threshold):
	if clickData is None:
		return "Click on a cell in the heatmap to see token information."
	try:
		token_clicked = clickData["points"][0]["x"]
	except (KeyError, IndexError):
		return "Error retrieving token info from click data."

	token_id = tokenizer.convert_tokens_to_ids(token_clicked)
	embedding = model.wte.weight[token_id]
	embedding_norm = torch.norm(embedding).item()
	info = f"Token: {token_clicked}\nToken ID: {token_id}\nEmbedding Norm: {embedding_norm:.4f}"

	full_input_ids = tokenizer.encode(input_text, return_tensors='pt')
	full_tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0])
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1
	truncated_ids = full_input_ids[:, :token_index+1]

	with torch.no_grad():
		baseline_outputs = lm_model(truncated_ids)
	baseline_logits = baseline_outputs.logits[0, -1, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)

	baseline_topk = 5
	baseline_top_probs, baseline_top_indices = torch.topk(baseline_probs, baseline_topk)
	baseline_top_tokens = tokenizer.convert_ids_to_tokens(baseline_top_indices.tolist())

	baseline_info = "\n\nBaseline Next Token Predictions:\n"
	for token, prob in zip(baseline_top_tokens, baseline_top_probs.tolist()):
		baseline_info += f"{token}: {prob:.4f}\n"

	# Updated: use radio selection value (not checklist) for ablation method.
	if causal_intervention != 'none':
		combo_list = combo_dropdown if isinstance(combo_dropdown, list) else [combo_dropdown]
		hook_handles = []
		for combo in combo_list:
			try:
				layer_str, head_str = combo.split("-")
				layer = int(layer_str)
				head = int(head_str)
			except Exception:
				continue
			if causal_intervention == 'standard':
				hook = make_ablate_hook(head, scale=ablation_scale)
			elif causal_intervention == 'permute':
				hook = make_permutation_hook(head)
			elif causal_intervention == 'sparsify':
				hook = make_sparsification_hook(head, sparsity_threshold)
			else:
				hook = make_ablate_hook(head, scale=ablation_scale)
			hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(hook)
			hook_handles.append(hook_handle)

		with torch.no_grad():
			ablated_outputs = lm_model(truncated_ids)

		for handle in hook_handles:
			handle.remove()

		ablated_logits = ablated_outputs.logits[0, -1, :]
		ablated_probs = torch.softmax(ablated_logits, dim=-1)
		ablated_topk = 5
		ablated_top_probs, ablated_top_indices = torch.topk(ablated_probs, ablated_topk)
		ablated_top_tokens = tokenizer.convert_ids_to_tokens(ablated_top_indices.tolist())

		ablated_info = "\n\nAblated Next Token Predictions:\n"
		for token, prob in zip(ablated_top_tokens, ablated_top_probs.tolist()):
			ablated_info += f"{token}: {prob:.4f}\n"

		# Compute additional metrics
		metrics = compute_extra_metrics(baseline_probs, ablated_probs, token_clicked)
		extra_metrics = "\n\nDeeper Analysis Metrics:\n"
		extra_metrics += f"KL Divergence: {metrics['KL Divergence']:.4f}\n"
		extra_metrics += f"Delta Top Token Probability: {metrics['Delta Top Token Probability']:.4f}\n"
		extra_metrics += f"Baseline Entropy: {metrics['Baseline Entropy']:.4f}\n"
		extra_metrics += f"Ablated Entropy: {metrics['Ablated Entropy']:.4f}\n"
		extra_metrics += f"Entropy Increase: {metrics['Entropy Increase']:.4f}\n"
		extra_metrics += f"Baseline Rank: {metrics['Baseline Rank']}\n"
		extra_metrics += f"Ablated Rank: {metrics['Ablated Rank']}\n"
		extra_metrics += f"Rank Change: {metrics['Rank Change']}\n"

		info += baseline_info + ablated_info + extra_metrics
	else:
		info += baseline_info

	return html.Pre(info)

# -------------------------------
# Long callback for ablation study
# -------------------------------
@app.long_callback(
	output=[Output("ablation-result", "children"),
			Output("combo-dropdown", "value")],
	inputs=[Input("run-ablation-study", "n_clicks")],
	state=[State("input-text", "value"),
		   State("attention-heatmap", "clickData"),
		   State("combo-dropdown", "value"),
		   State("ablation-scale-slider", "value"),
		   State("causal-intervention", "value"),
		   State("sparsity-threshold-slider", "value")],
	progress=[Output("ablation-progress", "value")],
	running=[(Output("run-ablation-study", "disabled"), True, False)],
	manager=long_callback_manager,
	prevent_initial_call=True
)
def run_ablation_study(progress, n_clicks, input_text, clickData, current_combos, ablation_scale, causal_intervention, sparsity_threshold):
	if clickData is None:
		return "Click on a token in the heatmap before running the ablation study.", current_combos

	try:
		token_clicked = clickData["points"][0]["x"]
	except (KeyError, IndexError):
		return "Error retrieving token info from click data.", current_combos

	full_input_ids = tokenizer.encode(input_text, return_tensors="pt")
	full_tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0])
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1
	truncated_ids = full_input_ids[:, :token_index+1]

	with torch.no_grad():
		baseline_logits = lm_model(truncated_ids).logits[0, -1, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)

	# Use iterative search strategy to refine the best ablation combo.
	ablation_method = causal_intervention if not isinstance(causal_intervention, list) else (causal_intervention[0] if causal_intervention else 'none')
	best_set, best_score = find_best_ablation_combo(truncated_ids, baseline_probs, max_heads=10, scale=ablation_scale, ablation_method=ablation_method, sparsity_threshold=sparsity_threshold, progress_callback=progress, search_strategy='iterative')
	progress(100)

	best_set_str = [f"{layer}-{head}" for (layer, head) in best_set]
	table_rows = []
	for (layer, head) in best_set:
		table_rows.append(html.Tr([
			html.Td(layer, style={'border': '1px solid black', 'padding': '4px'}),
			html.Td(head, style={'border': '1px solid black', 'padding': '4px'})
		]))

	table = html.Table(
		[
			html.Thead(html.Tr([
				html.Th("Layer", style={'border': '1px solid black', 'padding': '4px'}),
				html.Th("Head", style={'border': '1px solid black', 'padding': '4px'})
			])),
			html.Tbody(table_rows)
		],
		style={'width': '100%', 'borderCollapse': 'collapse'}
	)

	result_text = f"Best ablation combo (ablating {len(best_set)} heads):"
	result_text += f"\nCombined Score: {best_score:.4f}"

	# Compute extra metrics for the best combo
	hook_handles = []
	for (layer, head) in best_set:
		if ablation_method == 'standard':
			hook = make_ablate_hook(head, scale=ablation_scale)
		elif ablation_method == 'permute':
			hook = make_permutation_hook(head)
		elif ablation_method == 'sparsify':
			hook = make_sparsification_hook(head, sparsity_threshold)
		else:
			hook = make_ablate_hook(head, scale=ablation_scale)
		hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(hook)
		hook_handles.append(hook_handle)
	with torch.no_grad():
		best_ablated_logits = lm_model(truncated_ids).logits[0, -1, :]
	for handle in hook_handles:
		handle.remove()
	best_ablated_probs = torch.softmax(best_ablated_logits, dim=-1)
	metrics = compute_extra_metrics(baseline_probs, best_ablated_probs, token_clicked)
	extra_metrics_text = "\n\nDeeper Analysis Metrics (Best Combo):\n"
	extra_metrics_text += f"KL Divergence: {metrics['KL Divergence']:.4f}\n"
	extra_metrics_text += f"Delta Top Token Probability: {metrics['Delta Top Token Probability']:.4f}\n"
	extra_metrics_text += f"Baseline Entropy: {metrics['Baseline Entropy']:.4f}\n"
	extra_metrics_text += f"Ablated Entropy: {metrics['Ablated Entropy']:.4f}\n"
	extra_metrics_text += f"Entropy Increase: {metrics['Entropy Increase']:.4f}\n"
	extra_metrics_text += f"Baseline Rank: {metrics['Baseline Rank']}\n"
	extra_metrics_text += f"Ablated Rank: {metrics['Ablated Rank']}\n"
	extra_metrics_text += f"Rank Change: {metrics['Rank Change']}\n"

	final_result = html.Div([html.Pre(result_text + extra_metrics_text), table])
	final_result = final_result.to_plotly_json()
	return final_result, best_set_str

if __name__ == '__main__':
	app.run_server(debug=True)
