import multiprocessing as mp
# Force the start method to 'fork' to avoid pickling issues (Unix/Linux only).
try:
	mp.set_start_method('fork', force=True)
except RuntimeError:
	pass

import dash                            # Dash for building web apps
from dash import dcc, html             # Core and HTML components
from dash.dependencies import Input, Output, State  # Interactive callbacks
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import torch
from transformers import GPT2Tokenizer, GPT2Model  # Hugging Face Transformers for pre-trained models
from transformers import GPT2LMHeadModel  # Added for next-token predictions
import numpy as np
import dash_bootstrap_components as dbc  # For improved UI styling with Bootstrap.
from tqdm import tqdm

import diskcache  # Diskcache for caching intermediate results in long callbacks
from dash.long_callback import DiskcacheLongCallbackManager  # Manager for long callbacks

# Create a diskcache instance and long callback manager.
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Use Bootstrap stylesheet for better styling.
external_stylesheets = [dbc.themes.BOOTSTRAP]

# -------------------------------
# STEP 1: LOAD THE MODEL AND TOKENIZER
# -------------------------------
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
model.eval()
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
lm_model.eval()
NUM_LAYERS = model.config.n_layer
NUM_HEADS = model.config.n_head

# -------------------------------
# STEP 2: DEFINE A FUNCTION TO EXTRACT AND FILTER ATTENTION DATA
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

# -------------------------------
# STEP 3: CREATE THE DASH APP LAYOUT
# -------------------------------
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, 
				long_callback_manager=long_callback_manager)
layer_and_head_options=[
	{'label': f"Layer {layer}, Head {head}", 'value': f"{layer}-{head}"}
	for layer in range(NUM_LAYERS)
	for head in range(NUM_HEADS)
]
sidebar = dbc.Card(
	[dbc.CardHeader("Instructions"),
	 dbc.CardBody([
		 html.P("- The input text on the right is modifiable.", className="card-text"),
		 html.P("- Select one or more layers and heads to view their attention heatmaps.", className="card-text"),
		 html.P("- Adjust the attention threshold slider to filter weak connections.", className="card-text"),
		 html.P("- Click on a cell in any heatmap to view token details.", className="card-text"),
		 html.P("- Reading intuition: pick a token and examine how much it attends to others.", className="card-text"),
		 html.P("- Enable causal tracing (ablate selected heads) to see changes in next-token predictions.", className="card-text")
	 ])],
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
			dcc.Checklist(id="causal-intervention",
						  options=[{'label': '\tEnable Causal Tracing (Ablate Selected Heads)', 'value': 'ablate'}],
						  value=[])
		], style={'marginTop': '20px', 'marginBottom': '20px'}),
		dcc.Loading(id="loading-graph", type="circle",
					children=dcc.Graph(id="attention-heatmap")),
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
# STEP 4: CALLBACK TO UPDATE THE HEATMAP
# -------------------------------
@app.callback(
	Output("attention-heatmap", "figure"),
	[Input("input-text", "value"),
	 Input("combo-dropdown", "value"),
	 Input("threshold-slider", "value")]
)
def update_heatmap(input_text, selected_combos, threshold):
	if not isinstance(selected_combos, list):
		selected_combos = [selected_combos]
	combos = []
	for combo in selected_combos:
		try:
			layer_str, head_str = combo.split("-")
			combos.append((int(layer_str), int(head_str)))
		except Exception:
			continue
	rows = len(combos)
	cols = 1
	subplot_titles = [f"Layer {layer}, Head {head}" for layer, head in combos]
	fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
	for i, (layer, head) in enumerate(combos):
		attn_data, tokens = get_attention_data(input_text, layer, head, threshold)
		heatmap = go.Heatmap(z=attn_data, x=tokens, y=tokens,
							 colorscale='Viridis',
							 colorbar=dict(title="Attention Weight"))
		fig.add_trace(heatmap, row=i+1, col=1)
	return fig

def make_ablate_hook(selected_head):
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			attn_output_clone[:, :, start:end] = 0
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			output_clone[:, :, start:end] = 0
			return output_clone
	return hook

# -------------------------------
# STEP 5: CALLBACK TO UPDATE TOKEN INFO ON CLICK
# -------------------------------
@app.callback(
	Output("token-info", "children"),
	[Input("attention-heatmap", "clickData"),
	 Input("input-text", "value"),
	 Input("causal-intervention", "value"),
	 Input("combo-dropdown", "value")]
)
def update_token_info(clickData, input_text, causal_intervention, combo_dropdown):
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
	if 'ablate' in causal_intervention:
		combo_list = combo_dropdown if isinstance(combo_dropdown, list) else [combo_dropdown]
		hook_handles = []
		for combo in combo_list:
			try:
				layer_str, head_str = combo.split("-")
				layer = int(layer_str)
				head = int(head_str)
			except Exception:
				continue
			hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(
				make_ablate_hook(head)
			)
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
		epsilon = 1e-10
		kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
		baseline_top_token = baseline_top_tokens[0]
		ablated_top_token = ablated_top_tokens[0]
		top_token_change = (f"Top token changed from '{baseline_top_token}' to '{ablated_top_token}'"
							if baseline_top_token != ablated_top_token 
							else "Top token remains unchanged")
		deeper_metrics = "\n\nDeeper Analysis Metrics:\n"
		deeper_metrics += f"KL Divergence: {kl_div:.4f}\n"
		deeper_metrics += f"{top_token_change}\n"
		info += baseline_info + ablated_info + deeper_metrics
	else:
		info += baseline_info
	return html.Pre(info)

def evaluate_ablation(truncated_ids, baseline_probs, ablation_set, epsilon=1e-10):
	hook_handles = []
	for (layer, head) in ablation_set:
		hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(
			make_ablate_hook(head)
		)
		hook_handles.append(hook_handle)
	with torch.no_grad():
		ablated_logits = lm_model(truncated_ids).logits[0, -1, :]
	for handle in hook_handles:
		handle.remove()
	ablated_probs = torch.softmax(ablated_logits, dim=-1)
	kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
	return kl_div

def find_best_ablation_combo(truncated_ids, baseline_probs, max_heads=10):
	candidate_list = []
	for layer in range(lm_model.config.n_layer):
		for head in range(lm_model.config.n_head):
			candidate_list.append((layer, head))
	best_set = []  
	best_kl = evaluate_ablation(truncated_ids, baseline_probs, best_set)
	improved = True
	while improved and len(best_set) < max_heads:
		improved = False
		best_candidate = None
		candidate_kl = best_kl
		for candidate in tqdm(candidate_list, desc="Evaluating candidates", leave=False):
			if candidate in best_set:
				continue
			test_set = best_set + [candidate]
			kl = evaluate_ablation(truncated_ids, baseline_probs, test_set)
			if kl > candidate_kl:
				candidate_kl = kl
				best_candidate = candidate
				improved = True
		if best_candidate is not None:
			best_set.append(best_candidate)
			best_kl = candidate_kl
	return best_set, best_kl

# -------------------------------
# STEP 6: ABLATION STUDY LONG CALLBACK (Using progress() calls)
# -------------------------------
@app.long_callback(
	output=[Output("ablation-result", "children"),
			Output("combo-dropdown", "value")],
	inputs=[Input("run-ablation-study", "n_clicks")],
	state=[State("input-text", "value"),
		   State("attention-heatmap", "clickData"),
		   State("combo-dropdown", "value")],
	progress=[Output("ablation-progress", "value")],
	running=[(Output("run-ablation-study", "disabled"), True, False)],
	manager=long_callback_manager,
	prevent_initial_call=True
)
def run_ablation_study(progress, n_clicks, input_text, clickData, current_combos):
	# Early exit if no token was clicked.
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
	
	candidate_list = []
	for layer in range(lm_model.config.n_layer):
		for head in range(lm_model.config.n_head):
			candidate_list.append((layer, head))
	
	best_set = []  
	best_kl = evaluate_ablation(truncated_ids, baseline_probs, best_set)
	
	max_heads = 3
	total_expected = len(candidate_list) * max_heads
	evaluations_done = 0
	
	improved = True
	while improved and len(best_set) < max_heads:
		improved = False
		best_candidate = None
		candidate_kl = best_kl
		for candidate in candidate_list:
			if candidate in best_set:
				evaluations_done += 1
				progress(int((evaluations_done / total_expected) * 100))
				continue
			test_set = best_set + [candidate]
			kl = evaluate_ablation(truncated_ids, baseline_probs, test_set)
			evaluations_done += 1
			progress(int((evaluations_done / total_expected) * 100))
			if kl > candidate_kl:
				candidate_kl = kl
				best_candidate = candidate
				improved = True
		if best_candidate is not None:
			best_set.append(best_candidate)
			best_kl = candidate_kl
	
	progress(100)
	
	best_set_str = [f"{layer}-{head}" for (layer, head) in best_set]
	table_rows = []
	for (layer, head) in best_set:
		table_rows.append(html.Tr([
			html.Td(layer, style={'border': '1px solid black', 'padding': '4px'}),
			html.Td(head, style={'border': '1px solid black', 'padding': '4px'})
		]))
	table = html.Table(
		[html.Thead(html.Tr([html.Th("Layer", style={'border': '1px solid black', 'padding': '4px'}),
							  html.Th("Head", style={'border': '1px solid black', 'padding': '4px'})])),
		 html.Tbody(table_rows)],
		style={'width': '100%', 'borderCollapse': 'collapse'}
	)
	
	result_text = f"Best ablation combo (simultaneously ablating {len(best_set)} heads):"
	result_text += f"\nKL Divergence: {best_kl:.4f}"
	final_result = html.Div([html.Pre(result_text), table])
	# Convert final_result to a JSON-serializable dict.
	final_result = final_result.to_plotly_json()
	return final_result, best_set_str

# -------------------------------
# STEP 7: RUN THE DASH APP
# -------------------------------
if __name__ == '__main__':
	app.run_server(debug=True)
