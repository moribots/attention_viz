"""
Main entry point for the interactive attention visualization tool.
Sets up the Dash app, defines the layout, and registers all callbacks.
"""

import multiprocessing as mp
try:
	mp.set_start_method('fork', force=True)
except RuntimeError:
	pass

import math
import io
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import diskcache
from dash.long_callback import DiskcacheLongCallbackManager
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from PIL import Image

# Our custom modules
from model_manager import TransformerModel
import ablation
import metrics
from circuit_finder import CircuitFinder
from config import NUM_COMBOS, SUBPLOT_SIZE, HEATMAP_MARGIN, TOP_K, SLIDER_MIN, SLIDER_MAX, SLIDER_STEP, SLIDER_MARKS

# Set up caching for long callbacks
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

external_stylesheets = [dbc.themes.BOOTSTRAP]

# Instantiate the GPT-2 model manager
transformer = TransformerModel()


# ============================================================================
# Helper Functions
# ============================================================================

def parse_combo(combo_str):
	"""
	Parse a combo string formatted as "layer-head" and return a tuple of integers.
	
	:param combo_str: String in the format "layer-head".
	:return: Tuple (layer, head) or None if parsing fails.
	"""
	try:
		layer_str, head_str = combo_str.split("-")
		return int(layer_str), int(head_str)
	except Exception:
		return None

def extract_clicked_token(clickData):
	"""
	Extract the clicked token from clickData.
	
	:param clickData: Dictionary containing click event data.
	:return: The token (str) if found, otherwise None.
	"""
	if clickData is None:
		return None
	try:
		return clickData["points"][0]["x"]
	except (KeyError, IndexError):
		return None

def convert_combos(combo_list):
	"""
	Convert a list of "layer-head" string combos into a list of (layer, head) tuples.
	
	:param combo_list: List of strings.
	:return: List of (layer, head) tuples.
	"""
	tuples = []
	for combo in combo_list:
		parsed = parse_combo(combo)
		if parsed is not None:
			tuples.append(parsed)
	return tuples


# ============================================================================
# Layout Building
# ============================================================================

def build_layout():
	"""
	Constructs and returns the layout for the Dash app.
	
	:return: A Dash layout (dbc.Container) including sidebar instructions,
			 main content with input fields, graphs, and ablation controls.
	"""
	# Build options for layer-head combo dropdown
	layer_and_head_options = [
		{'label': f"Layer {layer}, Head {head}", 'value': f"{layer}-{head}"}
		for layer in range(transformer.num_layers)
		for head in range(transformer.num_heads)
	]
	
	# Define example prompt options
	example_prompts = [
		{"label": "When Mary and John went to the store, John gave a drink to", 
		 "value": "When Mary and John went to the store, John gave a drink to"},
		{"label": "The key to the cabinets", "value": "The key to the cabinets"},
		{"label": "The trophy doesn't fit in the brown suitcase because it's too",
		 "value": "The trophy doesn't fit in the brown suitcase because it's too"}
	]
	
	sidebar = dbc.Card(
		[
			dbc.CardHeader("Instructions"),
			dbc.CardBody([
				html.P("- You can modify the input text on the right.", className="card-text"),
				html.P("- Select one or more layer-head pairs to view their attention heatmaps.", className="card-text"),
				html.P("- Use the slider to set a threshold and filter weak attention links.", className="card-text"),
				html.P("- Click a cell in any heatmap to view details about that token.", className="card-text"),
				html.P("- Ablation (causal tracing) lets you see how head removal changes predictions.", className="card-text"),
				html.P("- Click on 'Evaluate all Heads' to determine the impact of ablating each Layer-Head pair on the current predicted token.", className="card-text"),
				html.P("- Click on 'Discover Circuit' to do a basic circuit analysis based on 'Interpretability in the Wild[...]' by Wang et al.", className="card-text")
			])
		],
		style={"width": "100%", "marginBottom": "20px"}
	)
	
	main_content = dbc.Card(
		dbc.CardBody([
			# Example prompts dropdown
			html.Div([
				html.Label("Select Example Prompt:"),
				dcc.Dropdown(
					id="example-dropdown",
					options=example_prompts,
					placeholder="Select an example prompt or enter your own below",
					clearable=True,
					style={'width': '100%'}
				)
			], style={'marginBottom': '20px'}),
			
			# Input text area
			html.Div([
				html.Label("Input Text:"),
				dcc.Input(id="input-text", type="text",
						  value="When Mary and John went to the store, John gave a drink to",
						  style={'width': '100%'})
			], style={'marginBottom': '20px'}),
	
			# Combo dropdown with reset button
			html.Div([
				html.Div([
					html.Label("Select Layer–Head Combos:"),
					dcc.Dropdown(id="combo-dropdown",
								 options=layer_and_head_options,
								 value=["0-0"],
								 multi=True, clearable=False)
				], style={'width': '80%', 'display': 'inline-block'}),
				html.Div([
					html.Button("Reset", id="reset-button", className="btn btn-outline-secondary",
								style={'marginLeft': '10px', 'marginTop': '25px'})
				], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'})
			], style={'display': 'flex', 'marginBottom': '20px'}),
	
			# Threshold slider
			html.Div([
				html.Label("Attention Threshold (0.0 - 1.0):"),
				dcc.Slider(id="threshold-slider", min=SLIDER_MIN, max=SLIDER_MAX, step=SLIDER_STEP, value=0.0,
						   marks=SLIDER_MARKS)
			], style={'marginTop': '20px', 'marginBottom': '20px'}),
	
			# Ablation scale slider
			html.Div([
				html.Label("Ablation Scale Factor (0.0 = full ablation, 1.0 = no ablation):"),
				dcc.Slider(id="ablation-scale-slider", min=SLIDER_MIN, max=SLIDER_MAX, step=SLIDER_STEP, value=0.0,
						   marks=SLIDER_MARKS)
			], style={'marginTop': '20px', 'marginBottom': '20px'}),
	
			# Sparsity threshold slider
			html.Div([
				html.Label("Sparsity Threshold (for Structured Sparsification):"),
				dcc.Slider(id="sparsity-threshold-slider", min=SLIDER_MIN, max=SLIDER_MAX, step=SLIDER_STEP, value=0.1,
						   marks=SLIDER_MARKS)
			], style={'marginTop': '20px', 'marginBottom': '20px'}),
	
			# Ablation method radio items
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
	
			# Prev/Next page buttons
			html.Div([
				html.Button("Previous Page", id="prev-page-btn", n_clicks=0,
							style={'marginRight': '10px'}),
				html.Button("Next Page", id="next-page-btn", n_clicks=0),
				dcc.Store(id="page-store", data=0)
			], style={'marginBottom': '20px'}),
	
			# The heatmap
			dcc.Loading(
				id="loading-graph",
				type="circle",
				children=dcc.Graph(
					id="attention-heatmap",
					style={"width": "100%", "minHeight": "700px"}
				)
			),
	
			# Display area for token info
			html.Div(id="token-info", style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'}),
	
			# Ablation study controls
			html.Div([
				html.Button("Run Ablation Study", id="run-ablation-study", n_clicks=0, style={'marginRight': '10px'}),
				html.Label("Target token:", style={'marginRight': '10px', 'marginLeft': '10px'}),
				dcc.Dropdown(
					id="target-token-dropdown",
					options=[{'label': 'N/A (maximize change)', 'value': 'N/A'}],
					value='N/A',
					style={'width': '250px', 'display': 'inline-block'}
				)
			], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '20px'}),
	
			dbc.Progress(id="ablation-progress", striped=True, animated=True,
						 style={'marginTop': '20px', 'height': '20px'}),
			html.Div(id="ablation-result", style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'}),
	
			# Evaluate all heads controls
			html.Br(),
			html.Button("Evaluate All Heads", id="evaluate-all-heads", n_clicks=0),
			dcc.Graph(id="all-heads-bar-chart", style={"width": "100%", "minHeight": "500px"}),
	
			# Circuit discovery controls
			html.Div([
				html.Button("Discover Circuit", id="discover-circuit-btn", 
							n_clicks=0, className="btn btn-primary",
							style={'marginRight': '10px'}),
				dbc.Progress(id="circuit-progress", striped=True, animated=True,
							 style={'marginTop': '10px', 'height': '20px', 'width': '90%'}),
			], style={'marginTop': '20px'}),
			
			# Container for circuit visualization
			html.Div(id="circuit-container", style={
				'marginTop': '20px', 
				'padding': '10px', 
				'border': '1px solid #ccc',
				'display': 'none'
			}),
	
		]),
		style={"width": "100%"}
	)
	
	layout = dbc.Container([
		dbc.Row(dbc.Col(html.H1("Interactive Attention Visualization"), width=12),
				style={"marginTop": "20px", "marginBottom": "20px"}),
		dbc.Row([dbc.Col(sidebar, width=3), dbc.Col(main_content, width=9)])
	], fluid=True)
	
	return layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
				long_callback_manager=long_callback_manager)
app.layout = build_layout()


# ============================================================================
# Callbacks
# ============================================================================

@app.callback(
	Output("input-text", "value"),
	Input("example-dropdown", "value"),
	prevent_initial_call=True
)
def update_input_text_from_example(selected_example):
	"""
	Updates the input text field when an example prompt is selected.
	
	:param selected_example: The selected example prompt.
	:return: The selected prompt for the input text.
	"""
	if selected_example:
		return selected_example
	raise dash.exceptions.PreventUpdate

@app.callback(
	Output("combo-dropdown", "value", allow_duplicate=True),
	Input("reset-button", "n_clicks"),
	prevent_initial_call=True
)
def reset_combos(n_clicks):
	"""
	Resets the layer-head combo selection to the default value "0-0".
	
	:param n_clicks: Number of times the reset button has been clicked.
	:return: Default selection ["0-0"].
	"""
	return ["0-0"]

@app.callback(
	Output("page-store", "data"),
	[Input("prev-page-btn", "n_clicks"), Input("next-page-btn", "n_clicks")],
	[State("combo-dropdown", "value"), State("page-store", "data")]
)
def update_page(prev_clicks, next_clicks, combos, current_page):
	"""
	Adjusts the page index based on Prev/Next button clicks.
	
	:param prev_clicks: Number of times the "Previous Page" button was clicked.
	:param next_clicks: Number of times the "Next Page" button was clicked.
	:param combos: List of selected layer-head combos.
	:param current_page: The current page index.
	:return: Updated page index.
	"""
	if not combos:
		return 0
	ctx = dash.callback_context
	if not ctx.triggered:
		return current_page
	button_id = ctx.triggered[0]["prop_id"].split(".")[0]
	
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

@app.callback(
	Output("attention-heatmap", "figure"),
	[Input("input-text", "value"),
	 Input("combo-dropdown", "value"),
	 Input("threshold-slider", "value"),
	 Input("page-store", "data"),
	 Input("attention-heatmap", "clickData")]
)
def update_heatmap(input_text, selected_combos, threshold, current_page, clickData):
	"""
	Renders the attention heatmap for up to NUM_COMBOS combos on the current page.
	Highlights the selected token with red rectangles.
	
	:param input_text: The user-provided text.
	:param selected_combos: List of selected (layer-head) combos.
	:param threshold: Minimum attention value for display.
	:param current_page: Current page index.
	:param clickData: Click event data from the heatmap.
	:return: Plotly figure with heatmaps.
	"""
	if not isinstance(selected_combos, list):
		selected_combos = [selected_combos]
	
	combos = convert_combos(selected_combos)
	
	page_size = NUM_COMBOS
	start_idx = current_page * page_size
	end_idx = start_idx + page_size
	combos_on_page = combos[start_idx:end_idx]
	
	n = len(combos_on_page)
	if n == 0:
		fig = go.Figure()
		fig.add_annotation(text="No combos selected.", showarrow=False)
		fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
		return fig
	
	# Decide grid size for subplots
	if n <= 2:
		rows, cols = 1, n
	else:
		rows, cols = 2, 2
	
	subplot_titles = [f"Layer {layer}, Head {head}" for (layer, head) in combos_on_page]
	while len(subplot_titles) < rows * cols:
		subplot_titles.append("")
	
	fig = make_subplots(
		rows=rows, cols=cols,
		subplot_titles=subplot_titles,
		horizontal_spacing=0.2,
		vertical_spacing=0.2,
		row_heights=[1.0 / rows] * rows,
		column_widths=[1.0 / cols] * cols
	)
	
	selected_token = extract_clicked_token(clickData)
	
	for i, (layer, head) in enumerate(combos_on_page):
		attn_data, tokens = transformer.get_attention_data(input_text, layer, head, threshold)
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
		
		if selected_token is not None and selected_token in tokens:
			token_idx = tokens.index(selected_token)
			fig.add_shape(
				type="rect",
				x0=-0.5,
				x1=len(tokens) - 0.5,
				y0=token_idx - 0.5,
				y1=token_idx + 0.5,
				line=dict(color="red", width=3),
				fillcolor="rgba(0,0,0,0)",
				row=row,
				col=col
			)
			fig.add_shape(
				type="rect",
				x0=token_idx - 0.5,
				x1=token_idx + 0.5,
				y0=-0.5,
				y1=len(tokens) - 0.5,
				line=dict(color="red", width=3),
				fillcolor="rgba(0,0,0,0)",
				row=row,
				col=col
			)
	
	# Force subplots to be square
	for i in range(1, rows * cols + 1):
		x_str = "x" if i == 1 else f"x{i}"
		y_str = "y" if i == 1 else f"y{i}"
		if x_str in fig.layout and y_str in fig.layout:
			fig.layout[y_str].scaleanchor = x_str
			fig.layout[y_str].scaleratio = 1
	
	fig_width = SUBPLOT_SIZE * cols
	fig_height = SUBPLOT_SIZE * rows
	fig.update_layout(
		autosize=False,
		width=fig_width,
		height=fig_height,
		margin=dict(l=0, r=0, t=50, b=0),
		paper_bgcolor="white",
		plot_bgcolor="white"
	)
	
	for i in range(1, rows * cols + 1):
		x_str = "x" if i == 1 else f"x{i}"
		y_str = "y" if i == 1 else f"y{i}"
		if x_str in fig.layout and y_str in fig.layout:
			fig.layout[x_str].showgrid = False
			fig.layout[x_str].zeroline = False
			fig.layout[x_str].showline = False
			fig.layout[y_str].showgrid = False
			fig.layout[y_str].zeroline = False
			fig.layout[y_str].showline = False
	
	for i in range(1, rows * cols + 1):
		x_str = "x" if i == 1 else f"x{i}"
		y_str = "y" if i == 1 else f"y{i}"
		if x_str in fig.layout and y_str in fig.layout:
			fig.layout[x_str].update(
				tickmode='array',
				tickvals=list(range(len(tokens))),
				ticktext=tokens,
				tickangle=45
			)
			fig.layout[y_str].update(
				tickmode='array',
				tickvals=list(range(len(tokens))),
				ticktext=tokens
			)
	
	fig.update_layout(
		autosize=False,
		width=fig_width,
		height=fig_height,
		margin=HEATMAP_MARGIN,
		paper_bgcolor="white",
		plot_bgcolor="white"
	)
	
	return fig

@app.callback(
	Output("all-heads-bar-chart", "figure"),
	Input("evaluate-all-heads", "n_clicks"),
	[State("input-text", "value"),
	 State("attention-heatmap", "clickData"),
	 State("causal-intervention", "value"),
	 State("ablation-scale-slider", "value"),
	 State("sparsity-threshold-slider", "value"),
	 State("target-token-dropdown", "value")]
)
def update_all_heads_chart(n_clicks, input_text, clickData, causal_intervention, 
						   ablation_scale, sparsity_threshold, target_token_id):
	"""
	Evaluates all attention heads for a selected token and displays their ablation scores.
	
	:param n_clicks: Number of clicks on the "Evaluate All Heads" button.
	:param input_text: The full input text.
	:param clickData: Click event data from the heatmap.
	:param causal_intervention: Selected ablation method.
	:param ablation_scale: Ablation scale factor.
	:param sparsity_threshold: Threshold for sparsification.
	:param target_token_id: Target token ID (unused here, kept for consistency).
	:return: Plotly bar chart figure.
	"""
	if n_clicks == 0:
		return go.Figure()
		
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors="pt")
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	
	token_clicked = extract_clicked_token(clickData)
	if token_clicked is None:
		token_index = len(full_tokens) - 1
		token_clicked = full_tokens[token_index]
	else:
		# Clean and match token
		clean_token = token_clicked.split('_')[0] if '_' in token_clicked else token_clicked
		matches = [i for i, token in enumerate(full_tokens) if token.lstrip('Ġ') == clean_token]
		token_index = matches[0] if matches else len(full_tokens) - 1
	
	truncated_ids = full_input_ids[:, :token_index+1]
	
	with torch.no_grad():
		baseline_logits = transformer.lm_model(truncated_ids).logits[0, token_index, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	method = causal_intervention if causal_intervention != 'none' else 'standard'
	
	head_scores = ablation.evaluate_all_heads(
		truncated_ids, baseline_probs, transformer.lm_model,
		token_index=token_index, scale=ablation_scale, 
		ablation_method=method, sparsity_threshold=sparsity_threshold
	)
	
	labels = [f"{layer}-{head}" for (layer, head) in sorted(head_scores.keys())]
	scores = [head_scores[(layer, head)] for (layer, head) in sorted(head_scores.keys())]
	
	fig = go.Figure(data=go.Bar(x=labels, y=scores))
	title_text = f"Ablation Scores for Each Attention Head (Token: {token_clicked})"
	y_axis_title = "Ablation Score (KL Divergence + Delta Top Token Probability)"
	
	fig.update_layout(
		title=title_text,
		xaxis=dict(type='category', title="Layer-Head", tickangle=-45),
		yaxis=dict(title=y_axis_title),
		template="plotly_white",
		height=600,
		margin=HEATMAP_MARGIN
	)
	return fig

@app.callback(
	Output("token-info", "children"),
	[Input("attention-heatmap", "clickData"),
	 Input("input-text", "value"),
	 Input("causal-intervention", "value"),
	 Input("combo-dropdown", "value"),
	 Input("ablation-scale-slider", "value"),
	 Input("sparsity-threshold-slider", "value")]
)
def update_token_info(clickData, input_text, causal_intervention,
					  combo_dropdown, ablation_scale, sparsity_threshold):
	"""
	Displays information about the clicked token including baseline predictions
	and, if ablation is active, ablated predictions with extra metrics.
	
	:param clickData: Click event data from the heatmap.
	:param input_text: The user-provided text.
	:param causal_intervention: Selected ablation method.
	:param combo_dropdown: Selected layer-head combos.
	:param ablation_scale: Scale factor for standard ablation.
	:param sparsity_threshold: Threshold for structured sparsification.
	:return: A preformatted text block with token information.
	"""
	if clickData is None:
		return "Click on a cell in the heatmap to see token information."
	
	token_clicked = extract_clicked_token(clickData)
	if token_clicked is None:
		return "Error retrieving token info from click data."
	
	token_id = transformer.tokenizer.convert_tokens_to_ids(token_clicked)
	embedding = transformer.model.wte.weight[token_id]
	embedding_norm = torch.norm(embedding).item()
	info_str = f"Token: {token_clicked}\nToken ID: {token_id}\nEmbedding Norm: {embedding_norm:.4f}"
	
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors='pt')
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1
	truncated_ids = full_input_ids[:, :token_index+1]
	
	with torch.no_grad():
		baseline_outputs = transformer.lm_model(truncated_ids)
	baseline_logits = baseline_outputs.logits[0, -1, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	baseline_info = "\n\nBaseline Next Token Predictions:\n"
	baseline_top_probs, baseline_top_indices = torch.topk(baseline_probs, TOP_K)
	baseline_top_tokens = transformer.tokenizer.convert_ids_to_tokens(baseline_top_indices.tolist())
	for token, prob in zip(baseline_top_tokens, baseline_top_probs.tolist()):
		baseline_info += f"{token}: {prob:.4f}\n"
	
	if causal_intervention != 'none':
		combo_list = combo_dropdown if isinstance(combo_dropdown, list) else [combo_dropdown]
		hook_handles = []
		for combo in combo_list:
			parsed = parse_combo(combo)
			if not parsed:
				continue
			layer, head = parsed
			if causal_intervention == 'standard':
				hook = ablation.make_ablate_hook(head, scale=ablation_scale, lm_model=transformer.lm_model)
			elif causal_intervention == 'permute':
				hook = ablation.make_permutation_hook(head, lm_model=transformer.lm_model)
			elif causal_intervention == 'sparsify':
				hook = ablation.make_sparsification_hook(head, sparsity_threshold, lm_model=transformer.lm_model)
			else:
				hook = ablation.make_ablate_hook(head, scale=ablation_scale, lm_model=transformer.lm_model)
			hook_handle = transformer.lm_model.transformer.h[layer].attn.register_forward_hook(hook)
			hook_handles.append(hook_handle)
		
		with torch.no_grad():
			ablated_outputs = transformer.lm_model(truncated_ids)
		for handle in hook_handles:
			handle.remove()
		
		ablated_logits = ablated_outputs.logits[0, -1, :]
		ablated_probs = torch.softmax(ablated_logits, dim=-1)
		ablated_info = "\n\nAblated Next Token Predictions:\n"
		ablated_top_probs, ablated_top_indices = torch.topk(ablated_probs, TOP_K)
		ablated_top_tokens = transformer.tokenizer.convert_ids_to_tokens(ablated_top_indices.tolist())
		for token, prob in zip(ablated_top_tokens, ablated_top_probs.tolist()):
			ablated_info += f"{token}: {prob:.4f}\n"
		
		extra = metrics.compute_extra_metrics(baseline_probs, ablated_probs, token_clicked, transformer.tokenizer)
		extra_metrics_str = "\n\nDeeper Analysis Metrics:\n"
		extra_metrics_str += f"KL Divergence: {extra['KL Divergence']:.4f}\n"
		extra_metrics_str += f"Delta Top Token Probability: {extra['Delta Top Token Probability']:.4f}\n"
		extra_metrics_str += f"Baseline Entropy: {extra['Baseline Entropy']:.4f}\n"
		extra_metrics_str += f"Ablated Entropy: {extra['Ablated Entropy']:.4f}\n"
		extra_metrics_str += f"Entropy Increase: {extra['Entropy Increase']:.4f}\n"
		extra_metrics_str += f"Baseline Rank: {extra['Baseline Rank']}\n"
		extra_metrics_str += f"Ablated Rank: {extra['Ablated Rank']}\n"
		extra_metrics_str += f"Rank Change: {extra['Rank Change']}\n"
		
		info_str += baseline_info + ablated_info + extra_metrics_str
	else:
		info_str += baseline_info
	
	return html.Pre(info_str)

@app.long_callback(
	output=[Output("ablation-result", "children"),
			Output("combo-dropdown", "value")],
	inputs=[Input("run-ablation-study", "n_clicks")],
	state=[State("input-text", "value"),
		   State("attention-heatmap", "clickData"),
		   State("combo-dropdown", "value"),
		   State("ablation-scale-slider", "value"),
		   State("causal-intervention", "value"),
		   State("sparsity-threshold-slider", "value"),
		   State("target-token-dropdown", "value")],
	progress=[Output("ablation-progress", "value")],
	running=[(Output("run-ablation-study", "disabled"), True, False)],
	manager=long_callback_manager,
	prevent_initial_call=True
)
def run_ablation_study(progress, n_clicks, input_text, clickData, current_combos,
					   ablation_scale, causal_intervention, sparsity_threshold, target_token_id):
	"""
	Searches for the best set of heads to ablate based on the selected optimization target.
	If target_token_id is 'N/A', maximizes overall distribution change.
	
	:param progress: Callback to update progress.
	:param n_clicks: Number of times the study button was clicked.
	:param input_text: The input text.
	:param clickData: Click event data from the heatmap.
	:param current_combos: Selected layer-head combos.
	:param ablation_scale: Scale factor for ablation.
	:param causal_intervention: Selected ablation method.
	:param sparsity_threshold: Threshold for sparsification.
	:param target_token_id: Target token ID or 'N/A'.
	:return: Tuple of updated ablation result HTML and updated combo selections.
	"""
	if clickData is None:
		return "Click on a token in the heatmap before running the ablation study.", current_combos
	
	token_clicked = extract_clicked_token(clickData)
	
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors="pt")
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1
	
	truncated_ids = full_input_ids[:, :token_index+1]
	
	with torch.no_grad():
		baseline_logits = transformer.lm_model(truncated_ids).logits[0, token_index, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	ablation_method = causal_intervention if causal_intervention != 'none' else 'standard'
	progress(10)
	
	specific_token_id = None if target_token_id == 'N/A' else int(target_token_id)
	
	best_set, best_score = ablation.find_best_ablation_combo(
		truncated_ids, baseline_probs, token_index=token_index,
		max_head_layer_pairs=20, scale=ablation_scale,
		ablation_method=ablation_method,
		sparsity_threshold=sparsity_threshold,
		lm_model=transformer.lm_model,
		progress_callback=lambda x: progress(x),
		search_strategy='iterative',
		target_token_id=specific_token_id
	)
	
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
	
	if specific_token_id is not None:
		target_token = transformer.tokenizer.convert_ids_to_tokens([specific_token_id])[0]
		result_text = f"Best ablation combo to maximize '{target_token}' probability:"
		result_text += f"\nFinal Probability: {best_score:.4f}"
	else:
		result_text = f"Best ablation combo (ablating {len(best_set)} heads):"
		result_text += f"\nCombined Score: {best_score:.4f}"
	
	hook_handles = []
	for (layer, head) in best_set:
		if ablation_method == 'standard':
			hook = ablation.make_ablate_hook(head, scale=ablation_scale, lm_model=transformer.lm_model)
		elif ablation_method == 'permute':
			hook = ablation.make_permutation_hook(head, lm_model=transformer.lm_model)
		elif ablation_method == 'sparsify':
			hook = ablation.make_sparsification_hook(head, sparsity_threshold, lm_model=transformer.lm_model)
		else:
			hook = ablation.make_ablate_hook(head, scale=ablation_scale, lm_model=transformer.lm_model)
		hook_handle = transformer.lm_model.transformer.h[layer].attn.register_forward_hook(hook)
		hook_handles.append(hook_handle)
	
	with torch.no_grad():
		best_ablated_logits = transformer.lm_model(truncated_ids).logits[0, token_index, :]
	for handle in hook_handles:
		handle.remove()
	
	best_ablated_probs = torch.softmax(best_ablated_logits, dim=-1)
	
	if specific_token_id is not None:
		target_token = transformer.tokenizer.convert_ids_to_tokens([specific_token_id])[0]
		baseline_target_prob = baseline_probs[specific_token_id].item()
		ablated_target_prob = best_ablated_probs[specific_token_id].item()
		extra_metrics_text = f"\n\nTarget Token: {target_token}"
		extra_metrics_text += f"\nBaseline Probability: {baseline_target_prob:.4f}"
		extra_metrics_text += f"\nAblated Probability: {ablated_target_prob:.4f}"
		extra_metrics_text += f"\nProbability Increase: {ablated_target_prob - baseline_target_prob:.4f}"
	else:
		extra = metrics.compute_extra_metrics(baseline_probs, best_ablated_probs, token_clicked, transformer.tokenizer)
		extra_metrics_text = "\n\nDeeper Analysis Metrics (Best Combo):\n"
		extra_metrics_text += f"KL Divergence: {extra['KL Divergence']:.4f}\n"
		extra_metrics_text += f"Delta Top Token Probability: {extra['Delta Top Token Probability']:.4f}\n"
		extra_metrics_text += f"Baseline Entropy: {extra['Baseline Entropy']:.4f}\n"
		extra_metrics_text += f"Ablated Entropy: {extra['Ablated Entropy']:.4f}\n"
		extra_metrics_text += f"Entropy Increase: {extra['Entropy Increase']:.4f}\n"
		extra_metrics_text += f"Baseline Rank: {extra['Baseline Rank']}\n"
		extra_metrics_text += f"Ablated Rank: {extra['Ablated Rank']}\n"
		extra_metrics_text += f"Rank Change: {extra['Rank Change']}\n"
	
	final_result = html.Div([html.Pre(result_text + extra_metrics_text), table])
	final_result = final_result.to_plotly_json()
	
	return final_result, best_set_str

@app.callback(
	Output("target-token-dropdown", "options"),
	[Input("attention-heatmap", "clickData"),
	 Input("input-text", "value")]
)
def update_target_dropdown(clickData, input_text):
	"""
	Updates the target token dropdown with N/A and the top 5 predicted tokens when a token is clicked.
	
	:param clickData: Click event data from the heatmap.
	:param input_text: The user-provided text.
	:return: List of dropdown options.
	"""
	if clickData is None:
		return [{'label': 'N/A (maximize change)', 'value': 'N/A'}]
	
	token_clicked = extract_clicked_token(clickData)
	if token_clicked is None:
		return [{'label': 'N/A (maximize change)', 'value': 'N/A'}]
	
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors="pt")
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1
	
	truncated_ids = full_input_ids[:, :token_index+1]
	with torch.no_grad():
		baseline_logits = transformer.lm_model(truncated_ids).logits[0, token_index, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	baseline_top_probs, baseline_top_indices = torch.topk(baseline_probs, TOP_K)
	baseline_top_tokens = transformer.tokenizer.convert_ids_to_tokens(baseline_top_indices.tolist())
	
	options = [{'label': 'N/A (maximize change)', 'value': 'N/A'}]
	for token, prob, token_id in zip(baseline_top_tokens, baseline_top_probs.tolist(), baseline_top_indices.tolist()):
		options.append({'label': f"{token} ({prob:.4f})", 'value': str(token_id)})
	
	return options

@app.long_callback(
	output=[
		Output("circuit-container", "children"),
		Output("circuit-container", "style"),
		Output("circuit-progress", "value")
	],
	inputs=[Input("discover-circuit-btn", "n_clicks")],
	state=[
		State("input-text", "value"),
		State("combo-dropdown", "value"),
		State("target-token-dropdown", "value")
	],
	progress=[Output("circuit-progress", "value")],
	running=[(Output("discover-circuit-btn", "disabled"), True, False)],
	manager=long_callback_manager,
	prevent_initial_call=True
)
def discover_attention_circuit(progress, n_clicks, input_text, selected_combos, target_token_id):
	"""
	Discovers and visualizes attention circuits using selected attention heads.
	
	:param progress: Callback function to update progress.
	:param n_clicks: Number of clicks on the discover circuit button.
	:param input_text: Input text for analysis.
	:param selected_combos: Selected layer-head combos.
	:param target_token_id: Target token ID or 'N/A'.
	:return: Tuple of (circuit visualization HTML, container style, progress value).
	"""
	if not selected_combos or not input_text:
		return html.Div("Please select attention heads and provide input text."), {'display': 'block'}, 100
	
	important_heads = []
	for combo in selected_combos:
		parsed = parse_combo(combo)
		if parsed is not None:
			important_heads.append(parsed)
			
	if not important_heads:
		return html.Div("Could not parse selected attention heads."), {'display': 'block'}, 100
	
	progress(10)
	
	try:
		if target_token_id == 'N/A' or target_token_id is None:
			specific_token_id = None
		else:
			specific_token_id = int(target_token_id) if isinstance(target_token_id, str) else int(target_token_id)
	except (ValueError, TypeError) as e:
		print(f"Error converting target token ID: {e}, using None instead")
		specific_token_id = None
	
	print(f"Using target token ID: {specific_token_id}, type: {type(specific_token_id)}")
	
	circuit_finder = CircuitFinder(transformer.lm_model, transformer.tokenizer)
	
	try:
		circuit_graph = circuit_finder.build_circuit_from_ablation(
			important_heads=important_heads,
			input_text=input_text,
			target_token_id=specific_token_id,
			progress_callback=progress
		)
	except Exception as e:
		import traceback
		traceback_str = traceback.format_exc()
		error_message = f"Error building circuit: {str(e)}\n\nTraceback:\n{traceback_str}"
		return html.Div([
			html.Pre(error_message, style={'whiteSpace': 'pre-wrap', 'overflowX': 'auto'})
		]), {'display': 'block'}, 100
	
	plt_buf = io.BytesIO()
	try:
		circuit_finder.visualize_circuit(save_path=plt_buf)
		plt_buf.seek(0)
		img_str = base64.b64encode(plt_buf.read()).decode('utf-8')
		edge_count = circuit_graph.number_of_edges()
		node_count = circuit_graph.number_of_nodes()
		explanation = f"""
### Attention Circuit Analysis

This visualization shows how information flows through the {len(important_heads)} 
selected attention heads for the input: "{input_text}"

- **Blue squares**: Input tokens
- **Green circles**: Attention heads (Layer-Head)
- **Red circle**: Target token (if specified)
- **Edge thickness**: Strength of connection between components

Circuit Statistics:
- {node_count} total nodes
- {edge_count} significant connections
- {len([n for n in circuit_graph.nodes if circuit_graph.nodes[n].get('type') == 'attention_head'])} attention heads in circuit
		"""
		circuit_viz = html.Div([
			html.H4("Transformer Circuit Visualization"),
			dcc.Markdown(explanation),
			html.Img(src=f'data:image/png;base64,{img_str}', style={'width': '100%', 'marginTop': '20px'})
		])
	except Exception as viz_error:
		circuit_viz = html.Div([
			html.H4("Circuit Discovery Successful, Visualization Error"),
			html.P("The circuit was successfully built, but there was an error generating the visualization:"),
			html.Pre(str(viz_error))
		])
	
	progress(100)
	
	return circuit_viz, {'display': 'block', 'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'}, 100

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
	app.run_server(debug=True)
