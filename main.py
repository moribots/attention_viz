"""
Main entry point for the interactive attention visualization tool.
Sets up the Dash app, defines the layout, and registers all callbacks.
"""

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
import dash_bootstrap_components as dbc
import diskcache
from dash.long_callback import DiskcacheLongCallbackManager
import math
import torch

# Our custom modules
from model_manager import TransformerModel
import ablation
import metrics

# Set up caching for long callbacks
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

external_stylesheets = [dbc.themes.BOOTSTRAP]

# We'll display up to 4 combos per page in the heatmap
NUM_COMBOS = 4

# Instantiate the GPT-2 model manager
transformer = TransformerModel()

def build_layout():
	"""
	Constructs and returns the layout for the Dash app.
	
	:return: A Dash layout (dbc.Container) that includes sidebar instructions,
			 main content with input fields, graphs, and ablation controls.
	"""
	# We'll build a list of (layer, head) combos for the user to choose from
	layer_and_head_options = [
		{'label': f"Layer {layer}, Head {head}", 'value': f"{layer}-{head}"}
		for layer in range(transformer.num_layers)
		for head in range(transformer.num_heads)
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
			# Input text area
			html.Div([
				html.Label("Input Text:"),
				dcc.Input(id="input-text", type="text",
						  value="The quick brown fox jumps over the lazy dog.",
						  style={'width': '100%'})
			], style={'marginBottom': '20px'}),
	
			# Dropdown for selecting combos
			html.Div([
				html.Label("Select Layer–Head Combos:"),
				dcc.Dropdown(id="combo-dropdown",
							 options=layer_and_head_options,
							 value=["0-0"],
							 multi=True, clearable=False)
			], style={'width': '90%', 'marginBottom': '20px'}),
	
			# Threshold slider
			html.Div([
				html.Label("Attention Threshold (0.0 - 1.0):"),
				dcc.Slider(id="threshold-slider", min=0.0, max=1.0, step=0.01, value=0.0,
						   marks={i/10: f"{i/10}" for i in range(0, 11)})
			], style={'marginTop': '20px', 'marginBottom': '20px'}),
	
			# Ablation scale slider
			html.Div([
				html.Label("Ablation Scale Factor (0.0 = full ablation, 1.0 = no ablation):"),
				dcc.Slider(id="ablation-scale-slider", min=0.0, max=1.0, step=0.01, value=0.0,
						   marks={i/10: f"{i/10}" for i in range(0, 11)})
			], style={'marginTop': '20px', 'marginBottom': '20px'}),
	
			# Sparsity threshold slider
			html.Div([
				html.Label("Sparsity Threshold (for Structured Sparsification):"),
				dcc.Slider(id="sparsity-threshold-slider", min=0.0, max=1.0, step=0.01, value=0.1,
						   marks={i/10: f"{i/10}" for i in range(0, 11)})
			], style={'marginTop': '20px', 'marginBottom': '20px'}),
	
			# RadioItems for ablation method
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
	
			# The heatmap itself
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
	
			# Button to run the ablation study
			html.Button("Run Ablation Study", id="run-ablation-study", n_clicks=0),
			dbc.Progress(id="ablation-progress", striped=True, animated=True,
						 style={'marginTop': '20px', 'height': '20px'}),
			html.Div(id="ablation-result", style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'})
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

@app.callback(
	Output("page-store", "data"),
	[Input("prev-page-btn", "n_clicks"), Input("next-page-btn", "n_clicks")],
	[State("combo-dropdown", "value"), State("page-store", "data")]
)
def update_page(prev_clicks, next_clicks, combos, current_page):
	"""
	Adjusts the page index based on Prev/Next button clicks.
	We display up to NUM_COMBOS combos per page in the heatmap.

	:param prev_clicks: int
		Number of times the "Previous Page" button was clicked.
	:param next_clicks: int
		Number of times the "Next Page" button was clicked.
	:param combos: list
		The selected (layer-head) combos in the dropdown.
	:param current_page: int
		The current page index.
	:return: int
		The updated page index after handling the clicks.
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
	 Input("page-store", "data")]
)
def update_heatmap(input_text, selected_combos, threshold, current_page):
	"""
	Renders the attention heatmap for up to NUM_COMBOS combos on the current "page".

	:param input_text: str
		The user-provided text.
	:param selected_combos: list
		The selected (layer-head) combos in the dropdown.
	:param threshold: float
		Minimum attention value for display.
	:param current_page: int
		The current page index (for combos).
	:return: plotly.graph_objects.Figure
		The figure containing one or more heatmaps.
	"""
	if not isinstance(selected_combos, list):
		selected_combos = [selected_combos]
	
	# Convert "layer-head" strings into (layer, head) tuples
	combos = []
	for combo in selected_combos:
		try:
			layer_str, head_str = combo.split("-")
			combos.append((int(layer_str), int(head_str)))
		except:
			continue
	
	page_size = NUM_COMBOS
	start_idx = current_page * page_size
	end_idx = start_idx + page_size
	combos_on_page = combos[start_idx:end_idx]
	
	n = len(combos_on_page)
	if n == 0:
		# If no combos selected on this page, show a placeholder
		fig = go.Figure()
		fig.add_annotation(text="No combos selected.", showarrow=False)
		fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
		return fig
	
	# Decide how many rows/cols to make in the subplot
	if n <= 2:
		rows, cols = 1, n
	else:
		rows, cols = 2, 2
	
	# Titles for each subplot
	subplot_titles = [f"Layer {layer}, Head {head}" for (layer, head) in combos_on_page]
	# If we have fewer combos than the full grid, pad the titles
	while len(subplot_titles) < rows * cols:
		subplot_titles.append("")
	
	# Create a subplot grid
	fig = make_subplots(
		rows=rows, cols=cols,
		subplot_titles=subplot_titles,
		horizontal_spacing=0.2,
		vertical_spacing=0.2,
		row_heights=[1.0 / rows] * rows, # The number of columns always increases before the number of rows, and the max is 2.
		column_widths=[1.0 / cols] * cols
	)

	# Add a heatmap for each (layer, head)
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
	
	# Suppose rows and cols are determined by how many combos you have (max 2×2).
	# After you create your subplots and add your heatmaps:

	# 1) Force each subplot to remain square via scaleanchor/scaleratio
	for i in range(1, rows * cols + 1):
		x_str = "x" if i == 1 else f"x{i}"
		y_str = "y" if i == 1 else f"y{i}"
		if x_str in fig.layout and y_str in fig.layout:
			# Lock the y-axis to the x-axis so each subplot is a square
			fig.layout[y_str].scaleanchor = x_str
			fig.layout[y_str].scaleratio = 1

	# 2) Dynamically size the figure based on rows and cols.
	#    We'll define a base "subplot_size" that sets how big each square is.
	#    Then figure width = subplot_size * cols + some margin
	#    and figure height = subplot_size * rows + some margin.

	subplot_size = 650  # Adjust this to make subplots bigger or smaller

	fig_width = subplot_size * cols
	fig_height = subplot_size * rows * 2.0 / 3.0

	# If you only have 1 row and 2 columns, for instance, 
	#   fig_width = 300 * 2 + 100 = 700
	#   fig_height = 300 * 1 + 100 = 400
	# That yields a fairly wide but not super tall figure.

	fig.update_layout(
		autosize=False,
		width=fig_width,
		height=fig_height,
		margin=dict(l=0, r=0, t=50, b=0),
		paper_bgcolor="white",
		plot_bgcolor="white"
	)

	# Optionally remove grid lines and zero lines if you see extra background lines:
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
	Displays info about the clicked token, including baseline predictions,
	and if ablation is active, also shows the ablated predictions + metrics.

	:param clickData: dict
		Data about the clicked cell in the heatmap.
	:param input_text: str
		The user-provided text.
	:param causal_intervention: str
		The selected ablation method ('none', 'standard', 'permute', 'sparsify').
	:param combo_dropdown: list
		The (layer, head) combos to ablate if ablation is active.
	:param ablation_scale: float
		Scale factor for standard ablation.
	:param sparsity_threshold: float
		Threshold for structured sparsification.
	:return: dash.html.Pre
		A text block with baseline and ablated predictions, plus extra metrics if ablation is active.
	"""
	if clickData is None:
		return "Click on a cell in the heatmap to see token information."
	
	try:
		token_clicked = clickData["points"][0]["x"]
	except (KeyError, IndexError):
		return "Error retrieving token info from click data."
	
	# Convert token to ID and retrieve embedding norm
	token_id = transformer.tokenizer.convert_tokens_to_ids(token_clicked)
	embedding = transformer.model.wte.weight[token_id]
	embedding_norm = torch.norm(embedding).item()
	info_str = f"Token: {token_clicked}\nToken ID: {token_id}\nEmbedding Norm: {embedding_norm:.4f}"
	
	# Encode the full input
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors='pt')
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	
	# Find the index of the clicked token in the input
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1
	truncated_ids = full_input_ids[:, :token_index+1]
	
	# Get baseline probabilities
	with torch.no_grad():
		baseline_outputs = transformer.lm_model(truncated_ids)
	baseline_logits = baseline_outputs.logits[0, -1, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	# Show top-5 baseline predictions
	baseline_topk = 5
	baseline_top_probs, baseline_top_indices = torch.topk(baseline_probs, baseline_topk)
	baseline_top_tokens = transformer.tokenizer.convert_ids_to_tokens(baseline_top_indices.tolist())
	baseline_info = "\n\nBaseline Next Token Predictions:\n"
	for token, prob in zip(baseline_top_tokens, baseline_top_probs.tolist()):
		baseline_info += f"{token}: {prob:.4f}\n"
	
	# If ablation is not 'none', apply the chosen method
	if causal_intervention != 'none':
		# Convert combo_dropdown to a list of (layer, head) combos
		combo_list = combo_dropdown if isinstance(combo_dropdown, list) else [combo_dropdown]
		hook_handles = []
		for combo in combo_list:
			try:
				layer_str, head_str = combo.split("-")
				layer = int(layer_str)
				head = int(head_str)
			except:
				continue
			
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
		ablated_topk = 5
		ablated_top_probs, ablated_top_indices = torch.topk(ablated_probs, ablated_topk)
		ablated_top_tokens = transformer.tokenizer.convert_ids_to_tokens(ablated_top_indices.tolist())
		
		ablated_info = "\n\nAblated Next Token Predictions:\n"
		for token, prob in zip(ablated_top_tokens, ablated_top_probs.tolist()):
			ablated_info += f"{token}: {prob:.4f}\n"
		
		# Compute extra metrics for the clicked token
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
		   State("sparsity-threshold-slider", "value")],
	progress=[Output("ablation-progress", "value")],
	running=[(Output("run-ablation-study", "disabled"), True, False)],
	manager=long_callback_manager,
	prevent_initial_call=True
)
def run_ablation_study(progress, n_clicks, input_text, clickData, current_combos,
					   ablation_scale, causal_intervention, sparsity_threshold):
	"""
	Searches for the best set of heads to ablate (up to 10),
	using an iterative strategy, then reports the results.

	:param progress: function
		Callback used to update the progress bar (0-100%).
	:param n_clicks: int
		Number of times the "Run Ablation Study" button was clicked.
	:param input_text: str
		The user-provided text to run the model on.
	:param clickData: dict
		Data about the clicked token in the heatmap (we need to pick a token).
	:param current_combos: list
		The combos selected in the dropdown (layer-head pairs).
	:param ablation_scale: float
		Scale factor for standard ablation.
	:param causal_intervention: str
		Which ablation method is chosen.
	:param sparsity_threshold: float
		Threshold for structured sparsification.
	:return: tuple
		(jsonified HTML with results, updated combos).
	"""
	if clickData is None:
		return "Click on a token in the heatmap before running the ablation study.", current_combos
	
	try:
		token_clicked = clickData["points"][0]["x"]
	except (KeyError, IndexError):
		return "Error retrieving token info from click data.", current_combos
	
	# Encode input and find index of the clicked token
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors="pt")
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1
	
	truncated_ids = full_input_ids[:, :token_index+1]
	
	# Get baseline probabilities
	with torch.no_grad():
		baseline_logits = transformer.lm_model(truncated_ids).logits[0, -1, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	# If the user selected 'none', let's interpret that as 'standard'
	# so we can see some effect from the search. (Alternatively, we could skip.)
	ablation_method = causal_intervention if causal_intervention != 'none' else 'standard'

	progress(10)
	
	# Use our ablation find_best_ablation_combo
	best_set, best_score = ablation.find_best_ablation_combo(
		truncated_ids, baseline_probs,
		max_heads=10, scale=ablation_scale,
		ablation_method=ablation_method,
		sparsity_threshold=sparsity_threshold,
		lm_model=transformer.lm_model,
		progress_callback=lambda x: progress(10 + (x * 80) // 100),  # Normalize within 10-90%,
		search_strategy='iterative'
	)
	
	# Mark progress as done
	progress(100)
	
	best_set_str = [f"{layer}-{head}" for (layer, head) in best_set]
	
	# Build a small HTML table of the best set
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
	
	# For completeness, let's also compute deeper analysis metrics for that best combo
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
		best_ablated_logits = transformer.lm_model(truncated_ids).logits[0, -1, :]
	for handle in hook_handles:
		handle.remove()
	
	best_ablated_probs = torch.softmax(best_ablated_logits, dim=-1)
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

if __name__ == '__main__':
	app.run_server(debug=True)
