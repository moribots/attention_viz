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
from circuit_finder import CircuitFinder
import io
import base64
from PIL import Image

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
	
	# Define example prompt options
	example_prompts = [
		{"label": "The key to the cabinets", "value": "The key to the cabinets"}, # subject-verb agreement ambiguity (singular vs plural)
		{"label": "When Mary and John went to the store, John gave a drink to", # co-reference resolution
         "value": "When Mary and John went to the store, John gave a drink to"},
		{"label": "The trophy doesn't fit in the brown suitcase because it's too", # ambiguous pronoun reference.
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
				html.P("- Ablation (causal tracing) lets you see how head removal changes predictions.", className="card-text")
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
						  value="The key to the cabinets",
						  style={'width': '100%'})
			], style={'marginBottom': '20px'}),
	
			# Dropdown for selecting combos with reset button
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
	
			# Button to run the ablation study with target token dropdown
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

			# Button and graph for evaluating all heads individually
			html.Br(),
			html.Button("Evaluate All Heads", id="evaluate-all-heads", n_clicks=0),
			dcc.Graph(id="all-heads-bar-chart", style={"width": "100%", "minHeight": "500px"}),

			# Add button for circuit discovery after the ablation results section
			html.Div([
				html.Button("Discover Circuit", id="discover-circuit-btn", 
						   n_clicks=0, className="btn btn-primary",
						   style={'marginRight': '10px'}),
				dbc.Progress(id="circuit-progress", striped=True, animated=True,
							style={'marginTop': '10px', 'height': '20px', 'width': '90%'}),
			], style={'marginTop': '20px'}),
			
			# Add container for circuit visualization
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

# Add callback for example prompt selection
@app.callback(
	Output("input-text", "value"),
	Input("example-dropdown", "value"),
	prevent_initial_call=True
)
def update_input_text_from_example(selected_example):
	"""
	Updates the input text field when an example prompt is selected.
	
	:param selected_example: The selected example prompt.
	:return: The selected prompt to be used as input text.
	"""
	if selected_example:
		return selected_example
	# If dropdown is cleared, keep the current value
	raise dash.exceptions.PreventUpdate

# Add a new callback for the reset button
@app.callback(
	Output("combo-dropdown", "value", allow_duplicate=True),
	Input("reset-button", "n_clicks"),
	prevent_initial_call=True
)
def reset_combos(n_clicks):
	"""
	Resets the layer-head combo selection to just show Layer 0, Head 0.
	
	:param n_clicks: Number of times the reset button has been clicked.
	:return: Default list with just "0-0" selected.
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
	 Input("page-store", "data"),
	 Input("attention-heatmap", "clickData")]  # Add clickData as input
)
def update_heatmap(input_text, selected_combos, threshold, current_page, clickData):
	"""
	Renders the attention heatmap for up to NUM_COMBOS combos on the current "page".
	Highlights the selected token and its attention patterns with red rectangular borders.

	:param input_text: str
		The user-provided text.
	:param selected_combos: list
		The selected (layer-head) combos in the dropdown.
	:param threshold: float
		Minimum attention value for display.
	:param current_page: int
		The current page index (for combos).
	:param clickData: dict
		Click data containing information about the selected token.
	:return: plotly.graph_objects.Figure
		The figure containing one or more heatmaps with highlight rectangles.
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
	
	# Extract selected token from clickData (if available)
	selected_token = None
	if clickData is not None:
		try:
			selected_token = clickData["points"][0]["x"]
		except (KeyError, IndexError):
			selected_token = None

	# Add a heatmap for each (layer, head)
	for i, (layer, head) in enumerate(combos_on_page):
		attn_data, tokens = transformer.get_attention_data(input_text, layer, head, threshold)
		print(f'tokens: {tokens}')
		
		# Create heatmap with explicit axis settings
		heatmap = go.Heatmap(
			z=attn_data,
			x=tokens,  # to tokens
			y=tokens,  # from tokens
			colorscale='Viridis',
			colorbar=dict(title="Attention Weight"),
			hovertemplate="From Token: %{y}<br>To Token: %{x}<br>Attention: %{z:.4f}<extra></extra>"
		)

		row = (i // cols) + 1
		col = (i % cols) + 1
		fig.add_trace(heatmap, row=row, col=col)
		
		# Add highlighting for selected token if available
		if selected_token is not None and selected_token in tokens:
			token_idx = tokens.index(selected_token)
			
			# Add shape to highlight row (tokens this token attends to)
			fig.add_shape(
				type="rect",
				x0=-0.5,  # Start slightly before first token
				x1=len(tokens) - 0.5,  # End slightly after last token
				y0=token_idx - 0.5,  # Selected token's row
				y1=token_idx + 0.5,
				line=dict(color="red", width=3),
				fillcolor="rgba(0,0,0,0)",  # Transparent fill
				row=row,
				col=col
			)
			
			# Add shape to highlight column (tokens that attend to this token)
			fig.add_shape(
				type="rect",
				x0=token_idx - 0.5,  # Selected token's column
				x1=token_idx + 0.5,
				y0=-0.5,  # Start slightly before first token
				y1=len(tokens) - 0.5,  # End slightly after last token
				line=dict(color="red", width=3),
				fillcolor="rgba(0,0,0,0)",  # Transparent fill
				row=row,
				col=col
			)
	
	# 1) Force each subplot to remain square via scaleanchor/scaleratio
	for i in range(1, rows * cols + 1):
		x_str = "x" if i == 1 else f"x{i}"
		y_str = "y" if i == 1 else f"y{i}"
		if x_str in fig.layout and y_str in fig.layout:
			# Lock the y-axis to the x-axis so each subplot is a square
			fig.layout[y_str].scaleanchor = x_str
			fig.layout[y_str].scaleratio = 1

	# 2) Dynamically size the figure based on rows and cols
	subplot_size = 650  # Adjust this to make subplots bigger or smaller

	fig_width = subplot_size * cols
	fig_height = subplot_size * rows * 2.0 / 3.0

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

	# Set explicit axis configurations to ensure all tokens are visible
	for i in range(1, rows * cols + 1):
		x_str = "x" if i == 1 else f"x{i}"
		y_str = "y" if i == 1 else f"y{i}"
		if x_str in fig.layout and y_str in fig.layout:
			# Make sure axes show all ticks and don't skip any tokens
			fig.layout[x_str].update(
				showgrid=False,
				zeroline=False,
				showline=False,
				tickmode='array',
				tickvals=list(range(len(tokens))),
				ticktext=tokens,
				tickangle=45
			)
			fig.layout[y_str].update(
				showgrid=False,
				zeroline=False,
				showline=False,
				tickmode='array',
				tickvals=list(range(len(tokens))),
				ticktext=tokens
				)

	# Increase margins to ensure all labels are visible
	fig.update_layout(
		autosize=False,
		width=fig_width,
		height=fig_height,
		margin=dict(l=80, r=40, t=50, b=80),  # Increased bottom and left margins
		paper_bgcolor="white",
		plot_bgcolor="white"
	)

	return fig

@app.callback(
	Output("all-heads-bar-chart", "figure"),
	Input("evaluate-all-heads", "n_clicks"),
	State("input-text", "value"),
	State("attention-heatmap", "clickData"),
	State("causal-intervention", "value"),
	State("ablation-scale-slider", "value"),
	State("sparsity-threshold-slider", "value"),
	State("target-token-dropdown", "value")  # Keep this parameter for UI consistency
)
def update_all_heads_chart(n_clicks, input_text, clickData, causal_intervention, 
						  ablation_scale, sparsity_threshold, target_token_id):
	"""
	Evaluates all attention heads individually for a selected token and plots their ablation scores.
	This always evaluates based on overall distribution change regardless of target token selection.

	:param n_clicks: Number of times the "Evaluate All Heads" button was clicked.
	:param input_text: The full input text.
	:param clickData: Click event data from the heatmap.
	:param causal_intervention: Selected ablation method.
	:param ablation_scale: Ablation scale factor.
	:param sparsity_threshold: Threshold for sparsification.
	:param target_token_id: ID of token (unused in this function, maintained for UI consistency).
	:return: Plotly figure displaying a bar chart of ablation scores.
	"""
	# Check if button was clicked
	if n_clicks == 0:
		# Return empty figure if button hasn't been clicked yet
		return go.Figure()
		
	# Encode the input text to get tokens
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors="pt")
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	
	# Default to last token if no token is clicked
	token_clicked = None
	if clickData is not None:
		try:
			token_clicked = clickData["points"][0]["x"]
		except (KeyError, IndexError) as e:
			print(f"Error extracting clicked token: {e}")
	
	if token_clicked is None:
		# If no token was clicked, explicitly set to evaluate the last token
		token_index = len(full_tokens) - 1
		token_clicked = full_tokens[token_index]
		print(f"No token selected, defaulting to last token: {token_clicked}")
	else:
		# We need to handle the display formatting that might be present in token names
		# Extract base token without positional suffix if present
		clean_token = token_clicked.split('_')[0] if '_' in token_clicked else token_clicked
		
		# Find matching tokens in the full token list
		matches = []
		for i, token in enumerate(full_tokens):
			# Clean the token for comparison
			if token.startswith('Ġ'):  # Handle GPT-2's space prefix
				clean_full_token = token[1:]
			else:
				clean_full_token = token
				
			if clean_full_token == clean_token:
				matches.append(i)
		
		# Use the match if found
		if matches:
			# If there are multiple matches, try to determine which one was clicked
			# based on position suffix or default to the first occurrence
			if '_' in token_clicked and len(matches) > 1:
				try:
					position = int(token_clicked.split('_')[1])
					if 1 <= position <= len(matches):
						token_index = matches[position - 1]
					else:
						token_index = matches[0]
				except (ValueError, IndexError):
					token_index = matches[0]
			else:
				token_index = matches[0]
		else:
			# If no match found, default to the last token
			token_index = len(full_tokens) - 1
			token_clicked = full_tokens[token_index]
			print(f"Could not find token match, defaulting to last token: {token_clicked}")
		
	
	# Truncate input IDs up to the selected token (inclusive)
	truncated_ids = full_input_ids[:, :token_index+1]
	
	# Compute baseline logits for the selected token
	with torch.no_grad():
		baseline_logits = transformer.lm_model(truncated_ids).logits[0, token_index, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	# Use the selected ablation method; default to 'standard' if none selected
	method = causal_intervention if causal_intervention != 'none' else 'standard'
	
	# Evaluate all heads WITHOUT target token parameter - reverted to original behavior
	head_scores = ablation.evaluate_all_heads(
		truncated_ids, baseline_probs, transformer.lm_model,
		token_index=token_index, scale=ablation_scale, 
		ablation_method=method, sparsity_threshold=sparsity_threshold
	)
	
	# Prepare labels and corresponding scores
	labels = [f"{layer}-{head}" for (layer, head) in sorted(head_scores.keys())]
	scores = [head_scores[(layer, head)] for (layer, head) in sorted(head_scores.keys())]
	
	# Create the bar chart using Plotly
	fig = go.Figure(data=go.Bar(x=labels, y=scores))
	
	# Update title and y-axis - always showing distribution change metrics
	title_text = f"Ablation Scores for Each Attention Head (Token: {token_clicked})"
	y_axis_title = "Ablation Score (KL Divergence + Delta Top Token Probability)"
	
	fig.update_layout(
		title=title_text,
		xaxis=dict(type='category'),
		xaxis_title="Layer-Head",
		yaxis_title=y_axis_title,
		xaxis_tickangle=-45,
		template="plotly_white",
		height=600,
		margin=dict(l=40, r=40, t=60, b=150)
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
		   State("sparsity-threshold-slider", "value"),
		   State("target-token-dropdown", "value")],  # Add target token state
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
	If a token is selected, finds heads that maximize that token's probability.
	
	:param progress: Callback function to update progress.
	:param n_clicks: Number of times the study button was clicked.
	:param input_text: The input sentence.
	:param clickData: Click event data from the heatmap.
	:param current_combos: Selected layer-head combos.
	:param ablation_scale: Scale factor for ablation.
	:param causal_intervention: Selected ablation method.
	:param sparsity_threshold: Threshold for sparsification.
	:param target_token_id: ID of the token to maximize probability for, or 'N/A'.
	:return: Tuple of updated ablation result HTML and updated combo selections.
	"""
	if clickData is None:
		return "Click on a token in the heatmap before running the ablation study.", current_combos
	
	try:
		# Extract token from x-axis (assuming heatmap is now set up for "from" tokens)
		token_clicked = clickData["points"][0]["x"]
	except (KeyError, IndexError):
		return "Error retrieving token info from click data.", current_combos
	
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors="pt")
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1  # Default to last token if not found
	
	truncated_ids = full_input_ids[:, :token_index+1]
	
	with torch.no_grad():
		baseline_logits = transformer.lm_model(truncated_ids).logits[0, token_index, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	ablation_method = causal_intervention if causal_intervention != 'none' else 'standard'
	
	progress(10)
	
	# Parse target_token_id, passing None if 'N/A' is selected
	specific_token_id = None if target_token_id == 'N/A' else int(target_token_id)
	
	# Find the best ablation combo using our updated function that supports token maximization
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
	
	# Update result text based on what we were optimizing for
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
	
	# If we're targeting a specific token, show its probability before and after
	if specific_token_id is not None:
		target_token = transformer.tokenizer.convert_ids_to_tokens([specific_token_id])[0]
		baseline_target_prob = baseline_probs[specific_token_id].item()
		ablated_target_prob = best_ablated_probs[specific_token_id].item()
		extra_metrics_text = f"\n\nTarget Token: {target_token}"
		extra_metrics_text += f"\nBaseline Probability: {baseline_target_prob:.4f}"
		extra_metrics_text += f"\nAblated Probability: {ablated_target_prob:.4f}"
		extra_metrics_text += f"\nProbability Increase: {ablated_target_prob - baseline_target_prob:.4f}"
	else:
		# Original metrics
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
	Updates the target token dropdown with the top 5 predicted next tokens when a token is clicked.
	
	:param clickData: Click event data from the heatmap.
	:param input_text: The user-provided text.
	:return: Updated dropdown options with N/A and top 5 predicted tokens.
	"""
	if clickData is None:
		return [{'label': 'N/A (maximize change)', 'value': 'N/A'}]
	
	try:
		token_clicked = clickData["points"][0]["x"]
	except (KeyError, IndexError):
		return [{'label': 'N/A (maximize change)', 'value': 'N/A'}]
	
	# Encode the input text
	full_input_ids = transformer.tokenizer.encode(input_text, return_tensors="pt")
	full_tokens = transformer.tokenizer.convert_ids_to_tokens(full_input_ids[0])
	
	# Find token index, defaulting to last token if not found
	try:
		token_index = full_tokens.index(token_clicked)
	except ValueError:
		token_index = len(full_tokens) - 1
	
	truncated_ids = full_input_ids[:, :token_index+1]
	
	# Get baseline predictions for the token
	with torch.no_grad():
		baseline_logits = transformer.lm_model(truncated_ids).logits[0, token_index, :]
	baseline_probs = torch.softmax(baseline_logits, dim=-1)
	
	# Get top 5 tokens
	baseline_topk = 5
	baseline_top_probs, baseline_top_indices = torch.topk(baseline_probs, baseline_topk)
	baseline_top_tokens = transformer.tokenizer.convert_ids_to_tokens(baseline_top_indices.tolist())
	
	# Create dropdown options with N/A and top 5 tokens
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
	running=[
		(Output("discover-circuit-btn", "disabled"), True, False)
	],
	manager=long_callback_manager,
	prevent_initial_call=True
)
def discover_attention_circuit(progress, n_clicks, input_text, selected_combos, target_token_id):
	"""
	Discovers and visualizes attention circuits in the transformer model.
	Uses optimized algorithms for circuit discovery to improve performance.
	
	:param progress: Callback function to update progress.
	:param n_clicks: Number of times the discover circuit button was clicked.
	:param input_text: The input text to analyze.
	:param selected_combos: Selected layer-head combos from the dropdown.
	:param target_token_id: ID of target token, or 'N/A'.
	:return: Tuple of (circuit visualization HTML, container style dict, progress value)
	"""
	if not selected_combos or not input_text:
		return html.Div("Please select attention heads and provide input text."), {'display': 'block'}, 100
	
	# Parse selected combos
	important_heads = []
	for combo in selected_combos:
		try:
			layer_str, head_str = combo.split("-")
			important_heads.append((int(layer_str), int(head_str)))
		except:
			continue
			
	if not important_heads:
		return html.Div("Could not parse selected attention heads."), {'display': 'block'}, 100
	
	# Initial progress update
	progress(10)
	
	# Proper handling of target token ID - ensure it's an integer or None
	try:
		if target_token_id == 'N/A' or target_token_id is None:
			specific_token_id = None
		else:
			# Convert to integer explicitly, handling both string and tensor cases
			if isinstance(target_token_id, str):
				specific_token_id = int(target_token_id)
			elif isinstance(target_token_id, torch.Tensor):
				specific_token_id = target_token_id.item()
			else:
				specific_token_id = int(target_token_id)
				
		print(f"Using target token ID: {specific_token_id}, type: {type(specific_token_id)}")
	except (ValueError, TypeError) as e:
		print(f"Error converting target token ID: {e}, using None instead")
		specific_token_id = None
	
	# Create circuit finder with progress monitoring
	circuit_finder = CircuitFinder(transformer.lm_model, transformer.tokenizer)
	
	# Build circuit with progress reporting
	try:
		circuit_graph = circuit_finder.build_circuit_from_ablation(
			important_heads=important_heads,
			input_text=input_text,
			target_token_id=specific_token_id,  # Pass the properly converted ID
			progress_callback=progress
		)
	except Exception as e:
		import traceback
		traceback_str = traceback.format_exc()
		error_message = f"Error building circuit: {str(e)}\n\nTraceback:\n{traceback_str}"
		return html.Div([
			html.Pre(error_message, style={'whiteSpace': 'pre-wrap', 'overflowX': 'auto'})
		]), {'display': 'block'}, 100
	
	# Generate circuit visualization
	plt_buf = io.BytesIO()
	try:
		circuit_finder.visualize_circuit(save_path=plt_buf)
		plt_buf.seek(0)
		
		# Convert plot to base64 image
		img_str = base64.b64encode(plt_buf.read()).decode('utf-8')
		
		# Create figure explanation based on graph properties
		edge_count = circuit_graph.number_of_edges()
		node_count = circuit_graph.number_of_nodes()
		
		# Generate helpful explanation text with performance note
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
		
		Note: This visualization uses optimized algorithms that prioritize connections between adjacent layers
		for better performance.
		"""
		
		# Create circuit visualization with explanation
		circuit_viz = html.Div([
			html.H4("Transformer Circuit Visualization"),
			dcc.Markdown(explanation),
			html.Img(src=f'data:image/png;base64,{img_str}', style={'width': '100%', 'marginTop': '20px'})
		])
	except Exception as viz_error:
		circuit_viz = html.Div([
			html.H4("Circuit Discovery Successful, Visualization Error"),
			html.P(f"The circuit was successfully built with {circuit_graph.number_of_nodes()} nodes, but there was an error generating the visualization:"),
			html.Pre(str(viz_error))
		])
	
	# Final progress update
	progress(100)
	
	# Return visualization with display style and progress 
	return circuit_viz, {'display': 'block', 'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'}, 100

if __name__ == '__main__':
	app.run_server(debug=True)

	"""

	1. Standard Ablation
	What it does: Zeroes out or scales down the output of specific attention heads (controlled by the ablation scale slider).

	Works best when: You want to understand the basic importance of a head by completely removing its contribution.

	Why it's useful: Provides a straightforward measure of head importance - if removing a head drastically changes predictions, it's likely critical.

	2. Permutation Ablation
	What it does: Maintains the same attention weights but randomly shuffles which tokens they connect to.

	Works best when: The specific pattern or ordering of attention matters.

	Why it's useful: If a head's function depends on attending to particular tokens in a specific order (like attending to a subject to determine a verb's form), permutation will disrupt this while keeping the overall "energy" intact. Heads that show larger effects under permutation than standard ablation likely perform precise targeting rather than general information aggregation.

	3. Structured Sparsification
	What it does: Retains only the strongest attention connections above the sparsity threshold, zeroing out weaker ones.

	Works best when: Testing if partial information is sufficient for a head's function.

	Why it's useful: If a head works well with only its top connections, weaker ones may be redundant. Heads resilient to sparsification likely rely on just a few key connections rather than broad patterns. This method helps identify heads that perform specialized, focused tasks versus those that need complete information.
	"""