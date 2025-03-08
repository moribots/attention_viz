# Attention Visualizer

A custom attention visualizer implementation. This isn't nearly as good as BertViz and the like, but I used the process of implementation as a way of learning more about Mechanistic Interpretability. My hope is to use what I learn here to augment this viz to apply to Vision Language Action Models. I am invested in controllable and reliable Robotics, and given the recent trend towards ML-based control, studying interpretability seems like the way to go.

## Demo

### Playing around with the viz

You can choose to display any number of layer-head combinations to visualize attention. In addition, you can perform various methods of ablation to study the effect on the next token prediction.

![select_prompt_and_heads](media/select_prompt_and_heads.gif)

### Running the Ablation Study

You can take your prompt and run two types of ablation studies:
- Generic: maximizes KL-divergence change
- Targeted: pick an alternate token prediction in the dropdown and ablate heads until it becomes the most likely option (this is what I do here)

![run_ablation_study](media/un_ablation_study.gif)

### Discover Circuit

TODO

## Usage

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

```bash
python3 main.py

# --> ctrl + click on the link in the terminal.
```

# Potential studies:

## Causal Role of Attention Heads

- Which attention heads are critical for predicting the next token?
- Can we identify redundant heads?

## Compositional Structure in Attention

- Do certain heads focus on specific linguistic features (e.g., subject-verb agreement)?
- Can permuting attention outputs break coherence?

## Sparse Representations in Transformer Attention

-Does structured sparsification reveal essential vs. inessential heads?
- Are there clusters of heads that work together?

## Progression Across Layers

- Do early layers focus on local dependencies while later layers capture abstract meaning?
- How do attention patterns change from input tokens to final predictions?
