# Circuit Selection Levels: edge vs node vs neuron (with mean‑positional evaluation)

## TL;DR
- **edge**: most precise and compact but noisier and harder to manage.
- **node**: stable, interpretable (“k heads / k MLPs”), best default under mean‑positional.
- **neuron**: ultra‑sparse inside MLPs; combine with node‑level for fine pruning.

If you’re using `intervention="mean-positional"`, start with **node‑level** selection and optionally refine MLPs with **neuron‑level**. Use edge‑level when you need path‑specific causal stories and can tolerate more complexity/variance.

---

## What each level selects
- **edge** — individual connections (e.g., head→head, head→MLP, MLP→residual) that carry signal along a route.
- **node** — whole components (attention head or MLP block) as atomic units.
- **neuron** — individual hidden units inside MLPs (feature channels).

> Note: “neuron” is meaningful for MLPs, not attention heads.

---

## Pros & cons

### edge
**Pros**
- Most fine‑grained; preserves only the *routes that matter*.
- Enables clear pathway narratives (“head A → head B → MLP C”).
- Often gives best faithfulness/size trade‑off for **tiny circuits**.

**Cons**
- Higher variance; small data/baseline shifts can reorder top edges.
- Harder to visualize and reason about at scale.
- Heavier to patch/ablate (more hooks; larger search space).
- With `mean-positional`, averaging *partial connections* can be off‑manifold / less stable.

**Use when:** you want pathway‑level explanations or the smallest faithful circuit and accept extra complexity.

---

### node
**Pros**
- Natural unit for interpretation (e.g., “6 heads do X”).  
- More stable than edges; simple hooks; faster evaluation.
- Plays nicely with `intervention="mean-positional"`.
- Great for size‑vs‑faithfulness sweeps.

**Cons**
- Coarser: keeps *all* routes of a head/MLP, including irrelevant ones.
- Circuits can be larger than edge‑level.

**Use when:** you want robust, interpretable circuits and clean reporting under mean‑positional. **Recommended default.**

---

### neuron
**Pros**
- Maximum sparsity inside MLPs; isolates specific features.
- Reduces collateral effects of dropping entire MLPs.
- Mean/mean‑positional baselines are naturally defined per‑feature.

**Cons**
- Huge search space; needs more data for stable attribution.
- Doesn’t apply to attention heads (use with nodes).
- Can hurt faithfulness if interactions require groups of neurons.

**Use when:** you suspect specific MLP features drive the behavior and want ultra‑compact MLP sub‑circuits.

---

## Mean‑positional vs. patching (context)
- Under **mean‑positional**, **node**/**neuron** ablations are generally **more stable** than **edge**.
- **Edge** ablation still works but is more baseline‑sensitive because “mean of a connection” is less well‑defined than “mean of a component/feature”.

**Rule of thumb:** Evaluate with span‑aware metrics under mean‑positional; select at **node‑level** first, then optionally refine MLPs with **neuron‑level**.

---

## How to set it in code

You likely have something like:
```python
graph.apply_topn(topn, absolute=True, level="edge", reset=True, prune=True)
```

Switch the granularity with `level=`:

### Node‑level (recommended default under mean‑positional)
```python
graph.apply_topn(
    topn,
    absolute=True,   # rank by |score|; use False if sign matters
    level="node",    # "edge" | "node" | "neuron"
    reset=True,
    prune=True
)
```

### Neuron‑level (MLPs only)
```python
graph.apply_topn(
    topn_neurons,
    absolute=True,
    level="neuron",
    reset=True,
    prune=True
)
```

### Edge‑level
```python
graph.apply_topn(
    topn_edges,
    absolute=True,
    level="edge",
    reset=True,
    prune=True
)
```

---

## A practical two‑stage pattern (mean‑positional)

1) **Pick nodes** (heads/MLPs): small, interpretable backbone.
```python
graph.apply_topn(k_nodes, absolute=True, level="node", reset=True, prune=False)
```

2) **Prune neurons** **within** the selected MLPs to tighten the circuit:
```python
graph.apply_topn(k_neurons, absolute=True, level="neuron", reset=False, prune=True)
```

This keeps the overall structure simple (few nodes) while making MLPs compact.

---

## Tips
- Start with node‑level sweeps (e.g., k ∈ {4, 6, 8, 12}) and pick the smallest k that preserves your **span‑aware** faithfulness metrics (KL/JS over span, agreement over span, NL(c) over span).
- Keep `absolute=True` to capture strong positive *and* negative contributors; set `False` if you care about directionality.
- When reporting, include both **size** (edges/nodes/neurons kept) and **faithfulness** (your span metrics) to make trade‑offs clear.
- If edge vs node results diverge, prefer node‑level for stability and interpretability under mean‑positional; use edges to tell a fine‑grained causal story.
