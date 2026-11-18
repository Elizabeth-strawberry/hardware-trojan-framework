# Filter the Spurious, Recover the Missing: A Framework for Hardware Trojan Detection
## Usage

Requires Python 3.10+. Select a model configuration in `configs/` and set its API credential. `TOP_K` specifies the number of highest-ranked detector nodes used as initial candidates.

| Command | Purpose |
| --- | --- |
| `prepare` | Load detector predictions and run PayloadChecker and ModuleChecker to construct and recover candidate graphs. |
| `review` | Use the LLM to refine candidate nodes based on recorded structural results and retrieved references. |
| `score` | Compare candidates with ground-truth labels to compute recall and inspection size. Labels are used only for evaluation. |

```bash
python -m pip install -e '.[api]'
python examples/create_demo.py

trojan-refine prepare \
  --report outputs/demo/predictions.json \
  --variant synthetic_demo --top-k "$TOP_K" --trojan-type combinational \
  --output outputs/demo/prepared

trojan-refine review \
  --input outputs/demo/prepared --config configs/openai.example.json \
  --output outputs/demo/review.json

trojan-refine score \
  --input outputs/demo/prepared --result outputs/demo/review.json \
  --labels outputs/demo/labels.json --output outputs/demo/metrics.json
```

For experiments, supply detector predictions and the corresponding netlist; FSM and Counter circuits also require `--library`.

## Examples

- [create_demo.py](examples/create_demo.py) generates a sample circuit, simulated detector predictions, and labels for the `prepare → review → score` workflow.
- [create_training_corpus.py](examples/create_training_corpus.py) generates module-annotated reference graphs for GraphMentor training and FAISS index construction.

Both scripts use synthetic data for demonstration, not the paper's experimental data.

## Training and retrieval

Install `pip install -e '.[research,api]'` for GraphMentor and FAISS. Use `train-mentor` to train on annotated reference graphs and `build-vector-index` to build the retrieval bank. Exclude test hosts from training.

Pass `--mentor-index` to enable learned retrieval and `--neo4j` during preparation to enable connectivity queries. Configure Neo4j through `NEO4J_HTTP_URI`, `NEO4J_DATABASE`, `NEO4J_USER`, and `NEO4J_PASSWORD`.

Use `trojan-refine <command> --help` for options. Run tests with `python -m unittest discover -s tests`.
