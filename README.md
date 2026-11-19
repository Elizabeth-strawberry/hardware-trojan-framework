# Filter the Spurious, Recover the Missing: A Framework for Hardware Trojan Detection

The proposed framework uses structural analysis and LLM reasoning to refine predictions from node-level hardware Trojan detection algorithms, reducing false positives and recovering missed Trojan nodes.

## Run the example

Use Python 3.10 or newer. Run these commands from the repository directory:

```bash
python -m pip install -e '.[api]'
python examples/create_demo.py
```

This creates a small example circuit, simulated detector predictions, and labels in `outputs/demo/`. These are demonstration files, not the paper's experimental data.

Set your API key and enter how many of the highest-ranked detector nodes to start with:

```bash
export OPENAI_API_KEY="your-api-key"
read -r -p "Number of initial detector nodes: " TOP_K
```

`TOP_K` is a node count, not a multiple of the number of Trojan nodes. The example contains six ranked nodes; choose a positive integer up to six.

Run the three steps below:

| Command | What it does |
| --- | --- |
| `prepare` | Load the predictions. PayloadChecker searches for payload gates and expands the connected logic. ModuleChecker checks the Trojan modules and recovers missing gates. |
| `review` | Ask the LLM to read the saved detector and structural-check results and select the nodes to keep. |
| `score` | Compare the selected nodes with the labels to calculate recall and the number of nodes to inspect. Only this step uses the labels. |

```bash
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

The selected nodes are saved in `outputs/demo/review.json`; the evaluation results are saved in `outputs/demo/metrics.json`. The `review` step sends requests to the configured model API.

## Use your own circuit

Run your detector first, then supply its saved predictions to `--report`. Use the generated `outputs/demo/predictions.json` as a format example. Set `--variant` to your circuit's name and use `--netlist` to specify its netlist file. For FSM or Counter Trojans, set `--trojan-type` to `fsm` or `counter` and provide the standard-cell Verilog library with `--library`.

Choose a model configuration from `configs/`, set the API key named by its `api_key_env` field, and pass that configuration to `review --config`.

## Optional: reference retrieval

Reference retrieval lets ModuleChecker and the LLM compare a suspected Trojan with known examples. Install the extra dependencies with `python -m pip install -e '.[research,api]'`. Train GraphMentor with `train-mentor`, then create a searchable FAISS index with `build-vector-index`. Pass the index directory through `--mentor-index` to both `prepare` and `review`. Keep test circuits out of the training references.

[examples/create_training_corpus.py](examples/create_training_corpus.py) creates sample reference graphs with module labels to demonstrate training. To use Neo4j for tracing connections during preparation, configure the connection variables listed in `.env.example` and add `--neo4j` to `prepare`. Set these variables in your shell; `.env.example` is not loaded automatically.

Run `trojan-refine <command> --help` for command options.
