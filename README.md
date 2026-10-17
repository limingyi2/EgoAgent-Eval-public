# EgoAgent-Eval

EgoAgent-Eval is a Python framework for evaluating video agents that choose which frames to inspect. It provides video decoding, uniform and candidate-based retrieval, hierarchical observation policies, provider adapters, protocol validation, trace recording, and an evaluation harness.

The repository contains implementation code and synthetic/offline tests. It does not include private datasets, benchmark answers, source videos, model responses, API credentials, or historical experiment traces. Results from synthetic providers are control-flow checks and do not measure visual accuracy.

## Requirements and offline checks

Python 3.11 or later is required. Install the package and test dependencies, then run the offline suite:

```powershell
python -m pip install -e ".[dev,dataset]"
$env:PYTHON_DOTENV_DISABLED = "1"
python -m pytest -ra
```

The test suite uses synthetic videos and in-memory HTTP transports. Real-provider integration tests are not included in the public test set. Running the CLI with a real provider requires explicit local configuration and is outside the scope of the offline tests.

## Basic fake run

```powershell
python -m scripts.make_test_video --output .\artifacts\synthetic.avi
python -m egoagent_eval --video .\artifacts\synthetic.avi --question "Describe the visible shapes." --provider fake --method all --output .\artifacts
```

Use only data you have permission to process. Keep credentials, videos, generated frames, traces, and experiment outputs outside version control. See `docs/architecture.md` and `docs/experiment-ledger.md` for the public architecture and ledger contract.
