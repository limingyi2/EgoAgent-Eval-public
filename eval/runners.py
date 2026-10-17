"""Runner contract and deliberately synthetic integration runner."""

import importlib
import json

from .data import uniform_times


class FakeRunner:
    is_mock = True
    supports_seed = True
    name = "fake-fixture-v1"

    def run(self, task, strategy, config, recorder):
        initial = uniform_times(task["duration_seconds"], 4)
        timestamps = uniform_times(task["duration_seconds"], 8) if strategy == "Uniform-8" else initial
        recorder.tool("observe_frames", {"timestamps": timestamps},
                      lambda: recorder.frames(timestamps, [f"mock://{task['task_id']}/{t}" for t in timestamps]))
        fixture_id = task["task_id"]
        if fixture_id == "fixture-tool-error":
            recorder.tool("inspect_frame", {"timestamp": 5}, lambda: self._tool_error())
        if fixture_id == "fixture-budget":
            recorder.frames([1] * 9, ["mock://over-budget"] * 9)
        if strategy == "Active-4+4":
            selected = [4.5, 5.0, 5.5, 6.0]
            recorder.request({"phase": "select", "question": task["question"], "images": recorder.observation_images()},
                             lambda: {"output": json.dumps({"observe_timestamps": selected}),
                                      "usage": {"input_tokens": 80, "output_tokens": 20}})
            recorder.tool("observe_frames", {"timestamps": selected},
                          lambda: recorder.frames(selected, [f"mock://{fixture_id}/{t}" for t in selected]))
        answer = "blue" if fixture_id != "fixture-incorrect" else "red"
        output = {"answer": answer, "evidence_timestamps": [5.0], "evidence_interval": [4.5, 5.5]}
        if fixture_id == "fixture-alias":
            output["answer"] = " AZURE "
        if fixture_id == "fixture-missing-evidence":
            output = {"answer": "blue"}
        def respond():
            if fixture_id == "fixture-provider-error":
                raise ConnectionError("synthetic provider failure")
            return {"output": "{broken" if fixture_id == "fixture-invalid-json" else json.dumps(output),
                    "usage": None if fixture_id == "fixture-missing-usage" else {"input_tokens": 120, "output_tokens": 30}}
        return recorder.request({"phase": "answer", "question": task["question"], "images": recorder.observation_images()}, respond)

    @staticmethod
    def _tool_error():
        raise OSError("synthetic frame extraction failure")


def load_runner(spec):
    if spec == "fake":
        return FakeRunner()
    module, separator, factory = spec.partition(":")
    if not separator:
        raise ValueError("runner must be fake or module:factory")
    runner = getattr(importlib.import_module(module), factory)()
    for attribute in ("is_mock", "supports_seed", "name", "run"):
        if not hasattr(runner, attribute):
            raise ValueError(f"runner missing {attribute}")
    if type(runner.is_mock) is not bool or type(runner.supports_seed) is not bool:
        raise ValueError("runner is_mock/supports_seed must be booleans")
    if not isinstance(runner.name, str) or not runner.name.strip() or not callable(runner.run):
        raise ValueError("runner requires a nonempty name and callable run method")
    return runner
