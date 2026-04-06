import pandas as pd
import pytest

import agent.LLM as llm_module


class DummyRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, description, parameters, func):
        self._tools[name] = func

    def execute(self, fn, arguments):
        return self._tools[fn](**arguments)

    def ollama_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "dummy",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                },
            }
            for name in self._tools
        ]


class DummyClient:
    def __init__(self, responses):
        self.responses = responses
        self.idx = 0

    def chat(self, model, messages, tools, stream=False):
        if self.idx >= len(self.responses):
            raise RuntimeError("No more dummy responses configured.")
        response = self.responses[self.idx]
        self.idx += 1
        return response


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "amt": [10.0, 20.0, 30.0, 1000.0],
            "city_pop": [1000, 2000, 3000, 4000],
            "merchant": ["a", "b", "a", "c"],
            "category": ["food", "tech", "food", "other"],
            "is_fraud": [0, 0, 1, 1],
        }
    )


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setattr(llm_module, "ToolRegistry", DummyRegistry)
    return llm_module.LLM()


def get_tool_result(llm, tool_name: str) -> dict:
    """Return the last recorded result for *tool_name* from last_run_trace."""
    results = llm.last_run_trace["tool_results"]
    matching = [r for r in results if r["name"] == tool_name]
    assert matching, f"No tool result recorded for: {tool_name}"
    return matching[-1]["result"]


def make_tool_call_response(tool_name: str, arguments: dict):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    }
                }
            ],
        }
    }


def make_final_response(text: str):
    return {
        "message": {
            "role": "assistant",
            "content": text,
            "tool_calls": [],
        }
    }


def test_run_loads_dataset_via_find_dataset(monkeypatch, llm, sample_df):
    monkeypatch.setattr(
        llm_module,
        "find_dataset",
        lambda query: {"match_found": True, "dataset_name": "Fraud detection"},
    )
    monkeypatch.setattr(
        llm_module,
        "load_dataset_by_name",
        lambda dataset_name: sample_df,
    )

    llm.client = DummyClient(
        [
            make_tool_call_response("find_dataset", {"query": "Load Fraud dataset"}),
            make_final_response('The dataset "Fraud detection" was loaded successfully.'),
        ]
    )

    answer = llm.run("Load Fraud dataset")

    assert "loaded successfully" in answer
    assert llm.current_dataset_name == "Fraud detection"
    assert "Fraud detection" in llm.loaded_datasets
    pd.testing.assert_frame_equal(llm.loaded_datasets["Fraud detection"], sample_df)

    result = get_tool_result(llm, "find_dataset")
    assert result["match_found"] is True
    assert result["dataset_name"] == "Fraud detection"
    assert result["loaded"] is True
    assert result["rows"] == 4
    assert result["columns"] == 5


def test_run_get_info_on_loaded_dataset(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("get_info", {}),
            make_final_response("Here is the dataset information."),
        ]
    )

    answer = llm.run("Give me information on the dataset")

    assert "information" in answer.lower()

    result = get_tool_result(llm, "get_info")
    assert result["rows"] == 4
    assert result["columns"] == 5
    assert set(result["column_names"]) == {"amt", "city_pop", "merchant", "category", "is_fraud"}
    assert set(result["numeric_columns"]) == {"amt", "city_pop", "is_fraud"}
    assert set(result["categorical_columns"]) == {"merchant", "category"}
    assert all(v == 0 for v in result["missing_values"].values())


def test_run_describe_dataset(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("describe_dataset", {}),
            make_final_response("Here is the dataset description."),
        ]
    )

    answer = llm.run("Describe the dataset")
    assert "description" in answer.lower()

    result = get_tool_result(llm, "describe_dataset")
    assert result["rows"] == 4
    assert result["columns"] == 5
    assert set(result["numeric_columns"]) == {"amt", "city_pop", "is_fraud"}
    assert set(result["categorical_columns"]) == {"merchant", "category"}
    assert isinstance(result["summary_preview"], dict)


def test_run_calc_mean_column(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("calc_mean_column", {"col": "amt"}),
            make_final_response("The mean was calculated."),
        ]
    )

    answer = llm.run("Calculate the mean of amt")
    assert "mean" in answer.lower()

    # amt = [10, 20, 30, 1000]  →  mean = 265.0
    result = get_tool_result(llm, "calc_mean_column")
    assert result["column"] == "amt"
    assert result["mean"] == pytest.approx(265.0)


def test_run_scatter_plot(llm, sample_df, monkeypatch):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    monkeypatch.setattr(
        llm_module,
        "scatter_plot",
        lambda df, col1, col2, plot_description="", linewidths=0.7: {
            "plot_created": True,
            "plot_path": "/tmp/test_scatter.png",
            "x": col1,
            "y": col2,
        },
    )
    llm.dataset_tools["scatter_plot"] = llm_module.scatter_plot

    llm.client = DummyClient(
        [
            make_tool_call_response(
                "scatter_plot",
                {"col1": "amt", "col2": "city_pop", "plot_description": "Test plot"},
            ),
            make_final_response("Scatter plot created."),
        ]
    )

    answer = llm.run("Create a scatter plot")
    assert "scatter plot" in answer.lower()

    result = get_tool_result(llm, "scatter_plot")
    assert result["plot_created"] is True
    assert result["x"] == "amt"
    assert result["y"] == "city_pop"


def test_run_pca_feature_selection(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("pca_feature_selection", {"n_features": 2}),
            make_final_response("PCA feature selection completed."),
        ]
    )

    answer = llm.run("Perform PCA feature selection")
    assert "pca" in answer.lower() or "feature selection" in answer.lower()

    # sample_df has 3 numeric cols → 2 components are valid
    result = get_tool_result(llm, "pca_feature_selection")
    assert result["n_features"] == 2
    assert len(result["explained_variance_ratio"]) == 2
    assert sum(result["explained_variance_ratio"]) <= 1.0 + 1e-9
    assert len(result["components_preview"]) > 0
    assert "PC1" in result["components_preview"][0]
    assert "PC2" in result["components_preview"][0]


def test_run_filter_outliers(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("filter_outliers", {"threshold": 3.5}),
            make_final_response("Outliers filtered."),
        ]
    )

    answer = llm.run("Filter outliers")
    assert "outlier" in answer.lower()

    # amt=1000 is the only outlier (robust-z ≈ 65.8 >> 3.5)
    result = get_tool_result(llm, "filter_outliers")
    assert result["rows_before"] == 4
    assert result["rows_removed"] == 1
    assert result["rows_after"] == 3
    assert result["threshold"] == pytest.approx(3.5)


def test_run_normalize(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("normalize", {}),
            make_final_response("Dataset normalized."),
        ]
    )

    answer = llm.run("Normalize the dataset")
    assert "normal" in answer.lower()

    result = get_tool_result(llm, "normalize")
    assert set(result["normalized_columns"]) == {"amt", "city_pop", "is_fraud"}
    # After z-score normalization the mean of each numeric column must be ~0
    preview = pd.DataFrame(result["preview"])
    for col in ["amt", "city_pop", "is_fraud"]:
        assert preview[col].mean() == pytest.approx(0.0, abs=1e-9)


def test_run_missing_values_report(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("missing_values_report", {}),
            make_final_response("Missing values report generated."),
        ]
    )

    answer = llm.run("Create a missing values report")
    assert "missing" in answer.lower()

    result = get_tool_result(llm, "missing_values_report")
    report = {row["column"]: row for row in result["report"]}
    assert set(report.keys()) == {"amt", "city_pop", "merchant", "category", "is_fraud"}
    assert all(row["missing_count"] == 0 for row in result["report"])
    assert all(row["missing_percent"] == pytest.approx(0.0) for row in result["report"])


def test_run_column_type_report(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("column_type_report", {}),
            make_final_response("Column type report generated."),
        ]
    )

    answer = llm.run("Create a column type report")
    assert "column type" in answer.lower() or "report" in answer.lower()

    result = get_tool_result(llm, "column_type_report")
    by_col = {row["column"]: row for row in result["report"]}
    assert len(by_col) == 5
    assert by_col["amt"]["dtype"] == "float64"
    assert by_col["city_pop"]["dtype"] == "int64"
    assert by_col["merchant"]["dtype"] == "object"
    assert by_col["merchant"]["n_unique"] == 3
    assert by_col["category"]["n_unique"] == 3
    assert by_col["is_fraud"]["n_unique"] == 2


def test_run_correlation_report(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("correlation_report", {"method": "pearson"}),
            make_final_response("Correlation report generated."),
        ]
    )

    answer = llm.run("Create a correlation report")
    assert "correlation" in answer.lower()

    result = get_tool_result(llm, "correlation_report")
    assert result["method"] == "pearson"
    assert set(result["columns"]) == {"amt", "city_pop", "is_fraud"}
    matrix = result["correlation_matrix"]
    # Diagonal of a correlation matrix must be 1.0
    for col in result["columns"]:
        assert matrix[col][col] == pytest.approx(1.0)


def test_run_categorical_summary(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("categorical_summary", {"top_n": 5}),
            make_final_response("Categorical summary generated."),
        ]
    )

    answer = llm.run("Summarize categorical columns")
    assert "categorical" in answer.lower() or "summary" in answer.lower()

    result = get_tool_result(llm, "categorical_summary")
    assert result["top_n"] == 5
    summary = result["summary"]
    assert "merchant" in summary and "category" in summary
    # "a" appears twice in merchant → most frequent value
    assert summary["merchant"]["a"] == 2
    # "food" appears twice in category → most frequent value
    assert summary["category"]["food"] == 2


def test_run_duplicate_report(llm, sample_df):
    llm.loaded_datasets["Fraud detection"] = sample_df
    llm.current_dataset_name = "Fraud detection"

    llm.client = DummyClient(
        [
            make_tool_call_response("duplicate_report", {}),
            make_final_response("Duplicate report generated."),
        ]
    )

    answer = llm.run("Check duplicates")
    assert "duplicate" in answer.lower()

    result = get_tool_result(llm, "duplicate_report")
    assert result["duplicate_rows"] == 0
    assert result["duplicate_percent"] == pytest.approx(0.0)


def test_run_fails_if_dataset_tool_called_without_loaded_dataset(llm):
    llm.client = DummyClient(
        [
            make_tool_call_response("get_info", {}),
            make_final_response("No dataset loaded."),
        ]
    )

    answer = llm.run("Give me information on the dataset")
    assert isinstance(answer, str)


def test_run_returns_reached_max_steps_if_model_never_finishes(llm):
    llm.client = DummyClient(
        [
            make_tool_call_response("get_info", {}),
        ] * llm.max_steps
    )

    llm.loaded_datasets["Fraud detection"] = pd.DataFrame({"a": [1, 2, 3]})
    llm.current_dataset_name = "Fraud detection"

    answer = llm.run("Loop forever")
    assert answer == "Reached max steps."