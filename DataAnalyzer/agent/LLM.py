from ollama import Client
from typing import Any
import pandas as pd
from .ToolRegistry import ToolRegistry
import json
import difflib
import inspect

from .helpers import (
    list_datasets,
    find_dataset,
    load_dataset_by_name,
    get_info,
    describe_dataset,
    calc_mean_column,
    scatter_plot,
    pca_feature_selection,
    filter_outliers,
    normalize,
    missing_values_report,
    column_type_report,
    correlation_report,
    categorical_summary,
    duplicate_report,
)

class LLM():
    def __init__(self, model: str = "functiongemma",
        host: str = "http://localhost:11434",
        max_steps: int = 8,
    ):
        self.client = Client(host=host)
        self.model = model
        self.client = Client(host="http://localhost:11434")
        self.registry = ToolRegistry()
        self.max_steps = max_steps

        self.loaded_datasets: dict[str, pd.DataFrame] = {}
        self.current_dataset_name: str | None = None
        
        self.dataset_tools = {
            "get_info": get_info,
            "describe_dataset": describe_dataset,
            "calc_mean_column": calc_mean_column,
            "scatter_plot": scatter_plot,
            "pca_feature_selection": pca_feature_selection,
            "filter_outliers": filter_outliers,
            "normalize": normalize,
            "missing_values_report": missing_values_report,
            "column_type_report": column_type_report,
            "correlation_report": correlation_report,
            "categorical_summary": categorical_summary,
            "duplicate_report": duplicate_report,
        }
        
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                        "You are a data analyst. "
                        "Use tools whenever possible. "
                        "Only use tool results. "
                        "If the user asks to load a dataset by topic or approximate name, "
                        "call find_dataset immediately. "
                        "If find_dataset returns a match, only talk about the matched dataset. "
                        "Do not mention any other dataset names in the answer. "
                        "Do not call list_datasets unless the user explicitly asks to see the available datasets. "
                        "If a dataset is already loaded and the user asks for information, details, summary, columns, or schema, "
                        "call get_info. "
                        "Do not answer with raw JSON unless explicitly requested. "
                        "If the user's request requires multiple steps (e.g. load a dataset and then analyze it), "
                        "call all required tools in sequence before giving a final answer. "
                        "Do not stop after the first tool call if there are more steps to complete. "
                        "Only give a final text answer once all requested operations have been completed. "
                        "Do not mention results from earlier conversation turns or unrelated tools. "
                    ),
            }
        ]
        
        self.registry.register(
            name="list_datasets",
            description="List all available datasets that can be loaded by the agent.",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=list_datasets
        )
        
        self.registry.register(
            name="find_dataset",
            description=(
                    "Find the best matching dataset for the user request. "
                    "Always set the argument 'query' to the full last user message exactly as written. "
                ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                            "type": "string",
                            "description": "Copy the user's request verbatim."
                        }
                },
                "required": ["query"]
            },
            func=find_dataset
        )
        
        self.registry.register(
            name="get_info",
            description=(
                "Get information about the currently loaded dataset. "
                "Use this when the user asks for information, details, overview, shape, columns, schema, or summary "
                "of the dataset that is already loaded."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=lambda: {}
        )
        
        self.registry.register(
            name="describe_dataset",
            description=(
                   "Get descriptive statistics for the currently loaded dataset. "
                   "Use this when the user asks to describe the dataset statistically. "
                   "This includes summary statistics such as count, mean, std, min, max, and category summaries. "
                   "Do not use this for a simple overview of rows, columns, or dtypes."
               ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=lambda: {}
        )

        self.registry.register(
            name="calc_mean_column",
            description="Calculate the mean of a numeric column in the currently loaded dataset.",
            parameters={
                "type": "object",
                "properties": {
                    "col": {
                        "type": "string",
                        "description": "Column name."
                    }
                },
                "required": ["col"]
            },
            func=lambda col: {"col": col}
        )

        self.registry.register(
            name="scatter_plot",
            description="Create a scatter plot for two columns in the currently loaded dataset.",
            parameters={
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "X-axis column."},
                    "col2": {"type": "string", "description": "Y-axis column."},
                    "plot_description": {"type": "string", "description": "Optional title."},
                    "linewidths": {"type": "number", "description": "Marker linewidth."}
                },
                "required": ["col1", "col2"]
            },
            func=lambda **kwargs: kwargs
        )

        self.registry.register(
            name="pca_feature_selection",
            description=(
                "Perform PCA-based dimensionality reduction on the numeric columns of the currently loaded dataset. "
                "Use this when the user asks for PCA, principal components, dimensionality reduction, or feature selection using PCA."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n_features": {
                        "type": "integer",
                        "description": "Number of principal components to compute."
                    }
                },
                "required": []
            },
            func=lambda **kwargs: kwargs
        )

        self.registry.register(
            name="filter_outliers",
            description=(
                "Filter outliers from numeric columns in the currently loaded dataset using a robust z-score method. "
                "Use this when the user asks to remove or filter outliers. "
                "This returns how many rows were removed and a preview of the filtered data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Optional robust z-score threshold. Default is 3.5."
                    }
                },
                "required": []
            },
            func=lambda **kwargs: kwargs
        )

        self.registry.register(
            name="normalize",
            description=(
                "Normalize all numeric columns in the currently loaded dataset using z-score normalization. "
                "Use this when the user asks to normalize or standardize the dataset."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=lambda: {}
        )

        self.registry.register(
            name="missing_values_report",
            description=(
                "Create a report of missing values for each column in the currently loaded dataset. "
                "Use this when the user asks about missing values, nulls, NaNs, or completeness."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=lambda: {}
        )

        self.registry.register(
            name="column_type_report",
            description=(
                "Create a report of column types for the currently loaded dataset. "
                "Use this when the user asks about dtypes, column types, unique values, or example values per column. "
                "Do not use this for missing values."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=lambda: {}
        )
        self.registry.register(
            name="correlation_report",
            description=(
                "Create a correlation report for numeric columns in the currently loaded dataset. "
                "Use this when the user asks for correlations or a correlation matrix. "
                "If the user specifies Pearson, Spearman, or Kendall, set 'method' accordingly."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Correlation method: pearson, spearman, or kendall."
                    }
                },
                "required": []
            },
            func=lambda **kwargs: kwargs
        )

        self.registry.register(
            name="categorical_summary",
            description=(
                "Summarize categorical columns in the currently loaded dataset using top value counts. "
                "Use this when the user asks for a summary of categorical columns, frequent categories, or top values per category. "
                "Do not use this for dtypes or missing values."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "How many top values to include per categorical column."
                    }
                },
                "required": []
            },
            func=lambda **kwargs: kwargs
        )

        self.registry.register(
            name="duplicate_report",
            description=(
                "Report duplicate rows in the currently loaded dataset. "
                "Use this when the user asks about duplicate rows or duplicate records."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            func=lambda: {}
        )
        
        self.last_run_trace = {
            "user_query": '',
            "tool_calls": [],
            "tool_results": [],
            "final_answer": None,
            "errors": [],
        }
    

    def find_best_tool_match(self, tool_name: str, allowed_tool_names: set[str] | None = None) -> str:
        available_tools = set(self.dataset_tools.keys()) | {"list_datasets", "find_dataset"}
    
        if allowed_tool_names is not None:
            available_tools &= set(allowed_tool_names)
    
        if not tool_name:
            if allowed_tool_names is not None and len(allowed_tool_names) == 1:
                return next(iter(allowed_tool_names))
            raise ValueError("Empty tool name received.")
    
        if tool_name in available_tools:
            return tool_name
    
        matches = difflib.get_close_matches(tool_name, list(available_tools), n=1, cutoff=0.75)
        if matches:
            return matches[0]
    
        raise ValueError(f"Unknown tool: {tool_name}")
    
    def reset_conversation(self):
        system_message = self.messages[0]
        self.messages = [system_message]
        if self.current_dataset_name:
            self.messages.append({
                "role": "assistant",
                "content": f"The dataset '{self.current_dataset_name}' is currently loaded and ready for analysis.",
            })

    def run(self, user_query: str) -> str:

        self.last_run_trace = {
            "user_query": user_query,
            "tool_calls": [],
            "tool_results": [],
            "final_answer": None,
            "errors": [],
        }

        self.messages.append({
            "role": "user",
            "content": user_query,
        })

        for _ in range(self.max_steps):
            tools_for_turn = self._select_tools_for_query(user_query)
            allowed_tool_names = {tool["function"]["name"] for tool in tools_for_turn}
            #print(allowed_tool_names)
            
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=tools_for_turn,
                stream=False,
            )

            message = response["message"]
            self.messages.append(message)

            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                final_answer = message.get("content", "").strip()
                self.last_run_trace["final_answer"] = final_answer
                return final_answer

            for tool_call in tool_calls:
                try:
                    raw_fn = tool_call["function"].get("name", "")
                    arguments = tool_call["function"].get("arguments", {})
                    fn = self.find_best_tool_match(raw_fn, allowed_tool_names)
                    
                    self.last_run_trace["tool_calls"].append({
                        "name": fn,
                        "arguments": arguments,
                    })
                    
                    if fn == "find_dataset":
                        result = self.registry.execute(fn, arguments)
                    
                        if not result.get("match_found"):
                            result_for_model = result
                        else:
                            dataset_name = result["dataset_name"]
                            df = load_dataset_by_name(dataset_name)
                    
                            self.loaded_datasets[dataset_name] = df
                            self.current_dataset_name = dataset_name
                    
                            result_for_model = {
                                "match_found": True,
                                "dataset_name": dataset_name,
                                "loaded": True,
                                "rows": int(df.shape[0]),
                                "columns": int(df.shape[1]),
                                "column_names": list(df.columns),
                                "message": f'The dataset "{dataset_name}" was loaded successfully. If the user requested additional operations, continue calling the necessary tools now.'
                            }
                    elif fn in self.dataset_tools.keys():
                        
                        if self.current_dataset_name is None:
                            raise ValueError("No dataset is currently loaded.")
                            
                        df = pd.DataFrame(self.loaded_datasets[self.current_dataset_name])
                        dataset_func = self.dataset_tools[fn]

                        sig = inspect.signature(dataset_func)
                        valid_params = set(sig.parameters.keys()) - {"df"}
                        filtered_args = {k: v for k, v in arguments.items() if k in valid_params}

                        result_for_model = dataset_func(df, **filtered_args)
                    
                        if isinstance(result_for_model, dict):
                            result_for_model = {
                                "dataset_name": self.current_dataset_name,
                                **result_for_model
                            }
                    else:
                        result = self.registry.execute(fn, arguments)
                        result_for_model = result
                        
                    tool_message = {
                        "role": "tool",
                        "name": fn,
                        "content": json.dumps(result_for_model, ensure_ascii=False),
                    }

                    self.last_run_trace["tool_results"].append({
                        "name": fn,
                        "result": result_for_model,
                    })
                except Exception as e:
                    tool_message = {
                        "role": "tool",
                        "name": raw_fn,
                        "content": json.dumps(
                            {"error": str(e)},
                            ensure_ascii=False,
                        ),
                    }
                    
                    self.last_run_trace["errors"].append(str(e))

                self.messages.append(tool_message)
                
            self.last_run_trace["final_answer"] = message.get("content", "").strip()
                
        return "Reached max steps."       
    
    def _select_tools_for_query(self, user_query: str):
        all_tools = self.registry.ollama_tools()

        if self.current_dataset_name is None:
            listing_intent = any(
                kw in user_query.lower()
                for kw in ("list", "show", "available", "which dataset", "what dataset")
            )
            nav_name = "list_datasets" if listing_intent else "find_dataset"
            nav_tools = [t for t in all_tools if t["function"]["name"] == nav_name]
            return nav_tools or all_tools

        tool_texts = [
            f"{t['function']['name'].replace('_', ' ')}: {t['function']['description']}"
            for t in all_tools
        ]

        try:
            import numpy as np
            response = self.client.embed(
                model=self.model,
                input=tool_texts + [user_query],
            )
            embeddings = np.array(response["embeddings"])
            tool_embs = embeddings[:-1]
            query_emb = embeddings[-1]

            denom = (
                np.linalg.norm(tool_embs, axis=1) * np.linalg.norm(query_emb)
                + 1e-9
            )
            sims = np.dot(tool_embs, query_emb) / denom

            best = float(sims.max())
            threshold = max(best * 0.75, 0.25)
            selected = [t for t, s in zip(all_tools, sims) if float(s) >= threshold]
            if selected:
                return selected
        except Exception:
            pass


        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            corpus = tool_texts + [user_query]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            matrix = vec.fit_transform(corpus)

            sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
            best = float(sims.max())
            if best >= 0.1:
                threshold = best * 0.75
                selected = [t for t, s in zip(all_tools, sims) if s >= threshold]
                if selected:
                    return selected
        except Exception:
            pass

        return all_tools