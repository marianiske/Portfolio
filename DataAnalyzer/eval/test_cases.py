BASIC_CASES = [
    {
        "name": "list_datasets",
        "prompt": "Which datasets are available?",
        "expected_tools": ["list_datasets"],
        "forbidden_tools": ["find_dataset"],
        "must_not_hallucinate": True,
        "must_contain": ["Fraud detection", "House prices", "Productivity"]
    },
    {
        "name": "load_fraud_dataset",
        "prompt": "Load Fraud dataset",
        "expected_tools": ["find_dataset"],
        "expected_dataset": "Fraud detection",
        "must_load_dataset": True,
        "must_not_hallucinate": True,
        "forbidden_answer_terms": ["House prices", "Productivity"],
        "must_contain": []
    },
    {
        "name": "get_info_after_load",
        "prompt": "Give me information on the dataset",
        "expected_tools": ["get_info"],
        "expected_dataset": "Fraud detection",
        "must_not_hallucinate": True,
        "must_contain": [],
        "forbidden_answer_terms": ["House prices", "Productivity"],
        "must_not_dump_raw_json": True,
    },
    {
        "name": "describe_dataset",
        "prompt": "Describe the dataset",
        "expected_tools": ["describe_dataset"],
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "mean_amt",
        "prompt": "Calculate the mean of the column amt",
        "expected_tools": ["calc_mean_column"],
        "expected_args": {"col": "amt"},
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "scatter_amt_city_pop",
        "prompt": "Create a scatter plot of amt against city_pop",
        "expected_tools": ["scatter_plot"],
        "expected_args": {"col1": "amt", "col2": "city_pop"},
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "pca_feature_selection",
        "prompt": "Perform PCA feature selection with 2 components",
        "expected_tools": ["pca_feature_selection"],
        "expected_args": {"n_features": 2},
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "filter_outliers_default",
        "prompt": "Filter outliers in the dataset",
        "expected_tools": ["filter_outliers"],
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "filter_outliers_threshold",
        "prompt": "Filter outliers with threshold 3.5",
        "expected_tools": ["filter_outliers"],
        "expected_args": {"threshold": 3.5},
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "normalize_dataset",
        "prompt": "Normalize the dataset",
        "expected_tools": ["normalize"],
        "expected_dataset": "Fraud detection",
        "must_not_hallucinate": True,
        "must_contain": [],
        "forbidden_answer_terms": ["failed", "error"],
    },
    {
        "name": "missing_values_report",
        "prompt": "Create a missing values report",
        "expected_tools": ["missing_values_report"],
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "column_type_report",
        "prompt": "Create a column type report",
        "expected_tools": ["column_type_report"],
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "correlation_report_default",
        "prompt": "Create a correlation report",
        "expected_tools": ["correlation_report"],
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "correlation_report_spearman",
        "prompt": "Create a Spearman correlation report",
        "expected_tools": ["correlation_report"],
        "expected_args": {"method": "spearman"},
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "categorical_summary_default",
        "prompt": "Summarize the categorical columns",
        "expected_tools": ["categorical_summary"],
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "categorical_summary_top_5",
        "prompt": "Summarize the categorical columns with top 5 values",
        "expected_tools": ["categorical_summary"],
        "expected_args": {"top_n": 5},
        "must_not_hallucinate": True,
        "must_contain": []
    },
    {
        "name": "duplicate_report",
        "prompt": "Check for duplicate rows",
        "expected_tools": ["duplicate_report"],
        "must_not_hallucinate": True,
        "must_contain": []
    },
]