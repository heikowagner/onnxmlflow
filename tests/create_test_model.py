"""Creates a simple test ONNX model (Iris multi-class classifier).

Run directly:
    python tests/create_test_model.py
Or call create_iris_model() from test fixtures.
"""
from __future__ import annotations

import os


def create_iris_model(output_path: str = "tests/fixtures/iris_classifier.onnx") -> str:
    """Train a LogisticRegression on Iris data and export it as ONNX.

    Parameters
    ----------
    output_path:
        Destination ``.onnx`` file path.

    Returns
    -------
    str
        The *output_path* that was written.
    """
    try:
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise ImportError(
            "Install dev extras to create test models:\n"
            "  pip install 'onnxmlflow[dev]'"
        ) from exc

    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=500, multi_class="ovr")
    clf.fit(X_train, y_train)

    initial_types = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_types)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())

    print(f"ONNX model saved → {output_path}")
    return output_path


if __name__ == "__main__":
    create_iris_model()
