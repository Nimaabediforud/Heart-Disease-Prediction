from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib


class trainer:
    """
    Final trainer class for building a stacking classifier (or any chosen model)
    inside a preprocessing + model pipeline.
    """

    def __init__(self, random_state=42, max_iter=1000):
        self.random_state = random_state
        self.max_iter = max_iter

    def train(self, preprocessor, X_train, y_train, model=None, save_path=None):
        """
        Train the final model inside a pipeline and optionally save it.

        Parameters
        ----------
        preprocessor : ColumnTransformer
            Already fitted on training data.
        X_train, y_train : array-like
            Training features and labels.
        model : estimator, optional
            If None, uses the default stacking ensemble.
        save_path : str, optional
            Path to save the trained pipeline (joblib).

        Returns
        -------
        pipeline : Pipeline
            Fitted pipeline (preprocessor + model).
        """
        if model is None:
            # Default stacking ensemble
            model = StackingClassifier(
                estimators=[
                    ("RF", RandomForestClassifier(random_state=self.random_state)),
                    ("GB", GradientBoostingClassifier(random_state=self.random_state)),
                    ("ADA", AdaBoostClassifier(random_state=self.random_state)),
                    ("ET", ExtraTreesClassifier(random_state=self.random_state)),
                    ("SVC", SVC(probability=True, random_state=self.random_state))
                ],
                final_estimator=LogisticRegression(
                    max_iter=self.max_iter, random_state=self.random_state
                )
            )

        pipeline = Pipeline([
            ("preproc", preprocessor),
            ("model", clone(model))
        ])
        pipeline.fit(X_train, y_train)

        if save_path:
            joblib.dump(pipeline, save_path)

        return pipeline

    def evaluate(self, pipeline_or_path, X_val, y_val):
        """
        Evaluate a trained pipeline (or load from disk) on validation data.

        Returns
        -------
        metrics : dict
            F1 (weighted), classification report, confusion matrix.
        """
        if isinstance(pipeline_or_path, str):
            pipe = joblib.load(pipeline_or_path)
        else:
            pipe = pipeline_or_path

        y_pred = pipe.predict(X_val)

        metrics = {
            'f1_score_weighted': f1_score(y_val, y_pred, average='weighted'),
            'classification_report': classification_report(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred)
        }
        print(f"F1-Score (weighted): {round(metrics['f1_score_weighted']*100,3)}%")
        print(f"Classification Report:\n{metrics['classification_report']}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        return metrics