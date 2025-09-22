from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


class trainer:
    """
    Trainer class for building and evaluating a stacking ensemble model.

    This class focuses on training a StackingClassifier using strong candidate
    models and evaluating its performance using standard classification metrics.

    Attributes:
    -----------
    random_state : int
        Random seed for reproducibility of model results.
    max_iter : int
        Maximum number of iterations for the LogisticRegression final estimator.
    """

    def __init__(self, random_state=42, max_iter=1000):
        """
        Initializes the trainer with random state and max iterations.

        Parameters:
        -----------
        random_state : int, optional
            Seed for reproducibility. Default is 42.
        max_iter : int, optional
            Maximum iterations for LogisticRegression. Default is 1000.
        """
        self.random_state = random_state
        self.max_iter = max_iter


    def train(self, final_X_train, y_train):
        """
        Train a StackingClassifier on the training data.

        The stacking classifier combines:
        - RandomForestClassifier
        - GradientBoostingClassifier
        - AdaBoostClassifier
        - ExtraTreesClassifier
        - SVC (with probability=True)
        
        The final estimator is LogisticRegression.

        Parameters:
        -----------
        final_X_train : pd.DataFrame
            Preprocessed training features.
        y_train : pd.Series
            Training labels.

        Returns:
        --------
        model : StackingClassifier
            Trained stacking classifier.
        """

        # ----- Stacking Classifier -----
        stacking_model = StackingClassifier(
            estimators=[
                ("RF", RandomForestClassifier(random_state=self.random_state)),
                ("GB", GradientBoostingClassifier(random_state=self.random_state)),
                ("ADA", AdaBoostClassifier(random_state=self.random_state)),
                ("ET", ExtraTreesClassifier(random_state=self.random_state)),
                ("SVC", SVC(probability=True, random_state=self.random_state))
            ],
            final_estimator= LogisticRegression(max_iter=self.max_iter,random_state=self.random_state)
        )

        # Fit
        model = stacking_model.fit(final_X_train, y_train)

        return model
    

    def evaluate(self, model_path, final_X_val, y_val):
        """
        Evaluate a saved model on validation data and return key metrics.

        Parameters:
        -----------
        model_path : str
            Path to the saved model file (e.g., using joblib).
        final_X_val : pd.DataFrame
            Preprocessed validation features.
        y_val : pd.Series
            Validation labels.

        Returns:
        --------
        metrics : dict
            Dictionary containing:
            - 'f1_score_weighted': weighted F1-score
            - 'classification_report': detailed classification report
            - 'confusion_matrix': confusion matrix
        """

        # Load & Initialize the model
        model = joblib.load(model_path)

        # Predict
        stacking_y_pred = model.predict(final_X_val)

        # Compute metrics
        f1_weighted = f1_score(y_val, stacking_y_pred, average='weighted')
        clf_report = classification_report(y_val, stacking_y_pred)
        cm = confusion_matrix(y_val, stacking_y_pred)

        # Print metrics
        print(f"F1-Score (weighted): {round(f1_weighted*100,3)}%")
        print(f"Classification Report:\n{clf_report}")
        print(f"Confusion Matrix:\n{cm}")

        # Return metrics as dictionary
        metrics = {
            'f1_score_weighted': f1_weighted,
            'classification_report': clf_report,
            'confusion_matrix': cm
        }

        return metrics

