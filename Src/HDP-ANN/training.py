from sklearn.metrics import f1_score, classification_report, confusion_matrix
import tensorflow as tf



class trainer:
    """
    Trainer class for building, training, and evaluating an artificial neural network (ANN)
    for binary classification tasks, specifically predicting heart disease in this project.

    Attributes:
    -----------
    dropout : float
        Dropout rate applied after the hidden layers to reduce overfitting (default: 0.3).
    decision_threshold : float
        Threshold for converting predicted probabilities into binary labels (default: 0.45).
    """

    def __init__(self, dropout=0.3, decision_threshold=0.45):
        self.dropout = dropout
        self.decision_threshold = decision_threshold

    
    def train(self, final_X_train, final_X_val, y_train, y_val):
        """
        Build, compile, and train the artificial neural network on the training data.
        
        The network consists of:
        - Input layer matching the number of features.
        - Two hidden layers (16 units, then 8 units) with ReLU activation.
        - Dropout layer to reduce overfitting.
        - Output layer with 1 unit and sigmoid activation for binary classification.

        Early stopping is applied to monitor validation loss with patience of 5 epochs.

        Parameters:
        -----------
        final_X_train : np.ndarray
            Training features as a NumPy array.
        final_X_val : np.ndarray
            Validation features as a NumPy array.
        y_train : np.ndarray
            Training labels.
        y_val : np.ndarray
            Validation labels.

        Returns:
        --------
        model : tf.keras.Model
            Trained Keras model.
        history : tf.keras.callbacks.History
            Training history containing loss and accuracy metrics per epoch.
        """
        # Create neural networks architecture 
        model = tf.keras.models.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(16,)),
            # First hidden layer with 16 units
            tf.keras.layers.Dense(16, activation='relu'),
            # Second hidden layer with 8 units 
            tf.keras.layers.Dense(8, activation='relu'),
            # Set dropout to prevent overfitting
            tf.keras.layers.Dropout(self.dropout),
            # Third hidden layer with 1 unit (output layer)
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Stop early if validation loss doesn’t improve for 5 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=5,
                                                        restore_best_weights=True
                                                        )

        # Fit
        history = model.fit(
            final_X_train,
            y_train,
            epochs=30,
            validation_data=(final_X_val, y_val),
            callbacks=[early_stop]
        )

        # Evaluate during training
        model.evaluate(final_X_val, y_val, verbose=2)

        return model, history
    
    def evaluate(self, model_path, final_X_val, y_val):
        """
        Load a saved ANN model and evaluate its performance on the validation set.

        Predictions are converted to binary labels based on `decision_threshold`.
        The evaluation metrics returned include:
        - Weighted F1 score
        - Classification report
        - Confusion matrix

        Parameters:
        -----------
        model_path : str
            Path to the saved Keras model file.
        final_X_val : np.ndarray
            Validation features as a NumPy array.
        y_val : np.ndarray
            Validation labels.

        Returns:
        --------
        metrics : dict
            Dictionary containing:
            - 'f1_score_weighted': Weighted F1 score (float)
            - 'classification_report': Classification report as string
            - 'confusion_matrix': Confusion matrix as np.ndarray
        """
        # Load & Initialize the model
        model = tf.keras.models.load_model(model_path)

        # Predict
        y_pred_probs = model.predict(final_X_val)

        # Convert probabilities into binary labels
        y_pred = (y_pred_probs > self.decision_threshold).astype("int32").flatten()

        # Compute metrics
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        clf_report = classification_report(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

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

