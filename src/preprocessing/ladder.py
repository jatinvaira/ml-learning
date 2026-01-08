from typing import Dict, Any, List
import copy

class LadderBuilder:
    """
    Builds a cumulative ladder of preprocessing strategies.

    Structure:
    rung_0: baseline (minimal)
    rung_1: rung_0 + step_1
    rung_2: rung_1 + step_2
    ...
    """
    def __init__(self, ladder_config: List[Dict[str, Any]], base_config: Dict[str, Any]):
        """
        Args:
            ladder_config: List of steps to add sequentially.
            base_config: The starting strategy configuration (rung 0).
        """
        self.ladder_config = ladder_config
        self.base_config = base_config

    def build_strategies(self) -> Dict[str, Dict[str, Any]]:
        strategies = {}

        # Rung 0: Baseline
        current_config = copy.deepcopy(self.base_config)
        strategies["rung_0"] = self._finalize_config(current_config, 0)

        for i, step in enumerate(self.ladder_config):
            rung_id = f"rung_{i+1}"

            # Apply step to current config
            current_config = self._apply_step(current_config, step)

            # Finalize (add complexity metric)
            strategies[rung_id] = self._finalize_config(current_config, i + 1)

        return strategies

    def _apply_step(self, config: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies a single step (e.g., "impute", "encode") to the config.
        We assume 'numerical' and 'categorical' sections in 'preprocessing'.
        """
        new_config = copy.deepcopy(config)
        step_type = step["type"]

        if step_type == "impute":
            # Add imputer to both num and cat if they exist
            self._add_step_to_section(new_config, "numerical", {"name": "imputer", "type": "simple_imputer", "args": step.get("numerical_args", {"strategy": "median"})})
            self._add_step_to_section(new_config, "categorical", {"name": "imputer", "type": "simple_imputer", "args": step.get("categorical_args", {"strategy": "most_frequent"})})

        elif step_type == "encode":
            # Add encoder to categorical
            encoder_type = step.get("encoder_type", "one_hot")
            self._add_step_to_section(new_config, "categorical", {"name": "encoder", "type": encoder_type, "args": step.get("args", {})})

        elif step_type == "scale":
            # Add scaler to numerical
            scaler_type = step.get("scaler_type", "standard_scaler")
            self._add_step_to_section(new_config, "numerical", {"name": "scaler", "type": scaler_type, "args": step.get("args", {})})

        elif step_type == "clip":
            # Add outlier clipper to numerical
            self._add_step_to_section(new_config, "numerical", {"name": "clipper", "type": "outlier_clipper", "args": step.get("args", {})})

        elif step_type == "feature_select":
            # This would likely be a separate component in the pipeline, typically after preprocessor.
            # However, our 'create_preprocessor' returns a ColumnTransformer.
            # Feature selection usually happens *after* ColumnTransformer.
            # The current 'create_preprocessor' only handles column transformations.
            # To strictly follow the "one preprocessor object" design, we might need to rely on the fact
            # that we can't easily add feature selection inside the ColumnTransformer unless we wrap it.
            # For this exercise, if 'feature_select' is requested, we might mark it in the config
            # and handle it in build_full_pipeline, OR (simpler) skip it for the preprocessor
            # and assume the user's "pipeline" definition will handle it.
            # BUT, the task says "Extend pipeline...".
            # Let's add a metadata flag to the config that the ExperimentRunner can use
            # to wrap the preprocessor or add a step.
            new_config["feature_selection"] = step.get("args", {"method": "variance_threshold"})

        elif step_type == "resample":
            # Similar to feature selection, resampling happens on X, y.
            # Usually handled by imblearn pipeline or manually.
            new_config["resampling"] = step.get("args", {"method": "smote"})

        return new_config

    def _add_step_to_section(self, config, section, step_def):
        if "preprocessing" not in config:
            config["preprocessing"] = {}
        if section not in config["preprocessing"]:
            config["preprocessing"][section] = {"features": [], "steps": []}

        # Ensure 'steps' list exists if it was initialized differently (e.g., just features)
        if "steps" not in config["preprocessing"][section]:
             config["preprocessing"][section]["steps"] = []

        # Append step
        config["preprocessing"][section]["steps"].append(step_def)

    def _finalize_config(self, config: Dict[str, Any], complexity: int) -> Dict[str, Any]:
        """Adds complexity metadata."""
        config["complexity_steps"] = complexity
        return config
