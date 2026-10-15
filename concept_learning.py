"""
concept_learning.py
===================
Implementation of the Find-S (Find-Specific) Concept Learning Algorithm
for Student Performance Prediction.

This module provides:
1. Configurable numeric-to-categorical feature transformation supporting
   attendance, internal assessment, study hours, assignment completion,
   and previous examination marks.
2. The Find-S algorithm implementation with step-by-step hypothesis evolution.
3. Handling of edge cases such as datasets with zero positive examples.
4. Clear explanations of the theoretical limitations of Find-S.
5. Synthetic demonstration training dataset.
6. Pedagogical academic suggestions tailored to student strengths and deficits.
"""

from typing import Dict, List, Tuple, Any, Optional

# =====================================================================
# 1. FEATURE DEFINITIONS & CONFIGURABLE THRESHOLDS
# =====================================================================

# Categorical features used by the Find-S concept learning model
FEATURE_NAMES = [
    "Attendance",
    "Internal Assessment",
    "Study Hours",
    "Assignment Completion",
    "Previous Examination Marks"
]

# Configurable cutoff thresholds used to convert numeric inputs into "High" or "Low"
# These can be customized or overridden by passing a dictionary to convert_numeric_to_categorical.
DEFAULT_THRESHOLDS = {
    # Attendance percentage (0 to 100): 75.0% and above is categorized as "High"
    "attendance": 75.0,
    
    # Internal assessment marks (0 to 50): 30.0 marks and above is categorized as "High"
    "internal_assessment": 30.0,

    # Daily study hours (0 to 24): 3.0 hours/day and above is categorized as "High"
    "study_hours": 3.0,

    # Assignment completion percentage (0 to 100): 75.0% and above is categorized as "High"
    "assignment_completion": 75.0,

    # Previous examination marks (0 to 100): 60.0% and above is categorized as "High"
    "previous_exam_marks": 60.0
}

# Target concept labels
POSITIVE_LABEL = "Good Performer"
NEGATIVE_LABEL = "Needs Academic Support"


# =====================================================================
# 2. NUMERIC TO CATEGORICAL CONVERSION
# =====================================================================

def convert_numeric_to_categorical(
    attendance: float,
    internal_assessment: float,
    study_hours: float,
    assignment_completion: float,
    previous_exam_marks: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, str]:
    """
    Converts raw numeric student metrics into binary categorical values ("High" or "Low")
    expected by the Find-S concept learning model.

    Parameters:
      attendance (float): Attendance percentage (0 - 100).
      internal_assessment (float): Internal marks (0 - 50).
      study_hours (float): Daily study hours (0 - 24).
      assignment_completion (float): Assignment completion percentage (0 - 100).
      previous_exam_marks (float, optional): Previous examination marks (0 - 100).
      thresholds (dict, optional): Custom cutoff thresholds for each feature.

    Returns:
      dict: A dictionary mapping each feature name to "High" or "Low".
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Retrieve threshold benchmarks
    att_bench = thresholds.get("attendance", DEFAULT_THRESHOLDS["attendance"])
    ia_bench = thresholds.get("internal_assessment", DEFAULT_THRESHOLDS["internal_assessment"])
    sh_bench = thresholds.get("study_hours", DEFAULT_THRESHOLDS["study_hours"])
    ac_bench = thresholds.get("assignment_completion", DEFAULT_THRESHOLDS["assignment_completion"])

    # Discretize core features: "High" if value >= threshold, otherwise "Low"
    categorical = {
        "Attendance": "High" if float(attendance) >= att_bench else "Low",
        "Internal Assessment": "High" if float(internal_assessment) >= ia_bench else "Low",
        "Assignment Completion": "High" if float(assignment_completion) >= ac_bench else "Low",
        "Study Hours": "High" if float(study_hours) >= sh_bench else "Low"
    }

    # Include previous examination marks if provided
    if previous_exam_marks is not None:
        pm_bench = thresholds.get("previous_exam_marks", DEFAULT_THRESHOLDS.get("previous_exam_marks", 60.0))
        categorical["Previous Examination Marks"] = "High" if float(previous_exam_marks) >= pm_bench else "Low"

    return categorical


# =====================================================================
# 3. SYNTHETIC DEMONSTRATION TRAINING DATASET
# =====================================================================

def get_sample_training_data() -> List[Dict[str, Any]]:
    """
    Returns a clearly labelled synthetic sample dataset containing both
    positive and negative student examples.

    NOTE ON DATASET:
    This dataset is synthetic and designed solely for academic demonstration
    of the Find-S concept learning mechanics in machine learning coursework.
    """
    return [
        {
            "student_id": "STU-001",
            "Attendance": "High",
            "Internal Assessment": "High",
            "Assignment Completion": "High",
            "Study Hours": "High",
            "Previous Examination Marks": "High",
            "label": POSITIVE_LABEL  # Positive Example 1: Initializes null hypothesis
        },
        {
            "student_id": "STU-002",
            "Attendance": "High",
            "Internal Assessment": "High",
            "Assignment Completion": "High",
            "Study Hours": "Low",
            "Previous Examination Marks": "High",
            "label": POSITIVE_LABEL  # Positive Example 2: Study Hours varies -> generalizes Study Hours to '?'
        },
        {
            "student_id": "STU-003",
            "Attendance": "Low",
            "Internal Assessment": "Low",
            "Assignment Completion": "Low",
            "Study Hours": "Low",
            "Previous Examination Marks": "Low",
            "label": NEGATIVE_LABEL  # Negative Example (Find-S ignores this)
        },
        {
            "student_id": "STU-004",
            "Attendance": "High",
            "Internal Assessment": "Low",
            "Assignment Completion": "Low",
            "Study Hours": "High",
            "Previous Examination Marks": "Low",
            "label": NEGATIVE_LABEL  # Negative Example (Find-S ignores this)
        },
        {
            "student_id": "STU-005",
            "Attendance": "High",
            "Internal Assessment": "High",
            "Assignment Completion": "High",
            "Study Hours": "High",
            "Previous Examination Marks": "High",
            "label": POSITIVE_LABEL  # Positive Example 3: Re-confirms existing constraints
        },
        {
            "student_id": "STU-006",
            "Attendance": "Low",
            "Internal Assessment": "High",
            "Assignment Completion": "High",
            "Study Hours": "High",
            "Previous Examination Marks": "High",
            "label": NEGATIVE_LABEL  # Negative Example: Low attendance causes support label (ignored by Find-S)
        }
    ]


# =====================================================================
# 4. FIND-S CONCEPT LEARNER CLASS
# =====================================================================

class FindSConceptLearner:
    """
    Find-S (Find-Specific) Concept Learning Algorithm.

    Theoretical foundation:
    Find-S begins with the most specific hypothesis in the hypothesis space:
        h = < 'Ø', 'Ø', 'Ø', 'Ø', 'Ø' >  ('Ø' = null / rejects all instances)
    
    Upon observing each POSITIVE training example:
    - If the current constraint is 'Ø', replace it with the observed feature value.
    - If the current constraint differs from the observed feature value,
      generalize it to '?' ('?' = wildcard / accepts any value).
    - If the constraint already matches the observed value or is '?', keep it unchanged.
    
    NEGATIVE examples are deliberately ignored by Find-S.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_NAMES
        # Initialize hypothesis to the most specific hypothesis: all 'Ø' (null)
        self.hypothesis: List[str] = ["Ø"] * len(self.feature_names)
        self.training_steps: List[Dict[str, Any]] = []
        self.has_positive_examples: bool = False
        self.is_trained: bool = False

    def fit(self, training_data: List[Dict[str, Any]]) -> "FindSConceptLearner":
        """
        Trains the Find-S model on the provided training instances.

        Parameters:
          training_data (list): List of dictionaries containing feature values and a 'label'.

        Returns:
          self: Trained instance of FindSConceptLearner.

        Handles:
          - Dataset with no positive examples: hypothesis remains the most specific null vector
            and a clear warning flag is raised.
        """
        if not training_data:
            self.hypothesis = ["Ø"] * len(self.feature_names)
            self.has_positive_examples = False
            self.is_trained = True
            return self

        # Dynamically align feature_names with the features present in training data
        sample_keys = [k for k in training_data[0].keys() if k not in ("student_id", "label")]
        if sample_keys:
            # Preserve designated order if subset/superset
            ordered = [f for f in self.feature_names if f in sample_keys]
            # Add any other keys present
            for k in sample_keys:
                if k not in ordered:
                    ordered.append(k)
            self.feature_names = ordered

        # Step 1: Reset hypothesis to most specific state: < Ø, Ø, ... >
        self.hypothesis = ["Ø"] * len(self.feature_names)
        self.training_steps = []
        self.has_positive_examples = False

        positive_count = 0

        # Step 2: Iterate through each training example
        for idx, instance in enumerate(training_data, start=1):
            label = instance.get("label")
            is_positive = (label == POSITIVE_LABEL)
            prev_hyp = list(self.hypothesis)
            action_desc = ""

            if is_positive:
                positive_count += 1
                self.has_positive_examples = True
                
                # Check each attribute constraint in the hypothesis
                for i, feat in enumerate(self.feature_names):
                    val = instance.get(feat, "?")
                    
                    if self.hypothesis[i] == "Ø":
                        # First positive example initializes the null constraint
                        self.hypothesis[i] = val
                    elif self.hypothesis[i] != val and self.hypothesis[i] != "?":
                        # Generalize constraint to '?' because positive examples have different values
                        self.hypothesis[i] = "?"

                action_desc = "Positive example: updated hypothesis constraints."
            else:
                # IMPORTANT FIND-S BEHAVIOR: Negative examples are ignored
                action_desc = "Negative example ignored: Find-S does not update from negative instances."

            # Record step trace for educational transparency
            self.training_steps.append({
                "step": idx,
                "student_id": instance.get("student_id", f"INST-{idx}"),
                "features": {feat: instance.get(feat, "N/A") for feat in self.feature_names},
                "label": label,
                "is_positive": is_positive,
                "hypothesis_before": prev_hyp,
                "hypothesis_after": list(self.hypothesis),
                "action": action_desc
            })

        # Step 3: Handle edge case when no positive examples exist
        if positive_count == 0:
            self.has_positive_examples = False
            self.is_trained = True
            return self

        self.is_trained = True
        return self

    def predict(self, categorical_features: Dict[str, str]) -> Dict[str, Any]:
        """
        Classifies a single student based on their categorical features.

        Matching rule:
        An instance is classified as POSITIVE (Good Performer) if and only if
        for every feature constraint h_i in the learned hypothesis:
            h_i == '?'  OR  h_i == instance[feature_i]
        If any constraint is violated or if hypothesis is still 'Ø',
        the instance is classified as NEGATIVE (Needs Academic Support).

        Parameters:
          categorical_features (dict): e.g., {'Attendance': 'High', ...}

        Returns:
          dict: Prediction outcome, feature-by-feature evaluation, and explanations.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained with fit() before making predictions.")

        # Edge case check: No positive examples were available during training
        if not self.has_positive_examples:
            return {
                "prediction": NEGATIVE_LABEL,
                "predicted_category": NEGATIVE_LABEL,
                "matches_hypothesis": False,
                "learned_hypothesis": self.hypothesis,
                "formatted_hypothesis": f"⟨ {', '.join(self.hypothesis)} ⟩",
                "feature_breakdown": [],
                "explanation": (
                    "No positive training examples were present in the dataset. "
                    "The hypothesis remains at the most specific null state ⟨ Ø, Ø, ... ⟩, "
                    f"classifying all instances as '{NEGATIVE_LABEL}'."
                ),
                "academic_suggestion": (
                    "Model training dataset contained no positive examples. Please retrain with valid "
                    "positive training instances to establish performance concept boundaries."
                ),
                "warning": "No positive training examples were present in the training set."
            }

        feature_breakdown = []
        all_satisfied = True
        violated_features = []

        for i, feat in enumerate(self.feature_names):
            if feat not in categorical_features:
                # Allow prediction on the core features if an optional feature is omitted
                continue

            h_val = self.hypothesis[i]
            x_val = categorical_features[feat]

            # Check if current feature satisfies the hypothesis constraint
            satisfied = (h_val == "?" or h_val == x_val)
            if not satisfied:
                all_satisfied = False
                violated_features.append(feat)

            feature_breakdown.append({
                "feature": feat,
                "student_value": x_val,
                "hypothesis_constraint": h_val,
                "satisfied": satisfied,
                "reason": (
                    f"Matches required constraint '{h_val}'" if h_val != "?" and satisfied
                    else "Wildcard '?' accepts any value" if h_val == "?"
                    else f"Value '{x_val}' does not match required constraint '{h_val}'"
                )
            })

        prediction = POSITIVE_LABEL if all_satisfied else NEGATIVE_LABEL

        # Generate simple explanation
        if all_satisfied:
            explanation = (
                "The student's features satisfy all constraints in the learned Find-S hypothesis. "
                "Classified as 'Good Performer'."
            )
        else:
            violated_str = ", ".join(violated_features)
            explanation = (
                f"The student's features violated hypothesis constraint(s) on: {violated_str}. "
                "Classified as 'Needs Academic Support'."
            )

        # Generate appropriate primary academic suggestion
        academic_suggestion = self._generate_academic_suggestion(all_satisfied, violated_features)

        return {
            "prediction": prediction,
            "predicted_category": prediction,
            "matches_hypothesis": all_satisfied,
            "learned_hypothesis": self.hypothesis,
            "formatted_hypothesis": f"⟨ {', '.join(self.hypothesis)} ⟩",
            "feature_breakdown": feature_breakdown,
            "explanation": explanation,
            "academic_suggestion": academic_suggestion
        }

    def _generate_academic_suggestion(self, all_satisfied: bool, violated_features: List[str]) -> str:
        """
        Produces a concise, highly relevant academic improvement suggestion
        based on which features violated the concept learning hypothesis.
        """
        if all_satisfied:
            return (
                "Maintain consistent study routines, explore advanced elective topics, "
                "and consider peer mentoring or undergraduate research participation."
            )

        advices = []
        if "Attendance" in violated_features:
            advices.append("Improve classroom attendance above 75% to stay aligned with lecture concepts.")
        if "Internal Assessment" in violated_features:
            advices.append("Attend faculty office hours and solve past internal assessment problem sets.")
        if "Assignment Completion" in violated_features:
            advices.append("Establish a weekly coursework schedule to submit assignments promptly.")
        if "Study Hours" in violated_features:
            advices.append("Increase dedicated independent study time to at least 3 hours per day.")
        if "Previous Examination Marks" in violated_features:
            advices.append("Review prerequisite core topics and practice timed mock examination papers.")

        return " ".join(advices) if advices else "Focus on continuous academic engagement and seek mentor guidance."


# =====================================================================
# 5. FIND-S LIMITATIONS EXPLANATION
# =====================================================================

def get_find_s_limitations() -> List[Dict[str, str]]:
    """
    Returns simple, educational descriptions of the key limitations of the Find-S algorithm.
    """
    return [
        {
            "title": "Ignores All Negative Examples",
            "description": (
                "Find-S completely disregards negative examples when generalizing its hypothesis. "
                "Because of this, it cannot detect whether negative examples are inconsistent with "
                "positive examples or whether the training data contains conflicting labels."
            )
        },
        {
            "title": "Zero Resistance to Corrupted Data (Noise)",
            "description": (
                "A single noisy or incorrectly labeled positive student example can force an attribute "
                "constraint to become '?' permanently. This broadens the hypothesis incorrectly and "
                "leads to false-positive classifications."
            )
        },
        {
            "title": "Cannot Detect Alternative Consistent Hypotheses",
            "description": (
                "Find-S only finds one maximally specific hypothesis. It cannot indicate whether multiple "
                "competing hypotheses could also explain the data, unlike Version Space and Candidate "
                "Elimination algorithms."
            )
        },
        {
            "title": "Strictly Conjunctive Hypothesis Space",
            "description": (
                "Find-S can only learn concepts represented by conjunctions (AND logic) of attribute "
                "constraints. It cannot learn disjunctive concepts (e.g., 'High Attendance OR High Internal Assessment')."
            )
        }
    ]


# =====================================================================
# 6. SINGLETON MODEL HELPER FOR APPLICATION USAGE
# =====================================================================

_demo_learner_instance: Optional[FindSConceptLearner] = None

def get_trained_find_s_model() -> Tuple[FindSConceptLearner, List[Dict[str, Any]]]:
    """
    Returns a trained singleton Find-S model on the demonstration dataset.
    """
    global _demo_learner_instance
    sample_data = get_sample_training_data()
    if _demo_learner_instance is None:
        learner = FindSConceptLearner()
        learner.fit(sample_data)
        _demo_learner_instance = learner
    return _demo_learner_instance, sample_data
