"""Multi-label exoplanet cascade: 3 calibrated heads (transit / EB / rotation) + OOD gate.
UI-agnostic — import pipeline.* from Streamlit later."""
from .config import ExoConfig, DIP_FEATURES, SPOT_FEATURES, ALL_FEATURES, label_bits
