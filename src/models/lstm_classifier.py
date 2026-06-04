from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from src.utils.device import resolve_torch_device


@dataclass(frozen=True)
class LSTMClassifierConfig:
    num_keypoints: int = 17
    coords_per_keypoint: int = 3
    input_size: int | None = None
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 2

    @property
    def resolved_input_size(self) -> int:
        if self.input_size is not None:
            return int(self.input_size)
        return self.num_keypoints * self.coords_per_keypoint


class LSTMFallClassifier(nn.Module):
    """

    Sequence classifier for per-frame fall features. ^v^
    
    """

    def __init__(self, config: LSTMClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or LSTMClassifierConfig()
        dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.config.resolved_input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.num_classes),
        )

    def forward(self, feature_sequence: torch.Tensor) -> torch.Tensor:
        if feature_sequence.ndim == 4:
            batch, steps, joints, coords = feature_sequence.shape
            feature_sequence = feature_sequence.reshape(batch, steps, joints * coords)
        _, (hidden, _) = self.lstm(feature_sequence)
        return self.classifier(hidden[-1])


class LSTMFeatureSequenceClassifier:
    """
    
    Inference helper around an LSTM fall feature classifier. ^v^


    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | int | None = "auto",
        prefer_cuda: bool = True,
        cuda_index: int = 0,
        config: LSTMClassifierConfig | None = None,
    ) -> None:
        self.device = resolve_torch_device(
            device=device,
            prefer_cuda=prefer_cuda,
            cuda_index=cuda_index,
        )
        self.model = LSTMFallClassifier(config=config).to(self.device)
        if weights_path:
            state = torch.load(
                Path(weights_path),
                map_location=self.device,
                weights_only=True,
            )
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            weight_input_size = self._state_input_size(state)
            if weight_input_size is not None and weight_input_size != self.expected_input_size:
                raise ValueError(
                    "LSTM weights input size mismatch: weights expect "
                    f"{weight_input_size}, but config expects {self.expected_input_size}. "
                    "Retrain/export the LSTM on the current per-frame feature vector "
                    "or set lstm.model.input_size to match the weights."
                )
            self.model.load_state_dict(state)
        self.model.eval()

    @property
    def expected_input_size(self) -> int:
        return int(self.model.config.resolved_input_size)

    def predict_proba(self, sequence_features: np.ndarray) -> float:
        sequence = np.asarray(sequence_features, dtype=np.float32)
        if sequence.ndim == 2:
            sequence = sequence[None, ...]
        elif (
            sequence.ndim == 3
            and sequence.shape[-2] == self.model.config.num_keypoints
            and sequence.shape[-1] == self.model.config.coords_per_keypoint
        ):
            sequence = sequence[None, ...]
        elif sequence.ndim not in (3, 4):
            raise ValueError(
                "Expected an LSTM sequence shaped (T,F), (B,T,F), "
                "(T,J,C), or (B,T,J,C)."
            )
        if sequence.ndim == 4:
            batch, steps, joints, coords = sequence.shape
            sequence = sequence.reshape(batch, steps, joints * coords)
        if sequence.shape[-1] != self.expected_input_size:
            raise ValueError(
                "LSTM input feature size mismatch: expected "
                f"{self.expected_input_size}, got {sequence.shape[-1]}. "
                "Train weights with the same feature_size as config.features.feature_size "
                "or update lstm.model.input_size."
            )
        tensor = torch.from_numpy(sequence).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=-1)
        return float(probabilities[0, 1].cpu().item())

    @staticmethod
    def _state_input_size(state: dict) -> int | None:
        weight = state.get("lstm.weight_ih_l0") if isinstance(state, dict) else None
        if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
            return None
        return int(weight.shape[1])


LSTMSequenceClassifier = LSTMFeatureSequenceClassifier
