"""
Face Recognition & Embedding Generation Module using ArcFace / InsightFace (w600k_mbf).
Extracts 512-dimensional SOTA facial embeddings and performs cosine similarity matching
for re-identification and auto-registration.
"""

import os
import cv2
import time
import numpy as np
from typing import Tuple, Optional, Dict, List
import onnxruntime as ort


class FaceRecognizer:
    """ArcFace / InsightFace ONNX feature extractor."""

    def __init__(
        self,
        model_path: str = "models/w600k_mbf.onnx",
        similarity_threshold: float = 0.50,
        use_gpu: bool = True
    ):
        self.model_path = model_path
        self.similarity_threshold = similarity_threshold

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ArcFace model not found at {self.model_path}")

        # Configure ONNX Runtime execution providers
        providers = []
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options=session_options, providers=providers)

        # Get input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # usually [batch, 3, 112, 112]
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses cropped face for ArcFace input (112x112, RGB, normalized).
        """
        # Resize to standard ArcFace input 112x112
        resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize: (pixel - 127.5) / 128.0
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0
        # Transpose HWC to CHW and add batch dimension -> (1, 3, 112, 112)
        transposed = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(transposed, axis=0).astype(np.float32)
        return batch

    def generate_embedding(self, face_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generates a 512-dimensional normalized embedding for a given face crop.
        Returns:
            (embedding, elapsed_ms)
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(512, dtype=np.float32), 0.0

        t0 = time.perf_counter()
        inp = self.preprocess(face_crop)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        raw_emb = outputs[0][0].astype(np.float32)

        # L2 normalize embedding for cosine similarity
        norm = np.linalg.norm(raw_emb)
        if norm > 0:
            emb = raw_emb / norm
        else:
            emb = raw_emb

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return emb, elapsed_ms

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity between two unit-normalized embeddings."""
        return float(np.dot(emb1, emb2))

    def match_face(
        self,
        query_emb: np.ndarray,
        registered_embeddings: Dict[str, np.ndarray]
    ) -> Tuple[Optional[str], float]:
        """
        Compares query embedding with all registered visitor embeddings.
        Returns:
            (matched_visitor_id, best_similarity)
            If no match passes threshold, returns (None, best_similarity).
        """
        if not registered_embeddings or query_emb is None:
            return None, 0.0

        best_id = None
        best_sim = -1.0

        for visitor_id, reg_emb in registered_embeddings.items():
            sim = self.compute_similarity(query_emb, reg_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = visitor_id

        if best_sim >= self.similarity_threshold:
            return best_id, best_sim
        else:
            return None, best_sim
