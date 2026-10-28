"""Consent-based enrollment from a folder of images.

Images are read, embedded and discarded in memory - nothing but the ArcFace
vectors is persisted (``images_stored`` is always 0 in the report).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from face.analyzer import FaceAnalyzer
from face.gallery import FaceGallery

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def enroll_from_directory(gallery: FaceGallery, analyzer: FaceAnalyzer, name: str, directory: Path, *,
                          consent_reference: str, purpose: str, operator: str, expires_days: Optional[int] = None,
                          min_samples: int = 3, max_samples: int = 20, min_score: float = 0.6) -> Dict[str, Any]:
    import cv2

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    if not analyzer.provides_embeddings:
        raise RuntimeError("the active face backend does not produce ArcFace embeddings")
    embeddings: List[np.ndarray] = []
    rejected: List[Dict[str, str]] = []
    for path in sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMG_EXT):
        if len(embeddings) >= max_samples:
            break
        img = cv2.imread(str(path))
        if img is None:
            rejected.append({"file": path.name, "reason": "unreadable"})
            continue
        faces = [f for f in analyzer.detect(img) if f.score >= min_score and f.embedding is not None]
        del img  # pixels discarded immediately
        if len(faces) != 1:
            rejected.append({"file": path.name, "reason": f"expected exactly 1 face, found {len(faces)}"})
            continue
        embeddings.append(faces[0].embedding)
    if len(embeddings) < min_samples:
        raise ValueError(f"only {len(embeddings)} usable image(s); need >= {min_samples}. Rejected: {rejected}")
    mat = np.stack(embeddings)
    centroid = mat.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    keep = [e for e, s in zip(embeddings, mat @ centroid) if s >= 0.5]  # drop outliers (wrong person in folder)
    if len(keep) < min_samples:
        raise ValueError("samples are inconsistent (different people?) - enrollment aborted")
    identity_id = gallery.enroll(name, keep, consent_reference=consent_reference, purpose=purpose,
                                 operator=operator, expires_days=expires_days)
    return {"identity_id": identity_id, "name": name, "accepted": len(keep),
            "outliers_removed": len(embeddings) - len(keep), "rejected": rejected, "images_stored": 0}
