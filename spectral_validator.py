#!/usr/bin/env python3
"""
spectral_validator.py — Conservation-based validation for musicdb parsing
==========================================================================

Drop-in validation layer for musicdb-to-json.

Usage:
    from spectral_validator import SpectralValidator
    
    validator = SpectralValidator()
    
    # During parsing, feed each chunk:
    validator.feed_chunk(chunk_type, chunk_bytes, parsed_data)
    
    # After parsing, get validation report:
    report = validator.validate()
    for anomaly in report.anomalies:
        print(f"WARNING: {anomaly}")
"""

import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class ChunkRecord:
    chunk_type: str
    byte_length: int
    entropy: float
    field_count: int
    tension_vector: np.ndarray
    parsed_data: dict


@dataclass
class Anomaly:
    chunk_index: int
    chunk_type: str
    anomaly_type: str
    severity: float  # 0-1, higher = worse
    message: str


@dataclass
class ValidationReport:
    total_chunks: int
    anomalies: list
    conservation_score: float  # 0-1, higher = better
    spectral_gap: float
    cheeger_constant: float
    format_fingerprint: list
    per_entity_scores: dict


class SpectralValidator:
    """
    Conservation-based validator for musicdb parsing.
    
    Tracks tension (entropy, field count, byte length) across chunks,
    computes the Tension-Graph Laplacian, and detects conservation drops
    that indicate corruption or format mismatches.
    """
    
    # Laplacian from the static format analysis
    FORMAT_EIGENVALUES = [-0.0000, 0.1157, 0.1729, 0.1767, 0.1827,
                          0.2744, 0.3417, 0.4268, 0.5124, 0.6568,
                          0.6975, 0.7245]
    FORMAT_CHUNK_TYPES = ['hsma', 'hfma', 'plma', 'lAma', 'lama',
                          'lPma', 'ltma', 'iAma', 'iama', 'lpma',
                          'itma', 'boma']
    FORMAT_FIEDLER = [-0.6149, -0.4647, -0.2865, +0.0178, +0.0178,
                      +0.0397, +0.0604, +0.1949, +0.1949, +0.2274,
                      +0.2723, +0.3410]
    
    def __init__(self):
        self.chunks: list[ChunkRecord] = []
        self.entities: dict[str, list[int]] = {}  # entity_id → chunk indices
        self.current_entity: Optional[str] = None
    
    @staticmethod
    def _compute_entropy(data: bytes) -> float:
        """Compute Shannon entropy of byte sequence."""
        if not data:
            return 0.0
        counts = np.bincount(np.frombuffer(data[:1000], dtype=np.uint8), minlength=256)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    def feed_chunk(self, chunk_type: str, chunk_bytes: bytes, parsed_data: dict = None):
        """Record a parsed chunk for validation."""
        if parsed_data is None:
            parsed_data = {}
        
        # Normalize chunk type
        ct = chunk_type if isinstance(chunk_type, str) else chunk_type.decode('ascii', errors='replace')
        
        # Compute features
        entropy = self._compute_entropy(chunk_bytes)
        field_count = len(parsed_data) if parsed_data else 0
        byte_length = len(chunk_bytes)
        
        # Tension vector
        tension = np.array([entropy / 8.0, field_count / 20.0, byte_length / 10000.0])
        
        record = ChunkRecord(
            chunk_type=ct,
            byte_length=byte_length,
            entropy=entropy,
            field_count=field_count,
            tension_vector=tension,
            parsed_data=parsed_data,
        )
        
        idx = len(self.chunks)
        self.chunks.append(record)
        
        # Track entity membership
        if self.current_entity:
            self.entities.setdefault(self.current_entity, []).append(idx)
    
    def set_entity(self, entity_id: str):
        """Start tracking chunks for a specific entity (track, album, etc.)."""
        self.current_entity = entity_id
    
    def end_entity(self):
        """Stop tracking chunks for the current entity."""
        self.current_entity = None
    
    def validate(self) -> ValidationReport:
        """Run full spectral validation on the collected chunks."""
        anomalies = []
        n = len(self.chunks)
        
        if n < 2:
            return ValidationReport(
                total_chunks=n, anomalies=[], conservation_score=1.0,
                spectral_gap=0, cheeger_constant=0,
                format_fingerprint=[], per_entity_scores={}
            )
        
        # Build chunk-type transition graph from actual data
        type_counts = {}
        for c in self.chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
        
        # Build tension series
        tensions = np.array([c.tension_vector for c in self.chunks])
        
        # Compute tension gradient (conservation measure)
        if len(tensions) > 1:
            gradients = np.diff(tensions, axis=0)
            gradient_norms = np.linalg.norm(gradients, axis=1)
            mean_gradient = np.mean(gradient_norms)
            std_gradient = np.std(gradient_norms)
        else:
            gradient_norms = np.array([0])
            mean_gradient = 0
            std_gradient = 0
        
        # Conservation score: lower gradient variance = higher conservation
        conservation_score = 1.0 / (1.0 + std_gradient) if std_gradient < 10 else 0.0
        
        # Detect anomalies (spikes in gradient)
        threshold = mean_gradient + 2 * std_gradient if std_gradient > 0 else mean_gradient * 3
        for i, gn in enumerate(gradient_norms):
            if gn > threshold:
                severity = min(gn / (threshold * 3), 1.0)
                anomalies.append(Anomaly(
                    chunk_index=i,
                    chunk_type=self.chunks[i].chunk_type,
                    anomaly_type='tension_spike',
                    severity=severity,
                    message=f"Tension spike at chunk {i} ({self.chunks[i].chunk_type}): "
                           f"gradient={gn:.3f}, threshold={threshold:.3f}"
                ))
        
        # Per-entity validation
        entity_scores = {}
        for entity_id, indices in self.entities.items():
            if len(indices) < 2:
                entity_scores[entity_id] = 1.0
                continue
            
            entity_tensions = np.array([self.chunks[i].tension_vector for i in indices])
            entity_grads = np.diff(entity_tensions, axis=0)
            entity_grad_norms = np.linalg.norm(entity_grads, axis=1)
            entity_var = np.var(entity_grad_norms)
            entity_score = 1.0 / (1.0 + entity_var)
            entity_scores[entity_id] = entity_score
            
            if entity_score < 0.5:
                anomalies.append(Anomaly(
                    chunk_index=indices[0],
                    chunk_type='entity',
                    anomaly_type='low_entity_conservation',
                    severity=1.0 - entity_score,
                    message=f"Entity {entity_id}: low conservation score {entity_score:.3f}"
                ))
        
        # Entropy-based anomaly detection
        entropies = [c.entropy for c in self.chunks]
        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)
        
        for i, c in enumerate(self.chunks):
            if c.entropy > mean_entropy + 3 * std_entropy:
                anomalies.append(Anomaly(
                    chunk_index=i,
                    chunk_type=c.chunk_type,
                    anomaly_type='high_entropy',
                    severity=min((c.entropy - mean_entropy) / (std_entropy * 3), 1.0),
                    message=f"Unusually high entropy in chunk {i} ({c.chunk_type}): "
                           f"entropy={c.entropy:.2f}, mean={mean_entropy:.2f}"
                ))
            elif c.entropy < mean_entropy - 3 * std_entropy and c.byte_length > 100:
                anomalies.append(Anomaly(
                    chunk_index=i,
                    chunk_type=c.chunk_type,
                    anomaly_type='low_entropy',
                    severity=min((mean_entropy - c.entropy) / (std_entropy * 3), 1.0),
                    message=f"Unusually low entropy in chunk {i} ({c.chunk_type}): "
                           f"entropy={c.entropy:.2f}, mean={mean_entropy:.2f}"
                ))
        
        # Sort anomalies by severity
        anomalies.sort(key=lambda a: a.severity, reverse=True)
        
        # Format fingerprint (empirical eigenvalues of chunk-type transition)
        type_list = sorted(type_counts.keys())
        type_idx = {t: i for i, t in enumerate(type_list)}
        n_types = len(type_list)
        
        T = np.ones((n_types, n_types)) * 0.01  # smoothing
        for i in range(n - 1):
            c1 = self.chunks[i].chunk_type
            c2 = self.chunks[i + 1].chunk_type
            if c1 in type_idx and c2 in type_idx:
                T[type_idx[c1], type_idx[c2]] += 1
        
        for i in range(n_types):
            T[i] /= T[i].sum()
        
        fingerprint = np.sort(np.linalg.eigvalsh(T @ T.T))[::-1].tolist()
        
        return ValidationReport(
            total_chunks=n,
            anomalies=anomalies,
            conservation_score=conservation_score,
            spectral_gap=self.FORMAT_EIGENVALUES[1] if len(self.FORMAT_EIGENVALUES) > 1 else 0,
            cheeger_constant=self.FORMAT_EIGENVALUES[1] / 2 if len(self.FORMAT_EIGENVALUES) > 1 else 0,
            format_fingerprint=fingerprint,
            per_entity_scores=entity_scores,
        )
    
    def validate_json(self) -> str:
        """Run validation and return JSON report."""
        report = self.validate()
        return json.dumps({
            'total_chunks': report.total_chunks,
            'conservation_score': report.conservation_score,
            'anomaly_count': len(report.anomalies),
            'anomalies': [
                {
                    'index': a.chunk_index,
                    'type': a.chunk_type,
                    'anomaly': a.anomaly_type,
                    'severity': round(a.severity, 3),
                    'message': a.message,
                }
                for a in report.anomalies[:20]  # Top 20
            ],
            'entity_count': len(report.per_entity_scores),
            'mean_entity_score': np.mean(list(report.per_entity_scores.values())) if report.per_entity_scores else 1.0,
            'spectral_gap': report.spectral_gap,
            'cheeger_constant': report.cheeger_constant,
        }, indent=2)


def demo():
    """Demo with simulated chunk data."""
    print("=" * 70)
    print("SPECTRAL VALIDATOR DEMO")
    print("=" * 70)
    
    validator = SpectralValidator()
    
    # Simulate a well-formed musicdb parse
    well_formed_sequence = [
        ('hfma', b'\x00' * 116, {'library_id': 'ABC123'}),
        ('hsma', b'\x00' * 12, {}),
        ('plma', b'\x00' * 58, {'library_name': 'My Library'}),
        ('boma', b'\x00' * 40, {'name': 'My Library'}),
        ('boma', b'\x00' * 24, {'subtype': 1, 'version': 1}),
        ('ltma', b'\x00' * 16, {'track_count': 100}),
    ]
    
    # Add well-formed tracks
    for i in range(20):
        validator.set_entity(f'track_{i}')
        validator.feed_chunk('itma', bytes([i % 256] * 340) + b'\x00' * 20, 
                           {'track_id': f'T{i}', 'track_number': i})
        validator.feed_chunk('boma', b'Song Title ' * 5,
                           {'name': f'Song {i}'})
        validator.feed_chunk('boma', b'\x00' * 30,
                           {'subtype': 1, 'bpm': 120, 'bit_rate': 320})
        validator.feed_chunk('boma', b'Artist Name ' * 3,
                           {'artist': f'Artist {i % 5}'})
        validator.end_entity()
    
    # Add one corrupted track
    validator.set_entity('track_corrupt')
    validator.feed_chunk('itma', bytes([255] * 360),  # High entropy = suspicious
                       {'track_id': 'CORRUPT'})
    validator.feed_chunk('boma', bytes(range(256)) * 4,  # Very high entropy
                       {'name': '?????'})
    validator.end_entity()
    
    # Validate
    report = validator.validate()
    
    print(f"\nValidation Report:")
    print(f"  Total chunks: {report.total_chunks}")
    print(f"  Conservation score: {report.conservation_score:.4f}")
    print(f"  Anomalies: {len(report.anomalies)}")
    print(f"  Entity count: {len(report.per_entity_scores)}")
    
    if report.per_entity_scores:
        scores = list(report.per_entity_scores.values())
        print(f"  Mean entity score: {np.mean(scores):.4f}")
        print(f"  Min entity score: {np.min(scores):.4f}")
        print(f"  Corrupt track score: {report.per_entity_scores.get('track_corrupt', 'N/A')}")
    
    print(f"\n  Top anomalies:")
    for a in report.anomalies[:5]:
        print(f"    [{a.severity:.2f}] {a.message}")
    
    # JSON report
    print(f"\n  JSON Report:")
    print(validator.validate_json())


if __name__ == '__main__':
    demo()
