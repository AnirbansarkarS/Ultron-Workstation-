"""
Fast lookup (optional).
"""

class SpatialHash:
    def __init__(self, chunk_size=4):
        self.chunk_size = chunk_size
        self.chunks = {}
