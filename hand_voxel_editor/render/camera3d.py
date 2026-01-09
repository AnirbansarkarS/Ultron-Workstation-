"""
Virtual camera math.
"""

class Camera3D:
    def __init__(self, position=(0, 0, -5), rotation=(0, 0, 0)):
        self.position = position
        self.rotation = rotation
