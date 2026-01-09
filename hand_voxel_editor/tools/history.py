"""
Undo / redo history.
"""

class History:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
