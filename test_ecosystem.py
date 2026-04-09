#!/usr/bin/env python3
"""
Tests for Learning Sandbox ecosystem simulation.
Run with: python -m pytest test_ecosystem.py -v
"""

import sys
import math

# Mock pygame before importing ecosystem
class MockPygame:
    class display:
        @staticmethod
        def set_mode(size): return MockSurface(size)
        @staticmethod
        def set_caption(s): pass
        @staticmethod
        def flip(): pass
        @staticmethod
        def get_surface(): return MockSurface((1, 1))

    class draw:
        @staticmethod
        def circle(surf, color, pos, radius): pass
        @staticmethod
        def rect(surf, color, rect): pass
        @staticmethod
        def line(surf, color, start, end, width): pass
        @staticmethod
        def lines(surf, color, closed, points, width): pass

    class event:
        QUIT = 1
        KEYDOWN = 2
        MOUSEBUTTONDOWN = 3
        K_ESCAPE = 27
        K_SPACE = 32
        K_r = 114
        K_g = 103

    class key:
        pass

    class mouse:
        @staticmethod
        def get_pos(): return (0, 0)

    class time:
        @staticmethod
        def Clock(): return MockClock()
        @staticmethod
        def tick(fps): return 16

    class font:
        Font = object


    KEYDOWN = 2
    MOUSEBUTTONDOWN = 3

class MockSurface:
    def __init__(self, size): self.size = size
    def fill(self, color): pass
    def blit(self, src, dest): pass
    def get_width(self): return self.size[0]
    def get_height(self): return self.size[1]

class MockClock:
    def tick(self, fps): return 16

# Apply mocks
sys.modules['pygame'] = MockPygame()

from ecosystem import MLP, Prey, Predator, Food, SpatialHash, Stats, ARENA_W, ARENA_H


class TestMLP:
    def test_forward(self):
        mlp = MLP(input_dim=5, hidden_dim=8, output_dim=2)
        x = numpy.random.randn(5)
        out = mlp.forward(x)
        assert out.shape == (2,), f"Expected shape (2,), got {out.shape}"

    def test_copy(self):
        mlp = MLP(input_dim=5, hidden_dim=8, output_dim=2)
        mlp.fc1[:] = 0.5
        mlp.fc2[:] = 0.5
        copy = mlp.copy()
        assert copy.fc1.shape == mlp.fc1.shape
        assert copy.fc2.shape == mlp.fc2.shape

    def test_mutate(self):
        mlp = MLP(input_dim=5, hidden_dim=8, output_dim=2)
        before = mlp.fc1.copy()
        mlp.mutate(rate=0.0)
        # With rate=0, weights shouldn't change much
        # Just check it runs without error


class TestSpatialHash:
    def test_insert_and_query(self):
        sh = SpatialHash(cell_size=40)
        food = Food(50, 60)
        sh.insert(food, 50, 60)
        results = sh.query(50, 60, 50)
        assert len(results) > 0

    def test_clear(self):
        sh = SpatialHash(cell_size=40)
        food = Food(50, 60)
        sh.insert(food, 50, 60)
        sh.clear()
        results = sh.query(50, 60, 50)
        assert len(results) == 0


class TestPrey:
    def test_creation(self):
        p = Prey(100, 100)
        assert p.energy == 0.5
        assert 10 <= p.detection_radius <= 80

    def test_reproduce(self):
        p = Prey(100, 100)
        p.energy = 1.5
        child = p.reproduce()
        assert child.energy == 0.5
        assert abs(child.detection_radius - p.detection_radius) < 5


class TestPredator:
    def test_creation(self):
        pr = Predator(100, 100)
        assert pr.energy == 0.5
        assert 1.0 <= pr.speed <= 4.0

    def test_reproduce(self):
        pr = Predator(100, 100)
        pr.energy = 2.0
        child = pr.reproduce()
        assert child.energy == 0.5


class TestStats:
    def test_record_and_history(self):
        s = Stats()
        s.record(100, 20)
        s.record(110, 22)
        assert len(s.prey_history) == 2
        assert len(s.pred_history) == 2
        assert s.prey_history[-1] == 110

    def test_max_history(self):
        s = Stats()
        for i in range(400):
            s.record(i, i)
        assert len(s.prey_history) == 300


class TestFood:
    def test_respawn(self):
        f = Food(10, 10)
        orig_x, orig_y = f.x, f.y
        f.respawn()
        # After respawn, should be within arena
        assert 0 < f.x < ARENA_W
        assert 0 < f.y < ARENA_H


if __name__ == "__main__":
    import numpy
    print("Running tests...")

    TestMLP().test_forward()
    TestMLP().test_copy()
    TestMLP().test_mutate()
    print("MLP tests passed")

    TestSpatialHash().test_insert_and_query()
    TestSpatialHash().test_clear()
    print("SpatialHash tests passed")

    TestPrey().test_creation()
    TestPrey().test_reproduce()
    print("Prey tests passed")

    TestPredator().test_creation()
    TestPredator().test_reproduce()
    print("Predator tests passed")

    TestStats().test_record_and_history()
    TestStats().test_max_history()
    print("Stats tests passed")

    TestFood().test_respawn()
    print("Food tests passed")

    print("All tests passed!")
