#!/usr/bin/env python3
"""
Learning Sandbox — Ecosystem Simulation
Prey and predators both have neural nets, both learn, both inherit weights on reproduction.
"""

import pygame
import random
import math
import numpy as np

# Initialize pygame
pygame.init()

# Screen
WIDTH, HEIGHT = 1200, 700
ARENA_W, ARENA_H = 900, 600
GRAPH_X, GRAPH_Y = ARENA_W + 20, 50
GRAPH_W, GRAPH_H = 250, 200

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Learning Sandbox")
clock = pygame.time.Clock()

# Font — create after display is ready to avoid Python 3.14 + pygame circular import bug
try:
    font_renderer = pygame.font.Font(None, 14)
except Exception:
    font_renderer = None


def draw_text(surface, text, pos, color=WHITE):
    if font_renderer:
        surf = font_renderer.render(text, True, color)
        surface.blit(surf, pos)
    else:
        # Fallback: draw text as colored rectangles at approximate size
        w, h = len(text) * 7, 12
        pygame.draw.rect(surface, color, (*pos, w, h), border_radius=1)

# Colors
BG = (10, 10, 20)
PREY_COLOR = (50, 220, 100)
PRED_COLOR = (220, 60, 60)
FOOD_COLOR = (255, 230, 80)
GRID_COLOR = (30, 30, 50)
WHITE = (220, 220, 220)
PANEL_BG = (20, 20, 35)

# Simulation params
FPS = 60
ARENA_RECT = pygame.Rect(10, 50, ARENA_W, ARENA_H)
INITIAL_PREY = 80
INITIAL_PRED = 15
FOOD_COUNT = 60
FOOD_RESPAWN_RATE = 0.02
PREY_REPRODUCE_THRESHOLD = 1.0
PRED_REPRODUCE_THRESHOLD = 1.5
PREY_DEATH_THRESHOLD = 0.0
PRED_STARVE_STEPS = 200
WEIGHT_MUTATION = 0.1
REWARD_GATHER = 0.3
REWARD_CATCH = 2.0
PENALTY_DANGER = 0.1
PENALTY_STARVE = 0.5
SPEED_MULTIPLIER = 1

# --- Neural Net ---
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = np.random.randn(input_dim, hidden_dim) * 0.3
        self.fc2 = np.random.randn(hidden_dim, output_dim) * 0.3
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = np.tanh(self.fc1.T @ x)
        x = self.fc2.T @ x
        return x

    def copy(self):
        other = MLP(self.input_dim, self.fc1.shape[0], self.output_dim)
        other.fc1 = self.fc1.copy()
        other.fc2 = self.fc2.copy()
        return other

    def mutate(self, rate=WEIGHT_MUTATION):
        self.fc1 += np.random.randn(*self.fc1.shape) * rate
        self.fc2 += np.random.randn(*self.fc2.shape) * rate


# --- Spatial Hash ---
class SpatialHash:
    def __init__(self, cell_size=40):
        self.cell_size = cell_size
        self.cells = {}

    def clear(self):
        self.cells = {}

    def hash_pos(self, x, y):
        return int(x // self.cell_size), int(y // self.cell_size)

    def insert(self, obj, x, y):
        k = self.hash_pos(x, y)
        if k not in self.cells:
            self.cells[k] = []
        self.cells[k].append(obj)

    def query(self, x, y, radius):
        cx, cy = self.hash_pos(x, y)
        results = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                k = (cx + dx, cy + dy)
                if k in self.cells:
                    for obj in self.cells[k]:
                        ox, oy = obj.x, obj.y
                        dist = math.hypot(x - ox, y - oy)
                        if dist <= radius:
                            results.append((obj, dist))
        return results


# --- Prey ---
class Prey:
    def __init__(self, x, y, brain=None):
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        self.energy = 0.5
        self.detection_radius = random.uniform(30, 60)
        self.speed = random.uniform(0.8, 1.5)
        self.age = 0
        self.last_danger = 0
        self.brain = brain if brain else MLP(input_dim=5, hidden_dim=8, output_dim=3)
        self.accumulated_gradient_fc1 = np.zeros_like(self.brain.fc1)
        self.accumulated_gradient_fc2 = np.zeros_like(self.brain.fc2)
        self.activation_history = []

    def perceive(self, food_list, predators):
        # Normalized inputs
        nearest_food = min(food_list, key=lambda f: math.hypot(f.x - self.x, f.y - self.y), default=None)
        nearest_pred = min(predators, key=lambda p: math.hypot(p.x - self.x, p.y - self.y), default=None)

        food_dist = nearest_food.dist / self.detection_radius if nearest_food else 1.0
        food_angle = math.atan2(nearest_food.y - self.y, nearest_food.x - self.x) / math.pi if nearest_food else 0.0
        pred_dist = nearest_pred.detection_radius / self.detection_radius if nearest_pred else 1.0
        pred_angle = math.atan2(nearest_pred.y - self.y, nearest_pred.x - self.x) / math.pi if nearest_pred else 0.0
        energy = self.energy

        return np.array([food_dist, food_angle, pred_dist, pred_angle, energy])

    def act(self, perception):
        logits = self.brain.forward(perception)
        action = np.argmax(logits)
        # Actions: 0=seek food, 1=flee predator, 2=wander
        target_x, target_y = self.x, self.y

        if action == 0 and perception[0] < 1.0:
            # Seek nearest food
            angle = perception[1] * math.pi
            target_x = self.x + math.cos(angle) * 50
            target_y = self.y + math.sin(angle) * 50
        elif action == 1 and perception[2] < 1.0:
            # Flee from predator
            angle = perception[3] * math.pi
            target_x = self.x - math.cos(angle) * 50
            target_y = self.y - math.sin(angle) * 50
        else:
            # Wander
            target_x = self.x + random.uniform(-20, 20)
            target_y = self.y + random.uniform(-20, 20)

        # Move toward target
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0.1:
            self.vx = (dx / dist) * self.speed
            self.vy = (dy / dist) * self.speed
        else:
            self.vx *= 0.8
            self.vy *= 0.8

        self.x += self.vx
        self.y += self.vy

        # Bounce off walls
        if self.x < 10:
            self.x = 10
            self.vx *= -1
        if self.x > ARENA_W + 10:
            self.x = ARENA_W + 10
            self.vx *= -1
        if self.y < 50:
            self.y = 50
            self.vy *= -1
        if self.y > ARENA_H + 50:
            self.y = ARENA_H + 50
            self.vy *= -1

    def learn(self, reward):
        # Simple REINFORCE-like weight update
        delta = reward * 0.01
        self.brain.fc1 += np.random.randn(*self.brain.fc1.shape) * delta
        self.brain.fc2 += np.random.randn(*self.brain.fc2.shape) * delta

    def reproduce(self):
        self.energy *= 0.5
        child = Prey(self.x + random.uniform(-10, 10), self.y + random.uniform(-10, 10), self.brain.copy())
        child.detection_radius = self.detection_radius + random.uniform(-5, 5)
        child.speed = max(0.5, min(2.5, self.speed + random.uniform(-0.1, 0.1)))
        child.energy = 0.5
        return child


# --- Predator ---
class Predator:
    def __init__(self, x, y, brain=None):
        self.x, self.y = x, y
        self.vx, self.vy = 0, 0
        self.energy = 0.5
        self.starve_counter = 0
        self.detection_radius = random.uniform(60, 100)
        self.speed = random.uniform(1.5, 2.5)
        self.brain = brain if brain else MLP(input_dim=6, hidden_dim=10, output_dim=3)
        self.last_reward = 0

    def perceive(self, prey_list):
        nearest_prey = sorted(prey_list, key=lambda p: math.hypot(p.x - self.x, p.y - self.y))[:3]
        inputs = []
        for i in range(3):
            if i < len(nearest_prey):
                p = nearest_prey[i]
                inputs.append((p.x - self.x) / ARENA_W)
                inputs.append((p.y - self.y) / ARENA_H)
            else:
                inputs.append(0.0)
                inputs.append(0.0)
        inputs.append(self.vx / 3.0)
        inputs.append(self.vy / 3.0)
        inputs.append(self.energy)
        return np.array(inputs)

    def act(self, perception):
        logits = self.brain.forward(perception)
        action = np.argmax(logits)
        # Actions: 0=left, 1=right, 2=forward
        angle = random.uniform(0, 2 * math.pi)
        if action == 0:
            angle = math.atan2(self.vy, self.vx) - 0.4
        elif action == 1:
            angle = math.atan2(self.vy, self.vx) + 0.4
        else:
            angle = math.atan2(self.vy, self.vx)
        self.vx = math.cos(angle) * self.speed
        self.vy = math.sin(angle) * self.speed
        self.x += self.vx * SPEED_MULTIPLIER
        self.y += self.vy * SPEED_MULTIPLIER
        # Bounce off walls
        if self.x < 10:
            self.x = 10
            self.vx *= -1
        if self.x > ARENA_W + 10:
            self.x = ARENA_W + 10
            self.vx *= -1
        if self.y < 50:
            self.y = 50
            self.vy *= -1
        if self.y > ARENA_H + 50:
            self.y = ARENA_H + 50
            self.vy *= -1

    def learn(self, reward):
        self.last_reward = reward
        delta = reward * 0.01
        self.brain.fc1 += np.random.randn(*self.brain.fc1.shape) * delta
        self.brain.fc2 += np.random.randn(*self.brain.fc2.shape) * delta

    def reproduce(self):
        self.energy *= 0.5
        child = Predator(self.x, self.y, self.brain.copy())
        child.speed = max(1.0, min(4.0, self.speed + random.uniform(-0.2, 0.2)))
        child.energy = 0.5
        return child


# --- Food ---
class Food:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.dist = random.uniform(3, 6)

    def respawn(self):
        self.x = random.uniform(20, ARENA_W - 20)
        self.y = random.uniform(60, ARENA_H - 20)


# --- Stats ---
class Stats:
    def __init__(self):
        self.prey_history = []
        self.pred_history = []
        self.max_history = 300

    def record(self, prey_count, pred_count):
        self.prey_history.append(prey_count)
        self.pred_history.append(pred_count)
        if len(self.prey_history) > self.max_history:
            self.prey_history.pop(0)
            self.pred_history.pop(0)


# --- World ---
prey_list = []
pred_list = []
food_list = []
stats = Stats()
spatial = SpatialHash()
gen = 0

def init():
    global prey_list, pred_list, food_list, stats, gen
    prey_list = [Prey(random.uniform(50, ARENA_W - 50), random.uniform(80, ARENA_H - 50)) for _ in range(INITIAL_PREY)]
    pred_list = [Predator(random.uniform(50, ARENA_W - 50), random.uniform(80, ARENA_H - 50)) for _ in range(INITIAL_PRED)]
    food_list = [Food(random.uniform(20, ARENA_W - 20), random.uniform(60, ARENA_H - 20)) for _ in range(FOOD_COUNT)]
    stats = Stats()
    gen = 0

init()

running = True
speed = 1
show_graph = True

while running:
    dt = clock.tick(FPS) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                speed = 5 if speed == 1 else 1
            elif event.key == pygame.K_r:
                init()
            elif event.key == pygame.K_g:
                show_graph = not show_graph
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if ARENA_RECT.collidepoint(mx, my):
                # Click to add food
                food_list.append(Food(mx, my))

    # --- Simulation step ---
    for _ in range(speed):
        gen += 1
        spatial.clear()

        # Spatial index
        for p in prey_list:
            spatial.insert(p, p.x, p.y)
        for f in food_list:
            spatial.insert(f, f.x, f.y)

        # Prey step
        new_prey = []
        for p in prey_list:
            perception = p.perceive(food_list, pred_list)
            p.act(perception)

            # Gather food
            eaten = False
            for food in food_list:
                if math.hypot(p.x - food.x, p.y - food.y) < p.detection_radius:
                    p.energy += REWARD_GATHER
                    p.learn(REWARD_GATHER)
                    food.respawn()
                    eaten = True
                    break

            # Check danger - flee if predator nearby
            nearest_pred = min(pred_list, key=lambda pr: math.hypot(p.x - pr.x, p.y - pr.y), default=None)
            if nearest_pred and math.hypot(p.x - nearest_pred.x, p.y - nearest_pred.y) < p.detection_radius * 2:
                p.learn(-PENALTY_DANGER)

            # Reproduce
            if p.energy > PREY_REPRODUCE_THRESHOLD:
                new_prey.append(p.reproduce())

            # Death
            if p.energy < PREY_DEATH_THRESHOLD:
                p.energy -= 0.01
            else:
                p.energy += 0.005

        prey_list.extend(new_prey)
        prey_list = [p for p in prey_list if p.energy > 0]

        # Predator step
        new_pred = []
        for pr in pred_list:
            perception = pr.perceive(prey_list)
            pr.act(perception)
            pr.starve_counter += 1
            pr.energy -= PENALTY_STARVE * 0.01

            # Catch prey
            caught = None
            for prey in prey_list:
                if math.hypot(pr.x - prey.x, pr.y - prey.y) < 12:
                    caught = prey
                    pr.energy += REWARD_CATCH
                    pr.starve_counter = 0
                    pr.learn(REWARD_CATCH)
                    break

            if caught:
                prey_list.remove(caught)

            # Reproduce
            if pr.energy > PRED_REPRODUCE_THRESHOLD:
                new_pred.append(pr.reproduce())

            # Starvation
            if pr.starve_counter > PRED_STARVE_STEPS:
                pr.energy -= 0.05

        pred_list.extend(new_pred)
        pred_list = [p for p in pred_list if p.energy > 0]

        # Food respawn
        for f in food_list:
            if random.random() < FOOD_RESPAWN_RATE * 0.1:
                f.respawn()

        stats.record(len(prey_list), len(pred_list))

    # --- Render ---
    screen.fill(BG)

    # Arena background
    pygame.draw.rect(screen, (15, 15, 30), ARENA_RECT)
    pygame.draw.rect(screen, GRID_COLOR, ARENA_RECT, 1)

    # Food
    for f in food_list:
        pygame.draw.circle(screen, FOOD_COLOR, (int(f.x), int(f.y)), int(f.dist))

    # Prey
    for p in prey_list:
        r = max(3, int(p.detection_radius * 0.15))
        pygame.draw.circle(screen, PREY_COLOR, (int(p.x), int(p.y)), r)
        # Direction indicator
        if abs(p.vx) > 0.1 or abs(p.vy) > 0.1:
            pygame.draw.line(screen, (100, 255, 150), (int(p.x), int(p.y)),
                             (int(p.x + p.vx * 6), int(p.y + p.vy * 6)), 1)

    # Predators
    for pr in pred_list:
        pygame.draw.circle(screen, PRED_COLOR, (int(pr.x), int(pr.y)), 6)
        # Direction indicator
        if abs(pr.vx) > 0.1 or abs(pr.vy) > 0.1:
            pygame.draw.line(screen, (255, 150, 150), (int(pr.x), int(pr.y)),
                             (int(pr.x + pr.vx * 8), int(pr.y + pr.vy * 8)), 2)

    # Panel
    panel_rect = pygame.Rect(ARENA_W + 10, 10, WIDTH - ARENA_W - 20, HEIGHT - 20)
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=6)

    # Stats
    draw_text(screen, f"Gen: {gen}", (ARENA_W + 20, 20), WHITE)
    draw_text(screen, f"Prey: {len(prey_list)}", (ARENA_W + 20, 40), PREY_COLOR)
    draw_text(screen, f"Predators: {len(pred_list)}", (ARENA_W + 20, 60), PRED_COLOR)
    draw_text(screen, f"Speed: {speed}x [SPACE]", (ARENA_W + 20, HEIGHT - 70), WHITE)
    draw_text(screen, "[R] Reset", (ARENA_W + 20, HEIGHT - 50), WHITE)
    draw_text(screen, "[Click] Add food", (ARENA_W + 20, HEIGHT - 30), WHITE)

    # Graph
    if show_graph:
        graph_rect = pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)
        pygame.draw.rect(screen, (10, 10, 25), graph_rect, border_radius=4)
        pygame.draw.rect(screen, GRID_COLOR, graph_rect, 1)

        if len(stats.prey_history) > 1:
            max_val = max(max(stats.prey_history), max(stats.pred_history), 1)
            # Prey line
            pts = []
            for i, cnt in enumerate(stats.prey_history):
                x = GRAPH_X + int(i / stats.max_history * GRAPH_W)
                y = GRAPH_Y + GRAPH_H - int(cnt / max_val * (GRAPH_H - 20))
                pts.append((x, y))
            if pts:
                pygame.draw.lines(screen, PREY_COLOR, False, pts, 2)

            # Predator line
            pts = []
            for i, cnt in enumerate(stats.pred_history):
                x = GRAPH_X + int(i / stats.max_history * GRAPH_W)
                y = GRAPH_Y + GRAPH_H - int(cnt / max_val * (GRAPH_H - 20))
                pts.append((x, y))
            if pts:
                pygame.draw.lines(screen, PRED_COLOR, False, pts, 2)

    pygame.display.flip()

pygame.quit()
