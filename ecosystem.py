#!/usr/bin/env python3
"""
Learning Sandbox — Ecosystem Simulation
Prey and predators both have neural net brains with DQN learning (actor-critic),
both inherit weights on reproduction.
"""

import pygame
import random
import math
import numpy as np

pygame.init()

# Screen
WIDTH, HEIGHT = 1200, 700
ARENA_W, ARENA_H = 900, 600
GRAPH_X, GRAPH_Y = ARENA_W + 20, 50
GRAPH_W, GRAPH_H = 250, 200

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Learning Sandbox")
clock = pygame.time.Clock()

BG = (10, 10, 20)
PREY_COLOR = (50, 220, 100)
PRED_COLOR = (220, 60, 60)
FOOD_COLOR = (255, 230, 80)
GRID_COLOR = (30, 30, 50)
WHITE = (220, 220, 220)
PANEL_BG = (20, 20, 35)

font_renderer = None


def draw_text(surface, text, pos, color=None):
    if color is None:
        color = WHITE
    if font_renderer:
        surf = font_renderer.render(text, True, color)
        surface.blit(surf, pos)
    else:
        w, h = len(text) * 7, 12
        pygame.draw.rect(surface, color, (*pos, w, h), border_radius=1)


# --- Constants ---
FPS = 60
ARENA_RECT = pygame.Rect(10, 50, ARENA_W, ARENA_H)
INITIAL_PREY = 80
INITIAL_PRED = 8
FOOD_COUNT = 60
PREY_REPRODUCE_THRESHOLD = 1.5
PRED_REPRODUCE_THRESHOLD = 4.0
PREY_DEATH_THRESHOLD = 0.0
PRED_STARVE_STEPS = 400
FOOD_RESPAWN_RATE = 0.03
WEIGHT_MUTATION = 0.05
REWARD_GATHER = 1.0
REWARD_CATCH = 4.0
REWARD_CLOSER = 0.1
PENALTY_DANGER = 0.3
PENALTY_STARVE = 0.1
CELL_SIZE = 60


# --- Spatial Hash ---
class SpatialHash:
    def __init__(self, cell_size=CELL_SIZE):
        self.cell_size = cell_size
        self.cells = {}

    def clear(self):
        self.cells = {}

    def _hash(self, x, y):
        return int(x // self.cell_size), int(y // self.cell_size)

    def insert(self, obj, x, y):
        k = self._hash(x, y)
        if k not in self.cells:
            self.cells[k] = []
        self.cells[k].append(obj)

    def query_radius(self, x, y, radius):
        """Return all (obj, dist) within radius. O(1) per cell."""
        cx, cy = self._hash(x, y)
        r = int(radius // self.cell_size) + 1
        results = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                k = (cx + dx, cy + dy)
                if k in self.cells:
                    for obj in self.cells[k]:
                        d = math.hypot(x - obj.x, y - obj.y)
                        if d <= radius:
                            results.append((obj, d))
        return results

    @staticmethod
    def nearest(x, y, candidates, max_n=3):
        if not candidates:
            return []
        dists = [(math.hypot(c.x - x, c.y - y), c) for c in candidates]
        dists.sort(key=lambda t: t[0])
        return [c for _, c in dists[:max_n]]


# --- Agent Base ---
class Agent:
    def __init__(self, x, y, brain=None):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.energy = 1.0
        self.age = 0
        self.speed = 1.0
        self.brain = brain  # must be set by subclass
        self.last_state = None
        self.last_action = None
        self.q_values = None  # for rendering/debug

    def _discretize(self, perception):
        """Coarse discretization for Q-table. Bin continuous inputs into 5 bins."""
        bins = np.zeros(len(perception))
        for i, v in enumerate(perception):
            bins[i] = int(np.clip(v * 5, 0, 4))
        return tuple(bins.astype(int))

    def choose_action(self, perception, epsilon):
        """Epsilon-greedy Q-learning action selection."""
        state = self._discretize(perception)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.brain))
        self.q_values = self.q_table[state]
        if random.random() < epsilon:
            return random.randrange(len(self.brain))
        return int(np.argmax(self.q_values))

    def learn(self, state, action, reward, next_state):
        """Q-learning update: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.brain))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.brain))
        target = reward + 0.9 * np.max(self.q_table[next_state])
        self.q_table[state][action] += 0.3 * (target - self.q_table[state][action])

    def act_greedy(self, perception):
        """Pick best action without exploration."""
        state = self._discretize(perception)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.brain))
        return int(np.argmax(self.q_table[state]))


# --- Prey ---
class Prey(Agent):
    def __init__(self, x, y, brain=None):
        super().__init__(x, y, brain=None)
        self.detection_radius = random.uniform(40, 70)
        self.speed = random.uniform(1.0, 2.0)
        self.q_table = {} if brain is None else brain.q_table.copy()
        self.brain = np.zeros(3)  # 3 actions: 0=wander, 1=seek_food, 2=flee_predator

    def perceive(self, food_near, pred_near):
        """5-D perception: food_dist, food_angle, pred_dist, pred_angle, energy"""
        if food_near:
            f = min(food_near, key=lambda i: i[1])
            fd = f[1] / self.detection_radius
            fa = math.atan2(f[0].y - self.y, f[0].x - self.x) / math.pi
        else:
            fd, fa = 1.0, 0.0
        if pred_near:
            p = min(pred_near, key=lambda i: i[1])
            pd = min(p[1] / (self.detection_radius * 2), 1.0)
            pa = math.atan2(p[0].y - self.y, p[0].x - self.x) / math.pi
        else:
            pd, pa = 1.0, 0.0
        return np.array([fd, fa, pd, pa, self.energy / 2.0])

    def act(self, perception, epsilon=0.1):
        action = self.choose_action(perception, epsilon)
        self.last_action = action
        self.last_state = self._discretize(perception)

        target_angle = random.uniform(0, 2 * math.pi)
        if action == 1 and perception[0] < 0.9:
            target_angle = perception[1] * math.pi
        elif action == 2 and perception[2] < 0.9:
            target_angle = perception[3] * math.pi + math.pi

        self.vx = math.cos(target_angle) * self.speed
        self.vy = math.sin(target_angle) * self.speed
        self.x += self.vx
        self.y += self.vy

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

        return self._discretize(perception)

    def learn_step(self, reward, next_perception):
        if self.last_state is not None and self.last_action is not None:
            next_state = self._discretize(next_perception)
            self.learn(self.last_state, self.last_action, reward, next_state)

    def reproduce(self):
        self.energy *= 0.5
        child = Prey(self.x + random.uniform(-10, 10), self.y + random.uniform(-10, 10))
        child.q_table = {k: v.copy() for k, v in self.q_table.items()}
        child.detection_radius = max(25, min(80, self.detection_radius + random.uniform(-5, 5)))
        child.speed = max(0.8, min(2.5, self.speed + random.uniform(-0.1, 0.1)))
        child.energy = 1.0
        return child


# --- Predator ---
class Predator(Agent):
    def __init__(self, x, y, brain=None):
        super().__init__(x, y, brain=None)
        self.detection_radius = random.uniform(80, 130)
        self.speed = random.uniform(2.0, 3.5)
        self.starve_counter = 0
        self.energy = 2.0  # start with buffer
        self.q_table = {} if brain is None else brain.q_table.copy()
        self.brain = np.zeros(3)  # 3 actions: 0=turn_left, 1=turn_right, 2=straight
        self.angle = random.uniform(0, 2 * math.pi)

    def perceive(self, nearest_prey):
        """7-D: 3 nearest prey (dx, dy each), own velocity, energy"""
        inputs = np.zeros(7)
        for i, p in enumerate(nearest_prey[:3]):
            inputs[i * 2] = (p.x - self.x) / ARENA_W
            inputs[i * 2 + 1] = (p.y - self.y) / ARENA_H
        inputs[6] = self.energy
        return inputs

    def act(self, perception, epsilon=0.1):
        action = self.choose_action(perception, epsilon)
        self.last_action = action
        self.last_state = self._discretize(perception)

        if action == 0:
            self.angle -= 0.4
        elif action == 1:
            self.angle += 0.4

        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed
        self.x += self.vx
        self.y += self.vy

        if self.x < 10:
            self.x = 10
            self.vx *= -1
            self.angle = math.atan2(self.vy, self.vx)
        if self.x > ARENA_W + 10:
            self.x = ARENA_W + 10
            self.vx *= -1
            self.angle = math.atan2(self.vy, self.vx)
        if self.y < 50:
            self.y = 50
            self.vy *= -1
            self.angle = math.atan2(self.vy, self.vx)
        if self.y > ARENA_H + 50:
            self.y = ARENA_H + 50
            self.vy *= -1
            self.angle = math.atan2(self.vy, self.vx)

        return self._discretize(perception)

    def learn_step(self, reward, next_perception):
        if self.last_state is not None and self.last_action is not None:
            next_state = self._discretize(next_perception)
            self.learn(self.last_state, self.last_action, reward, next_state)

    def reproduce(self):
        self.energy *= 0.5
        child = Predator(self.x, self.y)
        child.q_table = {k: v.copy() for k, v in self.q_table.items()}
        child.speed = max(1.5, min(4.0, self.speed + random.uniform(-0.2, 0.2)))
        child.energy = 2.0
        return child


# --- Food ---
class Food:
    def __init__(self, x=None, y=None):
        if x is None:
            self.respawn()
        else:
            self.x, self.y = x, y
        self.radius = random.uniform(4, 7)

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
    prey_list = [Prey(random.uniform(50, ARENA_W - 50), random.uniform(80, ARENA_H - 50))
                 for _ in range(INITIAL_PREY)]
    pred_list = [Predator(random.uniform(50, ARENA_W - 50), random.uniform(80, ARENA_H - 50))
                 for _ in range(INITIAL_PRED)]
    food_list = [Food() for _ in range(FOOD_COUNT)]
    stats = Stats()
    gen = 0


init()

running = True
speed = 1
show_graph = True
epsilon = 0.2  # exploration rate


def draw_agents():
    """Batch render all agents efficiently."""
    # Draw all food
    for f in food_list:
        pygame.draw.circle(screen, FOOD_COLOR, (int(f.x), int(f.y)), int(f.radius))

    # Draw all prey (simple circles, no direction lines for performance)
    prey_positions = [(int(p.x), int(p.y)) for p in prey_list]
    prey_radii = [max(3, int(p.detection_radius * 0.12)) for p in prey_list]
    for (x, y), r in zip(prey_positions, prey_radii):
        pygame.draw.circle(screen, PREY_COLOR, (x, y), r)

    # Draw all predators
    for pr in pred_list:
        pygame.draw.circle(screen, PRED_COLOR, (int(pr.x), int(pr.y)), 6)
        # Direction indicator
        pygame.draw.line(screen, (255, 120, 120), (int(pr.x), int(pr.y)),
                         (int(pr.x + pr.vx * 7), int(pr.y + pr.vy * 7)), 2)


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
                epsilon = 0.05 if speed > 1 else 0.2  # less explore at high speed
            elif event.key == pygame.K_r:
                init()
            elif event.key == pygame.K_g:
                show_graph = not show_graph
            elif event.key == pygame.K_PERIOD:
                epsilon = max(0, epsilon - 0.05)
            elif event.key == pygame.K_COMMA:
                epsilon = min(0.5, epsilon + 0.05)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if ARENA_RECT.collidepoint(mx, my):
                food_list.append(Food(mx, my))

    for _ in range(speed):
        gen += 1

        spatial.clear()
        for p in prey_list:
            spatial.insert(p, p.x, p.y)
        for f in food_list:
            spatial.insert(f, f.x, f.y)

        # --- Prey step ---
        new_prey = []
        for p in prey_list:
            food_near = spatial.query_radius(p.x, p.y, p.detection_radius)
            food_near = [(f, d) for f, d in food_near if isinstance(f, Food)]
            pred_near = spatial.query_radius(p.x, p.y, p.detection_radius * 2)
            pred_near = [(pr, d) for pr, d in pred_near if isinstance(pr, Predator)]

            perception = p.perceive(food_near, pred_near)
            next_state = p.act(perception, epsilon=epsilon)

            # Learning: reward for eating, small reward for being near food
            reward = -0.01  # default: slight penalty for existing
            eaten = False
            for food, dist in food_near:
                if dist < p.detection_radius * 0.7:
                    reward = REWARD_GATHER
                    p.energy += 0.5
                    food.respawn()
                    eaten = True
                    break

            if not eaten and food_near:
                nearest_dist = min(d for _, d in food_near)
                reward += (1.0 - nearest_dist / p.detection_radius) * REWARD_CLOSER

            if pred_near:
                nearest_pred_dist = min(d for _, d in pred_near)
                if nearest_pred_dist < p.detection_radius:
                    reward -= PENALTY_DANGER

            p.learn_step(reward, perception)
            p.energy += 0.005

            if p.energy > PREY_REPRODUCE_THRESHOLD:
                new_prey.append(p.reproduce())

            if p.energy < PREY_DEATH_THRESHOLD:
                p.energy -= 0.02

        prey_list.extend(new_prey)
        prey_list = [p for p in prey_list if p.energy > 0]

        # --- Predator step ---
        new_pred = []
        for pr in pred_list:
            prey_in_range = spatial.query_radius(pr.x, pr.y, pr.detection_radius)
            prey_in_range = [(p, d) for p, d in prey_in_range if isinstance(p, Prey)]
            prey_candidates = [p for p, _ in prey_in_range]
            nearest_prey = SpatialHash.nearest(pr.x, pr.y, prey_candidates, max_n=3)

            # Track previous distance to nearest prey
            prev_dist = float('inf')
            if prey_candidates:
                prev_dist = min(d for _, d in prey_in_range)

            perception = pr.perceive(nearest_prey)
            next_state = pr.act(perception, epsilon=epsilon)

            reward = -PENALTY_STARVE * 0.05
            pr.starve_counter += 1
            pr.energy -= PENALTY_STARVE * 0.02

            # Hunt learning: reward for getting closer
            if prey_in_range:
                curr_dist = min(d for _, d in prey_in_range)
                if curr_dist < prev_dist:
                    reward += REWARD_CLOSER * (1.0 - curr_dist / pr.detection_radius)

            # Catch
            caught = None
            for prey in list(prey_list):
                if math.hypot(pr.x - prey.x, pr.y - prey.y) < 14:
                    caught = prey
                    pr.energy += REWARD_CATCH
                    pr.starve_counter = 0
                    reward = REWARD_CATCH
                    break

            if caught:
                prey_list.remove(caught)

            pr.learn_step(reward, perception)

            if pr.energy > PRED_REPRODUCE_THRESHOLD and len(new_pred) < 2:
                new_pred.append(pr.reproduce())

            if pr.starve_counter > PRED_STARVE_STEPS:
                pr.energy -= 0.1

        pred_list.extend(new_pred)
        pred_list = [p for p in pred_list if p.energy > 0]

        # Food respawn
        for f in food_list:
            if random.random() < FOOD_RESPAWN_RATE:
                f.respawn()

        stats.record(len(prey_list), len(pred_list))

    # --- Render ---
    screen.fill(BG)
    pygame.draw.rect(screen, (15, 15, 30), ARENA_RECT)
    pygame.draw.rect(screen, GRID_COLOR, ARENA_RECT, 1)

    draw_agents()

    # Panel
    panel_rect = pygame.Rect(ARENA_W + 10, 10, WIDTH - ARENA_W - 20, HEIGHT - 20)
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=6)

    draw_text(screen, f"Gen: {gen}", (ARENA_W + 20, 20), WHITE)
    draw_text(screen, f"Prey: {len(prey_list)}", (ARENA_W + 20, 40), PREY_COLOR)
    draw_text(screen, f"Predators: {len(pred_list)}", (ARENA_W + 20, 60), PRED_COLOR)
    draw_text(screen, f"Speed: {speed}x [SPACE]", (ARENA_W + 20, HEIGHT - 90), WHITE)
    draw_text(screen, f"Explore: {epsilon:.0%} [,/.]", (ARENA_W + 20, HEIGHT - 70), WHITE)
    draw_text(screen, "[R] Reset", (ARENA_W + 20, HEIGHT - 50), WHITE)
    draw_text(screen, "[Click] Add food", (ARENA_W + 20, HEIGHT - 30), WHITE)

    # Graph
    if show_graph:
        graph_rect = pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)
        pygame.draw.rect(screen, (10, 10, 25), graph_rect, border_radius=4)
        pygame.draw.rect(screen, GRID_COLOR, graph_rect, 1)

        if len(stats.prey_history) > 1:
            max_val = max(max(stats.prey_history), max(stats.pred_history), 1)
            pts_prey = [(GRAPH_X + int(i / stats.max_history * GRAPH_W),
                         GRAPH_Y + GRAPH_H - int(cnt / max_val * (GRAPH_H - 20)))
                        for i, cnt in enumerate(stats.prey_history)]
            pts_pred = [(GRAPH_X + int(i / stats.max_history * GRAPH_W),
                         GRAPH_Y + GRAPH_H - int(cnt / max_val * (GRAPH_H - 20)))
                        for i, cnt in enumerate(stats.pred_history)]
            if pts_prey:
                pygame.draw.lines(screen, PREY_COLOR, False, pts_prey, 2)
            if pts_pred:
                pygame.draw.lines(screen, PRED_COLOR, False, pts_pred, 2)

    pygame.display.flip()

pygame.quit()
