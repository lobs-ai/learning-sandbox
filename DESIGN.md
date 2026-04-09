# Learning Sandbox — Design Doc

*A digital ecosystem. Many agents, co-evolution, survival. Watch selection happen in real time.*

---

## The Concept

An artificial life sandbox — not a single learning agent, but a population of agents that compete, survive, and evolve. Prey stand still to eat, predators hunt prey. The ones that are better at surviving reproduce, their offspring inherit traits, and the population changes over time. You watch selection happen.

Not watching one agent learn. Watching an ecosystem evolve.

---

## The Simulation

**World:** 2D grid. Continuous — agents exist at positions, not locked to cells. Food spawns in patches. Prey congregate around food. Predators hunt prey.

**Prey:**
- Stand still to gather food from patches
- Move away from nearby predators (flee)
- Reproduce when food intake exceeds threshold
- Offspring inherit parent's speed + perception radius with small random mutation
- Die if a predator catches them

**Predators:**
- Move toward nearby prey (hunt)
- Reproduce when prey consumption exceeds threshold
- Offspring inherit parent's speed + hunt accuracy with small random mutation
- Die if they don't eat for too long (starvation)

**What you watch:**
- Population curves: prey count and predator count over time. Classic Lotka-Volterra dynamics — boom and bust cycles, predator-prey oscillations, eventual equilibrium or collapse.
- Agent traits evolving: early prey are slow, clumsy. Over generations, you see faster prey that detect predators from farther away. Same for predators — better hunters emerge.
- Spatial patterns: prey flocking behavior emerges naturally. Predators form pack-like hunting patterns.

**The moment:** when a new predator mutation makes them significantly better hunters — prey population crashes, predators spike, then prey adapts, population rebounds. Watching the arms race in real time.

---

## Visual Design

Minimalist geometric. You need to see hundreds of agents at once.

- **Prey:** small green dots
- **Predators:** slightly larger red dots
- **Food patches:** soft yellow glow regions
- **Background:** near-black, subtle grid optional

No textures, no sprites. Pure shape and motion. The graph is as important as the simulation — population curves overlaid, updating in real time.

---

## Interactions

- **Spawn controls:** set initial prey and predator counts
- **Speed slider:** watch at 1x, 10x, or 100x speed
- **Reset:** new random population, fresh evolution
- **Trait inspector:** click any agent to see its stats (speed, perception radius, age, offspring count)
- **Add disturbance:** drop a predator boom or prey plague to see how the ecosystem responds

---

## Technical Approach

**Stack:** Python + Pygame (2D rendering), or Three.js if we want 3D later.

**Physics:** Simple 2D positions, velocity vectors, perception radius checks.

**Evolution:** Each agent has a trait vector [speed, perception_radius]. On reproduction, offspring traits = parent traits + small Gaussian noise. No genetic algorithms library needed — simple float vectors.

**Simulation loop:**
1. All predators move toward nearest prey in perception range
2. All prey move away from nearest predator in perception range (or toward food if no predator nearby)
3. Food patches grow slowly
4. Eating → energy gain. Energy threshold → reproduction.
5. No energy → death.
6. Render + update population graphs

**Performance:** 500+ agents should run smoothly. Use spatial hashing for O(n) nearest-neighbor queries instead of O(n²).

---

## MVP Scope

**First build:**
- 2D world, ~200 prey + 20 predators
- Simple flee/hunt behaviors
- Food patches, eating, reproduction
- Trait inheritance with mutation
- Population graphs (prey count, predator count over time)
- Speed controls, reset button

**Post-MVP:**
- Multiple predator species (different hunt strategies)
- Prey flocking behaviors
- Environmental changes (food scarcity events, predator plagues)
- 3D version

---

## Why This Is Interesting

Most ML demos show one agent learning. This shows an **ecosystem** learning. The emergent behaviors — flocking, pack hunting, boom-bust cycles — aren't programmed, they arise. Selection happens and you watch it.

The arms race between prey and predator speed/perception is visible in the trait distributions over time. You can see evolution happening.

---

## Open Questions

- **Grid vs continuous:** Locked to cells or free movement?
- **Reproduction:** Sexual (two parents) or asexual (one parent)?
- **Mutation rate:** How fast do traits drift?

---

## Environments

### 1. Avoid the Red

**World:** 2D grid. Red zones spawn randomly. Green zones are safe.

**Agent:** Moves in 4 directions. Reward = time spent in green. Penalty = time spent in red.

**What you watch:** The agent starts dying constantly. Within 2-3 minutes, it learns to identify and avoid red zones. When you move the red zones, it relearns. When you add more red, it adapts density tolerance.

**Why it's satisfying:** Immediate feedback loop. Every failure is visible. Every improvement is visible.

**Complexity:** Trivial. Single state = current cell color. 4 actions. Q-table works fine.

---

### 2. Predator / Prey

**World:** 2D grid. One predator agent, one prey agent.

**Predator goal:** Catch the prey. Reward = catching, penalty = time spent chasing.
**Prey goal:** Avoid the predator. Reward = surviving, penalty = getting caught.

**What you watch:** Early on, the prey moves randomly and gets caught often. The predator learns to corner. Then the prey learns to run to walls, use edges as shields. The predator adapts. Neither fully wins — it becomes an arms race that never fully settles.

**Why it's interesting:** The co-evolution is visible. Both agents are learning simultaneously, each other's learning changes the problem.

**Complexity:** Medium. Both agents see each other's position. Training is asymmetric — one learns to chase, one learns to flee.

---

### 3. Foraging

**World:** 2D grid. Food items scatter randomly. Agent must collect as many as possible.

**Agent goal:** Move to maximize food collected. Reward = food collected, small penalty per step (to encourage efficiency).

**What you watch:** Early path is chaotic, lots of backtracking. Over time the agent develops efficient routes, minimizes redundant movement. You can change food density and watch it adapt its strategy.

**Why it's satisfying:** Efficiency improvement is concrete and measurable. Path length graphs go down visibly.

**Complexity:** Low. State = nearby food positions + current position. Q-table or policy gradient.

---

### 4. Matching Pennies

**World:** A simple adversarial game. You (the human) play against the agent.

**Rules:** Both players choose heads or tails simultaneously. If they match, you win. If they don't, the agent wins.

**Agent goal:** Learn your pattern and exploit it.

**What you watch:** At first the agent plays randomly. Then it starts noticing patterns in your play. If you switch to a counter-strategy, it adapts. By minute 5, it usually has a significant edge. It's unsettling how fast it figures you out.

**Why it's compelling:** The agent is learning about *you*. It has a model of your behavior. When it exploits a pattern you didn't know you had, it's genuinely eerie.

**Complexity:** Very low. State = your last N choices. Actions = heads/tails.

---

### 5. Ant Trail

**World:** 2D grid. A food source at a fixed location. A trail of pheromone markers the agent lays down.

**Agent goal:** Find the food, return home. Reward = food collected. The trail is the memory — it decays over time, but the agent can lay new markers.

**What you watch:** Early on, the agent wanders randomly. Then you see faint trails form. The agent learns to follow its own trails. When food is relocated, the old trails decay and new ones form. Shows memory + relearning in a way that's visually intuitive.

**Why it's good:** Pheromone trails are a beautiful primitive for emergent behavior. Simple rules, complex looking strategies.

**Complexity:** Medium. State = current position + nearby pheromone levels + food/hom direction. Needs a simple physics layer for pheromone diffusion.

---

## Technical Approach

**Stack:** Python + Pygame (2D), or Three.js (3D). Local only.

**Agent:** Q-learning or policy gradient depending on complexity. Start with Q-table for simple environments.

**State management:** Single Python process. Agent state lives in memory, persists to disk on demand (JSON weights).

**No streaming:** Everything runs locally. You watch by having the window open.

**Adding new environments:** Each environment is a class with `reset()`, `step(action)`, `render()`. Swap environments by changing a config or flag.

---

## MVP Scope

**First build:**
- Avoid the Red (primary — fastest to show learning)
- Predator/Prey (secondary — shows multi-agent)
- Foraging (tertiary — shows efficiency)

**Not in first build:** Matching Pennies, Ant Trail. Add later as the platform matures.

**What ships:**
- One codebase
- Three environments
- A simple UI that lets you switch between them
- Weight persistence (save/restore agent state to disk)
- A "reset" button to retrain from scratch

---

## Why Not WebSockets / Streaming

This is intentionally local and simple. The value is in watching the simulation, not in having it accessible remotely. Running on Rafe's machine means zero latency, zero deployment complexity, zero server costs.

Streaming is a future feature if we ever want to share live simulations. For now, it adds nothing.

---

## The Platform Vision (Later)

Eventually this becomes a platform: a launcher that hosts multiple environments, tracks all your agents across sessions, lets you compare learning curves, and shares interesting agents with others.

But version 1 is just: one repo, three environments, runs locally, something you'd actually open.

---

## Open Questions

- **Visual style:** Retro pixel art? Clean geometric? Minimalist neon?
- **3D:** Does a 3D version of Avoid the Red make sense, or is 2D sufficient for the first build?
- **Interaction:** Should you be able to intervene in real-time (move obstacles, teleport agent) or just watch?
- **Human in the loop:** Is Matching Pennies the only human-facing one, or do other environments support human intervention?
