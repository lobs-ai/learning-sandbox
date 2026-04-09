# Learning Sandbox — Design Doc

*A collection of simple ML environments where you watch agents learn. Local-only, no streaming, no server. Run it on your machine.*

---

## The Concept

A personal sandbox of learning simulations — small worlds where agents figure things out, visually. Not a demo, not a research tool. Something you'd keep open on a second monitor and occasionally glance at while working, like a lava lamp for machine learning.

The rule: **if you can't watch it learn in under 5 minutes, it's too complex.**

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
