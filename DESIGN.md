# Learning Sandbox — Design Doc

*A personal sandbox where you watch evolution happen. Population dynamics, co-evolution, natural selection — visible in real time.*

---

## The Core Experience

You're not watching one agent learn. You're watching an **ecosystem**. Prey reproduce and die. Predators hunt and starve. Traits drift over generations. The population graph shows boom/bust cycles in real time. It never fully stabilizes.

The rule: **if you can't watch it evolve in under 5 minutes, it's too complex.**

---

## Predator / Prey — The Flagship Environment

### World

- 2D bounded arena (top-down, clean geometric)
- Prey are stationary. They don't move — they stand still and eat food that spawns around them.
- Predators move. They hunt.
- **Many of each:** 50–200 prey, 10–30 predators to start. Numbers fluctuate based on fitness.

### Prey Behavior

- Prey are stationary but have a **food detection radius**. They "gather" food within range.
- **Food** spawns randomly across the arena. Prey absorb food within their radius.
- **Evolutionary:** No neural net. Trait = detection radius. Reproduce with mutation. The "learning" is natural selection across generations.
- **Reproduction:** Offspring inherit detection radius ± Gaussian mutation.
- **Death:** Prey starve if food is scarce.

### Predator Behavior

- Predators move toward nearby prey and catch them on contact.
- **Catching a prey:** large reward, predator reproduces.
- **Starving:** predators die if they haven't caught prey in N steps.
- **Neural net brain:** MLP — inputs are nearby prey positions + own velocity + recent reward history, outputs are movement direction + speed.
- Predators learn via DQN or policy gradient: reward = catching prey, penalty = time spent chasing without result.
- **Reproduction:** Offspring inherit trained weights + Gaussian mutation. Then continues learning from own experience.

### Co-Evolution Dynamics

- More predators → prey population drops → predators starve → prey recover → predators rebound
- Prey evolve larger detection radii when food is scarce
- Predators evolve faster speeds when prey are evasive
- Neither side ever wins permanently — the ratio oscillates

### What You Watch

- The arena: prey as small green dots (size = detection radius), predators as red dots (size = speed)
- Food particles as white/yellow sparkles
- Population graph in the corner: prey count and predator count over time
- Trait histograms: distribution of detection radii and speeds
- **The key moment:** a regime change — food becomes scarce, prey radii drift up, predators starve, prey boom, predators rebound with new strategies

### Why It's Satisfying

You're watching natural selection happen in real time. Not a single agent — a whole gene pool. The boom/bust cycles are mesmerizing. You can change parameters (food spawn rate, arena size, predator starting count) and watch the ecosystem respond.

---

## Population Stats Panel

Real-time graphs:
- Prey population over time (green line)
- Predator population over time (red line)
- Average prey detection radius (drifting up/down)
- Average predator speed (drifting up/down)
- Food availability (white line)
- Capture rate per predator per minute

---

## Parameters (User-Controllable)

- Initial prey count
- Initial predator count
- Food spawn rate
- Food consumption = reproduction threshold
- Mutation rate for offspring traits
- Arena size
- Simulation speed (1x, 2x, 5x, 10x)

---

## Visual Style

- Dark background arena
- Prey: green circles, radius proportional to detection range
- Predators: red circles, radius proportional to speed
- Food: small white/yellow particles with subtle glow
- Population graph: dark panel, bright colored lines
- Trait histograms: overlaid on population graph or in a separate panel

Font: monospace or geometric sans. Not cartoony — scientific/minimal.

---

## Technical Approach

**Stack:** Python + Pygame (or HTML Canvas for web-based rendering)

**Agent learning:**
- Prey: small MLP neural net trained via policy gradient. Inputs = perception state. Outputs = action logits. Reward signal = food gathered.
- Predators: small MLP neural net trained via DQN or policy gradient. Inputs = prey positions + velocity + reward history. Outputs = movement. Reward signal = prey caught.
- **Both:** reproduction passes down trained weights with mutation. Offspring continues learning from own experience. The brain learns, not just the population.

**Population management:**
- Each step: predators act → prey feed → reproduction/death events
- Reproduction: asexual for simplicity (or sexual if we want to mix traits)
- Mutation: Gaussian noise added to offspring traits

**Simulation loop:**
```
while running:
    food_spawn()
    for predator in predators:
        predator.choose_action(state) → move
        if catches_prey(): predator.reproduce(), prey.die()
    for prey in prey:
        prey.gather_food()
        if food_threshold_met(): prey.reproduce()
    predators -= starvation()
    prey -= starvation()
    record_stats()
    render()
```

**Performance:** 200 agents + simple physics + canvas rendering → runs at 60fps on modern hardware.

---

## MVP Scope

**Version 1:**
- One arena
- Evolutionary prey (trait = detection radius)
- RL predators (neural net trained via reward)
- Population graph
- Adjustable simulation speed
- Adjustable starting parameters

**Post-MVP:**
- Multiple arenas (compare runs)
- Sexual reproduction
- Predator trait evolution (speed + sensing)
- Spatial structure (food gradients, walls)
- Export population data

---

## Why This Is Better Than the Original

Single-agent learning shows one mind figuring things out. Population dynamics show **evolution** — emergent strategies, trait drift, co-evolutionary arms races. The history of life on Earth in miniature.

It's also more visually interesting: you're watching hundreds of entities, not one cube. The population graph going up and down is inherently satisfying to watch.

---

## Open Questions

- **Predator learning vs. evolution:** Should predators also evolve (speed as a trait), or is the neural net enough?
- **Prey movement:** Keep them fully stationary, or give them a small random drift?
- **Rendering:** Python + Pygame (fast, simple) or HTML Canvas (sharper, easier to deploy)?
- **Multi-species:** Add a second predator type that uses a different strategy?
