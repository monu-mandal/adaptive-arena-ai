# Adaptive Arena AI

> A 2D combat game where the enemy learns to beat you — in real time, in the browser.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Built With](https://img.shields.io/badge/built%20with-React%20%2B%20Vanilla%20RL-purple)

---

## What Is This?

Adaptive Arena AI is a reinforcement learning portfolio project disguised as a fighting game.

You play as a human-controlled fighter. Your opponent is an RL agent running a neural network trained **live, in your browser** — no server, no pre-trained weights, no external ML library. Every round it plays against you, it updates its policy and gets harder.

The longer you play, the smarter it gets.

---

## Demo

```
npm install && npm run dev
```

Open `http://localhost:5173` — no setup, no API keys, no backend.

---

## Features

### Must-Have (V1 Complete)
- [x] **Playable game** — move, jump (double), attack, block, dash, combo system
- [x] **RL enemy agent** — Q-learning with neural network policy
- [x] **Online adaptation** — agent trains every 6 frames during live gameplay
- [x] **AI leveling system** — enemy levels up on player victory, epsilon drops, difficulty scales
- [x] **Behavior evolution** — watch the AI shift from random to deliberate over episodes
- [x] **Stats HUD** — live epsilon, win rate, episode count, reward sparkline
- [x] **GitHub-ready structure** — this repo

### Should-Have (V1 Complete)
- [x] **Reward history graph** — sparkline chart on game-over screen
- [x] **AI personalities** — 5 archetypes (Aggressive, Defensive, Adaptive, Berserker, Tactician)
- [x] **Behavior state display** — real-time AI action badge in-game

---

## Architecture

### Neural Network (from scratch)
No TensorFlow. No PyTorch. No ONNX. A hand-rolled feedforward net in ~60 lines of vanilla JS:

```
Input (12) → Hidden (24) → Hidden (18) → Output (7)
```

- **Activations:** ReLU (hidden), Softmax (output)
- **Init:** He initialization (`√(2/fan_in)`)
- **Update:** TD-error gradient on output layer weights

### Q-Learning
- **Algorithm:** DQN with experience replay
- **Replay buffer:** 800 transitions, uniform sampling, batch size 32
- **Target network:** hard sync every 50 update steps
- **Exploration:** ε-greedy, ε ∈ [0.05, 0.90], decays per training step
- **Discount factor (γ):** 0.92

### State Space (12 features)
| Index | Feature |
|-------|---------|
| 0 | Relative X position (normalized) |
| 1 | Relative Y position (normalized) |
| 2 | AI velocity X |
| 3 | AI velocity Y |
| 4 | Player velocity X |
| 5 | Player velocity Y |
| 6 | AI HP ratio |
| 7 | Player HP ratio |
| 8 | AI is attacking |
| 9 | Player is attacking |
| 10 | AI on ground |
| 11 | Frame phase (mod 60) |

### Action Space (7 discrete)
`idle` · `moveLeft` · `moveRight` · `jump` · `attack` · `block` · `dash`

### Reward Shaping
```
+0.3 × damage dealt to player
-0.3 × damage received from player
+0.02 per frame within 100px of player
+0.10 for blocking a player attack
+0.01 × (ai_hp - player_hp) / MAX_HP
```

### Personality System
Each personality applies a **prior bias** to the action distribution before training kicks in. This shapes early-episode behavior and persists as a soft constraint throughout learning.

| Personality | Bias |
|-------------|------|
| Aggressive | +attack, +moveRight |
| Defensive | +block, +idle |
| Adaptive | Balanced |
| Berserker | +attack, +jump |
| Tactician | +jump, +dash |

---

## Controls

| Key | Action |
|-----|--------|
| `← →` or `A D` | Move |
| `↑` or `W` or `Space` | Jump (double-jump supported) |
| `Z` or `J` | Attack (combo on repeat) |
| `X` or `K` | Block (75% damage reduction) |
| `C` or `L` | Dash |

---

## Project Structure

```
adaptive-arena-ai/
├── src/
│   └── AdaptiveArena.jsx     # Entire game: NN, RL agent, physics, rendering, UI
├── public/
│   └── index.html
├── package.json
├── vite.config.js
└── README.md
```

Everything lives in `AdaptiveArena.jsx` by design — this is a portfolio demo, not a production codebase. The goal is readability and auditability in one file.

---

## How the AI Evolves

| Episode | Behavior |
|---------|----------|
| 1–5 | Mostly random, shaped by personality bias |
| 6–15 | Starts closing distance, occasional attack timing |
| 16–30 | Learns to punish player attacks, uses block |
| 30+ | Adapts to player patterns, exploits openings |

Epsilon (exploration rate) is displayed live in the HUD. Watch it drop as the agent commits to its learned policy.

---

## Design Decisions

**Why no ML library?**
The entire RL stack fits in ~100 lines. Shipping PyTorch WASM or TF.js for a 3-layer net is overkill and obscures what's actually happening. The raw implementation is easier to read, audit, and extend.

**Why online learning instead of offline pre-training?**
The point is watching the agent adapt *to you*. Pre-training produces a fixed policy. Online learning produces an agent that reacts to your playstyle.

**Why one file?**
Portfolio legibility. Reviewers can read the entire system — physics, RL, rendering, UI — in one scroll.

---

## Extending This

Ideas for V2:
- **LSTM / recurrent policy** — let the agent remember sequences of player moves
- **Self-play training mode** — AI vs AI to bootstrap a stronger prior
- **Checkpoint export/import** — serialize `qNet.weights` to `localStorage` or JSON download
- **Replay recording** — log state history, render replays
- **Multiple arenas** — platforms, obstacles, hazards that change optimal strategy
- **Population-based training** — run N agents in parallel, keep the best weights

---

## License

MIT — use it, fork it, learn from it.

---

*Built as an RL engineering portfolio project. The AI genuinely learns. Give it 10+ rounds.*
