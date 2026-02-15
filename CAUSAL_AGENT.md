# Causal Structure Discovery Agent: Beyond Observed Counterfactuals

**Project Proposal: Intrinsically Motivated Causal Learning for Autonomous Agents**

---

## Intro

This document proposes an extension to the current Counterfactual Q-Learning architecture that addresses a fundamental limitation: **the agent currently requires an oracle or environment-provided counterfactual feedback to learn from unchosen actions**. In real-world scenarios, such feedback is rarely available. 

We propose developing an **intrinsically motivated causal discovery agent** inspired by how human children learn cause-effect relationships in their environment. Rather than being told "what would have happened," the agent actively explores to discover the **causal structure** of its world, enabling it to generate its own principled counterfactual predictions via Pearl's do-calculus.

This moves the system from **Level 3 (Counterfactual)** of Pearl's Causal Hierarchy — which assumes a causal model is given — to **Level 2 (Interventional)** — where the agent discovers the causal model itself through systematic experimentation.

---

## 1. Motivation: The Counterfactual Oracle Problem

### 1.1 Current System Limitations

The existing Counterfactual Q-Learning implementation operates in two distinct modes:

**Mode 1: Environment-Provided Counterfactuals** (Regret Grid World, Depleting Resources)
- Environment explicitly reveals: `info["counterfactual_rewards"] = {goal_A: 3.0, goal_C: 10.0}`
- Agent learns from unchosen options without needing to simulate
- **Problem:** This is unrealistic for most real-world tasks

**Mode 2: Model-Based Counterfactuals** (Shifting Hazards, Moving Obstacles)
- Agent's world model generates counterfactual outcomes: `cf_reward = world_model.predict(s, a_unchosen)`
- Accuracy depends on world model quality
- **Problem:** The world model is a simple tabular associational model, not a causal model

### 1.2 The Missing Piece: Discovering Causal Structure

Consider a robot learning to operate kitchen appliances:
- **Observation:** "When the toaster is plugged in AND the lever is down, toast appears"
- **Association:** $P(\text{toast} \mid \text{plugged}, \text{lever down})$ (what the current model learns)
- **Intervention:** $P(\text{toast} \mid do(\text{lever down}))$ — does the lever *cause* toast, or is it spurious?
- **Causal Graph:**
  ```
  Electricity → Toaster → Toast
       ↓          ↑
   PluggedIn   LeverDown
  ```

The current tabular world model cannot distinguish:
- **Confounder:** Electricity causes both lever engagement and heating
- **Mediator:** Lever down enables heating element
- **Collider:** Toast outcome is a collision of multiple causes

**Without the causal graph, counterfactual predictions are unreliable.**

---

## 2. Developmental Psychology: How Children Discover Cause and Effect

### 2.1 Alison Gopnik's "Scientist in the Crib"

Developmental psychologist **Alison Gopnik** (UC Berkeley) has demonstrated that children engage in sophisticated **interventional causal reasoning** much earlier than previously thought. Key developmental milestones:

| Age | Capability | Experimental Evidence |
|-----|-----------|----------------------|
| **8 months** | **Contingency detection** | Rovee-Collier mobile paradigm: Babies learn that kicking causes mobile movement vs. random motion |
| **18–24 months** | **Interventional learning** | "Blicket detector" (Gopnik & Sobel, 2000): Children systematically test which blocks activate a machine |
| **3–4 years** | **Observation vs. Intervention** | Children prefer intervening on ambiguous causal systems rather than passive observation |
| **4–5 years** | **Counterfactual reasoning** | Spontaneous "what if" reasoning about alternative outcomes (Harris et al., 1996) |

#### The Blicket Detector Paradigm

This elegant experiment demonstrates interventional causal learning:

**Setup:**
- A "blicket detector" machine lights up when certain blocks ("blickets") are placed on it
- Child observes: Blocks A+B together → machine activates
- Question: Which block(s) cause activation?

**Child's Strategy (18–24 months):**
1. **Intervene:** Test block A alone → machine activates
2. **Intervene:** Test block B alone → no activation
3. **Conclusion:** A is a blicket, B is not
4. **Generalization:** Child then correctly predicts new combinations (A+C, B+C)

**Key insight:** Children don't just learn $P(\text{light} \mid A)$ (associational). They learn $P(\text{light} \mid do(A))$ (interventional) — the causal effect of A. This is precisely Pearl's distinction between seeing and doing.

### 2.2 Two Types of Learning in Children

| Learning Type | Question | Brain System | Example |
|--------------|----------|--------------|---------|
| **Reward Learning** | "What should I choose?" | Ventral striatum (dopamine) | "Hot stove burned me → avoid" |
| **Structure Learning** | "How does the world work?" | Hippocampus + Prefrontal cortex | "Switches cause lights, not vice versa" |

Children pursue **structure learning even without immediate reward**. A 2-year-old will flip light switches repeatedly, not because it's rewarding, but because they are discovering the **causal mechanism**.

**Implication for RL:** Agents need two reward signals — extrinsic (task performance) and **intrinsic (causal discovery)**.

---

## 3. Neuroscience: Two Dopaminergic Circuits

Two distinct dopaminergic pathways support this dual learning:

### 3.1 Circuit 1: Reward Prediction Error (VTA → Ventral Striatum)

- **Classical Schultz (1997) signal:** $\delta = r - V(s)$
- Fires when reward is better/worse than expected
- Drives value-based learning (Q-learning, CF Q-learning)
- **What it learns:** "Which actions lead to reward?"

### 3.2 Circuit 2: Information Gain / Curiosity (VTA → Prefrontal Cortex / ACC)

- **Bromberg-Martin & Hikosaka (2009):** Monkeys prefer cues that reveal information, even with identical reward
- Anterior cingulate cortex (ACC) tracks **prediction error about world dynamics**, not reward
- **Kang et al. (2009):** Hippocampus + PFC activate when humans encounter "curiosity gaps"
- **What it learns:** "How does the environment work?"

**Key Finding:** Circuit 2 fires when the agent **resolves uncertainty about the causal structure**, independently of reward magnitude.

**Implications:**
- Current CF Q-learning uses only Circuit 1
- Children use both → they actively seek **structural discovery**
- We need to model Circuit 2 as **intrinsic motivation for causal learning**

---

## 4. Pearl's Causal Hierarchy and Developmental Stages

| Pearl Level | Formal | Child Development | Brain Region | Current Project |
|-------------|--------|------------------|--------------|-----------------|
| **1. Association** | $P(Y \mid X)$ | Pavlovian conditioning (birth–8 months) | Amygdala, cerebellum | ❌ Not the focus |
| **2. Intervention** | $P(Y \mid do(X))$ | Blicket detector experiments (18–24 months) | Dorsolateral PFC, ACC | **⚠️ Missing (this proposal)** |
| **3. Counterfactual** | $P(Y_x \mid X=x', Y=y')$ | "What if" reasoning (4+ years) | vmPFC, OFC | ✅ Current CF Q-learning |

**Problem:** The current project jumps directly to Level 3 by assuming a world model good enough for counterfactual inference. But **Level 2 is the foundation** — without learning what interventions *do*, counterfactual predictions are unreliable.

**This proposal targets Level 2:** An agent that discovers causal structure through systematic intervention, then uses that structure for principled Level 3 reasoning.

---

## 5. Formal Framework: Causal MDPs with Intrinsic Motivation

### 5.1 Extended MDP Definition

We extend the standard MDP to include explicit causal structure and dual rewards:

$$\mathcal{M}_{\text{causal}} = (\mathcal{S}, \mathcal{A}, \mathcal{G}, \text{SEM}, R_{\text{ext}}, R_{\text{int}}, \gamma)$$

**Components:**

- $\mathcal{S}$: State space (as before)
- $\mathcal{A}$: Action space — now interpreted as **interventions** $do(V_j = v)$
- $\mathcal{G} = (V, E)$: **Causal DAG** over state variables $V = \{V_1, \ldots, V_d\}$ (initially unknown)
- $\text{SEM}$: **Structural Equation Model** defining mechanisms:
  $$V_i = f_i(\text{Pa}(V_i), U_i)$$
  where $\text{Pa}(V_i)$ are parents in $\mathcal{G}$ and $U_i$ is exogenous noise
- $R_{\text{ext}}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: Extrinsic task reward (can be zero during exploration phase)
- $R_{\text{int}}: \mathcal{S} \times \mathcal{A} \times \mathcal{S}' \to \mathbb{R}$: **Intrinsic reward for causal discovery** (the key contribution)
- $\gamma$: Discount factor

### 5.2 The Causal Discovery Objective

The agent maintains a **posterior distribution** over possible causal structures:

$$P(\mathcal{G} \mid \mathcal{D}_t)$$

where $\mathcal{D}_t = \{(s_i, a_i, s'_i)\}_{i=1}^t$ is the history of state transitions.

**Goal during exploration:** Maximize information gain about $\mathcal{G}$:

$$\mathcal{G}^* = \arg\max_{\mathcal{G}} P(\mathcal{G} \mid \mathcal{D})$$

**Goal during exploitation:** Use learned $\mathcal{G}$ for counterfactual Q-learning with principled predictions.

---

## 6. Intrinsic Reward Strategies for Causal Discovery

We propose four complementary approaches to $R_{\text{int}}$, inspired by psychology and neuroscience:

### 6.1 Strategy A: Information Gain over Causal Graphs (Bayesian Viewpoint)

**Definition:**
$$R_{\text{int}}^{(A)}(t) = H[\mathcal{G} \mid \mathcal{D}_t] - \mathbb{E}[H[\mathcal{G} \mid \mathcal{D}_t, (s_t, a_t, s'_{t+1})]]$$

The agent receives intrinsic reward proportional to how much the new observation reduces uncertainty about the causal graph structure.

**Interpretation:**
- **High reward:** Observation disambiguates between competing graph hypotheses
- **Low reward:** Observation is consistent with most hypotheses (redundant)

**Connection to development:** This mirrors Gopnik's "little scientist" hypothesis — children preferentially intervene on variables where their structural uncertainty is highest.

**Implementation:**
- Maintain posterior over graph structures using constraint-based or score-based methods
- Compute expected KL divergence: $\mathbb{E}_{s'} [D_{KL}(P(\mathcal{G} \mid \mathcal{D}_t, s') \| P(\mathcal{G} \mid \mathcal{D}_t))]$
- Use this as the intrinsic reward signal

**Algorithms:**
- **BALD** (Bayesian Active Learning by Disagreement): Tong & Koller (2001)
- **CausalCuriosity**: Sontakke et al. (NeurIPS 2021)

---

### 6.2 Strategy B: Interventional Prediction Error (Surprise-Based)

**Definition:**
$$R_{\text{int}}^{(B)}(t) = \|P_{\hat{\mathcal{G}}}(s'_{t+1} \mid do(a_t), s_t) - \delta_{s'_{t+1}}\|^2$$

The agent receives reward when its causal model is **wrong** about the effect of an intervention.

**Interpretation:**
- **High reward:** Agent encountered an outcome it didn't predict → model is incomplete
- **Low reward:** Outcome matches prediction → model is accurate in this region

**Connection to neuroscience:** This corresponds to the **ACC prediction error signal** — not "was the reward unexpected?" but "was the *world dynamics* unexpected?"

**Implementation:**
- Use current $\hat{\mathcal{G}}$ and $\hat{\text{SEM}}$ to predict $\hat{s}' = \text{SEM}(do(a), s, \hat{\mathcal{G}})$
- Compare to actual $s'$
- Squared error serves as intrinsic reward
- Update $\hat{\mathcal{G}}$ to reduce this error

**Algorithms:**
- **ICM** (Intrinsic Curiosity Module): Pathak et al. (2017) — adapted for causal features
- **RND** (Random Network Distillation): Burda et al. (2019) — measures state novelty

---

### 6.3 Strategy C: Interventional Entropy Reduction (Optimal Experimental Design)

**Definition:**
For each candidate intervention $do(X=x)$, compute expected information gain about a target variable $Y$:

$$\text{IG}(X \to Y) = H[Y \mid \mathcal{D}_t] - \mathbb{E}_{do(X=x)}[H[Y \mid \mathcal{D}_t, do(X=x), Y_{\text{obs}}]]$$

Choose the intervention that maximally reduces uncertainty about $Y$:

$$R_{\text{int}}^{(C)}(t) = \max_{(X,x)} \text{IG}(X \to Y)$$

**Interpretation:**
- Agent selectively intervenes on variables that reveal the most about causal structure
- This is **optimal experimental design** (OED) for causal discovery

**Connection to science:** This is how scientists design experiments — choose the intervention most likely to distinguish between competing theories.

**Implementation:**
- For each candidate action (intervention), simulate outcomes under all $\mathcal{G}_i \in \text{hypothesis space}$
- Compute expected entropy reduction
- Take action with maximal expected IG
- Computationally expensive but optimal

**Algorithms:**
- **ABCD** (Adaptive Bayesian Causal Discovery): Agrawal et al. (2019)
- **DCDI** (Differentiable Causal Discovery with Interventions): Brouillard et al. (ICML 2020)

---

### 6.4 Strategy D: Causal Influence Detection (Lightweight Heuristic)

**Definition:**
$$R_{\text{int}}^{(D)}(t) = \sum_{i=1}^d \Big|P(V_i^{t+1} \mid do(a_t)) - P(V_i^{t+1} \mid \text{observe only})\Big|$$

The agent is rewarded for discovering **which variables it can influence** via its actions.

**Interpretation:**
- Measures the **causal power** of action $a_t$
- High reward if $do(a_t)$ changes many downstream variables
- Low reward if action has no effect

**Connection to infancy research:** This mirrors **Watson's (1966) contingency detection** — 8-month-olds kick more when their kick causes a mobile to move (contingent) versus random motion (non-contingent). They detect $|P(Y \mid do(\text{kick})) - P(Y)| > 0$.

**Implementation:**
- For each action, compare predicted distributions under intervention vs. observation
- Total variation distance serves as intrinsic reward
- Lightweight: doesn't require full graph inference
- Builds a "controllability map" of the environment

**Algorithms:**
- **CICA** (Causal Influence-based Curiosity Agent)
- **Empowerment** (Klyubin et al., 2005): Channel capacity $I(A; S_{future})$ between actions and future states

---

### 6.5 Comparison of Strategies

| Strategy | Complexity | Sample Efficiency | Theoretical Optimality | Best For |
|----------|------------|------------------|----------------------|----------|
| **A: Info Gain** | High (graph posterior) | Medium | Optimal (Bayesian) | Full structure discovery |
| **B: Prediction Error** | Low | High | Heuristic | Online learning with neural nets |
| **C: Interventional OED** | Very High | Highest | Optimal (OED) | High-stakes domains (robotics) |
| **D: Causal Influence** | Very Low | Medium | Heuristic | Real-time applications, large state spaces |

**Recommendation:** Start with **Strategy D** (simplest) for proof-of-concept, then incorporate **Strategy B** (prediction error) for online adaptation. Reserve **Strategy A/C** for domains where sample efficiency is critical and computational budget permits.

---

## 7. Integration with Existing CF Q-Learning

The complete system operates in three phases:

### Phase 1: Exploration with Intrinsic Motivation (NEW)

**Objective:** Discover causal structure $\mathcal{G}$ and structural equations $\text{SEM}$

```python
R_total = R_int(s, a, s')  # Only intrinsic reward
# Agent explores to maximize information gain about G
# World model learns: G, SEM = f_i(Pa(V_i), U_i)
```

**Termination Condition:**
- Graph posterior converges: $H[\mathcal{G} \mid \mathcal{D}] < \epsilon$
- Or: Fixed exploration budget (e.g., 1000 episodes)

**Output:**
- Causal DAG $\hat{\mathcal{G}}$
- Structural equations $\hat{\text{SEM}}$

---

### Phase 2: Exploitation with CF Q-Learning (EXISTING)

**Objective:** Maximize task reward using learned causal structure

```python
R_total = R_ext(s, a)  # Only extrinsic (task) reward
# Counterfactual queries now use learned G and SEM
cf_prediction = world_model.counterfactual_query(s, a_unchosen, G, SEM)
# CF Q-learning updates as before
```

**Key difference:** Counterfactual predictions are now **principled** because they use the true causal structure:

1. **Abduction:** Given $(X=x, Y=y)$, infer exogenous variables $U$ consistent with observations
2. **Action:** Replace structural equation for $X$ with $X := x'$ (intervention)
3. **Prediction:** Forward propagate through modified SEM to get $Y_{x'}$

This is Pearl's three-step counterfactual algorithm, mathematically guaranteed to handle confounders, mediators, and colliders correctly.

---

### Phase 3: Transfer and Adaptation (NEW)

**Objective:** Rapidly adapt to new tasks in the same causal environment

```python
# When task reward changes but physics stays the same:
R_total = R_ext_new(s, a) + λ·R_int(s, a, s')
# Reuse G for instant counterfactual reasoning
# Only reward mapping changes, not world dynamics
```

**Example:** Robot trained on "pick red block" with learned $\mathcal{G}$ (gripper physics, object dynamics). New task: "stack blue blocks." The causal structure ($\mathcal{G}$) remains valid → instant transfer.

**Advantage:** Causal structure is **task-independent**, only reward function is task-specific.

---

## 8. Architectural Changes to Current Codebase

Proposed extensions to `src/causalrl/`:

```
src/causalrl/
├── causal_discovery/          # NEW MODULE
│   ├── __init__.py
│   ├── causal_graph.py        # DAG representation (NetworkX-based)
│   │                          # - Methods: d_separation, backdoor criterion
│   │                          # - do-calculus operations
│   ├── structure_learner.py   # Graph structure learning
│   │                          # - Bayesian: posterior over G
│   │                          # - Score-based: BIC, MDL scoring
│   │                          # - Constraint-based: PC algorithm
│   ├── structural_equations.py # SEM V_i = f_i(Pa(V_i), U_i)
│   │                          # - Parametric (linear, additive noise)
│   │                          # - Neural (MLP for complex f_i)
│   ├── intrinsic_reward.py    # R_int computation
│   │                          # - InfoGain (Strategy A)
│   │                          # - PredictionError (Strategy B)
│   │                          # - InterventionalIG (Strategy C)
│   │                          # - CausalInfluence (Strategy D)
│   └── interventional_agent.py # Agent for Phase 1 (exploration)
│
├── world_models/
│   ├── tabular.py             # (existing) Count-based model
│   ├── oracle.py              # (existing) Perfect model
│   └── structural.py          # NEW: SCM-based world model
│                              # - Uses learned G and SEM
│                              # - Implements Pearl's 3-step CF algorithm
│                              # - Methods:
│                              #   - intervene(s, do(X=x))
│                              #   - counterfactual(s, a_actual, a_cf, r_actual)
│                              #   - abduction(evidence) → infer U
│
├── agents/
│   ├── counterfactual_rl.py  # (existing) CF Q-learning with composite errors
│   └── causal_agent.py        # NEW: Three-phase meta-agent
│                              # - Phase 1: Exploration (R_int)
│                              # - Phase 2: Exploitation (R_ext + CF updates)
│                              # - Phase 3: Transfer (reuse G)
│
├── core/
│   ├── prediction_error.py    # (existing) Composite δ signals
│   ├── ofc.py                 # (existing) α modulation
│   └── curiosity.py           # NEW: Meta-controller for R_int weighting
│                              # - Adaptive λ: balance R_int vs R_ext
│                              # - Confidence tracking in G
│
└── envs/
    ├── grid_world.py          # (existing)
    └── causal_gridworld.py    # NEW: Explicit causal structure
                               # - Ground truth G for validation
                               # - Interventional actions (set variable)
                               # - Measure: structural distance to true G
```

---

## 9. Pearl's Three-Step Counterfactual Algorithm (Implementation Detail)

The **structural world model** (`structural.py`) implements Pearl's counterfactual logic:

### Example: "What if I had gone right instead of up?"

**Observed:** Agent went up ($A=\text{up}$), hit wall ($R=-1$)

**Query:** What if I had gone right ($A=\text{right}$)?

**Step 1: Abduction (Infer exogenous variables)**
```python
# Given evidence (A=up, R=-1, S'=same_position)
# Infer U (noise variables) consistent with observation
U = abduction(G, SEM, evidence={'A': 'up', 'R': -1})
# Returns: U = {obstacle_at_north: True}
```

**Step 2: Action (Modify structural equation)**
```python
# Replace A equation with intervention
SEM_modified = SEM.copy()
SEM_modified['A'] = lambda: 'right'  # Fixed intervention
```

**Step 3: Prediction (Forward propagate)**
```python
# Compute outcome under modified SEM with same U
S_counterfactual = forward_propagate(SEM_modified, U)
R_counterfactual = reward_function(S_counterfactual)
# Returns: R = +1 (moved successfully)
```

**Key advantage:** This correctly handles:
- **Confounders:** If obstacle presence affects both action success and reward
- **Mediators:** If action causes intermediate state changes
- **Colliders:** If multiple causes converge on outcome

The current tabular world model cannot distinguish these — it only learns $P(R \mid A)$, not the causal mechanism.

---

## 10. Expected Outcomes and Validation

### 10.1 Quantitative Metrics

| Metric | Baseline (Current) | Target (Causal Agent) | Measurement |
|--------|-------------------|----------------------|-------------|
| **CF Accuracy** | ~70% (tabular model) | >95% (learned G) | $\|\hat{R}_{cf} - R^*_{cf}\|$ |
| **Sample Efficiency** | 1000 episodes to converge | 500 episodes | Episodes to 90% optimal |
| **Transfer Speed** | Start from scratch | 10× faster | Episodes on new task |
| **Structural Accuracy** | N/A | SHD < 5% | Structural Hamming Distance to true $\mathcal{G}$ |
| **Exploration Efficiency** | Random | 3× fewer interventions | Interventions to discover all edges in $\mathcal{G}$ |

### 10.2 Qualitative Capabilities

**Causal Reasoning Tests:**
1. **Confounder detection:** Does agent distinguish $X \to Y$ from $X \leftarrow Z \to Y$?
2. **Collider detection:** Does agent avoid conditioning on colliders (incorrect inference)?
3. **Mediator chains:** Does agent correctly predict indirect effects $X \to M \to Y$?
4. **Counterfactual consistency:** Do $Y_{x}$ predictions satisfy compositionality (nested CFs)?

**Developmental Benchmarks:**
5. **Blicket detector task:** Can agent replicate Gopnik's experiments?
6. **Tool use:** Does agent discover functional relationships (hammer → nail, not correlation)?

### 10.3 Ablation Studies

| Ablation | Purpose |
|----------|---------|
| **No intrinsic reward** | Baseline: standard Q-learning with tabular model |
| **Random exploration** | Does $R_{\text{int}}$ outperform uniform random? |
| **Oracle $\mathcal{G}$ given** | Upper bound: skip Phase 1, use ground truth |
| **Strategy comparison** | A vs. B vs. C vs. D: which $R_{\text{int}}$ works best? |
| **Phase 2 only** | Can CF Q-learning work without Phase 1 structure learning? |

---

## 11. Implementation Roadmap

### Stage 1: Foundation (Weeks 1–4)

**Deliverables:**
- [ ] `causal_graph.py`: DAG representation with d-separation, do-calculus
- [ ] `structural_equations.py`: Linear SEM for toy environments
- [ ] `intrinsic_reward.py`: Strategy D (Causal Influence) as baseline

**Test Environment:** 3-variable chain $X \to Y \to Z$ with known ground truth

**Validation:** Agent discovers correct edge directions in <100 episodes

---

### Stage 2: Intrinsic Motivation (Weeks 5–8)

**Deliverables:**
- [ ] `interventional_agent.py`: Agent with $R_{\text{int}}$ only
- [ ] `structure_learner.py`: Score-based (BIC) graph search
- [ ] Strategy B (Prediction Error) implementation

**Test Environment:** Fork structure $X \to Z \leftarrow Y$ (collider) + confounders

**Validation:** Agent correctly identifies collider (doesn't condition on $Z$)

---

### Stage 3: Integration with CF Q-Learning (Weeks 9–12)

**Deliverables:**
- [ ] `structural.py`: SCM-based world model with Pearl's CF algorithm
- [ ] `causal_agent.py`: Three-phase meta-controller
- [ ] Modified `counterfactual_rl.py` to use structural world model

**Test Environment:** Regret Grid World with explicit causal structure

**Validation:**
- Phase 1: Discover $\mathcal{G}$ in 200 episodes
- Phase 2: CF Q-learning converges 2× faster than baseline
- Phase 3: Transfer to new reward in 50 episodes (vs. 500 from scratch)

---

### Stage 4: Complex Environments (Weeks 13–16)

**Deliverables:**
- [ ] `causal_gridworld.py`: Environment with verifiable causal ground truth
- [ ] Neural SEM for high-dimensional state spaces
- [ ] Strategy A (Information Gain) for optimal exploration

**Test Environment:** 7×7 grid with 5 binary state variables, known $\mathcal{G}$

**Validation:**
- Structural Hamming Distance from learned $\hat{\mathcal{G}}$ to true $\mathcal{G}$ < 10%
- CF Q-learning with learned $\mathcal{G}$ matches Oracle CF performance (±5%)

---

### Stage 5: Benchmarking and Publication (Weeks 17–20)

**Deliverables:**
- [ ] Experiments on CausalWorld benchmark (Ahmed et al., 2021)
- [ ] Comparison with state-of-the-art: CRIB, CausalCuriosity, DCDI
- [ ] Conference paper (targeting NeurIPS/ICML/ICLR)

**Target Venues:**
- Conference on Robot Learning (CoRL)
- NeurIPS Workshop on Causal Machine Learning
- ICLR Track on Reinforcement Learning

---

## 12. Key Challenges and Mitigations

### Challenge 1: Combinatorial Graph Space

**Problem:** For $d$ variables, there are $O(2^{d^2})$ possible DAGs

**Mitigation:**
- Start with known graph skeletons (only learn edge directions)
- Use sparsity constraints (most real-world graphs are sparse)
- Modular learning: decompose graph into conditionally independent subgraphs

---

### Challenge 2: Identifiability

**Problem:** Markov equivalent graphs produce identical observational distributions

**Mitigation:**
- **Use interventions:** Interventional data breaks equivalence classes
- **Exploit temporal structure:** In sequential environments, time orders variables
- **Functional constraints:** Additive noise models have unique solutions (Shimizu et al., 2006)

---

### Challenge 3: Continuous State Spaces

**Problem:** Discrete causal graphs don't naturally extend to continuous states

**Mitigation:**
- **Causal feature learning:** Learn low-dimensional causal representations (Schölkopf et al., 2021)
- **Neural SEM:** Use neural networks for $f_i(\text{Pa}(V_i))$
- **Hybrid approach:** Discrete macro-variables, continuous micro-states

---

### Challenge 4: Partial Observability

**Problem:** Agent may not observe all causally relevant variables

**Mitigation:**
- **Latent variable models:** Infer hidden confounders (Silva & Scheines, 2006)
- **Causal sufficiency assumption:** Assume all confounders observed (reasonable for controlled environments)
- **PAG (Partial Ancestral Graph):** Learn under partial observability (Zhang, 2008)

---

## 13. Connections to Broader AI Research

This project sits at the intersection of multiple research communities:

| Community | Connection | Key Idea |
|-----------|-----------|----------|
| **Developmental Psychology** | Gopnik, Carey | Children as intuitive scientists |
| **Causal Inference** | Pearl, Spirtes & Glymour | Do-calculus, causal discovery algorithms |
| **Intrinsic Motivation RL** | Pathak, Burda, Oudeyer | Curiosity-driven exploration |
| **Meta-Learning** | Finn, Levine | Fast adaptation via structure learning |
| **World Models** | Ha, Schmidhuber | Learning environment dynamics |
| **Neuroscience** | Schultz, Doya | Dopamine as dual-purpose signal |

**Why this matters:** Most RL research focuses on **reward maximization** (Circuit 1). This project also models **structure discovery** (Circuit 2), making it cognitively more plausible and sample-efficient.

---

## 14. Expected Impact

### 14.1 Scientific Contributions

1. **First implementation** of developmentally-grounded causal RL combining:
   - Pearl's causal hierarchy
   - Gopnik's interventional learning theory
   - Dual dopaminergic reward signals

2. **Proof-of-concept** that intrinsic motivation for causal structure discovery improves:
   - Sample efficiency in RL
   - Transfer learning across tasks
   - Robustness to confounders

3. **Bridge** between:
   - Developmental psychology (how children learn)
   - Causal inference (mathematical formalism)
   - Reinforcement learning (computational implementation)

### 14.2 Practical Applications

| Domain | Current Limitation | Causal Agent Solution |
|--------|-------------------|----------------------|
| **Robotics** | Needs thousands of trials per task | Learns reusable causal model → 10× faster transfer |
| **Healthcare** | Observational data confounded | Discovers true treatment effects via $do()$ |
| **Autonomous Driving** | Spurious correlations (e.g., "sunny → safe") | Learns causal mechanisms (visibility, traction) |
| **Game AI** | AlphaGo needs millions of games | Could learn game mechanics explicitly |

**Killer app:** A robot that, like a child, systematically experiments with kitchen appliances for 1 hour, then **never needs retraining** when task changes ("make coffee" → "heat soup"), because it learned the **causal structure** (heating, mixing, pouring), not just task-specific policies.

---

## 15. Next Steps

### Immediate (Next 2 Weeks)
1. **Literature review:** Deep dive into CausalCuriosity (Sontakke et al.), DCDI (Brouillard et al.)
2. **Design toy environment:** 3-variable causal graph with ground truth
3. **Implement Strategy D:** Causal Influence detection (simplest $R_{\text{int}}$)

### Short-Term (Months 1–3)
4. **Validate Phase 1:** Agent discovers known causal graph in toy environment
5. **Implement `structural.py`:** Pearl's 3-step counterfactual algorithm
6. **Integrate with CF Q-learning:** End-to-end Phase 1 → Phase 2 pipeline

### Medium-Term (Months 4–6)
7. **Benchmark:** Compare against standard CF Q-learning on Regret Grid World
8. **Ablations:** Which $R_{\text{int}}$ strategy performs best?
9. **Neural SEM:** Scale to high-dimensional states

### Long-Term (Months 7–12)
10. **CausalWorld benchmark:** Industry-standard evaluation
11. **Transfer experiments:** Train on Task A, test on Task B with same physics
12. **Publication:** Submit to top-tier venue (NeurIPS, ICML, CoRL)

---

## 16. Key References and Further Reading

### Developmental Psychology
- **Gopnik, A., & Sobel, D. M.** (2000). Detecting blickets: How young children use information about novel causal powers in categorization and induction. *Child Development*, 71(5), 1205-1222.
- **Gopnik, A., et al.** (2004). A theory of causal learning in children: Causal maps and Bayes nets. *Psychological Review*, 111(1), 3.
- **Harris, P. L., German, T., & Mills, P.** (1996). Children's use of counterfactual thinking in causal reasoning. *Cognition*, 61(3), 233-259.

### Neuroscience
- **Schultz, W., Dayan, P., & Montague, P. R.** (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599.
- **Bromberg-Martin, E. S., & Hikosaka, O.** (2009). Midbrain dopamine neurons signal preference for advance information about upcoming rewards. *Neuron*, 63(1), 119-126.
- **Kang, M. J., et al.** (2009). The wick in the candle of learning: Epistemic curiosity activates reward circuitry and enhances memory. *Psychological Science*, 20(8), 963-973.

### Causal Inference
- **Pearl, J.** (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- **Spirtes, P., Glymour, C., & Scheines, R.** (2000). *Causation, Prediction, and Search*. MIT Press.
- **Peters, J., Janzing, D., & Schölkopf, B.** (2017). *Elements of Causal Inference*. MIT Press.

### Causal Reinforcement Learning
- **Sontakke, S. A., Mehrjou, A., Itti, L., & Schölkopf, B.** (2021). Causal curiosity: RL agents discovering self-supervised experiments for causal representation learning. *NeurIPS*, 34.
- **Brouillard, P., et al.** (2020). Differentiable causal discovery from interventional data. *ICML*, 4639-4649.
- **Zhu, S., Ng, I., & Chen, Z.** (2022). Causal discovery with reinforcement learning. *ICLR*.
- **Ahmed, O., et al.** (2021). CausalWorld: A robotic manipulation benchmark for causal structure and transfer learning. *ICLR*.

### Intrinsic Motivation
- **Pathak, D., et al.** (2017). Curiosity-driven exploration by self-supervised prediction. *ICML*, 2778-2787.
- **Burda, Y., et al.** (2019). Exploration by random network distillation. *ICLR*.
- **Oudeyer, P. Y., & Kaplan, F.** (2007). What is intrinsic motivation? A typology of computational approaches. *Frontiers in Neurorobotics*, 1, 6.

---

## 17. Conclusion

The current Counterfactual Q-Learning project demonstrates impressive sample efficiency by learning from unchosen actions. However, it operates under a critical assumption: **counterfactual feedback is provided by an oracle** (either the environment or a simple associational world model).

This proposal extends the architecture in a cognitively grounded direction, modeling how human children discover causal structure through **intrinsically motivated exploration**. By implementing Pearl's Level 2 (Interventional reasoning) as a precursor to Level 3 (Counterfactuals), we enable agents to:

1. **Discover their own world model** through systematic experimentation
2. **Generate principled counterfactuals** via learned causal graphs and structural equations
3. **Transfer knowledge** across tasks that share causal structure
4. **Operate in realistic environments** where counterfactual oracles are unavailable

**Key innovation:** Dual reward signals ($R_{\text{ext}}$ + $R_{\text{int}}$) mirroring the two dopaminergic circuits in the brain — one for **what to choose** (reward), one for **how the world works** (structure).

**Bottom line:** This moves from "learning what actions lead to reward" to "learning how the world works, then using that knowledge to reason counterfactually" — a more robust, general, and cognitively plausible approach to reinforcement learning.

---

**Status:** Proposal Stage  
**Next Milestone:** Implement Strategy D (Causal Influence) in toy environment  
**Timeline:** 6-month project with staged deliverables  
**Success Criteria:** 2× sample efficiency + successful transfer + published causal discovery benchmark

*Document Version: 1.0*  
*Date: February 15, 2026*  
*Author: Paolo Di Prodi*
