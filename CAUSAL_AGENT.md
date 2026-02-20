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
| **2. Intervention** | $P(Y \mid \text{do}(X))$ | Blicket detector experiments (18–24 months) | Dorsolateral PFC, ACC | **⚠️ Missing (this proposal)** |
| **3. Counterfactual** | $P(Y_x \mid X=x', Y=y')$ | "What if" reasoning (4+ years) | vmPFC, OFC | ✅ Current CF Q-learning |

**Problem:** The current project jumps directly to Level 3 by assuming a world model good enough for counterfactual inference. But **Level 2 is the foundation** — without learning what interventions *do*, counterfactual predictions are unreliable.

**This proposal targets Level 2:** An agent that discovers causal structure through systematic intervention, then uses that structure for principled Level 3 reasoning.

---

## 5. Alignment with the Alberta Plan for AI Research

### 5.1 The Alberta Plan's Core Principles

The **Alberta Plan** (Sutton, 2022–2025) provides a unified framework for building continual-learning, scalable agents based on four pillars:

1. **Agent-State Construction**: Agents build their own state representation $\hat{s}_t = f_\theta(o_1, a_1, \ldots, o_t)$ from observation history using function approximation
2. **General Value Functions (GVFs)**: All knowledge is represented as predictions of the form $v_\pi(s) = \mathbb{E}_\pi[\sum_k \gamma^k C_{t+k+1} \mid S_t = s]$ with varying cumulants $C$, policies $\pi$, and discount rates $\gamma$
3. **Options Framework**: Temporally extended actions (Sutton, Precup & Singh, 1999) enable hierarchical behavior
4. **Continual Learning**: A single, never-resetting stream of experience with no episodes or oracle resets

### 5.2 Pearl's Ladder Reformulated in OAK Terms

The three levels of Pearl's Causal Hierarchy map naturally onto the OAK architecture:

#### Ladder 1: Association → **Passive GVFs** (Observational Policies)

$$
P(Y \mid X) \quad \longleftrightarrow \quad v_{\pi_{\text{obs}}}(s; C_Y, \gamma)
$$

- **Pearl**: $P(Y \mid X)$ — seeing correlation
- **OAK**: A GVF with cumulant $C = Y$ under the **observation policy** $\pi_{\text{obs}}$ (passive observation)
- **Learning**: Standard on-policy TD($\lambda$) updates
- **Limitation**: Conflates correlation with causation; cannot distinguish $X \to Y$ from $X \leftarrow Z \to Y$

#### Ladder 2: Intervention → **Interventional GVFs** (Option Policies)

$$
P(Y \mid \text{do}(X)) \quad \longleftrightarrow \quad v_{\omega_{\text{do}(X)}}(s; C_Y, \gamma)
$$

- **Pearl**: $P(Y \mid do(X=x))$ — causal effect of forcing $X$
- **OAK**: A GVF with cumulant $C = Y$ under the **intervention option** $\omega_{do(X=x)}$
- **The Option Policy**: "Take actions until $X = x$, regardless of confounders" — literally implements graph surgery
- **Causal Discovery**: Compare $v_{\omega_{\text{do}(X)}}(s; C_Y) \text{ vs. } v_{\pi_{\text{obs}}}(s; C_Y)$ — if they differ, there's confounding
- **Learning**: On-policy TD under the intervention option

**Key insight:** The $do$-operator IS an option. Each potential causal edge $V_i \to V_j$ becomes a GVF that predicts "if I intervene on $V_i$, what happens to $V_j$?" The causal graph emerges from which edge-GVFs have non-zero, reliable predictions.

#### Ladder 3: Counterfactual → **Off-Policy GVF Evaluation**

$$
P(Y_x \mid X = x', Y = y') \quad \longleftrightarrow \quad v_{\omega_{\text{do}(X=x)}}(s; C_Y) \text{ via off-policy correction}
$$

- **Pearl**: "Given $(X=x', Y=y')$, what would $Y$ be under $do(X=x)$?"
- **OAK**: Evaluate intervention option $\omega_{do(X=x)}$ from agent-state $\hat{s}$ consistent with observations $(x', y')$ using off-policy GVF methods
- **Abduction**: Agent-state $\hat{s}_t$ already encodes relevant history including $(X=x', Y=y')$
- **Action**: Switch from behavior policy to counterfactual option $\omega_{do(X=x)}$
- **Prediction**: Off-policy GVF evaluation via importance sampling (GTD($\lambda$), Emphatic TD)
- **Learning**: Off-policy TD methods with importance sampling ratios $\rho_t = \pi(a|s)/\mu(a|s)$

This reformulation reveals that the **existing CF Q-learning mechanism** in the current codebase is already implementing Level 3, but it can be strengthened by explicitly building on Level 2 (interventional GVFs).

### 5.3 Continual Learning vs. Phased Learning

The Alberta Plan explicitly rejects episodic/phased learning. Instead of three discrete phases (explore → exploit → transfer), the agent must learn continuously from a **single stream of experience** with adaptive emphasis:

```python
R_total(t) = R_ext(t) + λ(t) · R_int(t)
λ(t) = f(confidence in interventional GVFs)
```

where $\lambda(t)$ decays as GVF predictions become reliable, analogous to how different GVF cumulants can shift emphasis without requiring mode switches.

---

## 6. Formal Framework: Causal MDPs with GVF-Based Knowledge

### 5.1 Extended MDP Definition

We extend the standard MDP to include explicit causal structure and dual rewards:

$$
\mathcal{M}_{\text{causal}} = (\mathcal{S}, \mathcal{A}, \mathcal{G}, \text{SEM}, R_{\text{ext}}, R_{\text{int}}, \gamma)
$$

**Components:**

- $\mathcal{S}$: State space (as before)
- $\mathcal{A}$: Action space — now interpreted as **interventions** $do(V_j = v)$
- $\mathcal{G} = (V, E)$: **Causal DAG** over state variables $V = \{V_1, \ldots, V_d\}$ (initially unknown)
- $\text{SEM}$: **Structural Equation Model** defining mechanisms:
$$
V_i = f_i(\text{Pa}(V_i), U_i)
$$
where $\text{Pa}(V_i)$ are parents in $\mathcal{G}$ and $U_i$ is exogenous noise
- $R_{\text{ext}}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: Extrinsic task reward (can be zero during exploration phase)
- $R_{\text{int}}: \mathcal{S} \times \mathcal{A} \times \mathcal{S}' \to \mathbb{R}$: **Intrinsic reward for causal discovery** (the key contribution)
- $\gamma$: Discount factor

### 5.2 The Causal Discovery Objective

The agent maintains a **posterior distribution** over possible causal structures:

$$
P(\mathcal{G} \mid \mathcal{D}_t)
$$

where $\mathcal{D}_t = \{(s_i, a_i, s'_i)\}_{i=1}^t$ is the history of state transitions.

**Goal during exploration:** Maximize information gain about $\mathcal{G}$:

$$
\mathcal{G}^* = \arg\max_{\mathcal{G}} P(\mathcal{G} \mid \mathcal{D})
$$

**Goal during exploitation:** Use learned $\mathcal{G}$ for counterfactual Q-learning with principled predictions.

---

## 7. Intrinsic Reward Strategies for Causal Discovery

We reformulate intrinsic motivation as **GVF prediction discrepancies** rather than graph posterior updates. All strategies are now expressed in terms of GVF learning signals:

### 7.1 Strategy A: GVF Uncertainty Reduction (Information-Theoretic)

**Definition (OAK formulation):**
$$
R_{\text{int}}^{(A)}(t) = \sum_{(V_i, V_j)} H[v_{\omega_{\text{do}(V_i)}}(\hat{s}_t; C_{V_j})] - H[v_{\omega_{\text{do}(V_i)}}(\hat{s}_{t+1}; C_{V_j})]
$$

The agent receives intrinsic reward proportional to **reduction in GVF prediction uncertainty** across the network.

**Interpretation:**
- **High reward:** New experience reduces uncertainty about interventional predictions
- **Low reward:** GVF predictions already confident

**Connection to development:** Gopnik's "little scientist" — children intervene where uncertainty is highest.

**Implementation:**
- Track prediction variance or ensemble disagreement for each GVF
- Reward experiences that reduce variance: $\text{Var}[v_\omega(\hat{s})]_{\text{before}} - \text{Var}[v_\omega(\hat{s})]_{\text{after}}$
- No explicit graph posterior needed — uncertainty is at the GVF level

**Algorithms:**
- Bootstrap ensemble of GVFs (Osband et al., 2016)
- Bayesian GVFs with distributional TD learning

---

### 7.2 Strategy B: GVF Prediction Error (Surprise-Based) — **CANONICAL ALBERTA PLAN APPROACH**

**Definition (OAK formulation):**
$$
R_{\text{int}}^{(B)}(t) = \sum_{j \in \Psi} |\delta_j(t)|
$$

where $\delta_j(t)$ is the **TD error for GVF** $\psi_j$:

$$
\delta_j(t) = C_{j,t+1} + \gamma_j v_{\pi_j}(\hat{s}_{t+1}) - v_{\pi_j}(\hat{s}_t)
$$

The agent is intrinsically rewarded when its **interventional GVFs make prediction errors**.

**Interpretation:**
- **High reward:** GVF predictions were wrong → need more experience in this part of state-action space
- **Low reward:** GVF predictions accurate → well-explored region

**Connection to neuroscience:** ACC prediction error signal — not "reward unexpected" but "**world dynamics unexpected**."

**Implementation:**
- Sum absolute TD errors across all interventional GVFs
- Intrinsic reward automatically focuses exploration on regions where causal predictions are poor
- **This is the Alberta Plan's core mechanism**: TD error drives all learning

**Algorithms:**
- Standard TD($\lambda$) for each GVF
- ICM adapted: prediction error in learned agent-state features
- The current CF Q-learning world model is already doing this!

---

### 6.3 Strategy C: Interventional Entropy Reduction (Optimal Experimental Design)

**Definition:**
For each candidate intervention $do(X=x)$, compute expected information gain about a target variable $Y$:

$$
\text{IG}(X \to Y) = H[Y \mid \mathcal{D}_t] - \mathbb{E}_{\text{do}(X=x)}[H[Y \mid \mathcal{D}_t, \text{do}(X=x), Y_{\text{obs}}]]
$$

Choose the intervention that maximally reduces uncertainty about $Y$:

$$
R_{\text{int}}^{(C)}(t) = \max_{(X,x)} \text{IG}(X \to Y)
$$

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

### 7.4 Strategy D: GVF Discrepancy (Causal Influence Detection) — **RECOMMENDED STARTING POINT**

**Definition (OAK formulation):**
$$
R_{\text{int}}^{(D)}(t) = \sum_{j} \Big|v_{\omega_{\text{do}(V_i)}}(\hat{s}_t; C_{V_j}) - v_{\pi_{\text{obs}}}(\hat{s}_t; C_{V_j})\Big|
$$

The agent is rewarded for discovering **where interventional GVFs differ from observational GVFs**.

**Interpretation:**
- Measures the **causal power** of interventions
- High reward if $do(V_i)$ predictions differ from passive observation → causal edge exists
- Low reward if intervention and observation predict the same → no causal edge or confounding

**Connection to infancy research:** Watson's (1966) contingency detection — babies detect $|P(Y \mid do(\text{kick})) - P(Y)| > 0$.

**Implementation:**
- Maintain two GVFs for each variable pair:
  - $v_{\omega_{do(V_i)}}(\hat{s}; C_{V_j})$ — interventional prediction
  - $v_{\pi_{\text{obs}}}(\hat{s}; C_{V_j})$ — observational prediction
- Intrinsic reward = absolute difference
- **This is a GVF prediction discrepancy**, not a probability comparison
- Lightweight: only 2 GVFs per edge, updated via standard TD

**Algorithms:**
- Empowerment (Klyubin et al., 2005): $I(A; S_{\text{future}})$ — special case where $C = S$
- **This directly implements the Alberta Plan vision**: knowledge = predictions, causal discovery = comparing predictions under different policies

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

## 8. Continual Learning Integration (Alberta Plan Compliant)

### 8.1 Single-Stream Learning Loop

Replacing the three-phase architecture with **continual learning** from a single, never-resetting experience stream:

```python
# Continual learning loop (no episodes, no resets)
for t in continual_experience_stream:
    # 1. Agent-state construction
    ŝ_t = φ(observation_history)
    
    # 2. Adaptive emphasis between extrinsic and intrinsic
    λ_t = compute_lambda(gvf_confidence)
    R_total(t) = R_ext(t) + λ_t · R_int(t)
    
    # 3. Action selection (option framework)
    if λ_t > threshold_explore:
        # Prefer intervention options (causal discovery)
        ω_t = select_intervention_option(gvf_uncertainty)
    else:
        # Prefer task-optimizing options
        ω_t = select_task_option(Q_values)
    
    # 4. Execute option, observe outcome
    o_{t+1}, r_{t+1} = execute_option(ω_t)
    ŝ_{t+1} = φ(observation_history + o_{t+1})
    
    # 5. Update ALL GVFs in parallel via TD(λ)
    for ψ_j in GVF_network:
        δ_j = C_j(t+1) + γ_j · v_j(ŝ_{t+1}) - v_j(ŝ_t)
        v_j(ŝ_t) += α_j · δ_j · e_j(t)  # eligibility traces
    
    # 6. Update Q-values using composite error (existing CF Q-learning)
    δ_actual = r_{t+1} + γ·max_a Q(ŝ_{t+1}, a) - Q(ŝ_t, a_t)
    δ_cf = max_{a≠a_t} Q(ŝ_t, a) - r_{t+1}  # from observational GVFs
    δ_composite = δ_actual + α·δ_cf
    Q(ŝ_t, a_t) += β·δ_composite
    
    # 7. Update unchosen actions using interventional GVFs
    for a ≠ a_t:
        Q(ŝ_t, a) += γ_cf · [v_ω_do(a)(ŝ_t) - Q(ŝ_t, a)]
    
    # 8. Adaptive λ decay as GVF confidence grows
    λ_{t+1} = λ_max · exp(-η · mean_gvf_confidence)
```

### 8.2 Adaptive Emphasis Function

The weighting $\lambda(t)$ shifts emphasis as the agent gains causal knowledge:

$$
\lambda(t) = \lambda_{\max} \cdot \exp\left(-\eta \cdot \frac{1}{|\Psi|} \sum_{j \in \Psi} \text{confidence}(v_j)\right)
$$

where confidence can be computed as:

$$
\text{confidence}(v_j) = 1 - \frac{|\delta_j|_{\text{MA}}}{|\delta_j|_{\text{init}}}
$$

(ratio of current TD error moving average to initial error)

**Behavior:**
- **Early experience** ($t \to 0$): Low GVF confidence → $\lambda \approx \lambda_{\max}$ → strong intrinsic motivation → causal discovery emphasis
- **Late experience** ($t \to \infty$): High GVF confidence → $\lambda \approx 0$ → weak intrinsic motivation → task performance emphasis
- **Non-stationarity**: If environment changes, GVF errors increase → $\lambda$ increases → automatic re-exploration

### 8.3 Pearl's Three-Step Algorithm in OAK Terms

The counterfactual query mechanism is reformulated:

**Original (Graph-Based):**
1. Abduction: Infer $U$ from evidence $(X=x, Y=y)$ and $\mathcal{G}$
2. Action: Replace equation $X = f_X(\text{Pa}(X), U_X)$ with $X := x'$
3. Prediction: Forward propagate through modified SEM

**OAK Formulation:**
1. **Abduction**: Agent-state $\hat{s}_t = \phi(\text{history including } X=x, Y=y)$ **already encodes** all relevant context
2. **Action**: Switch from behavior option $\omega_{\text{behavior}}$ to intervention option $\omega_{do(X=x')}$
3. **Prediction**: Evaluate interventional GVF **off-policy** using importance sampling:
$$
v_{\omega_{\text{do}(X=x')}}(\hat{s}_t; C_Y) \text{ via GTD}(\lambda) \text{ or Emphatic TD}
$$

**Key advantages:**
- No explicit inference of exogenous variables $U$ — agent-state $\hat{s}_t$ implicitly captures them
- No graph surgery — option switch implements the $do$ operator
- Off-policy GVF evaluation is a standard Alberta Plan technique
- The existing `counterfactual_query()` world model method becomes an off-policy GVF prediction

### 8.4 Transfer Learning via GVF Reuse

When task reward changes but environment dynamics stay constant:

```python
# New task with different reward function
R_ext_new(s, a) = new_task_reward(s, a)

# Interventional GVFs remain valid (physics unchanged)
for (V_i, V_j) in causal_edges:
    v_ω_do(V_i)(ŝ; C_V_j)  # Still accurate!

# Only need to learn new Q-values, not new GVFs
Q_new(ŝ, a) = R_ext_new + γ·Σ P(ŝ'|ŝ,a)·max_a' Q_new(ŝ', a')
# But P(ŝ'|ŝ,a) is predicted by existing GVFs!
```

**Transfer speed:** Because causal structure (GVF network) is **task-independent**, only the reward-to-action mapping needs relearning → 10× faster adaptation.

---

## 9. Architectural Changes to Current Codebase (OAK-Aligned)

Proposed extensions to `src/causalrl/` following the Alberta Plan architecture:

```
src/causalrl/
├── gvf/                       # NEW MODULE (Alberta Plan core)
│   ├── __init__.py
│   ├── gvf.py                 # GVF class: policy, cumulant, discount, value function
│   │                          # - Methods: td_update(ŝ, ŝ', C, γ)
│   │                          # - Eligibility traces for TD(λ)
│   │                          # - Off-policy importance sampling
│   ├── gvf_network.py         # Collection of GVFs for causal discovery
│   │                          # - Associational GVFs: v_π_obs(ŝ; C_V_j)
│   │                          # - Interventional GVFs: v_ω_do(V_i)(ŝ; C_V_j)
│   │                          # - Causal edge detection via GVF discrepancy
│   ├── agent_state.py         # Agent-state construction φ(o_1, a_1, ..., o_t)
│   │                          # - Neural: LSTM or Transformer
│   │                          # - Tabular: Recency-weighted feature vector
│   │                          # - Methods: update(o_t, a_t), get_state()
│   └── intrinsic_reward.py    # R_int computation from GVF errors
│                              # - Strategy A: GVF uncertainty reduction
│                              # - Strategy B: Sum of |δ_j| across GVFs
│                              # - Strategy D: GVF discrepancy |v_ω - v_π_obs|
│
├── options/                   # EXTENDED (existing option.py)
│   ├── __init__.py
│   ├── option.py              # (existing) Base Option class
│   ├── intervention_options.py # NEW: Options that implement do(V_i=x)
│   │                          # - Initiation: where V_i is manipulable
│   │                          # - Policy: primitives to set V_i to x
│   │                          # - Termination: V_i = x achieved
│   └── option_discovery.py    # NEW: Learn intervention options from GVFs
│                              # - Identify controllable variables
│                              # - Build option policies for setting them
│
├── world_models/
│   ├── tabular.py             # (existing) Count-based model
│   ├── oracle.py              # (existing) Perfect model
│   └── gvf_world_model.py     # NEW: World model as GVF network
│                              # - predict(ŝ, a): query observational GVFs
│                              # - counterfactual_query(ŝ, a_cf): 
│                              #   query interventional GVFs off-policy
│                              # - No explicit DAG or SEM
│                              # - Implements Pearl's 3-step via:
│                              #   1. ŝ (abduction built into agent-state)
│                              #   2. Switch to ω_do(a_cf) (action)
│                              #   3. Off-policy GVF eval (prediction)
│
├── agents/
│   ├── counterfactual_rl.py  # (existing) CF Q-learning with composite errors
│   └── continual_causal_agent.py # NEW: Single-loop continual learner
│                              # - No phases, adaptive λ(t) weighting
│                              # - Manages GVF network + option library
│                              # - Selects between task/intervention options
│
├── core/
│   ├── prediction_error.py    # (existing) Composite δ signals
│   ├── ofc.py                 # (existing) α modulation
│   ├── adaptive_lambda.py     # NEW: λ(t) schedule based on GVF confidence
│   │                          # - Decays as TD errors decrease
│   │                          # - Increases if non-stationarity detected
│   └── temporal_abstraction.py # NEW: Multi-timescale GVFs
│                              # - Short-term: γ ≈ 0.9 (immediate effects)
│                              # - Long-term: γ ≈ 0.99 (delayed effects)
│
└── envs/
    ├── grid_world.py          # (existing)
    └── interventional_env.py  # NEW: Environments exposing state features
                               # - NOT ground-truth graphs
                               # - State features V_1, ..., V_d observable
                               # - Agent discovers causal structure via GVFs
```

---

## 10. Temporal Abstraction: Causal Interventions as Options

### 10.1 Mapping Causal Mechanisms to Options

A key Alberta Plan principle is **temporal abstraction** via options. Causal mechanisms naturally map to options at multiple time scales:

| Causal Mechanism | Time Scale | Option Formulation |
|-----------------|-----------|--------------------|
| "Press lever" | Sub-second | $\omega_{\text{lever}}$: policy = move hand, terminate when lever down |
| "Heat water" | Seconds | $\omega_{\text{heat}}$: policy = turn knob, terminate when T > 100°C |
| "Make toast" | Minutes | $\omega_{\text{toast}}$: composition of $\omega_{\text{bread}} \circ \omega_{\text{lever}}$ |
| "Prepare breakfast" | Tens of minutes | Higher-level option composed of toast, coffee, eggs options |

### 10.2 Intervention Options

Each intervention $do(V_i = x)$ becomes an option:

```python
class InterventionOption(Option):
    def __init__(self, variable: str, target_value: float):
        self.variable = variable
        self.target = target_value
        
    def initiation_set(self, ŝ) -> bool:
        """Can we manipulate this variable from ŝ?"""
        return is_controllable(ŝ, self.variable)
    
    def policy(self, ŝ) -> Action:
        """Primitive actions to set variable to target"""
        # Learn via RL: which actions move V_i toward target?
        return argmax_a GVF_V_i(ŝ, a)  # GVF predicting effect on V_i
    
    def termination(self, ŝ) -> bool:
        """Have we achieved V_i = target?"""
        return abs(get_feature(ŝ, self.variable) - self.target) < ε
```

### 10.3 Hierarchical Causal Discovery

Causal discovery naturally becomes hierarchical:

1. **Low-level**: Discover primitive action → state feature edges (e.g., "move_right" → "x_position")
2. **Mid-level**: Compose into intervention options (e.g., $\omega_{\text{goto}(x,y)}$)
3. **High-level**: Discover option → outcome edges (e.g., $\omega_{\text{toast}}$ → "hunger satisfied")

This aligns with the Alberta Plan's vision of **scalable hierarchical agents**.

---

## 11. Pearl's Three-Step Algorithm in GVF Implementation

The **GVF world model** (`gvf_world_model.py`) implements Pearl's counterfactual logic without explicit graphs:

### Example: "What if I had gone right instead of up?"

**Observed:** Agent went up ($A=\text{up}$), hit wall ($R=-1$)

**Query:** What if I had gone right ($A=\text{right}$)?

**Step 1: Abduction (Agent-state encodes context)**
```python
# Agent-state already encodes: A=up, R=-1, S'=same_position
ŝ_t = φ(observation_history)  # includes wall collision
# No explicit inference of U needed - ŝ_t implicitly captures:
#   - Obstacle at north (inferred from collision)
#   - Agent position
#   - Action history
```

**Step 2: Action (Switch to intervention option)**
```python
# Switch from behavior policy to counterfactual intervention option
ω_cf = InterventionOption(variable='action', target='right')
# This option's policy will execute 'right' regardless of Q-values
```

**Step 3: Prediction (Off-policy GVF evaluation)**
```python
# Query interventional GVF for 'right' action, evaluated from ŝ_t
R_cf = v_ω_right(ŝ_t; C_reward, γ=0)  # Immediate reward GVF
# Off-policy correction: weight by ρ = π_ω(right|ŝ_t) / π_behavior(up|ŝ_t)
# Returns: R_cf ≈ +1 (right action succeeds)
```

**Key advantage in OAK formulation:**
- **No graph surgery**: Option switch implements $do$ operator
- **No explicit $U$**: Agent-state $\hat{s}_t$ implicitly captures all relevant context
- **Standard TD methods**: Off-policy GVF evaluation uses GTD($\lambda$) or Emphatic TD
- **Handles confounders**: If obstacle presence affected both action success and reward, the agent-state encodes this, and the interventional GVF prediction accounts for it

---

## 12. Expected Outcomes and Validation (Alberta Plan Compliant)

### 12.1 Behavioral Metrics (No Oracle Comparisons)

Following the Alberta Plan principle of no privileged information, validation focuses on **behavioral outcomes**:

| Metric | Baseline (Current) | Target (GVF-Based Agent) | Measurement |
|--------|-------------------|----------------------|-------------|
| **CF Prediction Accuracy** | ~70% (tabular model) | >95% (GVF network) | $\|\hat{R}_{cf} - R_{\text{actual}}\|$ on test interventions |
| **Sample Efficiency** | 1000 timesteps to converge | 500 timesteps | Steps to 90% optimal Q-values |
| **Transfer Speed** | Start from scratch | 10× faster | Steps on new task using same GVF network |
| **GVF Prediction Error** | N/A (no GVFs) | Mean $\|\delta_j\| < 0.05$ | TD error averaged across interventional GVFs |
| **Adaptive Re-exploration** | Manual reset | Automatic via λ(t) | Time to detect and adapt to environment change |

**Removed metric:** ~~Structural Hamming Distance to true $\mathcal{G}$~~ — violates Alberta Plan (no oracle ground truth)

### 12.2 GVF-Based Validation Tests

**Causal Reasoning Tests:**
1. **Confounder detection:** Does agent distinguish $X \to Y$ from $X \leftarrow Z \to Y$?
2. **Collider detection:** Does agent avoid conditioning on colliders (incorrect inference)?
3. **Mediator chains:** Does agent correctly predict indirect effects $X \to M \to Y$?
4. **Counterfactual consistency:** Do $Y_{x}$ predictions satisfy compositionality (nested CFs)?

**Developmental Benchmarks:**
5. **Blicket detector task:** Can agent replicate Gopnik's experiments?
6. **Tool use:** Does agent discover functional relationships (hammer → nail, not correlation)?

### 12.3 Ablation Studies (Alberta Plan Aligned)

| Ablation | Purpose |
|----------|---------||
| **No GVF network** | Baseline: standard Q-learning (associational learning only) |
| **No intrinsic reward** | $\lambda(t) = 0$ always: task reward only |
| **Observational GVFs only** | No intervention options: $v_{\pi_{\text{obs}}}$ only |
| **Interventional GVFs only** | No observational baseline for comparison |
| **Fixed λ** vs. **Adaptive λ(t)** | Does adaptive emphasis improve continual learning? |
| **Strategy comparison** | A vs. B vs. D: which $R_{\text{int}}$ GVF formulation works best? |
| **No agent-state** | Use environment-provided states: test importance of $\phi$ |
| **No options** | Primitive actions only: test temporal abstraction benefit |

---

## 13. Implementation Roadmap (OAK Architecture)

### Stage 1: GVF Infrastructure (Weeks 1–4)

**Deliverables:**
- [ ] `gvf/gvf.py`: Base GVF class with TD($\lambda$) updates
- [ ] `gvf/agent_state.py`: Agent-state construction $\phi(o_1, a_1, \ldots, o_t)$
- [ ] `gvf/intrinsic_reward.py`: Strategy D (GVF discrepancy) as baseline

**Test Environment:** 3-variable chain where agent observes features $V_1, V_2, V_3$

**Validation:** 
- Observational GVFs learn: $v_{\pi_{\text{obs}}}(\hat{s}; C_{V_2}) \approx V_1$, $v_{\pi_{\text{obs}}}(\hat{s}; C_{V_3}) \approx V_2$
- GVF TD errors decrease to < 0.1 in <500 timesteps

---

### Stage 2: Intervention Options (Weeks 5–8)

**Deliverables:**
- [ ] `options/intervention_options.py`: Options that implement $do(V_i = x)$
- [ ] `gvf/gvf_network.py`: Paired observational + interventional GVFs
- [ ] Strategy B (GVF prediction error) implementation

**Test Environment:** Confounded system where $X \leftarrow Z \to Y$ but agent can intervene on $X$

**Validation:** 
- $|v_{\omega_{do(X)}}(\hat{s}; C_Y) - v_{\pi_{\text{obs}}}(\hat{s}; C_Y)| > \epsilon$ detects confounding
- Agent learns $\omega_{do(X)}$ policy that successfully sets $X$ to target values

---

### Stage 3: Continual Learning Integration (Weeks 9–12)

**Deliverables:**
- [ ] `world_models/gvf_world_model.py`: World model as GVF network
- [ ] `agents/continual_causal_agent.py`: Single-loop continual learner with adaptive $\lambda(t)$
- [ ] Modified `counterfactual_rl.py` to use GVF-based counterfactual queries

**Test Environment:** Regret Grid World with observable state features

**Validation:**
- Continual learning: GVF network learns while Q-values improve (no phases)
- CF Q-learning converges 2× faster than baseline (500 vs. 1000 timesteps)
- Transfer: New task in same environment converges in 50 timesteps using existing GVFs

---

### Stage 4: Neural Agent-State and Function Approximation (Weeks 13–16)

**Deliverables:**
- [ ] `envs/interventional_env.py`: Environments with high-dimensional observations (no oracle graphs)
- [ ] Neural agent-state: LSTM or Transformer for $\phi(o_1, \ldots, o_t)$
- [ ] Neural GVF approximators for continuous state spaces
- [ ] Strategy A (GVF uncertainty) for optimal exploration

**Test Environment:** 7×7 grid with pixel observations, no state features exposed

**Validation:**
- Agent-state learns useful representations from pixels
- GVF network maintains prediction accuracy with neural function approximation
- Behavioral metrics (reward, transfer speed) match or exceed tabular case

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

1. **First Alberta Plan-compliant** implementation of causal RL combining:
   - Pearl's causal hierarchy reformulated as GVF networks
   - Gopnik's interventional learning theory
   - Dual dopaminergic reward signals ($R_{\text{ext}}$ + $R_{\text{int}}$)
   - Continual learning with no episodic resets

2. **Proof-of-concept** that GVF-based causal discovery improves:
   - Sample efficiency in RL
   - Transfer learning across tasks (GVF reuse)
   - Robustness to confounders (interventional vs. observational GVF comparison)

3. **Bridge** between:
   - Developmental psychology (how children learn)
   - Causal inference (Pearl's do-calculus)
   - Alberta Plan (GVFs, options, continual learning)
   - Reinforcement learning (computational implementation)

### 14.2 Practical Applications

| Domain | Current Limitation | GVF Causal Agent Solution |
|--------|-------------------|---------------------------|
| **Robotics** | Needs thousands of trials per task | Learns reusable GVF network → 10× faster transfer |
| **Healthcare** | Observational data confounded | Discovers true treatment effects via interventional GVFs |
| **Autonomous Driving** | Spurious correlations (e.g., "sunny → safe") | Learns causal mechanisms via $\|v_{\omega} - v_{\pi_{\text{obs}}}\|$ |
| **Game AI** | AlphaGo needs millions of games | Could learn game mechanics via interventional GVFs |

**Killer app:** A robot that, like a child, systematically experiments via intervention options for 1 hour (high $\lambda$), then **adapts instantly** when task changes ("make coffee" → "heat soup"), because it learned the **causal structure as GVF predictions** (heating, mixing, pouring), not just task-specific policies. As GVF confidence grows, $\lambda \to 0$ and the robot focuses on task performance.

---

## 15. Next Steps

### Immediate (Next 2 Weeks)
1. **Literature review:** Deep dive into:
   - Alberta Plan papers (Sutton et al., 2022–2025)
   - GVF networks (Modayil et al., 2014; Schaul et al., 2015)
   - Off-policy GVF learning (Sutton et al., GTD, Emphatic TD)
   - CausalCuriosity (Sontakke et al., NeurIPS 2021) — reinterpret via GVFs
2. **Design toy environment:** 3-variable chain where $V_1, V_2, V_3$ are observable features
3. **Implement base GVF:** TD($\lambda$) updates for single GVF as proof-of-concept

### Short-Term (Months 1–3)
4. **GVF infrastructure:** Build `gvf.py`, `agent_state.py`, `gvf_network.py`
5. **Intervention options:** Implement `InterventionOption` class
6. **Strategy D:** GVF discrepancy $|v_{\omega} - v_{\pi_{\text{obs}}}|$ as intrinsic reward

### Medium-Term (Months 4–6)
7. **Continual learning loop:** Single-stream agent with adaptive $\lambda(t)$
8. **GVF world model:** Integrate with existing CF Q-learning
9. **Benchmarks:** Compare against standard CF Q-learning on Regret Grid World

### Long-Term (Months 7–12)
10. **Neural function approximation**: LSTM agent-state + neural GVFs
11. **Transfer experiments**: Train on Task A, test on Task B with same GVF network
12. **Publication:** Submit to Alberta Plan-aligned venues (RL Conference, RLDM)

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

### Alberta Plan for AI Research
- **Sutton, R. S., et al.** (2022–2025). The Alberta Plan for AI Research. University of Alberta.
- **Sutton, R. S., Precup, D., & Singh, S.** (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1-2), 181-211.
- **Modayil, J., White, A., & Sutton, R. S.** (2014). Multi-timescale nexting in a reinforcement learning robot. *Adaptive Behavior*, 22(2), 146-160.
- **Schaul, T., Horgan, D., Gregor, K., & Silver, D.** (2015). Universal value function approximators. *ICML*, 1312-1320.
- **Sutton, R. S., Mahmood, A. R., & White, M.** (2016). An emphatic approach to the problem of off-policy temporal-difference learning. *JMLR*, 17(1), 2603-2631.

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

**Bottom line:** This moves from "learning what actions lead to reward" to "learning how the world works **via predictions (GVFs)**, then using that knowledge to reason counterfactually" — a more robust, general, and cognitively plausible approach aligned with the Alberta Plan's vision of continual learning agents.

### 17.1 Alberta Plan Compliance Summary

✅ **Agent-State**: Implemented via $\hat{s}_t = \phi(o_1, a_1, \ldots, o_t)$

✅ **GVFs as Knowledge**: Causal structure emerges from interventional GVF network, not explicit graphs

✅ **Options**: Temporal abstraction via intervention options $\omega_{do(V_i)}$

✅ **Continual Learning**: Single stream, adaptive $\lambda(t)$, no episodic resets

✅ **No Oracles**: Validation uses behavioral outcomes, not ground-truth graph comparisons

✅ **Off-Policy Learning**: Counterfactuals via GTD/Emphatic TD (standard Alberta Plan methods)

---

**Status:** Proposal Stage — Alberta Plan Aligned  
**Next Milestone:** Implement GVF infrastructure + Strategy D (GVF discrepancy)  
**Timeline:** 6-month project with staged deliverables  
**Success Criteria:** 2× sample efficiency + successful transfer + GVF prediction accuracy < 5% error

*Document Version: 2.0 — Alberta Plan Integration*  
*Date: February 18, 2026*  
*Author: Paolo Di Prodi*
