"""
PUCT DeepMind-style Synthetic Vector DB Generator v9.1 (Patched)
Features: 
- Animated 3D Evolution of AlphaZero building the Vector DB
- Fixed Coset generation and action mapping
- Functional Matryoshka representation truncation
- Deterministic evaluation for stable PUCT search
- Expanded Group Families
Dependencies: pip install torch numpy scipy scikit-learn plotly
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from itertools import product, permutations
from collections import defaultdict
import math
import random
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# ============================================================
# Parts 1-4: Group Theory, Embeddings, Dataset State
# ============================================================

class FiniteGroup:
    def __init__(self, elements: List, op: Callable, name: str = "G"):
        self.elements = list(elements)
        self.op = op
        self.name = name
        self._identity = self._find_identity()
        self._inverses = self._compute_inverses()

    def _find_identity(self):
        for e in self.elements:
            if all(self.op(e, x) == x for x in self.elements): return e
        raise ValueError("Not a group")

    def _compute_inverses(self):
        inv = {}
        for a in self.elements:
            for b in self.elements:
                if self.op(a, b) == self._identity: inv[a] = b; break
        return inv

    def inverse(self, a): return self._inverses[a]
    @property
    def identity(self): return self._identity
    @property
    def order(self): return len(self.elements)

    def generated_subgroup(self, generators) -> 'FiniteGroup':
        subgroup = set(generators) | {self._identity}
        while True:
            new = {self.op(a, b) for a in subgroup for b in subgroup} - subgroup
            if not new: break
            subgroup |= new
        return FiniteGroup(sorted(subgroup, key=str), self.op, f"<{generators}>")

    def left_cosets(self, H: 'FiniteGroup') -> List[List]:
        cosets, seen = [], set()
        for g in self.elements:
            if g not in seen:
                coset = [self.op(g, h) for h in H.elements]
                cosets.append(sorted(coset, key=str))
                seen.update(coset)
        return cosets

def cyclic_group(n: int) -> FiniteGroup:
    return FiniteGroup(list(range(n)), lambda a, b: (a + b) % n, f"Z_{n}")

def klein_four() -> FiniteGroup:
    elems = [(a, b) for a in range(2) for b in range(2)]
    return FiniteGroup(elems, lambda a, b: ((a[0]+b[0]) % 2, (a[1]+b[1]) % 2), "V4")

def symmetric_group_s3() -> FiniteGroup:
    elems = list(permutations(range(3)))
    return FiniteGroup(elems, lambda p, q: tuple(p[q[i]] for i in range(3)), "S_3")

def direct_product(g1: FiniteGroup, g2: FiniteGroup) -> FiniteGroup:
    elems = [(a, b) for a in g1.elements for b in g2.elements]
    op = lambda x, y: (g1.op(x[0], y[0]), g2.op(x[1], y[1]))
    return FiniteGroup(elems, op, f"{g1.name}x{g2.name}")

class GroupEmbedding:
    def __init__(self, group: FiniteGroup, dim: int = 64, noise_std: float = 0.1, seed: int = 0):
        self.group = group
        self.dim = dim
        self.noise_std = noise_std
        rng = np.random.default_rng(seed)
        self.projection = rng.standard_normal((group.order, dim)).astype(np.float32) / np.sqrt(dim)
        for d in range(dim):
            self.projection[:, d] *= (1.0 - d / (2 * dim))  # Matryoshka variance decay
        
        # Precompute Cholesky for fast multivariate noise sampling
        cov = rng.standard_normal((dim, dim)).astype(np.float32)
        self.cov = cov @ cov.T / dim + np.eye(dim) * 1e-5  # Ensure positive definite
        self.cov_chol = np.linalg.cholesky(self.cov * self.noise_std**2)
        
        self.element_index = {e: i for i, e in enumerate(group.elements)}

    def embed(self, element, noise: bool = True, trunc_dim: int = None) -> np.ndarray:
        vec = self.projection[self.element_index[element]].copy()
        if noise and self.noise_std > 0:
            # O(dim^2) -> O(dim^2) but avoids recomputing Cholesky internally
            vec += (self.cov_chol @ np.random.randn(self.dim)).astype(np.float32)
        
        # Apply Matryoshka Truncation (slice and zero-pad to maintain dimension size)
        if trunc_dim is not None and trunc_dim < self.dim:
            vec = np.pad(vec[:trunc_dim], (0, self.dim - trunc_dim), 'constant')
            
        return vec

    def embed_set(self, elements, samples: int, is_query: bool = False, trunc_dim: int = None, rng=None):
        rng = rng or np.random
        vecs, labels, meta = [], [], []
        for e in elements:
            for _ in range(samples):
                v = self.embed(e, noise=True, trunc_dim=trunc_dim)
                if is_query: v += rng.randn(*v.shape).astype(np.float32) * 0.05
                vecs.append(v)
                labels.append(self.element_index[e])
                meta.append({"element": str(e)})
        return np.stack(vecs), np.array(labels), meta

@dataclass
class DatasetState:
    groups: List[FiniteGroup] = field(default_factory=list)
    embeddings: List[GroupEmbedding] = field(default_factory=list)
    selected_cosets: List[Tuple[int, List, float]] = field(default_factory=list) 
    base_samples: int = 5
    noise_std: float = 0.1
    dim: int = 64
    ood_noise_ratio: float = 0.0
    matryoshka_trunc: int = 64

    def to_vectors(self, n_queries: int = 50, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else np.random
        corpus_v, corpus_l, corpus_g, meta = [], [], [], []
        
        # FIX: Iterate over selected_cosets directly to avoid zip truncation
        for ci, (gidx, coset, weight) in enumerate(self.selected_cosets):
            emb = self.embeddings[gidx]
            n_samples = max(1, int(self.base_samples * weight))
            v, l, m = emb.embed_set(coset, n_samples, is_query=False, trunc_dim=self.matryoshka_trunc, rng=rng)
            corpus_v.append(v); corpus_l.append(l); corpus_g.append(np.full(len(l), ci)) # ci = coset index
            for item in m: item["group_id"] = ci
            meta.extend(m)
            
        if not corpus_v: return None
        cv = np.concatenate(corpus_v); cl = np.concatenate(corpus_l); cg = np.concatenate(corpus_g)
        
        if self.ood_noise_ratio > 0:
            n_ood = int(len(cv) * self.ood_noise_ratio)
            if n_ood > 0:
                ood_v = rng.uniform(low=-1, high=1, size=(n_ood, self.dim)).astype(np.float32)
                cv = np.concatenate([cv, ood_v]); cl = np.concatenate([cl, np.full(n_ood, -1)])
                cg = np.concatenate([cg, np.full(n_ood, -1)])
                
        valid_idx = np.where(cl != -1)[0]
        query_indices = rng.choice(valid_idx, size=min(n_queries, len(valid_idx)), replace=False)
        qv = cv[query_indices] + rng.randn(len(query_indices), self.dim).astype(np.float32) * 0.05
        ql = cl[query_indices]
        qm = [meta[i] for i in query_indices]
        return {'corpus_vecs': cv, 'corpus_labels': cl, 'corpus_gids': cg,
                'query_vecs': qv, 'query_labels': ql, 'corpus_meta': meta, 'query_meta': qm}

# ============================================================
# Part 5: Quality Metrics
# ============================================================

class Quality:
    @staticmethod
    def recall_at_k(query_vecs, query_labels, corpus_vecs, corpus_labels, k=10):
        if len(query_vecs) == 0 or len(corpus_vecs) == 0: return 0.0
        q_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
        c_norm = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-8)
        sim = q_norm @ c_norm.T
        top_k_idx = np.argsort(-sim, axis=1)[:, :k]
        hits = sum(1 for i, q_lab in enumerate(query_labels) if q_lab in corpus_labels[top_k_idx[i]])
        return hits / len(query_labels)

    @staticmethod
    def overall(state: DatasetState):
        # FIX: Use fixed seed for deterministic evaluation during search
        data = state.to_vectors(seed=42)
        if not data or len(data['corpus_vecs']) == 0: return 0.0, 0.0
        cv_full, qv_full = data['corpus_vecs'], data['query_vecs']
        r1 = Quality.recall_at_k(qv_full, data['query_labels'], cv_full, data['corpus_labels'], k=1)
        r10 = Quality.recall_at_k(qv_full, data['query_labels'], cv_full, data['corpus_labels'], k=10)
        return float(0.3 * r10 + 0.2 * r1), float(r10)

# ============================================================
# Part 6: Action Space
# ============================================================

@dataclass
class Action:
    name: str
    params: dict = field(default_factory=dict)

class ActionSpace:
    # ENHANCEMENT: Expanded group families
    FAMILIES = [
        ("Z_3", lambda: cyclic_group(3)), ("Z_4", lambda: cyclic_group(4)),
        ("Z_5", lambda: cyclic_group(5)), ("Z_6", lambda: cyclic_group(6)),
        ("V4",  lambda: klein_four()), ("S_3", lambda: symmetric_group_s3()),
        ("Z_3xZ_2", lambda: direct_product(cyclic_group(3), cyclic_group(2))),
        ("Z_2xZ_2xZ_2", lambda: direct_product(klein_four(), cyclic_group(2)))
    ]

    @classmethod
    def all_actions(cls, state: DatasetState) -> List[Action]:
        actions = []
        for name, ctor in cls.FAMILIES:
            actions.append(Action("add_group", {"family": name, "ctor": ctor}))
            
        # FIX: Deduplicate coset actions
        seen_cosets = set()
        for i, g in enumerate(state.groups):
            non_id = [e for e in g.elements if e != g.identity]
            for gen_idx in range(min(2, len(non_id))):
                H = g.generated_subgroup([non_id[gen_idx]])
                for coset in g.left_cosets(H):
                    key = (i, tuple(str(e) for e in coset))
                    if key not in seen_cosets:
                        seen_cosets.add(key)
                        actions.append(Action("add_coset", {"group_idx": i, "coset": coset}))
                        
        for i in range(len(state.selected_cosets)):
            actions.append(Action("remove_coset", {"idx": i}))
            actions.append(Action("scale_cluster", {"idx": i, "mult": 2.0}))
            actions.append(Action("scale_cluster", {"idx": i, "mult": 0.5}))
        actions.append(Action("mod_noise", {"delta": 0.05}))
        actions.append(Action("mod_noise", {"delta": -0.05}))
        actions.append(Action("inject_ood", {"delta": 0.05}))
        actions.append(Action("remove_ood", {"delta": -0.05}))
        actions.append(Action("mod_trunc", {"delta": -8}))
        actions.append(Action("mod_trunc", {"delta": 8}))
        return actions

    @staticmethod
    def apply(state: DatasetState, action: Action) -> DatasetState:
        ns = DatasetState(groups=list(state.groups), embeddings=list(state.embeddings),
            selected_cosets=list(state.selected_cosets), base_samples=state.base_samples, 
            noise_std=state.noise_std, dim=state.dim, ood_noise_ratio=state.ood_noise_ratio,
            matryoshka_trunc=state.matryoshka_trunc)
            
        if action.name == "add_group":
            g = action.params["ctor"]()
            emb = GroupEmbedding(g, dim=state.dim, noise_std=state.noise_std, seed=len(state.groups)*7+13)
            ns.groups.append(g); ns.embeddings.append(emb)
            ns.selected_cosets.append((len(ns.groups) - 1, g.elements, 1.0))
        elif action.name == "add_coset":
            ns.selected_cosets.append((action.params["group_idx"], action.params["coset"], 1.0))
        elif action.name == "remove_coset":
            if action.params["idx"] < len(ns.selected_cosets):
                ns.selected_cosets.pop(action.params["idx"])
        elif action.name == "scale_cluster":
            idx = action.params["idx"]
            if idx < len(ns.selected_cosets):
                gidx, coset, w = ns.selected_cosets[idx]
                ns.selected_cosets[idx] = (gidx, coset, max(0.1, min(10.0, w * action.params["mult"])))
        elif action.name == "mod_noise":
            ns.noise_std = max(0.0, min(1.0, state.noise_std + action.params["delta"]))
            for e in ns.embeddings: e.noise_std = ns.noise_std
        elif action.name == "inject_ood":
            ns.ood_noise_ratio = min(0.5, state.ood_noise_ratio + action.params["delta"])
        elif action.name == "remove_ood":
            ns.ood_noise_ratio = max(0.0, state.ood_noise_ratio + action.params["delta"])
        elif action.name == "mod_trunc":
            ns.matryoshka_trunc = max(8, min(state.dim, state.matryoshka_trunc + action.params["delta"]))
        return ns

# ============================================================
# Part 7: Set Transformer & Networks
# ============================================================

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.fc_v = nn.Linear(dim_K, dim_V); self.fc_k = nn.Linear(dim_K, dim_V); self.fc_q = nn.Linear(dim_Q, dim_V)
        self.mha = nn.MultiheadAttention(dim_V, num_heads, batch_first=True)
    def forward(self, Q, K):
        Q = self.fc_q(Q); K, V = self.fc_k(K), self.fc_v(K)
        attn_out, _ = self.mha(Q, K, V)
        return Q + attn_out

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inducing=4):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim_out))
        self.mab1 = MAB(dim_out, dim_in, dim_out, num_heads); self.mab2 = MAB(dim_in, dim_out, dim_out, num_heads)
    def forward(self, X):
        H = self.mab1(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab2(X, H)

class SetTransformerEncoder(nn.Module):
    def __init__(self, dim=64, hidden=128, n_heads=4):
        super().__init__()
        self.dim = dim
        self.isab = ISAB(dim + 3, hidden, n_heads, num_inducing=4)
        # FIX: Added +1 for trunc_dim scalar
        self.rho = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.ReLU(), nn.Linear(hidden, 128))
    def forward(self, vecs, gids, num_cosets, noise_std, spe, ood_ratio, trunc_dim):
        if len(vecs) == 0: return torch.zeros(128)
        vt = torch.as_tensor(vecs, dtype=torch.float32)
        feats = []
        # FIX: Iterate over num_cosets instead of num_groups
        for c in range(num_cosets):
            mask = gids == c
            if mask.sum() == 0:
                feats.append(torch.zeros(self.dim + 3))
            else:
                gv = vt[mask]
                feats.append(torch.cat([gv.mean(0), gv.std(0).mean().unsqueeze(0), torch.tensor([float(mask.sum())]), torch.tensor([0.0])]))
        ood_mask = gids == -1
        if ood_mask.sum() > 0:
            ood_v = vt[ood_mask]
            feats.append(torch.cat([ood_v.mean(0), ood_v.std(0).mean().unsqueeze(0), torch.tensor([float(ood_mask.sum())]), torch.tensor([1.0])]))
        else:
            feats.append(torch.zeros(self.dim + 3))
        seq = torch.stack(feats).unsqueeze(0)
        attn_out = self.isab(seq).squeeze(0)
        pooled = attn_out.mean(dim=0)
        # FIX: Include trunc_dim in scalars
        scalars = torch.tensor([noise_std, spe, ood_ratio, float(trunc_dim)], dtype=torch.float32)
        return self.rho(torch.cat([pooled, scalars]))

class MultiTaskPolicyValueNet(nn.Module):
    def __init__(self, state_dim=128, n_actions=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128); self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1); self.aux_value_head = nn.Linear(128, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1), self.aux_value_head(x).squeeze(-1)

# ============================================================
# Part 8: PUCT Search
# ============================================================

@dataclass
class PUCTNode:
    state: DatasetState
    parent: Optional['PUCTNode'] = None
    action: Optional[Action] = None
    children: Dict[int, 'PUCTNode'] = field(default_factory=dict)
    visit_count: int = 0; total_value: float = 0.0; prior: float = 0.0
    action_list: List[Action] = field(default_factory=list); is_expanded: bool = False
    @property
    def q(self): return self.total_value / self.visit_count if self.visit_count else 0.0

class PUCTSearch:
    def __init__(self, pv_net, encoder, c_puct=1.5, max_actions=64, dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.pv_net, self.encoder = pv_net, encoder
        self.c_puct, self.max_actions = c_puct, max_actions
        self.dir_alpha, self.dir_eps = dirichlet_alpha, dirichlet_eps

    def _encode(self, state):
        data = state.to_vectors(n_queries=10)
        if not data: return torch.zeros(128)
        return self.encoder(data['corpus_vecs'], data['corpus_gids'], len(state.selected_cosets), state.noise_std, state.base_samples, state.ood_noise_ratio, state.matryoshka_trunc)

    def _get_actions(self, state):
        acts = ActionSpace.all_actions(state)
        if len(acts) > self.max_actions: acts = random.sample(acts, self.max_actions)
        elif not acts: acts = [Action("add_group", {"family": "Z_3", "ctor": lambda: cyclic_group(3)})]
        return acts

    def expand(self, node: PUCTNode, is_root: bool = False):
        if node.is_expanded: return
        node.action_list = self._get_actions(node.state)
        with torch.no_grad():
            logits, _, _ = self.pv_net(self._encode(node.state))
            k = min(len(node.action_list), logits.shape[0])
            probs = F.softmax(logits[:k], dim=-1).numpy()
        for i, action in enumerate(node.action_list):
            cs = ActionSpace.apply(node.state, action)
            prior = float(probs[i]) if i < len(probs) else 1.0 / len(node.action_list)
            if is_root:
                noise = np.random.dirichlet([self.dir_alpha] * len(node.action_list))
                prior = (1 - self.dir_eps) * prior + self.dir_eps * noise[i]
            node.children[i] = PUCTNode(state=cs, parent=node, action=action, prior=prior)
        node.is_expanded = True

    def select(self, node: PUCTNode) -> PUCTNode:
        while node.is_expanded and node.children:
            best, best_s = None, -float('inf')
            for c in node.children.values():
                u = self.c_puct * c.prior * math.sqrt(node.visit_count + 1) / (1 + c.visit_count)
                if c.q + u > best_s: best_s, best = c.q + u, c
            node = best
        return node

    def evaluate(self, node: PUCTNode) -> float:
        with torch.no_grad(): _, v_pred, _ = self.pv_net(self._encode(node.state))
        true_reward, _ = Quality.overall(node.state)
        return 0.4 * v_pred.item() + 0.6 * true_reward

    def backup(self, node: PUCTNode, value: float):
        while node is not None:
            node.visit_count += 1; node.total_value += value; node = node.parent

    def search(self, root_state: DatasetState, n_simulations: int = 100) -> PUCTNode:
        root = PUCTNode(state=root_state)
        self.expand(root, is_root=True)
        for _ in range(n_simulations):
            leaf = self.select(root)
            if leaf.visit_count > 0:
                self.expand(leaf)
                if leaf.children: leaf = max(leaf.children.values(), key=lambda c: c.prior)
            self.backup(leaf, self.evaluate(leaf))
        return root

# ============================================================
# Part 9: Trainer with Trajectory Baking
# ============================================================

class PUCTTrainer:
    def __init__(self, dim=64, lr=1e-3):
        self.dim = dim
        self.encoder = SetTransformerEncoder(dim=dim, hidden=128, n_heads=4)
        self.pv_net = MultiTaskPolicyValueNet(state_dim=128, n_actions=64)
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.pv_net.parameters()), lr=lr)
        self.searcher = PUCTSearch(self.pv_net, self.encoder)
        self.replay_buffer = []

    def self_play_episode(self, max_depth=6, n_simulations=50, record_trajectory=False):
        state = DatasetState(dim=self.dim, matryoshka_trunc=self.dim)
        examples = []
        trajectory = [] 
        
        if record_trajectory:
            trajectory.append({
                "state": state,
                "data": state.to_vectors(n_queries=20, seed=42),
                "action": "Initial State",
                "reward": 0.0
            })
            
        for d in range(max_depth):
            root = self.searcher.search(state, n_simulations=n_simulations)
            if not root.action_list: break
            visits = np.array([root.children[i].visit_count if i in root.children else 0 for i in range(len(root.action_list))])
            if visits.sum() == 0: break
            policy_target = visits / visits.sum()
            best = max(root.children.values(), key=lambda c: c.visit_count)
            value_target, aux_target = Quality.overall(best.state)
            with torch.no_grad(): sv = self.searcher._encode(state)
            examples.append((sv.numpy(), policy_target, value_target, aux_target, len(root.action_list)))
            
            state = ActionSpace.apply(state, best.action)
            
            if record_trajectory:
                act = best.action
                if act.name == "add_group": act_str = f"+ Group {act.params['family']}"
                elif act.name == "add_coset": act_str = f"+ Coset (G{act.params['group_idx']})"
                elif act.name == "scale_cluster": act_str = f"Scale Cluster {act.params['idx']} ({act.params['mult']}x)"
                else: act_str = act.name
                
                trajectory.append({
                    "state": state,
                    "data": state.to_vectors(n_queries=20, seed=42),
                    "action": act_str,
                    "reward": value_target
                })
                
        self.replay_buffer.extend(examples)
        if len(self.replay_buffer) > 2000: self.replay_buffer = self.replay_buffer[-2000:]
        return state, examples, trajectory

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size: return 0.0
        batch = random.sample(self.replay_buffer, batch_size)
        total_loss = 0.0
        for sv, pt, vt, at, n in batch:
            sv = torch.as_tensor(sv, dtype=torch.float32)
            logits, v_pred, a_pred = self.pv_net(sv)
            if n < logits.shape[0]: logits = logits[:n]
            log_p = F.log_softmax(logits, dim=-1)
            p_loss = -(torch.as_tensor(pt, dtype=torch.float32) * log_p).sum()
            v_loss = F.mse_loss(v_pred, torch.tensor([vt], dtype=torch.float32))
            a_loss = F.mse_loss(a_pred, torch.tensor([at], dtype=torch.float32))
            loss = p_loss + v_loss + 0.5 * a_loss
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            total_loss += loss.item()
        return total_loss / batch_size

    def train(self, n_episodes=10, max_depth=5, n_simulations=30):
        for ep in range(n_episodes):
            fs, _, _ = self.self_play_episode(max_depth, n_simulations, record_trajectory=False)
            loss = self.train_step(batch_size=min(32, len(self.replay_buffer)))
            score, _ = Quality.overall(fs)
            print(f"Ep {ep:3d}: loss={loss:.4f} score={score:.4f} groups={[g.name for g in fs.groups]}")

# ============================================================
# Part 10: Animated 3D Visualizer
# ============================================================

class AnimatedVisualizer:
    @staticmethod
    def plot_evolution(trajectory: list):
        print("\nGenerating Animated 3D Evolution HTML...")
        
        final_data = trajectory[-1]["data"]
        if not final_data: return
        pca = PCA(n_components=3)
        pca.fit(final_data['corpus_vecs'])
        
        frames = []
        
        for i, step in enumerate(trajectory):
            data = step["data"]
            if not data: continue
            
            cv = data['corpus_vecs']
            gids = data['corpus_gids']
            qv = data['query_vecs']
            
            cv_3d = pca.transform(cv)
            qv_3d = pca.transform(qv)
            
            traces = []
            
            unique_gids = np.unique(gids[gids >= 0])
            for gid in unique_gids:
                mask = gids == gid
                traces.append(go.Scatter3d(
                    x=cv_3d[mask, 0], y=cv_3d[mask, 1], z=cv_3d[mask, 2],
                    mode='markers', name=f'Cluster {gid}',
                    marker=dict(size=4, opacity=0.7),
                    hovertext=[f"GID: {gid}" for _ in range(np.sum(mask))]
                ))
                center = cv_3d[mask].mean(axis=0)
                traces.append(go.Scatter3d(
                    x=[center[0]], y=[center[1]], z=[center[2]],
                    mode='markers', name=f'Centroid {gid}',
                    marker=dict(size=10, color='black', symbol='circle-open', line=dict(width=2)),
                    hovertext=f"Centroid G{gid}"
                ))
                
            ood_mask = gids == -1
            if np.sum(ood_mask) > 0:
                traces.append(go.Scatter3d(
                    x=cv_3d[ood_mask, 0], y=cv_3d[ood_mask, 1], z=cv_3d[ood_mask, 2],
                    mode='markers', name='OOD Noise',
                    marker=dict(size=3, color='gray', opacity=0.4, symbol='x')
                ))
                
            if len(qv_3d) > 0:
                traces.append(go.Scatter3d(
                    x=qv_3d[:, 0], y=qv_3d[:, 1], z=qv_3d[:, 2],
                    mode='markers', name='Queries',
                    marker=dict(size=6, color='red', symbol='diamond')
                ))
                
            frames.append(go.Frame(
                data=traces,
                name=str(i),
                layout=go.layout.Layout(
                    title_text=f"Step {i}: <b>{step['action']}</b> | Reward (Q): {step['reward']:.3f}"
                )
            ))
            
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title=frames[0].layout.title,
                scene=dict(
                    xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="▶ Play",
                                  method="animate",
                                  args=[None, dict(frame=dict(duration=1500, redraw=True), fromcurrent=True)]),
                             dict(label="⏸ Pause",
                                  method="animate",
                                  args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                            ]
                )]
            )
        )
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Step: "},
            pad={"t": 50},
            steps=[dict(label=f"Step {i}", method="animate", args=[[str(i)], dict(mode="immediate")]) for i in range(len(frames))]
        )]
        fig.update_layout(sliders=sliders)
        
        fig.write_html("puct_animated_evolution.html")
        print("Saved Animated Evolution -> puct_animated_evolution.html")

# ============================================================
# Main Execution
# ============================================================

def main():
    print("=" * 88)
    print(" PUCT DeepMind Synthetic Vector DB Benchmark Generator v9.1 (Patched)")
    print(" Features: Animated 3D Evolution of AlphaZero Building the Database")
    print("=" * 88)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    trainer = PUCTTrainer(dim=64, lr=1e-3)

    print("\n[1/3] Training via self-play PUCT search...\n")
    trainer.train(n_episodes=10, max_depth=5, n_simulations=30)

    print("\n[2/3] Generating Final Trajectory for Visualization...\n")
    _, _, trajectory = trainer.self_play_episode(max_depth=8, n_simulations=50, record_trajectory=True)
    
    print("\n[3/3] Generating Animation...\n")
    AnimatedVisualizer.plot_evolution(trajectory)
    print("\nDone. Open puct_animated_evolution.html in your browser and press Play.")

if __name__ == "__main__":
    main()