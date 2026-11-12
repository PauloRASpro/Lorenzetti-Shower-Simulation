                                                                                       
                                                                            

import time
import gc
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import r2_score, mean_squared_error
from torch_geometric.nn import SAGEConv


                                                                
                       
                                                                

def _sanitize_numpy(x, nan_value=0.0, posinf_value=0.0, neginf_value=0.0):
    """
    Converte para float32 e elimina NaN/+Inf/-Inf.
    """
    x = np.asarray(x, dtype=np.float32)

    return np.nan_to_num(
        x,
        nan=nan_value,
        posinf=posinf_value,
        neginf=neginf_value,
    ).astype(np.float32)


def _sanitize_tensor(x):
    """
    Elimina NaN/+Inf/-Inf de tensores PyTorch.
    """
    return torch.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _fit_standardizer(X, eps=1e-8):
    """Ajusta média e desvio padrão exclusivamente no conjunto informado."""
    X = _sanitize_numpy(X)
    mean = _sanitize_numpy(np.mean(X, axis=0, keepdims=True))
    std = _sanitize_numpy(np.std(X, axis=0, keepdims=True))
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return {"mean": mean.astype(np.float32), "std": std}


def _apply_standardizer(X, stats):
    """Aplica uma padronização previamente ajustada."""
    X = _sanitize_numpy(X)
    return _sanitize_numpy((X - stats["mean"]) / stats["std"])


                                                                
                         
                                                                

def _make_chain_edge_index(n_nodes, ring_hops=1):
    """
    Cria grafo bidirecional local para uma região de anéis.

    ring_hops=1:
        0 <-> 1 <-> 2 <-> ...

    ring_hops=2:
        também inclui i <-> i+2.
    """
    n_nodes = int(n_nodes)
    ring_hops = int(ring_hops)

    if n_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)

    src_parts = []
    dst_parts = []

    for hop in range(1, ring_hops + 1):
        if hop >= n_nodes:
            break

        src = torch.arange(0, n_nodes - hop, dtype=torch.long)
        dst = src + hop

        src_parts.append(src)
        dst_parts.append(dst)

        src_parts.append(dst)
        dst_parts.append(src)

    if len(src_parts) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.stack(
        [
            torch.cat(src_parts, dim=0),
            torch.cat(dst_parts, dim=0),
        ],
        dim=0,
    )


def _repeat_edge_index(base_edge_index, batch_size, n_nodes, device):
    """
    Replica a topologia de um evento para vários eventos independentes.
    """
    batch_size = int(batch_size)
    n_nodes = int(n_nodes)

    if batch_size < 1 or base_edge_index.numel() == 0:
        return base_edge_index.to(device)

    base_edge_index = base_edge_index.to(device)

    n_edges = base_edge_index.shape[1]

    offsets = (
        torch.arange(batch_size, dtype=torch.long, device=device)
        .view(batch_size, 1, 1)
        * n_nodes
    )

    edges = base_edge_index.view(1, 2, n_edges) + offsets

    return edges.permute(1, 0, 2).reshape(2, batch_size * n_edges)


                                                                
                            
                                                                

def _nonnegative_energy_weights(values):
    """Uses nonnegative energy deposits to construct geometric node weights; negative deposits can arise from simulated noise."""
    values = _sanitize_numpy(values)
    return np.clip(values, 0.0, None).astype(np.float32)


def _weighted_radial_mean(values, eps=1e-8):
    """Média radial ponderada pela parcela não negativa da energia."""
    values = _sanitize_numpy(values)
    weights = _nonnegative_energy_weights(values)
    n_nodes = values.shape[1]

    if n_nodes < 2:
        return np.zeros((values.shape[0], 1), dtype=np.float32)

    positions = np.linspace(0.0, 1.0, n_nodes, dtype=np.float32)
    numerator = np.sum(weights * positions[None, :], axis=1, keepdims=True)
    denominator = np.sum(weights, axis=1, keepdims=True)

    result = np.zeros_like(denominator, dtype=np.float32)
    valid = denominator[:, 0] > eps
    result[valid, 0] = numerator[valid, 0] / denominator[valid, 0]
    return _sanitize_numpy(result)


def _weighted_radial_width(values, eps=1e-8):
    """Largura radial ponderada pela parcela não negativa da energia."""
    values = _sanitize_numpy(values)
    weights = _nonnegative_energy_weights(values)
    n_nodes = values.shape[1]

    if n_nodes < 2:
        return np.zeros((values.shape[0], 1), dtype=np.float32)

    positions = np.linspace(0.0, 1.0, n_nodes, dtype=np.float32)
    radial_mean = _weighted_radial_mean(values, eps=eps)
    numerator = np.sum(
        weights * (positions[None, :] - radial_mean) ** 2,
        axis=1,
        keepdims=True,
    )
    denominator = np.sum(weights, axis=1, keepdims=True)

    variance = np.zeros_like(denominator, dtype=np.float32)
    valid = denominator[:, 0] > eps
    variance[valid, 0] = numerator[valid, 0] / denominator[valid, 0]
    return _sanitize_numpy(np.sqrt(np.maximum(variance, 0.0)))


def _build_physics_features(rings, eps=1e-8):
    """
    Gera features físicas usando Rings[1:100].

    Divisão adotada no artigo:
        PS : Rings 1-8    -> índices Python 0:8
        EM : Rings 9-88   -> índices Python 8:88
        HAD: Rings 89-100 -> índices Python 88:100

    A subdivisão interna do bloco EM é mantida como agrupamento heurístico
    da arquitetura original, sem identificá-la como EM1/EM2/EM3.
    """
    rings = _sanitize_numpy(rings)

    if rings.ndim != 2 or rings.shape[1] != 100:
        raise ValueError(
            "Esta GNN requer exatamente 100 colunas: Rings[1:100]."
        )

    ps = rings[:, 0:8]
    em = rings[:, 8:88]
    had = rings[:, 88:100]

    sum_ps = np.sum(ps, axis=1, keepdims=True)
    sum_em = np.sum(em, axis=1, keepdims=True)
    sum_had = np.sum(had, axis=1, keepdims=True)

    sum_total = sum_ps + sum_em + sum_had
    safe_total = np.where(np.abs(sum_total) < eps, 1.0, sum_total)

    frac_ps = sum_ps / safe_total
    frac_em = sum_em / safe_total
    frac_had = sum_had / safe_total

    safe_sum_em = np.where(np.abs(sum_em) < eps, 1.0, sum_em)
    had_over_em = sum_had / safe_sum_em
    ps_over_em = sum_ps / safe_sum_em

    em_front = np.sum(rings[:, 8:13], axis=1, keepdims=True)
    em_late_core = np.sum(rings[:, 72:78], axis=1, keepdims=True)
    em_tail = np.sum(rings[:, 78:88], axis=1, keepdims=True)

    front_over_em = em_front / safe_sum_em
    late_core_over_em = em_late_core / safe_sum_em
    tail_over_em = em_tail / safe_sum_em

    ps_mean = _weighted_radial_mean(ps, eps=eps)
    em_mean = _weighted_radial_mean(em, eps=eps)
    had_mean = _weighted_radial_mean(had, eps=eps)

    ps_width = _weighted_radial_width(ps, eps=eps)
    em_width = _weighted_radial_width(em, eps=eps)
    had_width = _weighted_radial_width(had, eps=eps)

    features = np.hstack(
        [
            sum_ps,
            sum_em,
            sum_had,
            frac_ps,
            frac_em,
            frac_had,
            had_over_em,
            ps_over_em,
            em_front,
            em_late_core,
            em_tail,
            front_over_em,
            late_core_over_em,
            tail_over_em,
            ps_mean,
            em_mean,
            had_mean,
            ps_width,
            em_width,
            had_width,
        ]
    )

    return _sanitize_numpy(features)


def _standardize_features(X_train, X_test, eps=1e-8):
    """
    Padroniza features físicas com estatísticas do treino.
    """
    X_train = _sanitize_numpy(X_train)
    X_test = _sanitize_numpy(X_test)

    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)

    mean = _sanitize_numpy(mean)
    std = _sanitize_numpy(std)

    std = np.where(std < eps, 1.0, std)

    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std

    Xtr = _sanitize_numpy(Xtr)
    Xte = _sanitize_numpy(Xte)

    stats = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }

    return Xtr, Xte, stats


                                                                
                      
                                                                

class _RegionGraphBlock(nn.Module):
    """
    GNN para PS, EM ou HAD.
    """

    def __init__(
        self,
        n_nodes,
        region_id,
        node_hidden,
        graph_hidden,
        num_graph_layers,
        ring_hops,
        dropout,
    ):
        super().__init__()

        self.n_nodes = int(n_nodes)
        self.region_id = int(region_id)
        self.graph_hidden = int(graph_hidden)
        self.dropout = float(dropout)

        node_input_dim = 7

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden),
            nn.GELU(),
            nn.LayerNorm(node_hidden),
        )

        self.convs = nn.ModuleList()

        self.convs.append(
            SAGEConv(
                node_hidden,
                graph_hidden,
                aggr="mean",
            )
        )

        for _ in range(int(num_graph_layers) - 1):
            self.convs.append(
                SAGEConv(
                    graph_hidden,
                    graph_hidden,
                    aggr="mean",
                )
            )

        self.register_buffer(
            "base_edge_index",
            _make_chain_edge_index(
                n_nodes=self.n_nodes,
                ring_hops=ring_hops,
            ),
        )

        self.register_buffer(
            "local_positions",
            torch.linspace(
                0.0,
                1.0,
                self.n_nodes,
                dtype=torch.float32,
            ).view(1, self.n_nodes, 1),
        )

    def forward(self, values_z, values_raw, global_positions):
        """
        values_z:
            (batch, n_nodes)

        values_raw:
            (batch, n_nodes)
        """
        batch_size = values_z.shape[0]
        device = values_z.device

        values_z = _sanitize_tensor(values_z)
        values_raw = _sanitize_tensor(values_raw)

        value_z = values_z.unsqueeze(-1)
        value_raw = values_raw.unsqueeze(-1)

        local_pos = self.local_positions.to(device).expand(
            batch_size,
            -1,
            -1,
        )

        global_pos = global_positions.to(device).expand(
            batch_size,
            -1,
            -1,
        )

        region_one_hot = torch.zeros(
            batch_size,
            self.n_nodes,
            3,
            dtype=torch.float32,
            device=device,
        )

        region_one_hot[:, :, self.region_id] = 1.0

        node_features = torch.cat(
            [
                value_z,
                value_raw,
                local_pos,
                global_pos,
                region_one_hot,
            ],
            dim=-1,
        )

        h = node_features.reshape(
            batch_size * self.n_nodes,
            -1,
        )

        h = self.node_encoder(h)
        h = _sanitize_tensor(h)

        edge_index = _repeat_edge_index(
            base_edge_index=self.base_edge_index,
            batch_size=batch_size,
            n_nodes=self.n_nodes,
            device=device,
        )

        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = _sanitize_tensor(h)

            if i != len(self.convs) - 1:
                h = F.gelu(h)
                h = F.dropout(
                    h,
                    p=self.dropout,
                    training=self.training,
                )
                h = _sanitize_tensor(h)

        h = h.reshape(
            batch_size,
            self.n_nodes,
            self.graph_hidden,
        )

        pool_mean = torch.mean(h, dim=1)
        pool_max = torch.max(h, dim=1).values

        output = torch.cat(
            [pool_mean, pool_max],
            dim=1,
        )

        return _sanitize_tensor(output)


                                                                
                          
                                                                

class LayeredRingGraphHybrid(nn.Module):
    """
    PS-GNN + EM-GNN + HAD-GNN + branch MLP tabular.
    """

    def __init__(
        self,
        tabular_input_dim,
        node_hidden=64,
        graph_hidden=128,
        tab_hidden=256,
        num_graph_layers=2,
        ring_hops=1,
        dropout=0.0,
        graph_scale_init=0.05,
    ):
        super().__init__()

        self.graph_hidden = int(graph_hidden)
        self.tab_hidden = int(tab_hidden)
        self.dropout = float(dropout)

        self.ps_block = _RegionGraphBlock(
            n_nodes=8,
            region_id=0,
            node_hidden=node_hidden,
            graph_hidden=graph_hidden,
            num_graph_layers=num_graph_layers,
            ring_hops=ring_hops,
            dropout=dropout,
        )

        self.em_block = _RegionGraphBlock(
            n_nodes=80,
            region_id=1,
            node_hidden=node_hidden,
            graph_hidden=graph_hidden,
            num_graph_layers=num_graph_layers,
            ring_hops=ring_hops,
            dropout=dropout,
        )

        self.had_block = _RegionGraphBlock(
            n_nodes=12,
            region_id=2,
            node_hidden=node_hidden,
            graph_hidden=graph_hidden,
            num_graph_layers=num_graph_layers,
            ring_hops=ring_hops,
            dropout=dropout,
        )

        self.register_buffer(
            "ps_global_positions",
            torch.linspace(
                0.0,
                7.0 / 99.0,
                8,
                dtype=torch.float32,
            ).view(1, 8, 1),
        )

        self.register_buffer(
            "em_global_positions",
            torch.linspace(
                8.0 / 99.0,
                87.0 / 99.0,
                80,
                dtype=torch.float32,
            ).view(1, 80, 1),
        )

        self.register_buffer(
            "had_global_positions",
            torch.linspace(
                88.0 / 99.0,
                1.0,
                12,
                dtype=torch.float32,
            ).view(1, 12, 1),
        )

        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_input_dim, tab_hidden),
            nn.GELU(),
            nn.LayerNorm(tab_hidden),
            nn.Dropout(dropout),

            nn.Linear(tab_hidden, tab_hidden),
            nn.GELU(),
            nn.LayerNorm(tab_hidden),
            nn.Dropout(dropout),

            nn.Linear(tab_hidden, tab_hidden),
            nn.GELU(),
        )

        graph_embedding_dim = 3 * 2 * graph_hidden

        self.tabular_head = nn.Sequential(
            nn.Linear(tab_hidden, max(32, tab_hidden // 2)),
            nn.GELU(),
            nn.Linear(max(32, tab_hidden // 2), 1),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(
                graph_embedding_dim + tab_hidden,
                tab_hidden,
            ),
            nn.GELU(),
            nn.LayerNorm(tab_hidden),
            nn.Dropout(dropout),

            nn.Linear(
                tab_hidden,
                max(32, tab_hidden // 2),
            ),
            nn.GELU(),

            nn.Linear(
                max(32, tab_hidden // 2),
                1,
            ),
        )

        self.linear_skip = nn.Linear(tabular_input_dim, 1)

        self.graph_scale = nn.Parameter(
            torch.tensor(float(graph_scale_init), dtype=torch.float32)
        )

    def forward(self, rings_z, rings_raw, tabular_input):
        """
        Entrada:
            rings_z      -> (batch, 100)
            rings_raw    -> (batch, 100)
            tabular_input -> (batch, 120)
        """
        rings_z = _sanitize_tensor(rings_z)
        rings_raw = _sanitize_tensor(rings_raw)
        tabular_input = _sanitize_tensor(tabular_input)

        ps_z = rings_z[:, 0:8]
        em_z = rings_z[:, 8:88]
        had_z = rings_z[:, 88:100]

        ps_raw = rings_raw[:, 0:8]
        em_raw = rings_raw[:, 8:88]
        had_raw = rings_raw[:, 88:100]

        emb_ps = self.ps_block(
            values_z=ps_z,
            values_raw=ps_raw,
            global_positions=self.ps_global_positions,
        )

        emb_em = self.em_block(
            values_z=em_z,
            values_raw=em_raw,
            global_positions=self.em_global_positions,
        )

        emb_had = self.had_block(
            values_z=had_z,
            values_raw=had_raw,
            global_positions=self.had_global_positions,
        )

        graph_embedding = torch.cat(
            [
                emb_ps,
                emb_em,
                emb_had,
            ],
            dim=1,
        )

        graph_embedding = _sanitize_tensor(graph_embedding)

        tab_embedding = self.tabular_branch(tabular_input)
        tab_embedding = _sanitize_tensor(tab_embedding)

        tabular_prediction = self.tabular_head(tab_embedding)
        linear_prediction = self.linear_skip(tabular_input)

        fusion_input = torch.cat(
            [
                self.graph_scale * graph_embedding,
                tab_embedding,
            ],
            dim=1,
        )

        graph_correction = self.fusion_head(fusion_input)

        prediction = (
            linear_prediction
            + tabular_prediction
            + graph_correction
        )

        return _sanitize_tensor(prediction.view(-1))


                                                                
                  
                                                                

def _combined_loss(pred, target, lambda_mse, huber_beta):
    mse_loss = F.mse_loss(pred, target)
    huber_loss = F.smooth_l1_loss(pred, target, beta=huber_beta)
    return lambda_mse * mse_loss + (1.0 - lambda_mse) * huber_loss


def _predict_batches(model, rings_z, rings_raw, tabular, batch_size, device):
    n_samples = len(rings_z)
    output = np.empty(n_samples, dtype=np.float32)
    model.eval()

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            pred = model(
                rings_z=_sanitize_tensor(torch.as_tensor(
                    rings_z[start:end], dtype=torch.float32, device=device
                )),
                rings_raw=_sanitize_tensor(torch.as_tensor(
                    rings_raw[start:end], dtype=torch.float32, device=device
                )),
                tabular_input=_sanitize_tensor(torch.as_tensor(
                    tabular[start:end], dtype=torch.float32, device=device
                )),
            )

            output[start:end] = (
                _sanitize_tensor(pred).detach().cpu().numpy()
            )

    return _sanitize_numpy(output).ravel()


def gnn_pure_regression(X_train, X_test, y_train, y_test, **kwargs):
    """
    Arquitetura híbrida original com subgrafos sequenciais independentes
    para PS, EM e HAD, ramo MLP tabular e skip linear.

    Requer exclusivamente Rings[1:100].

    A interface e a saída permanecem inalteradas:
        model, ypred, r2, mse, corr, tpred
    """

    node_hidden = int(kwargs.get("node_hidden", 64))
    graph_hidden = int(kwargs.get("graph_hidden", 128))
    tab_hidden = int(kwargs.get("tab_hidden", 256))
    num_graph_layers = int(kwargs.get("num_graph_layers", 2))
    ring_hops = int(kwargs.get("ring_hops", 1))
    dropout = float(kwargs.get("dropout", 0.0))
    epochs = int(kwargs.get("epochs", 30))
    batch_size = int(kwargs.get("batch_size", 256))
    lr = float(kwargs.get("lr", 3e-4))
    weight_decay = float(kwargs.get("weight_decay", 1e-6))
    early_stopping_patience = int(kwargs.get("early_stopping_patience", 6))
    validation_fraction = float(kwargs.get("validation_fraction", 0.10))
    min_delta = float(kwargs.get("min_delta", 1e-8))
    lambda_mse = float(kwargs.get("lambda_mse", 0.7))
    huber_beta = float(kwargs.get("huber_beta", 1.0))
    max_train_samples = int(kwargs.get("max_train_samples", 20000))
    scale_y = bool(kwargs.get("scale_y", True))
    graph_scale_init = float(kwargs.get("graph_scale_init", 0.05))
    random_state = int(kwargs.get("random_state", 0))
    device = str(kwargs.get("device", "cpu"))
    verbose = bool(kwargs.get("verbose", False))

    if not 0.0 <= lambda_mse <= 1.0:
        raise ValueError("lambda_mse deve estar entre 0 e 1.")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction deve estar entre 0 e 0.5.")
    if node_hidden < 1 or graph_hidden < 1 or tab_hidden < 2:
        raise ValueError("Dimensões ocultas inválidas.")
    if num_graph_layers < 1 or ring_hops < 1:
        raise ValueError("num_graph_layers e ring_hops devem ser >= 1.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs e batch_size devem ser >= 1.")

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    rng = np.random.default_rng(random_state)

    Xtr_full_raw = _sanitize_numpy(X_train)
    Xte_raw = _sanitize_numpy(X_test)
    ytr_full = _sanitize_numpy(y_train).ravel()
    yte = None if y_test is None else _sanitize_numpy(y_test).ravel()

    if Xtr_full_raw.ndim != 2 or Xte_raw.ndim != 2:
        raise ValueError("X_train e X_test devem ser matrizes 2D.")
    if Xtr_full_raw.shape[1] != 100 or Xte_raw.shape[1] != 100:
        raise ValueError("Esta GNN requer exatamente 100 colunas: Rings[1:100].")
    if len(Xtr_full_raw) != len(ytr_full) or (yte is not None and len(Xte_raw) != len(yte)):
        raise ValueError("Tamanhos de X e y incompatíveis.")

    n_train_full = len(Xtr_full_raw)
    n_test = len(Xte_raw)

                                                                 
    if max_train_samples > 0 and n_train_full > max_train_samples:
        pool_idx = np.sort(rng.choice(
            n_train_full, size=max_train_samples, replace=False
        ))
    else:
        pool_idx = np.arange(n_train_full)

    if len(pool_idx) < 10:
        raise ValueError("Amostras insuficientes para treino e validação.")

    shuffled_pool = rng.permutation(pool_idx)
    n_val = max(1, int(round(validation_fraction * len(pool_idx))))
    n_val = min(n_val, len(pool_idx) - 1)
    val_idx = np.sort(shuffled_pool[:n_val])
    fit_idx = np.sort(shuffled_pool[n_val:])

    rings_raw_fit = Xtr_full_raw[fit_idx]
    rings_raw_val = Xtr_full_raw[val_idx]
    y_fit = ytr_full[fit_idx]
    y_val = ytr_full[val_idx]

                                                               
    ring_stats = _fit_standardizer(rings_raw_fit)
    rings_z_fit = _apply_standardizer(rings_raw_fit, ring_stats)
    rings_z_val = _apply_standardizer(rings_raw_val, ring_stats)
    rings_z_te = _apply_standardizer(Xte_raw, ring_stats)

    physical_fit_raw = _build_physics_features(rings_raw_fit)
    physical_val_raw = _build_physics_features(rings_raw_val)
    physical_te_raw = _build_physics_features(Xte_raw)

    physical_stats = _fit_standardizer(physical_fit_raw)
    physical_fit_z = _apply_standardizer(physical_fit_raw, physical_stats)
    physical_val_z = _apply_standardizer(physical_val_raw, physical_stats)
    physical_te_z = _apply_standardizer(physical_te_raw, physical_stats)

    tabular_fit = _sanitize_numpy(np.hstack([rings_z_fit, physical_fit_z]))
    tabular_val = _sanitize_numpy(np.hstack([rings_z_val, physical_val_z]))
    tabular_te = _sanitize_numpy(np.hstack([rings_z_te, physical_te_z]))

    if scale_y:
        y_mean = float(np.mean(y_fit))
        y_std = float(np.std(y_fit))
        if not np.isfinite(y_std) or y_std < 1e-12:
            y_std = 1.0
    else:
        y_mean, y_std = 0.0, 1.0

    y_fit_scaled = _sanitize_numpy((y_fit - y_mean) / y_std).ravel()
    y_val_scaled = _sanitize_numpy((y_val - y_mean) / y_std).ravel()

    model = LayeredRingGraphHybrid(
        tabular_input_dim=tabular_fit.shape[1],
        node_hidden=node_hidden,
        graph_hidden=graph_hidden,
        tab_hidden=tab_hidden,
        num_graph_layers=num_graph_layers,
        ring_hops=ring_hops,
        dropout=dropout,
        graph_scale_init=graph_scale_init,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    t0 = time.time()
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    best_epoch = 0
    patience_left = early_stopping_patience
    n_skipped_batches = 0
    history = []
    n_fit = len(fit_idx)

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n_fit)
        train_loss_sum = 0.0
        train_count = 0

        for start in range(0, n_fit, batch_size):
            end = min(start + batch_size, n_fit)
            idx = permutation[start:end]

            rings_z_batch = torch.as_tensor(
                rings_z_fit[idx], dtype=torch.float32, device=device
            )
            rings_raw_batch = torch.as_tensor(
                rings_raw_fit[idx], dtype=torch.float32, device=device
            )
            tabular_batch = torch.as_tensor(
                tabular_fit[idx], dtype=torch.float32, device=device
            )
            y_batch = torch.as_tensor(
                y_fit_scaled[idx], dtype=torch.float32, device=device
            )

            optimizer.zero_grad(set_to_none=True)
            pred = model(
                rings_z=_sanitize_tensor(rings_z_batch),
                rings_raw=_sanitize_tensor(rings_raw_batch),
                tabular_input=_sanitize_tensor(tabular_batch),
            )
            loss = _combined_loss(
                _sanitize_tensor(pred), _sanitize_tensor(y_batch),
                lambda_mse, huber_beta
            )

            if not torch.isfinite(loss):
                n_skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            gradients_finite = all(
                p.grad is None or torch.all(torch.isfinite(p.grad))
                for p in model.parameters()
            )
            if not gradients_finite:
                optimizer.zero_grad(set_to_none=True)
                n_skipped_batches += 1
                continue

            optimizer.step()

            parameters_finite = all(
                torch.all(torch.isfinite(p.data)) for p in model.parameters()
            )
            if not parameters_finite:
                model.load_state_dict(best_state)
                optimizer.zero_grad(set_to_none=True)
                n_skipped_batches += 1
                continue

            n_batch = len(idx)
            train_loss_sum += float(loss.item()) * n_batch
            train_count += n_batch

        train_loss = train_loss_sum / max(1, train_count)

                                                                   
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                end = min(start + batch_size, len(val_idx))
                pred = model(
                    rings_z=_sanitize_tensor(torch.as_tensor(
                        rings_z_val[start:end], dtype=torch.float32, device=device
                    )),
                    rings_raw=_sanitize_tensor(torch.as_tensor(
                        rings_raw_val[start:end], dtype=torch.float32, device=device
                    )),
                    tabular_input=_sanitize_tensor(torch.as_tensor(
                        tabular_val[start:end], dtype=torch.float32, device=device
                    )),
                )
                y_batch = torch.as_tensor(
                    y_val_scaled[start:end], dtype=torch.float32, device=device
                )
                val_loss_batch = _combined_loss(
                    _sanitize_tensor(pred), _sanitize_tensor(y_batch),
                    lambda_mse, huber_beta
                )
                if not torch.isfinite(val_loss_batch):
                    continue
                n_batch = end - start
                val_loss_sum += float(val_loss_batch.item()) * n_batch
                val_count += n_batch

        val_loss = val_loss_sum / max(1, val_count)
        history.append({
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        if verbose:
            print(
                f"[Regional Hybrid GNN] epoch={epoch + 1}/{epochs}, "
                f"train_loss={train_loss:.8f}, val_loss={val_loss:.8f}, "
                f"skipped_batches={n_skipped_batches}"
            )

        if np.isfinite(val_loss) and val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print("[Regional Hybrid GNN] early stopping.")
                break

    model.load_state_dict(best_state)
    ttrain = time.time() - t0

    t1 = time.time()
    ypred_scaled = _predict_batches(
        model, rings_z_te, Xte_raw, tabular_te, batch_size, device
    )
    tpred = time.time() - t1

    ypred = (ypred_scaled * y_std + y_mean).astype(np.float64)
    ypred = np.nan_to_num(
        ypred, nan=y_mean, posinf=y_mean, neginf=y_mean
    )

    if len(ypred) != len(Xte_raw):
        raise RuntimeError(
            f"len(ypred)={len(ypred)} é diferente de len(X_test)={len(Xte_raw)}."
        )

                                   
    if yte is None:
        r2=mse=corr=float('nan')
    else:
        r2 = r2_score(yte, ypred)
        mse = mean_squared_error(yte, ypred)
        if np.std(yte) < 1e-12 or np.std(ypred) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.nan_to_num(
                np.corrcoef(yte, ypred)[0, 1], nan=0.0
            ))

    model.t_train = ttrain
    model.t_pred_total = tpred
    model.t_pred_per_sample = tpred / max(1, len(ypred))
    model.scaler_stats_ = ring_stats
    model.physical_feature_stats_ = physical_stats
    model.y_scaler_stats_ = {
        "scale_y": scale_y, "mean": y_mean, "std": y_std
    }
    model.training_history_ = history
    model.n_parameters_ = int(sum(p.numel() for p in model.parameters()))

    model.graph_info_ = {
        "model_family": "regional_hybrid_graphsage_mlp",
        "topology": "independent_PS_EM_HAD_sequential_graphs",
        "PS_rings": "1:8",
        "EM_rings": "9:88",
        "HAD_rings": "89:100",
        "cross_region_edges": False,
        "n_train_full": int(n_train_full),
        "n_pool": int(len(pool_idx)),
        "n_fit": int(len(fit_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(n_test),
        "node_hidden": node_hidden,
        "graph_hidden": graph_hidden,
        "tab_hidden": tab_hidden,
        "num_graph_layers": num_graph_layers,
        "ring_hops": ring_hops,
        "dropout": dropout,
        "epochs_requested": epochs,
        "best_epoch": int(best_epoch),
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "lambda_mse": lambda_mse,
        "huber_beta": huber_beta,
        "graph_scale_init": graph_scale_init,
        "graph_scale_final": float(model.graph_scale.detach().cpu().item()),
        "best_val_loss": float(best_val_loss),
        "skipped_batches": int(n_skipped_batches),
        "validation_fraction": validation_fraction,
        "min_delta": min_delta,
    }

    model.subsample_info_ = {
        "pool_idx": pool_idx,
        "fit_idx": fit_idx,
        "val_idx": val_idx,
        "max_train_samples": max_train_samples,
    }

    gc.collect()
    return model, ypred, r2, mse, corr, tpred