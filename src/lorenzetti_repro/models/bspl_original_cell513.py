"""B-spline regressor extracted from the author's experimental notebook, cell 513."""
import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from lorenzetti_repro.scaling import z_norm_input

def b_splines_regression(X_train, X_test, y_train, y_test, **kwargs):
    """
    Regressão com base B-spline por feature + modelo linear.
    Retorna: (modelo, y_pred, r2, mse, corr, tpred)

    Kwargs principais:
      # B-splines (por feature, sem interações)
      - degree: int = 3               (grau da B-spline)
      - n_knots: int = 7              (nós por feature; controla flexibilidade)
      - knots: {"quantile","uniform"} = "quantile"
      - extrapolation: {"continue","linear","constant","error"} = "continue"
      - include_bias: bool = False
      - sparse_output: bool = True    (gera CSR; economiza RAM)

      # Estimador linear
      - estimator: {"ridge","sgd"} = "ridge"
      - alpha: float = 1.0
      - ridge_solver: {"sparse_cg","auto"} = "sparse_cg"   (bom p/ CSR)
      - max_iter: int = 10000
      - tol: float = 1e-4
      - random_state: int = 0
      - eta0 (SGD): float = 0.01
      - power_t (SGD): float = 0.25
      - average (SGD): bool = False
    """
    # 1) Normalização (consistência com o pipeline)
    X_train, X_test, stats = z_norm_input(X_train, X_test)

    # 2) Tipos estáveis
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test  = np.asarray(X_test,  dtype=np.float32)
    y_train = np.asarray(y_train).ravel()
    y_test  = None if y_test is None else np.asarray(y_test).ravel()

    # 3) Configuração da base B-spline
    degree        = int(kwargs.get("degree", 3))
    n_knots       = int(kwargs.get("n_knots", 7))
    knots         = str(kwargs.get("knots", "quantile"))
    extrapolation = str(kwargs.get("extrapolation", "continue"))
    include_bias  = bool(kwargs.get("include_bias", False))
    sparse_out    = bool(kwargs.get("sparse_output", True))

    spl = SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=knots,
        extrapolation=extrapolation,
        include_bias=include_bias,
        sparse_output=sparse_out
    )

    # 4) Ajuste/transformação
    Z_train = spl.fit_transform(X_train)   # CSR se sparse_output=True
    Z_test  = spl.transform(X_test)

    # 5) Estimador
    estimator    = str(kwargs.get("estimator", "ridge")).lower()
    alpha        = float(kwargs.get("alpha", 1.0))
    max_iter     = int(kwargs.get("max_iter", 10000))
    tol          = float(kwargs.get("tol", 1e-4))
    random_state = int(kwargs.get("random_state", 0))

    if estimator == "ridge":
        ridge_solver = str(kwargs.get("ridge_solver", "sparse_cg")).lower()
        # Garantir tipos adequados
        if sparse.issparse(Z_train):
            Z_train = Z_train.astype(np.float32, copy=False)
            Z_test  = Z_test.astype(np.float32, copy=False)
        else:
            Z_train = np.asarray(Z_train, dtype=np.float32)
            Z_test  = np.asarray(Z_test,  dtype=np.float32)

        model = Ridge(
            alpha=alpha,
            solver=ridge_solver,     # "sparse_cg" lida bem com CSR grande
            max_iter=max_iter,
            tol=tol
        )

    elif estimator == "sgd":
        # SGD trabalha bem com CSR e dados grandes; penalty L2 (ridge-like)
        if sparse.issparse(Z_train):
            Z_train = Z_train.astype(np.float32, copy=False)
            Z_test  = Z_test.astype(np.float32, copy=False)
        else:
            Z_train = np.asarray(Z_train, dtype=np.float32)
            Z_test  = np.asarray(Z_test,  dtype=np.float32)

        model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=alpha,
            learning_rate="invscaling",
            eta0=float(kwargs.get("eta0", 0.01)),
            power_t=float(kwargs.get("power_t", 0.25)),
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            average=bool(kwargs.get("average", False))
        )
    else:
        raise ValueError("estimator deve ser 'ridge' ou 'sgd'.")

    # 6) Treino
    t0 = time.time()
    model.fit(Z_train, y_train)
    ttrain = time.time() - t0

    # 7) Predição
    t1 = time.time()
    ypred = model.predict(Z_test)
    tpred = time.time() - t1

    # 8) Métricas
    if y_test is None:
        r2 = mse = corr = float('nan')
    else:
        r2 = r2_score(y_test, ypred)
        mse = mean_squared_error(y_test, ypred)
        corr = np.corrcoef(y_test.ravel(), np.asarray(ypred).ravel())[0, 1]

    # 9) Metadados úteis
    model.t_train = ttrain
    model.t_pred_total = tpred
    model.t_pred_per_sample = tpred / max(1, len(ypred))
    model.spline_transformer_ = spl
    model.scaler_stats_ = stats
    try:
        model.n_output_features_ = spl.n_features_out_
    except Exception:
        pass

    return model, ypred, r2, mse, corr, tpred