# Original DNN function from notebook cell #527; external dependencies made explicit.
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from lorenzetti_repro.scaling import z_norm_input

from sklearn.neural_network import MLPRegressor

def dnn_regression(X_train, X_test, y_train, y_test, **kwargs):
    """
    Rede neural profunda (MLP) no padrão do seu pipeline.
    Retorna: (modelo, y_pred, r2, mse, corr, tpred)

    Hiperparâmetros principais (para a nested CV):
      - n_layers: int >= 1              # número de camadas ocultas
      - width: int                      # neurônios na 1ª oculta
      - shape: {"constant","halving"}   # como gerar as demais camadas
      - min_width: int = 16             # piso ao usar "halving"

    Demais hiperparâmetros relevantes:
      - activation: {"relu","tanh"} (default "relu")
      - alpha: float (L2, weight decay; default 1e-4)
      - learning_rate_init: float (default 1e-3)
      - batch_size: int (default 256)
      - max_iter: int (default 200)
      - solver: {"adam","sgd"} (default "adam")
      - early_stopping: bool (default True)
      - n_iter_no_change: int (default 20)
      - validation_fraction: float (default 0.1)
      - random_state: int (default 0)
      - pred_batch_size: int opcional (lotes na predição)
    """
    # 1) Normalização (essencial p/ MLP) — consistência com seu pipeline
    X_train, X_test, stats = z_norm_input(X_train, X_test)

    # 2) Tipos/formatos estáveis
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test  = np.asarray(X_test,  dtype=np.float32)
    y_train = np.asarray(y_train).ravel()
    y_test  = None if y_test is None else np.asarray(y_test).ravel()

    # 3) Arquitetura a partir de (n_layers, width, shape)
    n_layers   = int(kwargs.pop("n_layers", 2))
    width      = int(kwargs.pop("width", 128))
    shape      = str(kwargs.pop("shape", "constant")).lower()
    min_width  = int(kwargs.pop("min_width", 16))
    if n_layers < 1:
        raise ValueError("n_layers deve ser >= 1")
    if width < 1:
        raise ValueError("width deve ser >= 1")

    if shape == "constant":
        hidden = tuple([width] * n_layers)
    elif shape == "halving":
        hs = []
        w = width
        for _ in range(n_layers):
            hs.append(max(min_width, int(round(w))))
            w = max(min_width, w / 2.0)
        hidden = tuple(hs)
    else:
        raise ValueError("shape deve ser 'constant' ou 'halving'.")

    # 4) Extras
    pred_batch_size = int(kwargs.pop("pred_batch_size", 0))  # 0 = predição única

    # 5) Defaults sensatos; permitem override via kwargs
    mlp_defaults = dict(
        hidden_layer_sizes=hidden,
        activation=kwargs.pop("activation", "relu"),
        alpha=float(kwargs.pop("alpha", 1e-4)),
        learning_rate_init=float(kwargs.pop("learning_rate_init", 1e-3)),
        batch_size=int(kwargs.pop("batch_size", 256)),
        max_iter=int(kwargs.pop("max_iter", 200)),
        solver=kwargs.pop("solver", "adam"),
        early_stopping=bool(kwargs.pop("early_stopping", True)),
        n_iter_no_change=int(kwargs.pop("n_iter_no_change", 20)),
        validation_fraction=float(kwargs.pop("validation_fraction", 0.1)),
        random_state=int(kwargs.pop("random_state", 0)),
        shuffle=True,
        verbose=False
    )
    # quaisquer kwargs restantes aceitos pelo MLPRegressor serão aplicados:
    mlp_defaults.update(kwargs)

    model = MLPRegressor(**mlp_defaults)

    # 6) Treino
    t0 = time.time()
    model.fit(X_train, y_train)
    ttrain = time.time() - t0

    # 7) Predição (em lotes opcional)
    t1 = time.time()
    if pred_batch_size > 0:
        m = X_test.shape[0]
        ypred = np.empty(m, dtype=float)
        for start in range(0, m, pred_batch_size):
            end = min(start + pred_batch_size, m)
            ypred[start:end] = model.predict(X_test[start:end])
    else:
        ypred = model.predict(X_test)
    tpred = time.time() - t1

    # 8) Métricas
    if y_test is None:
        r2=mse=corr=float('nan')
    else:
        r2  = r2_score(y_test, ypred)
        mse = mean_squared_error(y_test, ypred)
        corr = np.corrcoef(y_test.ravel(), np.asarray(ypred).ravel())[0, 1]

    # 9) Tempos no objeto (compatível com seu coletor)
    model.t_train = ttrain
    model.t_pred_total = tpred
    model.t_pred_per_sample = tpred / max(1, len(ypred))
    model.scaler_stats_ = stats

    return model, ypred, r2, mse, corr, tpred