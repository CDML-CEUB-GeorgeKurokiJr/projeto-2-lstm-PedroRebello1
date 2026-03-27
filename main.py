import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# adiciona indicadores tecnicos ao dataframe
def add_indicators(df, col='Close', window=14):
    df_out = df.copy()
    asset_prefix = col.replace('_Close', '')
    sma_col = f'{asset_prefix}_SMA_{window}'
    ema_col = f'{asset_prefix}_EMA_{window}'
    rsi_col = f'{asset_prefix}_RSI_{window}'

    df_out[sma_col] = df_out[col].rolling(window=window).mean()
    df_out[ema_col] = df_out[col].ewm(span=window, adjust=False).mean()

    delta = df_out[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df_out[rsi_col] = 100 - (100 / (1 + rs))
    return df_out

# COleta os dados
def fetch_data(tickers, start_date, end_date):
    dfs = []
    for t in tickers:
        try:
            print(f"Baixando dados para {t}...")
            df = yf.download(t, start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError(f"Nenhum dado retornado para {t}.")

            if isinstance(df.columns, pd.MultiIndex):
                close_data = df['Close']
                if isinstance(close_data, pd.Series):
                    df = close_data.to_frame(name=f'{t}_Close')
                else:
                    # Quando retorna DataFrame, pegamos a primeira coluna do ticker esperado.
                    df = close_data.iloc[:, [0]].rename(columns={close_data.columns[0]: f'{t}_Close'})
            else:
                df = df[['Close']].rename(columns={'Close': f'{t}_Close'})

            df = add_indicators(df, col=f'{t}_Close', window=14)
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao baixar dados de {t}: {e}")
            return None

    # Concatena os dados e remove linhas com valores nulos devido ao cálculo dos indicadores
    if not dfs:
        return None
    data = pd.concat(dfs, axis=1).dropna()
    return data

# Cria sequências para treinamento do modelo
def create_multivariate_sequences(data, seq_len, target_indices):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, target_indices])
    return np.array(X), np.array(y)

# Cria um dataset prórpio para o pytorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




class LSTM_GRU_Model(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        target_input_indices,
        hidden_lstm=128,
        hidden_gru=96,
        lstm_layers=2,
        gru_layers=1,
        dropout=0.10,
        bidirectional=False,
    ):
        super().__init__()
        self.register_buffer('target_input_indices', torch.tensor(target_input_indices, dtype=torch.long))

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_lstm,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_size = hidden_lstm * (2 if bidirectional else 1)

        self.gru = nn.GRU(
            input_size=lstm_out_size,
            hidden_size=hidden_gru,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )

        self.norm_lstm = nn.LayerNorm(lstm_out_size)
        self.norm_gru = nn.LayerNorm(hidden_gru)
        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Linear(hidden_gru, output_size)

    def forward(self, x):
        out_lstm, _ = self.lstm(x)
        out_lstm = self.norm_lstm(out_lstm)
        out_gru, _ = self.gru(out_lstm)
        out_gru = self.norm_gru(out_gru)
        deltas = self.delta_head(self.dropout(out_gru[:, -1, :]))
        last_closes = x[:, -1, :].index_select(dim=1, index=self.target_input_indices)
        final_output = last_closes + deltas
        return final_output





# Treina o modelo e salva os melhores pesos baseados na validação
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=80,
    lr=0.0007,
    weight_decay=1e-6,
    grad_clip=1.0,
    early_stopping_patience=20,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=8,
        min_lr=1e-6,
    )

    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss = criterion(preds, yb)
                val_batch_losses.append(val_loss.item())
        val_loss = np.mean(val_batch_losses)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 2 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:2d}: Train Loss = {train_loss:.6f}, "
                f"Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Parada antecipada no epoch {epoch} "
                    f"(sem melhora de validacao por {early_stopping_patience} epochs)."
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return train_losses, val_losses

#Avalia o modelo e plota a comparação entre real e previsto por ativo
def evaluate_model(model, test_loader, y_test, scaler, device, target_indices, target_names):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())

    if not preds:
        raise ValueError("Nao ha lotes de teste para avaliacao.")

    preds = np.vstack(preds)

    # Ajustar shape para inverse_transform mantendo apenas as colunas alvo
    preds_full = np.zeros((len(preds), scaler.n_features_in_))
    actual_full = np.zeros((len(y_test), scaler.n_features_in_))

    preds_full[:, target_indices] = preds
    actual_full[:, target_indices] = y_test

    preds_inverse = scaler.inverse_transform(preds_full)[:, target_indices]
    actual_inverse = scaler.inverse_transform(actual_full)[:, target_indices]

    print("\nMetricas por ativo:")
    for idx, name in enumerate(target_names):
        mae = np.mean(np.abs(actual_inverse[:, idx] - preds_inverse[:, idx]))
        rmse = np.sqrt(np.mean((actual_inverse[:, idx] - preds_inverse[:, idx]) ** 2))
        print(f"{name}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    n_targets = len(target_names)
    n_cols = 2 if n_targets > 1 else 1
    n_rows = math.ceil(n_targets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for idx, name in enumerate(target_names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.plot(actual_inverse[:, idx], label='Real')
        ax.plot(preds_inverse[:, idx], label='Previsto')
        ax.set_title(f'{name} - Real vs Previsto')
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Preco')
        ax.grid(True)
        ax.legend()

    for idx in range(n_targets, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.show()

#Calcula RSI simples com base nos ultimos fechamentos.
def _compute_rsi(closes, window=14):
    if len(closes) < 2:
        return 50.0

    deltas = np.diff(closes[-(window + 1):])
    gains = np.clip(deltas, a_min=0, a_max=None)
    losses = -np.clip(deltas, a_min=None, a_max=0)

    avg_gain = np.mean(gains) if len(gains) else 0.0
    avg_loss = np.mean(losses) if len(losses) else 0.0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Realiza previsões para 7 dias no futuro para todos os ativos
def predict_future(model, data, scaler, seq_len, device, target_indices, target_names, n_future=7, indicator_window=14):
    model.eval()
    current_sequence_unscaled = data[-seq_len:].values.copy()
    feature_names = list(data.columns)
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    future_preds_scaled = []

    per_asset_indices = {}
    for close_name in target_names:
        ticker = close_name.replace('_Close', '')
        sma_name = f'{ticker}_SMA_{indicator_window}'
        ema_name = f'{ticker}_EMA_{indicator_window}'
        rsi_name = f'{ticker}_RSI_{indicator_window}'
        required = [close_name, sma_name, ema_name, rsi_name]
        missing = [col for col in required if col not in feature_to_idx]
        if missing:
            raise ValueError(f'Colunas ausentes para previsao de {ticker}: {missing}')
        per_asset_indices[close_name] = {
            'close': feature_to_idx[close_name],
            'sma': feature_to_idx[sma_name],
            'ema': feature_to_idx[ema_name],
            'rsi': feature_to_idx[rsi_name],
        }

    for _ in range(n_future):
        current_scaled = scaler.transform(current_sequence_unscaled)
        current_input = torch.tensor(current_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            next_pred = model(current_input)
            pred_scaled = next_pred.cpu().numpy()[0]
            future_preds_scaled.append(pred_scaled)

        target_full_scaled = np.zeros((1, scaler.n_features_in_))
        target_full_scaled[0, target_indices] = pred_scaled
        pred_unscaled = scaler.inverse_transform(target_full_scaled)[0, target_indices]

        next_row = current_sequence_unscaled[-1].copy()

        for target_pos, close_name in enumerate(target_names):
            idx_map = per_asset_indices[close_name]
            close_idx = idx_map['close']
            sma_idx = idx_map['sma']
            ema_idx = idx_map['ema']
            rsi_idx = idx_map['rsi']

            new_close = pred_unscaled[target_pos]
            close_history = current_sequence_unscaled[:, close_idx]
            extended_closes = np.append(close_history, new_close)

            sma_val = np.mean(extended_closes[-indicator_window:])
            alpha = 2 / (indicator_window + 1)
            prev_ema = current_sequence_unscaled[-1, ema_idx]
            ema_val = alpha * new_close + (1 - alpha) * prev_ema
            rsi_val = _compute_rsi(extended_closes, window=indicator_window)

            next_row[close_idx] = new_close
            next_row[sma_idx] = sma_val
            next_row[ema_idx] = ema_val
            next_row[rsi_idx] = rsi_val

        current_sequence_unscaled = np.vstack([current_sequence_unscaled[1:], next_row])

    future_preds_scaled = np.array(future_preds_scaled)
    future_scaled_full = np.zeros((n_future, scaler.n_features_in_))
    future_scaled_full[:, target_indices] = future_preds_scaled
    future_prices = scaler.inverse_transform(future_scaled_full)[:, target_indices]

    plt.figure(figsize=(12, 6))
    for idx, name in enumerate(target_names):
        plt.plot(range(1, n_future + 1), future_prices[:, idx], marker='o', label=name)
    plt.title(f'Previsao de {n_future} dias futuros por ativo')
    plt.xlabel('Dias a frente')
    plt.ylabel('Preco previsto')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for i in range(n_future):
        per_day = ", ".join(
            [f"{name}: {future_prices[i, idx]:.2f}" for idx, name in enumerate(target_names)]
        )
        print(f"Dia +{i+1}: {per_day}")





# Configurações iniciais
tickers = ['NVDA', 'WDC', 'AMD', 'TSM', 'INTC']
start_date = '2017-01-01'
end_date = '2025-07-31'
seq_len = 60
set_seed(42)

# 1. Obter e preparar dados
raw_data = fetch_data(tickers, start_date, end_date)

if raw_data is not None and not raw_data.empty:
    close_cols = [col for col in raw_data.columns if col.endswith('_Close')]
    if not close_cols:
        raise ValueError("Nenhuma coluna de fechamento foi encontrada para os ativos informados.")

    target_indices = [raw_data.columns.get_loc(col) for col in close_cols]

    # 2. Divisao temporal em treino/val/teste sem vazamento
    n_total = len(raw_data)
    train_end = int(0.7 * n_total)
    val_end = int(0.85 * n_total)

    if train_end <= seq_len or val_end - train_end <= seq_len or n_total - val_end <= seq_len:
        raise ValueError(
            "Dados insuficientes para divisao temporal com o seq_len atual. "
            "Reduza 'seq_len' ou amplie o intervalo de datas."
        )

    # 3. Normalizacao ajustada apenas no treino
    scaler = MinMaxScaler()
    train_raw = raw_data.iloc[:train_end].values
    scaler.fit(train_raw)

    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(raw_data.iloc[train_end - seq_len:val_end].values)
    test_scaled = scaler.transform(raw_data.iloc[val_end - seq_len:].values)

    # 4. Criacao de sequencias por bloco temporal
    X_train, y_train = create_multivariate_sequences(train_scaled, seq_len, target_indices)
    X_val, y_val = create_multivariate_sequences(val_scaled, seq_len, target_indices)
    X_test, y_test = create_multivariate_sequences(test_scaled, seq_len, target_indices)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "Dados insuficientes para criar sequencias. "
            f"Ajuste 'seq_len' (atual: {seq_len}) ou amplie o intervalo de datas."
        )

    # 5. DataLoaders
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 6. Instanciar Modelo
    input_size = X_train.shape[2]
    output_size = len(target_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUtilizando dispositivo: {device}")

    model = LSTM_GRU_Model(
        input_size=input_size,
        output_size=output_size,
        target_input_indices=target_indices,
    ).to(device)

    # 7. Treinar
    print("\nIniciando treinamento...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, epochs=120)

    # Plot das curvas de perda
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Curvas de perda durante o treinamento')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 8. Avaliar
    print("\nAvaliando modelo nos dados de teste...")
    evaluate_model(model, test_loader, y_test, scaler, device, target_indices, close_cols)

    # 9. Prever
    print("\nGerando previsoes futuras...")
    predict_future(model, raw_data, scaler, seq_len, device, target_indices, close_cols, n_future=20)
else:
    raise ValueError("Falha ao obter dados validos do yfinance para os tickers informados.")