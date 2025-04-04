import os
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2, l1_l2
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.layers import BatchNormalization, Layer, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GRU, Concatenate, Add, GlobalAveragePooling1D
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import ccxt
import time
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import concurrent.futures
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Caminhos para os modelos
MODEL_PATH = "rede_neural/modelo_lstm_candlestick_manus.keras"
MODEL_PATH_ENSEMBLE = "rede_neural/modelo_ensemble.keras"

class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model  # Armazenar d_model para uso posterior
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=rate)
        self.ffn1 = Dense(dff, activation='relu')
        self.ffn2 = Dense(d_model)
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def build(self, input_shape):
        # Implementação explícita do método build
        super(TransformerEncoderLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Multi-Head Attention
        attn_output = self.mha(inputs, inputs, training=training)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) 
        
        # Feed Forward Network
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def compute_output_shape(self, input_shape):
        return input_shape

class TransformerEncoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model  # Armazenar d_model para uso posterior
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, rate) 
            for _ in range(num_layers)
        ]
        self.layernorm = LayerNormalization(epsilon=1e-6)
    
    def build(self, input_shape):
        # Implementação explícita do método build
        super(TransformerEncoder, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = inputs
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        return self.layernorm(x)
    
    def compute_output_shape(self, input_shape):
        return input_shape

class AttentionLayer(Layer):
    def __init__(self, output_dim=64, **kwargs):
        self.output_dim = output_dim
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W_q = self.add_weight(name="query_weight", shape=(input_shape[-1], self.output_dim),
                                  initializer="glorot_uniform", trainable=True)
        self.W_k = self.add_weight(name="key_weight", shape=(input_shape[-1], self.output_dim),
                                  initializer="glorot_uniform", trainable=True)
        self.W_v = self.add_weight(name="value_weight", shape=(input_shape[-1], self.output_dim),
                                  initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Linear projections
        q = tf.keras.backend.dot(inputs, self.W_q)
        k = tf.keras.backend.dot(inputs, self.W_k)
        v = tf.keras.backend.dot(inputs, self.W_v)
        
        # Attention scores
        scores = tf.keras.backend.batch_dot(q, k, axes=[2, 2]) / tf.math.sqrt(tf.cast(self.output_dim, tf.float32))
        
        # Attention distribution
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Output
        output = tf.keras.backend.batch_dot(attention_weights, v)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"output_dim": self.output_dim})
        return config

# Modelo LSTM melhorado com atenção e técnicas avançadas
def criar_modelo_melhorado(input_shape, dropout_rate=0.2, l2_reg=0.001):
    """
    Cria um modelo LSTM melhorado com atenção e técnicas avançadas para previsão de preços de criptomoedas.
    
    Parâmetros:
    - input_shape: Formato dos dados de entrada (sequência, features)
    - dropout_rate: Taxa de dropout para regularização
    - l2_reg: Valor de regularização L2
    
    Retorna:
    - Modelo compilado
    """
    inputs = Input(shape=input_shape)
    
    # Camada convolucional para extração de características locais
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # Segunda camada convolucional com diferentes tamanhos de kernel para capturar padrões em diferentes escalas
    conv2_1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(conv1)
    conv2_2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(conv1)
    conv2_3 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(conv1)
    
    # Concatenar as saídas das camadas convolucionais
    conv_concat = Concatenate()([conv2_1, conv2_2, conv2_3])
    conv_concat = BatchNormalization()(conv_concat)
    conv_concat = Dropout(dropout_rate)(conv_concat)
    
    # Camada Bidirectional LSTM para capturar dependências temporais
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.0001, l2=l2_reg)))(conv_concat)
    lstm1 = BatchNormalization()(lstm1)
    
    # Camada GRU para complementar o LSTM
    gru1 = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.0001, l2=l2_reg)))(lstm1)
    gru1 = BatchNormalization()(gru1)
    
    # Conexão residual
    lstm_gru_concat = Concatenate()([lstm1, gru1])
    
    # Obter a dimensão da saída concatenada para garantir compatibilidade
    # Importante: Precisamos garantir que a dimensão do transformer seja igual à dimensão da entrada
    concat_dim = 256 + 128  # 64*2 (bidirecional GRU) + 128*2 (bidirecional LSTM)
    
    # Projeção para dimensão compatível com o transformer
    projection = Dense(256)(lstm_gru_concat)
    projection = BatchNormalization()(projection)
    
    # Camada de Multi-Head Attention para focar em partes importantes da sequência
    attention = MultiHeadAttention(num_heads=8, key_dim=32)(projection, projection)
    attention = Add()([attention, projection])  # Conexão residual
    attention = LayerNormalization(epsilon=1e-6)(attention)
    
    # Camada do transformers para capturar relações complexas
    # Agora usando a mesma dimensão que a projeção anterior
    transformer = TransformerEncoder(num_layers=2, d_model=256, num_heads=8, dff=512, rate=dropout_rate)(attention)
    
    # Pooling global para reduzir a dimensão da sequência
    global_avg = GlobalAveragePooling1D()(transformer)
    
    # Camadas densas para previsão final
    dense1 = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=l2_reg))(global_avg)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(dropout_rate)(dense1)
    
    dense2 = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=l2_reg))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(dropout_rate/2)(dense2)
    
    # Camada de saída
    output = Dense(1)(dense2)
    
    # Criar modelo
    model = Model(inputs=inputs, outputs=output)
    
    # Compilar modelo com otimizador adaptativo e função de perda robusta
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='huber',  # Função de perda robusta para lidar com outliers
        metrics=['mae', 'mse']
    )
    
    return model

# Modelo GRU para ensemble
def criar_modelo_gru(input_shape, dropout_rate=0.2, l2_reg=0.001):
    """
    Cria um modelo GRU para ensemble.
    """
    inputs = Input(shape=input_shape)
    
    # Camada convolucional
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # Camada GRU
    gru1 = Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=l2(l2_reg)))(conv1)
    gru1 = BatchNormalization()(gru1)
    
    # Camada de atenção
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(gru1, gru1)
    attention = Add()([attention, gru1])
    attention = LayerNormalization(epsilon=1e-6)(attention)
    
    # Segunda camada GRU
    gru2 = Bidirectional(GRU(50, kernel_regularizer=l2(l2_reg)))(attention)
    gru2 = BatchNormalization()(gru2)
    gru2 = Dropout(dropout_rate)(gru2)
    
    # Camadas densas
    dense1 = Dense(50, activation='relu', kernel_regularizer=l2(l2_reg))(gru2)
    dense1 = Dropout(dropout_rate)(dense1)
    
    # Camada de saída
    output = Dense(1)(dense1)
    
    # Criar modelo
    model = Model(inputs=inputs, outputs=output)
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='huber',
        metrics=['mae']
    )
    
    return model

# Modelo TCN melhorado
def criar_modelo_tcn_melhorado(input_shape):
    """
    Cria um modelo TCN (Temporal Convolutional Network) melhorado.
    """
    inputs = Input(shape=input_shape)
    
    # Camadas TCN com dilatação exponencial
    x = Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu', padding='causal')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', padding='causal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='causal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=64, kernel_size=3, dilation_rate=8, activation='relu', padding='causal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Camadas densas
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Camada de saída
    output = Dense(1)(x)
    
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='huber', metrics=['mae'])
    
    return model

# Modelo Ensemble
def criar_modelo_ensemble(modelos, input_shape):
    """
    Cria um modelo ensemble combinando vários modelos.
    """
    inputs = Input(shape=input_shape)
    
    # Obter previsões de cada modelo
    outputs = [modelo(inputs) for modelo in modelos]
    
    # Média das previsões
    if len(outputs) > 1:
        avg_output = tf.keras.layers.Average()(outputs)
    else:
        avg_output = outputs[0]
    
    # Criar modelo ensemble
    ensemble_model = Model(inputs=inputs, outputs=avg_output)
    
    # Compilar modelo
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='huber',
        metrics=['mae']
    )
    
    return ensemble_model

def carregar_ou_criar_modelo(x, modelo_type='lstm'):
    """
    Carrega o modelo existente ou cria um novo se não existir.
    
    Parâmetros:
    - x: Dados de entrada para determinar o formato
    - modelo_type: Tipo de modelo a ser criado ('lstm', 'gru', 'tcn', 'ensemble')
    
    Retorna:
    - Modelo carregado ou recém-criado
    """
    try:
        if os.path.exists(MODEL_PATH):
            logging.info(f"Carregando modelo existente ({modelo_type})...")
            if modelo_type == 'ensemble' and os.path.exists(MODEL_PATH_ENSEMBLE):
                return load_model(MODEL_PATH_ENSEMBLE, 
                                 custom_objects={
                                     'TransformerEncoderLayer': TransformerEncoderLayer,
                                     'TransformerEncoder': TransformerEncoder,
                                     'AttentionLayer': AttentionLayer
                                 })
            return load_model(MODEL_PATH, 
                             custom_objects={
                                 'TransformerEncoderLayer': TransformerEncoderLayer,
                                 'TransformerEncoder': TransformerEncoder,
                                 'AttentionLayer': AttentionLayer
                             })
    except Exception as e:
        logging.error(f"Erro ao carregar modelo: {e}")
    
    logging.info(f"Criando novo modelo ({modelo_type})...")
    
    if modelo_type == 'lstm':
        return criar_modelo_melhorado((x.shape[1], x.shape[2]))
    elif modelo_type == 'gru':
        return criar_modelo_gru((x.shape[1], x.shape[2]))
    elif modelo_type == 'tcn':
        return criar_modelo_tcn_melhorado((x.shape[1], x.shape[2]))
    elif modelo_type == 'ensemble':
        # Criar modelos individuais para ensemble
        lstm_model = criar_modelo_melhorado((x.shape[1], x.shape[2]))
        gru_model = criar_modelo_gru((x.shape[1], x.shape[2]))
        tcn_model = criar_modelo_tcn_melhorado((x.shape[1], x.shape[2]))
        
        # Criar modelo ensemble
        return criar_modelo_ensemble([lstm_model, gru_model, tcn_model], (x.shape[1], x.shape[2]))
    else:
        return criar_modelo_melhorado((x.shape[1], x.shape[2]))

def criar_sequencias(data, seq_length, pred_steps=1):
    """
    Criar sequências para o modelo com suporte a previsão de múltiplos passos.
    
    Parâmetros:
    - data: Dados normalizados
    - seq_length: Comprimento da sequência de entrada
    - pred_steps: Número de passos futuros para prever (padrão: 1)
    
    Retorna:
    - x: Sequências de entrada
    - y: Valores alvo
    """
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 3]) 
    return np.array(x), np.array(y)

def detectar_outliers(data, contamination=0.05):
    """
    Detecta e marca outliers nos dados usando Isolation Forest.
    
    Parâmetros:
    - data: DataFrame com os dados
    - contamination: Proporção esperada de outliers
    
    Retorna:
    - DataFrame com coluna adicional indicando outliers
    """
    # Selecionar colunas numéricas para detecção de outliers
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Criar e treinar o modelo Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(data[numeric_cols])
    
    # Adicionar coluna de outliers (-1 para outliers, 1 para inliers)
    data['outlier'] = outliers
    
    # Converter para flag binária (1 para outliers, 0 para dados normais)
    data['is_outlier'] = data['outlier'].apply(lambda x: 1 if x == -1 else 0)
    
    # Remover coluna temporária
    data.drop('outlier', axis=1, inplace=True)
    
    logging.info(f"Detectados {data['is_outlier'].sum()} outliers em {len(data)} registros")
    
    return data

def adicionar_features_avancadas(data):
    """
    Adiciona features avançadas ao DataFrame de dados.
    
    Parâmetros:
    - data: DataFrame com dados OHLCV
    
    Retorna:
    - DataFrame com features adicionais
    """
    # Copiar o DataFrame para evitar modificações no original
    df = data.copy()
    
    # Indicadores de tendência
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Indicadores de volatilidade
    df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
    df['volatility'] = df['close'].rolling(window=10).std()
    
    # Indicadores de momentum
    df['rsi'] = calc_rsi(df['close'], window=14)
    df['rsi_7'] = calc_rsi(df['close'], window=7)
    df['rsi_21'] = calc_rsi(df['close'], window=21)
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Stochastic Oscillator
    df['lowest_14'] = df['low'].rolling(window=14).min()
    df['highest_14'] = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - df['lowest_14']) / (df['highest_14'] - df['lowest_14']))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Características de volume
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    
    # Características de preço
    df['price_rate_of_change'] = df['close'].pct_change(periods=1)
    df['price_momentum'] = df['close'].diff(periods=5)
    
    # Características temporais (dia da semana, hora do dia, etc.)
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        df['hour_of_day'] = df.index.hour
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Tentar adicionar decomposição sazonal se houver dados suficientes
    try:
        if len(df) > 30:  # Precisa de dados suficientes
            decomposition = seasonal_decompose(df['close'], model='additive', period=24)
            df['seasonal'] = decomposition.seasonal
            df['trend'] = decomposition.trend
            df['residual'] = decomposition.resid
    except Exception as e:
        logging.warning(f"Não foi possível adicionar decomposição sazonal: {e}")
    
    # Remover NaN criados pelos cálculos de janela
    df.dropna(inplace=True)
    
    return df

def calc_rsi(prices, window=14):
    """
    Calcula o Índice de Força Relativa (RSI).
    
    Parâmetros:
    - prices: Série de preços
    - window: Tamanho da janela para cálculo
    
    Retorna:
    - Série com valores de RSI
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Evitar divisão por zero
    avg_loss = avg_loss.replace(0, 0.00001)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def obter_dados_historicos(cripto, timeframe="15m", limit=1000):
    """
    Obter dados históricos da Binance com mais opções de timeframe.
    
    Parâmetros:
    - cripto: Par de criptomoedas (ex: 'BTC/USDT')
    - timeframe: Intervalo de tempo ('1m', '5m', '15m', '1h', '4h', '1d')
    - limit: Número máximo de candles a serem obtidos
    
    Retorna:
    - DataFrame com dados históricos e indicadores
    """
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        })
        
        # Obter dados históricos
        data = exchange.fetch_ohlcv(
            symbol=cripto,
            timeframe=timeframe,
            limit=limit
        )
        
        # Converter para DataFrame
        data = pd.DataFrame(
            data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        
        # Adicionar features avançadas
        data = adicionar_features_avancadas(data)
        
        # Detectar outliers
        data = detectar_outliers(data)
        
        return data
    except Exception as e:
        logging.error(f"Erro ao obter dados históricos: {e}")
        return None

def processar_par(par, logger, seq_length=60, batch_size=64, epochs=150, timeframe="15m"):
    """
    Processa um par de criptomoedas: obtém dados, treina modelo e faz previsões.
    
    Parâmetros:
    - par: Par de criptomoedas (ex: 'BTC/USDT')
    - logger: Logger para registrar informações
    - seq_length: Comprimento da sequência para o modelo
    - batch_size: Tamanho do lote para treinamento
    - epochs: Número máximo de épocas para treinamento
    - timeframe: Intervalo de tempo dos dados
    """
    try:
        # Obtém os dados históricos
        data = obter_dados_historicos(par, timeframe=timeframe, limit=1500)
        if data is None or len(data) < seq_length + 1:
            logger.error(f"Não foi possível obter dados suficientes para {par}")
            return
        
        # Remover a coluna de outliers para o treinamento, mas manter a informação
        is_outlier = data['is_outlier']
        data = data.drop('is_outlier', axis=1)
        
        # Features a serem usadas (todas as colunas numéricas)
        features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        data = data[features]
        
        # Escala os dados usando RobustScaler para lidar melhor com outliers
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Escalonador específico para o preço de fechamento (para desnormalização)
        close_idx = features.index('close')
        close_scaler = MinMaxScaler()
        close_scaler.fit(data[['close']])
        
        # Cria sequências
        X, y = criar_sequencias(scaled_data, seq_length)
        logger.info(f"Sequências criadas: {X.shape[0]} amostras, input_shape={X.shape[1:]}")
        
        # Divide em treino, validação e teste (70-15-15)
        train_split = int(len(X) * 0.7)
        val_split = int(len(X) * 0.85)
        
        X_train, X_val, X_test = X[:train_split], X[train_split:val_split], X[val_split:]
        y_train, y_val, y_test = y[:train_split], y[train_split:val_split], y[val_split:]
        
        logger.info(f"Treino: {X_train.shape[0]}, Validação: {X_val.shape[0]}, Teste: {X_test.shape[0]}")
        
        # Carrega ou cria o modelo (usando apenas LSTM para simplificar)
        modelo = carregar_ou_criar_modelo(X_train, modelo_type='lstm')
        
        # Diretório para checkpoints
        os.makedirs("rede_neural/checkpoints", exist_ok=True)
        
        # Callbacks para treinamento
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=30, 
            restore_best_weights=True,
            verbose=1
        )
        
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=10, 
            min_lr=1e-6,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            filepath=f"rede_neural/checkpoints/modelo_{par.replace('/', '_')}_{timeframe}.keras",
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        
        # Treina o modelo
        history = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_scheduler, checkpoint],
            verbose=1
        )
        
        # Salvar o modelo
        modelo.save(MODEL_PATH)
        logger.info(f"Modelo salvo em {MODEL_PATH}")
        
        # Faz previsões no conjunto de teste
        predictions = modelo.predict(X_test)
        
        # Desnormalizar previsões e valores reais
        predicted_prices = close_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        real_prices = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calcula métricas
        mae = mean_absolute_error(real_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
        mape = np.mean(np.abs((real_prices - predicted_prices) / real_prices)) * 100
        r2 = r2_score(real_prices, predicted_prices)
        
        logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        
        # Gráfico de previsões
        plt.figure(figsize=(14, 7))
        plt.plot(data.index[-len(predicted_prices):], real_prices, label='Real', color='blue')
        plt.plot(data.index[-len(predicted_prices):], predicted_prices, label='Previsão', color='orange')
        plt.title(f'Previsão de Preço do {par} ({timeframe})')
        plt.xlabel('Tempo')
        plt.ylabel('Preço (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Criar diretório para imagens se não existir
        os.makedirs("rede_neural/img", exist_ok=True)
        plt.savefig(f"rede_neural/img/previsao_{par.replace('/', '_')}_{timeframe}.png")
        plt.close()
        
        # Gráfico de histórico de treinamento
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title('Função de Perda')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Treino')
        plt.plot(history.history['val_mae'], label='Validação')
        plt.title('Erro Médio Absoluto')
        plt.xlabel('Época')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"rede_neural/img/historico_{par.replace('/', '_')}_{timeframe}.png")
        plt.close()
        
        # Salvar métricas em arquivo JSON
        metricas = {
            'par': par,
            'timeframe': timeframe,
            'metricas': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2)
            },
            'parametros': {
                'seq_length': seq_length,
                'batch_size': batch_size,
                'epochs_total': epochs,
                'epochs_treinadas': len(history.history['loss']),
                'learning_rate_final': float(history.history.get('lr', [0.0001])[-1])
            },
            'data_treinamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        os.makedirs("rede_neural/metricas", exist_ok=True)
        with open(f"rede_neural/metricas/{par.replace('/', '_')}_{timeframe}.json", 'w') as f:
            json.dump(metricas, f, indent=4)
        
        logger.info(f"Previsão do par {par} concluída com sucesso")
        return metricas
        
    except Exception as e:
        logger.error(f"Erro durante o processo para {par}: {e}")
        logger.error(traceback.format_exc())
        return None

def treinar_e_prever(timeframes=None):
    """
    Função para treinar modelo e fazer previsões com múltiplos timeframes.
    
    Parâmetros:
    - timeframes: Lista de timeframes para processar (ex: ['15m', '1h', '4h'])
    """
    try:
        if timeframes is None:
            timeframes = ['15m', '1h']  # Timeframes padrão
            
        pares = ['BNB/USDT', 'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
        
        # Configura o logger
        logger = logging.getLogger()
        
        resultados_gerais = {}
        
        # Processar cada par em cada timeframe
        for timeframe in timeframes:
            resultados_gerais[timeframe] = {}
            
            # Usando ThreadPoolExecutor para paralelizar o processamento
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Dispara a execução para cada par
                futures = {executor.submit(processar_par, par, logger, timeframe=timeframe): par for par in pares}
                
                # Aguarda a conclusão de todas as tarefas
                for future in concurrent.futures.as_completed(futures):
                    par = futures[future]
                    try:
                        resultado = future.result()
                        if resultado:
                            resultados_gerais[timeframe][par] = resultado
                    except Exception as e:
                        logger.error(f"Erro ao processar {par} em {timeframe}: {e}")
        
        # Salvar resultados gerais
        with open("rede_neural/resultados_gerais.json", 'w') as f:
            json.dump(resultados_gerais, f, indent=4)
                
        logging.info("Processamento de previsões concluído para todos os pares e timeframes.")
        
        return resultados_gerais

    except Exception as e:
        logging.error(f"Erro durante o processo global: {e}")
        logging.error(traceback.format_exc())
        return None

def obter_pares_usdt(top_volume=20):
    """
    Obtém pares USDT disponíveis na Binance, ordenados por volume.
    
    Parâmetros:
    - top_volume: Número de pares com maior volume a retornar
    
    Retorna:
    - Lista de pares USDT com maior volume
    """
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        })
        
        # Carregar mercados
        mercados = exchange.load_markets()
        
        # Filtrar pares USDT
        pares_usdt = [par for par in mercados.keys() if par.endswith('/USDT') and ':' not in par]
        
        # Se top_volume for None ou 0, retornar todos os pares
        if not top_volume:
            return pares_usdt
        
        # Obter informações de ticker para todos os pares USDT
        tickers = {}
        for par in pares_usdt:
            try:
                ticker = exchange.fetch_ticker(par)
                tickers[par] = ticker
            except Exception as e:
                logging.warning(f"Erro ao obter ticker para {par}: {e}")
        
        # Ordenar por volume (24h)
        pares_ordenados = sorted(tickers.keys(), key=lambda x: tickers[x]['quoteVolume'] if 'quoteVolume' in tickers[x] else 0, reverse=True)
        
        # Retornar os top N pares
        return pares_ordenados[:top_volume]
    
    except Exception as e:
        logging.error(f"Erro ao obter pares USDT: {e}")
        return []

def avaliar_modelo(par_moeda='BTC/USDT', seq_length=60, timeframe='15m'):
    """
    Função para avaliar o desempenho do modelo de previsão.
    
    Parâmetros:
    - par_moeda: Par de moeda para análise
    - seq_length: Comprimento da sequência para previsão
    - timeframe: Intervalo de tempo dos dados
    
    Retorna:
    - Dicionário com métricas de desempenho
    """
    try:
        # Carregar modelo existente
        try:
            modelo = load_model(MODEL_PATH, 
                               custom_objects={
                                   'TransformerEncoderLayer': TransformerEncoderLayer,
                                   'TransformerEncoder': TransformerEncoder,
                                   'AttentionLayer': AttentionLayer
                               })
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")
            return None

        # Obter dados históricos
        data = obter_dados_historicos(par_moeda, timeframe=timeframe, limit=1500)
        if data is None:
            return None
            
        # Remover a coluna de outliers para avaliação
        if 'is_outlier' in data.columns:
            data = data.drop('is_outlier', axis=1)
        
        # Features a serem usadas
        features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        data = data[features]
        
        # Escalonamento dos dados
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Escalonador específico para o preço de fechamento
        close_idx = features.index('close')
        close_scaler = MinMaxScaler()
        close_scaler.fit(data[['close']])
        
        # Criar sequências
        x, y = criar_sequencias(scaled_data, seq_length)
        
        # Divisão dos dados (usar últimos 20% para teste)
        split = int(len(x) * 0.8)
        x_test = x[split:]
        y_test = y[split:]
        
        # Previsões
        predictions = modelo.predict(x_test)
        
        # Inverter escalonamento
        predicted_prices = close_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        real_prices = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calcular métricas de desempenho
        mse = mean_squared_error(real_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(real_prices, predicted_prices)
        mape = np.mean(np.abs((real_prices - predicted_prices) / real_prices)) * 100
        r2 = r2_score(real_prices, predicted_prices)
        
        # Plotar resultados
        plt.figure(figsize=(14, 7))
        plt.plot(data.index[-len(predicted_prices):], real_prices, label='Preço Real', color='blue')
        plt.plot(data.index[-len(predicted_prices):], predicted_prices, label='Previsão', color='orange')
        plt.title(f'Avaliação de Previsão - {par_moeda} ({timeframe})')
        plt.xlabel('Tempo')
        plt.ylabel('Preço')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Criar diretório para imagens se não existir
        os.makedirs("rede_neural/img", exist_ok=True)
        plt.savefig(f"rede_neural/img/avaliacao_{par_moeda.replace('/', '_')}_{timeframe}.png")
        plt.close()
        
        # Resultados detalhados
        resultados = {
            'par_moeda': par_moeda,
            'timeframe': timeframe,
            'metricas': {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape),
                'R2': float(r2)
            },
            'interpretacao': {}
        }
        
        # Interpretar resultados
        if r2 > 0.8:
            resultados['interpretacao']['r2'] = 'Excelente ajuste do modelo'
        elif r2 > 0.6:
            resultados['interpretacao']['r2'] = 'Bom ajuste do modelo'
        elif r2 > 0.4:
            resultados['interpretacao']['r2'] = 'Ajuste moderado'
        else:
            resultados['interpretacao']['r2'] = 'Ajuste fraco'
        
        if mape < 5:
            resultados['interpretacao']['mape'] = 'Precisão muito alta'
        elif mape < 10:
            resultados['interpretacao']['mape'] = 'Boa precisão'
        elif mape < 20:
            resultados['interpretacao']['mape'] = 'Precisão aceitável'
        else:
            resultados['interpretacao']['mape'] = 'Baixa precisão'
        
        logging.info(f"Avaliação do modelo para {par_moeda} concluída")
        
        return resultados
    
    except Exception as e:
        logging.error(f"Erro durante a avaliação do modelo: {e}")
        logging.error(traceback.format_exc())
        return None

def main_avaliacao(timeframes=None):
    """
    Função principal para avaliação de múltiplos pares em diferentes timeframes.
    
    Parâmetros:
    - timeframes: Lista de timeframes para avaliar
    """
    if timeframes is None:
        timeframes = ['15m', '1h', '4h']
        
    # Obter top 5 pares por volume
    pares = obter_pares_usdt(top_volume=5)
    if not pares:
        pares = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
    
    resultados_gerais = {}
    
    for timeframe in timeframes:
        resultados_gerais[timeframe] = {}
        
        for par in pares:
            logging.info(f"Avaliando {par} em {timeframe}...")
            resultado = avaliar_modelo(par, timeframe=timeframe)
            if resultado:
                resultados_gerais[timeframe][par] = resultado
    
    # Criar diretório para resultados se não existir
    os.makedirs("rede_neural", exist_ok=True)
    
    with open('rede_neural/resultados_avaliacao_completa.json', 'w') as f:
        json.dump(resultados_gerais, f, indent=4)
    
    return resultados_gerais

def prever_proximo_preco(par_moeda='BTC/USDT', timeframe='15m', seq_length=60):
    """
    Prevê o próximo preço para um par de criptomoedas.
    
    Parâmetros:
    - par_moeda: Par de criptomoedas
    - timeframe: Intervalo de tempo
    - seq_length: Comprimento da sequência
    
    Retorna:
    - Dicionário com previsão e informações relacionadas
    """
    try:
        # Carregar modelo
        try:
            modelo = load_model(MODEL_PATH, 
                               custom_objects={
                                   'TransformerEncoderLayer': TransformerEncoderLayer,
                                   'TransformerEncoder': TransformerEncoder,
                                   'AttentionLayer': AttentionLayer
                               })
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")
            return None
        
        # Obter dados históricos
        data = obter_dados_historicos(par_moeda, timeframe=timeframe, limit=seq_length + 50)
        if data is None:
            return None
            
        # Último preço conhecido
        ultimo_preco = data['close'].iloc[-1]
        ultima_data = data.index[-1]
        
        # Remover a coluna de outliers
        if 'is_outlier' in data.columns:
            data = data.drop('is_outlier', axis=1)
        
        # Features a serem usadas
        features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        data = data[features]
        
        # Escalonamento dos dados
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Escalonador específico para o preço de fechamento
        close_idx = features.index('close')
        close_scaler = MinMaxScaler()
        close_scaler.fit(data[['close']])
        
        # Criar sequência para previsão (últimos seq_length pontos)
        x_pred = np.array([scaled_data[-seq_length:]])
        
        # Fazer previsão
        prediction = modelo.predict(x_pred)
        
        # Desnormalizar previsão
        predicted_price = close_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]
        
        # Calcular variação percentual
        variacao_pct = ((predicted_price - ultimo_preco) / ultimo_preco) * 100
        
        # Determinar próxima data com base no timeframe
        if timeframe == '1m':
            proxima_data = ultima_data + timedelta(minutes=1)
        elif timeframe == '5m':
            proxima_data = ultima_data + timedelta(minutes=5)
        elif timeframe == '15m':
            proxima_data = ultima_data + timedelta(minutes=15)
        elif timeframe == '30m':
            proxima_data = ultima_data + timedelta(minutes=30)
        elif timeframe == '1h':
            proxima_data = ultima_data + timedelta(hours=1)
        elif timeframe == '4h':
            proxima_data = ultima_data + timedelta(hours=4)
        elif timeframe == '1d':
            proxima_data = ultima_data + timedelta(days=1)
        else:
            proxima_data = ultima_data + timedelta(minutes=15)  # Padrão
        
        # Resultado
        resultado = {
            'par': par_moeda,
            'timeframe': timeframe,
            'ultimo_preco': float(ultimo_preco),
            'ultima_data': ultima_data.strftime('%Y-%m-%d %H:%M:%S'),
            'preco_previsto': float(predicted_price),
            'proxima_data': proxima_data.strftime('%Y-%m-%d %H:%M:%S'),
            'variacao_percentual': float(variacao_pct),
            'tendencia': 'alta' if variacao_pct > 0 else 'baixa',
            'data_previsao': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logging.info(f"Previsão para {par_moeda} ({timeframe}): {predicted_price:.4f} ({variacao_pct:+.2f}%)")
        
        return resultado
    
    except Exception as e:
        logging.error(f"Erro ao prever próximo preço: {e}")
        logging.error(traceback.format_exc())
        return None

def main():
    """
    Função principal que executa o ciclo de treinamento, avaliação e previsão.
    """
    try:
        # Criar diretórios necessários
        os.makedirs("rede_neural", exist_ok=True)
        os.makedirs("rede_neural/img", exist_ok=True)
        os.makedirs("rede_neural/checkpoints", exist_ok=True)
        os.makedirs("rede_neural/metricas", exist_ok=True)
        
        # Configurar logging para arquivo
        file_handler = logging.FileHandler("rede_neural/treinamento.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        while True:
            logging.info("=" * 50)
            logging.info("Iniciando ciclo de treinamento e previsão...")
            
            # Treinar e avaliar modelo em diferentes timeframes
            timeframes = ['15m', '1h']
            resultados = treinar_e_prever(timeframes)
            
            # Avaliar modelo em mais timeframes
            timeframes_avaliacao = ['15m', '1h', '4h']
            resultados_avaliacao = main_avaliacao(timeframes_avaliacao)
            
            # Fazer previsões para os próximos preços
            pares_principais = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
            previsoes = {}
            
            for par in pares_principais:
                for tf in timeframes_avaliacao:
                    previsao = prever_proximo_preco(par, timeframe=tf)
                    if previsao:
                        if par not in previsoes:
                            previsoes[par] = {}
                        previsoes[par][tf] = previsao
            
            # Salvar previsões
            with open('rede_neural/previsoes_atuais.json', 'w') as f:
                json.dump(previsoes, f, indent=4)
            
            logging.info("Ciclo completo. Aguardando para próxima iteração...")
            logging.info("=" * 50)
            
            # Aguardar para próxima iteração (30 minutos)
            time.sleep(30 * 60)
    
    except KeyboardInterrupt:
        logging.info("Programa interrompido pelo usuário")
    except Exception as e:
        logging.error(f"Erro crítico no programa principal: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
