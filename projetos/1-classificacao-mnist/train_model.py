# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ---------------------------------------------------------------------------
# Projeto 1 - Classificacao MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validacao (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saida (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validacao
#   6. Exibir a acuracia de validacao final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------


def load_and_preprocess_data():
    """Carrega e pre-processa o dataset MNIST."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizar para [0, 1] e adicionar dimensao do canal (grayscale)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]  # (60000, 28, 28) -> (60000, 28, 28, 1)
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Construi a CNN com 3 blocos convolucionais + Dropout + Softmax."""
    model = keras.Sequential([
        # --- Bloco 1 ---
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # --- Bloco 2 ---
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # --- Bloco 3 ---
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # --- Classificador ---
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    return model


def main():
    print("=" * 60)
    print("Projeto 1 - Classificacao MNIST")
    print("=" * 60)

    # 1. Carregar dados
    print("\n[1/4] Carregando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print("  Treino: {} amostras".format(x_train.shape[0]))
    print("  Teste:  {} amostras".format(x_test.shape[0]))

    # 2. Split treino/validacao (90%/10%)
    print("\n[2/4] Separando treino/validacao (90%/10%)...")
    split_idx = int(0.9 * len(x_train))
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train_split = x_train[:split_idx]
    y_train_split = y_train[:split_idx]
    print("  Treino:     {} amostras".format(len(x_train_split)))
    print("  Validacao:  {} amostras".format(len(x_val)))

    # 3. Construir modelo
    print("\n[3/4] Construindo arquitetura CNN...")
    model = build_model()
    model.summary()

    # 4. Compilar e treinar
    print("\n[4/4] Treinando o modelo...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        x_train_split, y_train_split,
        epochs=15,
        batch_size=128,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # 5. Avaliar no conjunto de validacao
    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print("\nAcuracia de validacao: {:.2f}%".format(val_acc * 100))

    # 6. Salvar modelo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    model.save(model_path)
    print("\nModelo salvo em: {}".format(model_path))
    print("Tamanho: {:.1f} KB".format(os.path.getsize(model_path) / 1024))
    print("=" * 60)


if __name__ == "__main__":
    main()
