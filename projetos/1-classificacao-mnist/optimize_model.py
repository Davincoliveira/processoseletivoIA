# -*- coding: utf-8 -*-
import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 - Otimizacao do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma tecnica de otimizacao (Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(script_dir, "model.h5")
    tflite_path = os.path.join(script_dir, "model.tflite")

    print("Carregando modelo Keras: {}".format(h5_path))
    model = tf.keras.models.load_model(h5_path)

    print("Convertendo para TensorFlow Lite com Dynamic Range Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    h5_size = os.path.getsize(h5_path) / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024
    reduction = (1 - tflite_size / h5_size) * 100

    print("\nTamanho original (model.h5):  {:.1f} KB".format(h5_size))
    print("Tamanho otimizado (model.tflite): {:.1f} KB".format(tflite_size))
    print("Reducao: {:.1f}%".format(reduction))
    print("\nModelo salvo em: {}".format(tflite_path))


if __name__ == "__main__":
    main()
