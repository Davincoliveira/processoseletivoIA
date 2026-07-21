# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Davi Neemias Cruz de Oliveira

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em `train_model.py` utiliza a arquitetura **Sequential** com **3 blocos convolucionais**, cada um composto por `Conv2D` + `BatchNormalization` + `MaxPooling2D`:

| Bloco | Filtros | Kernel | Função de Ativação |
| 1 | 32 | 3x3 | ReLU |
| 2 | 64 | 3x3 | ReLU |
| 3 | 128 | 3x3 | ReLU |

Após os blocos convolucionais, a rede possui:
- Camada `Flatten` para vetorizar as features (saída: 1.152 features)
- `Dropout(0.5)` para regularização após o flatten
- Camada densa `Dense(128, activation='relu')`
- `Dropout(0.3)` para regularização adicional
- Camada de saída `Dense(10, activation='softmax')` para classificação nas 10 classes

**Total de parâmetros:** 242.442 (241.994 treináveis).

**Estratégia de validação:** Split manual 90%/10% (54.000 treino / 6.000 validação).

**Early Stopping:** monitorando `val_loss` com `patience=3`, restaurando os pesos da melhor epoch (epoch 6).

### 2️⃣ Bibliotecas Utilizadas

 Python  3.9.23 
 TensorFlow  2.20.0 
 Keras  3.10.0 
 NumPy  2.0.2 

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a **Dynamic Range Quantization**, implementada via `tf.lite.Optimize.DEFAULT`. Esta técnica converte os pesos do modelo de ponto flutuante (float32) para inteiros de 8 bits (int8), reduzindo significativamente o tamanho do arquivo e acelerando a inferência em dispositivos Edge, com perda mínima de acurácia.

A conversão foi realizada com:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### 4️⃣ Resultados Obtidos

 Métrica - Valor 
 **Acurácia de validação** -- **99.33%** 
 Tamanho model.h5 (original) -- 2.910,3 KB (2,84 MB) 
 Tamanho model.tflite (otimizado) -- 249,0 KB (0,24 MB) 
 **Redução de tamanho** -- **91,4%** 

### 5️⃣ Comentários Adicionais

- **Hiperparâmetro escolhido:** `patience=3` no EarlyStopping. Justificativa: um valor menor (1-2) poderia interromper o treinamento antes do modelo convergir adequadamente, enquanto um valor maior (5+) aumentaria tempo de treino sem ganho significativo. O valor 3 equilibra eficiência e qualidade.
- O treinamento parou na epoch 9 e restaurou os pesos da epoch 6, demonstrando que o modelo atingiu seu melhor desempenho cedo e o EarlyStopping evitou overfitting.
- A Dynamic Range Quantization reduziu o modelo em 91,4%, tornando-o viável para execução em dispositivos com recursos limitados.
- O dataset MNIST é relativamente simples, permitindo alta acurácia com uma CNN de 3 blocos. Para datasets mais complexos, arquiteturas maiores ou data augmentation seriam necessários.

### 6️⃣ Exemplo de Inferência

Saída do terminal ao executar `python3.9 run_inference.py`:

```
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

**Análise:** As 5 amostras selecionadas automaticamente (primeiras imagens do conjunto de teste) foram todas classificadas corretamente, com predições idênticas aos rótulos reais. Isso é consistente com a acurácia de 99,33% obtida na validação — o modelo demonstra forte capacidade de generalização mesmo após a redução agressiva de 91,4% no tamanho via quantização.
