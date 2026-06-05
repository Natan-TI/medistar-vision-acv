# Arquiteturas dos Modelos — Medistar Vision

## CNN Simples — Baseline

A primeira arquitetura foi criada como modelo de referência, inspirada em uma CNN básica com duas etapas convolucionais seguidas por uma MLP.

Estrutura:

- Conv2D com 16 filtros, kernel 3x3, ativação ReLU
- MaxPooling2D 2x2
- Conv2D com 32 filtros, kernel 3x3, ativação ReLU
- MaxPooling2D 2x2
- Flatten
- Dense com 64 neurônios, ativação ReLU
- Dropout 0.3
- Dense de saída com 6 neurônios e ativação Softmax

Resultado no teste:

- Loss: 0.3585
- Acurácia: 90.39%

## CNN Intermediária

A segunda arquitetura foi criada para aumentar a capacidade de extração de características, usando mais filtros convolucionais, três blocos convolucionais e data augmentation leve.

Estrutura:

- RandomFlip horizontal
- RandomRotation 0.03
- RandomZoom 0.05
- Conv2D com 32 filtros, kernel 3x3, ativação ReLU
- MaxPooling2D 2x2
- Conv2D com 64 filtros, kernel 3x3, ativação ReLU
- MaxPooling2D 2x2
- Conv2D com 128 filtros, kernel 3x3, ativação ReLU
- MaxPooling2D 2x2
- Flatten
- Dense com 128 neurônios, ativação ReLU
- Dropout 0.3
- Dense de saída com 6 neurônios e ativação Softmax

Resultado no teste:

- Loss: 0.3136
- Acurácia: 90.05%

## Comparação

A CNN Simples foi escolhida como melhor modelo por apresentar a maior acurácia no conjunto de teste. A CNN Intermediária apresentou loss menor, indicando previsões médias mais estáveis, mas teve acurácia ligeiramente inferior.

Modelo selecionado:

- `outputs/models/best_model.keras`