# Adaline - Porta Lógica OU (One-Hot Encoding)

Este projeto implementa o algoritmo **Adaline (Adaptive Linear Neuron)** para aprender o comportamento da **porta lógica OU**, utilizando **representação bipolar** e **codificação One-Hot** para as classes.

O treinamento é realizado por meio da **Regra Delta**, e a convergência do erro é visualizada graficamente ao longo dos ciclos (épocas).

---

## 📌 Objetivo

Demonstrar o funcionamento do algoritmo Adaline aplicado a um problema clássico de classificação, explorando:

- Entradas bipolares (-1 e 1)
- Saídas em One-Hot Encoding
- Aprendizado supervisionado
- Convergência do erro quadrático

---

## 🧠 Descrição do Modelo

- **Entradas:** Dois neurônios (A e B)
- **Saídas:** Duas classes (Verdadeiro e Falso) em One-Hot
- **Função de ativação:** Degrau bipolar
- **Algoritmo de aprendizado:** Regra Delta (Adaline)
- **Critério de parada:** Erro total menor que uma tolerância definida

---

## 📂 Estrutura dos Dados

### Entradas (x)
Representação bipolar da porta lógica OU:

| A  | B  |
|----|----|
|  1 |  1 |
|  1 | -1 |
| -1 |  1 |
| -1 | -1 |

### Saídas (t)
Codificação One-Hot:

- Verdadeiro → `[1, -1]`
- Falso → `[-1, 1]`

---

## ⚙️ Parâmetros do Treinamento

- Taxa de aprendizado (`α`): `0.01`
- Limiar: `0.0`
- Tolerância do erro: `0.5`
- Pesos e bias inicializados aleatoriamente

---

## 📈 Visualização

Ao final do treinamento, é exibido um gráfico mostrando a **convergência do erro quadrático total** ao longo dos ciclos.

---

## 🛠️ Tecnologias Utilizadas

- Python 3
- NumPy
- Matplotlib

---

## ▶️ Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
