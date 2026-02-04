import numpy as np
import random as rd
import matplotlib.pyplot as plt

# 1. Configuração dos Dados (Porta Lógica OU com One-Hot)
# Entradas (x): [A, B]
x = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
# Alvos (t) usando One-Hot: [Verdadeiro, Falso]
t = np.array([[1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, -1.0, 1.0]])

amostras, entradas = np.shape(x)
numclasses, targets = np.shape(t)

# 2. Inicialização de Parâmetros
limiar = 0.0
alfa = 0.01
tolerancia = 0.5
erro = 10
ciclo = 0

# Pesos Aleatórios
v = np.random.uniform(-0.5, 0.5, (entradas, numclasses))
v0 = np.random.uniform(-0.5, 0.5, numclasses)

# Vetores para o gráfico
vetor1 = []
vetor2 = []

# 3. Laço de Treinamento (Ciclos/Épocas)
while erro > tolerancia:
    ciclo += 1
    erro = 0
    
    for i in range(amostras):
        xaux = x[i,:]
        yin = np.zeros(numclasses)
        y = np.zeros(numclasses)
        
        # Propagação (Cálculo da Saída)
        for m in range(numclasses):
            soma = 0
            for n in range(entradas):
                soma += xaux[n] * v[n][m]
            yin[m] = soma + v0[m]
            
            # Função de Ativação Bipolar
            y[m] = 1.0 if yin[m] >= limiar else -1.0
            
        # Cálculo do Erro (Quadrático)
        for j in range(numclasses):
            erro += 0.5 * ((t[j][i] - y[j])**2)
            
        # 4. Atualização dos Pesos (Regra Delta)
        # Atualiza pesos das conexões (v)
        for m in range(entradas):
            for n in range(numclasses):
                v[m][n] = v[m][n] + alfa * (t[n][i] - y[n]) * xaux[m]
        
        # Atualiza Bias (v0)
        for j in range(numclasses):
            v0[j] = v0[j] + alfa * (t[j][i] - y[j])

    print(f"Ciclo: {ciclo} | Erro: {erro:.4f}")
    vetor1.append(ciclo)
    vetor2.append(erro)

# 5. Visualização dos Resultados
plt.plot(vetor1, vetor2)
plt.title("Convergência do Erro - Adaline")
plt.xlabel("Ciclos")
plt.ylabel("Erro Quadrático")
plt.show()

print("Pesos Finais (v):\n", v)
print("Bias Finais (v0):\n", v0)
