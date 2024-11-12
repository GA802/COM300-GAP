import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

# Crear el entorno de Boxing de Atari con visualización en modo humano y observaciones en escala de grises
env = gym.make("ALE/Boxing-v5", render_mode="human", obs_type="grayscale")

# Reducir las dimensiones de la observación y discretizar los valores para simplificar el estado
def discretizar(observation):
    resized = cv2.resize(observation, (10, 10))  # Redimensionar a 10x10 píxeles
    discretized = (resized / 255 * 10).astype(np.int32)  # Escalar valores y convertir a enteros
    return tuple(discretized.flatten())  # Convertir a tupla para ser utilizado como clave en la tabla Q

def train(episodes):
    # Crear una tabla Q inicializada con ceros para todas las combinaciones posibles de estado y acción
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    # Definir los parámetros de aprendizaje
    learning_rate = 0.1            # Velocidad de aprendizaje
    discount_factor = 0.95         # Factor de descuento para recompensas futuras
    epsilon = 1                    # Probabilidad inicial de tomar acciones exploratorias
    epsilon_decay_rate = 0.0001    # Tasa de decremento de epsilon
    rng = np.random.default_rng()  # Generador de números aleatorios

    # Almacenar recompensas obtenidas por episodio
    rewards_per_episode = np.zeros(episodes)

    # Bucle principal para iterar sobre cada episodio
    for i in range(episodes): 
        state = env.reset()[0]  # Reiniciar el entorno y obtener el estado inicial
        state = discretizar(state)  # Discretizar el estado inicial
        terminated = False  # Indicador de fin del episodio
        truncated = False  # Indicador de truncamiento del episodio
        total_reward = 0  # Recompensa acumulada en el episodio

        # Ejecutar acciones hasta que el episodio termine
        while not terminated and not truncated:
            # Seleccionar acción basada en la estrategia epsilon-greedy
            if rng.random() < epsilon:  # Exploración
                action = env.action_space.sample()
            else:  # Explotación
                action = np.argmax(q_table[state])

            # Ejecutar la acción y obtener la transición al nuevo estado
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = discretizar(new_state)  # Discretizar el nuevo estado

            # Actualizar el valor Q utilizando la fórmula de Q-learning
            old_value = q_table[state][action]  # Valor Q actual
            next_max = np.max(q_table[new_state])  # Mejor valor Q del siguiente estado
            new_value = old_value + learning_rate * (
                reward + discount_factor * next_max - old_value
            )
            q_table[state][action] = new_value  # Actualizar el valor Q
            
            # Actualizar el estado actual y la recompensa acumulada
            state = new_state
            total_reward += reward

            # Renderizar el entorno para observar el progreso
            env.render()

        # Registrar la recompensa total obtenida en el episodio
        rewards_per_episode[i] = total_reward

        # Reducir epsilon gradualmente para favorecer la explotación en episodios futuros
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Mostrar progreso cada 100 episodios
        if (i + 1) % 100 == 0: 
            print(f"Episodio: {i + 1} - Recompensa total: {total_reward}")

    # Cerrar el entorno después del entrenamiento
    env.close()

    # Mostrar una muestra de los valores en la tabla Q
    print("\nMuestra de la Q-table:")
    for idx, (state, actions) in enumerate(list(q_table.items())[:10]):
        print(f"Estado: {state[:10]}..., Acciones: {actions}")

    # Graficar el rendimiento acumulado a lo largo de los episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])  # Sumar recompensas de los últimos 100 episodios
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas en 100 episodios')
    plt.title('Rendimiento acumulado en el entorno de Boxing')
    plt.show()

# Ejecutar el entrenamiento si el script se ejecuta directamente
if name == "main":
    train(5)