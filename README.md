# Gym
1 лаб работа

import gym

# Определяем функцию стратегии
def policy(state):
    action = None
    if state == 0:
        action = 1  # вниз
    elif state == 1:
        action = 1  # вниз
    elif state == 2:
        action = 3  # вправо
    elif state == 4:
        action = 1  # вниз
    elif state == 6:
        action = 1  # вниз
    elif state == 8:
        action = 3  # вправо
    elif state == 9:
        action = 3  # вправо
    elif state == 10:
        action = 1  # вниз
    elif state == 13:
        action = 2  # влево
    elif state == 14:
        action = 3  # вправо
    else:
        action = 0  # в качестве запасного варианта двигаемся вверх
    return action

# Создаём среду
env = gym.make('FrozenLake-v1', is_slippery=False)  # Отключаем скольжение для упрощения задачи
state = env.reset()

# Запускаем симуляцию
for _ in range(100):  # Ограничиваем количество шагов, чтобы избежать бесконечного цикла
    action = policy(state)
    t, state, reward, done, info = env.step(action)
    env.render()  # Отображаем текущее состояние среды
    if done:
        if reward == 1:
            print("Поздравляем! Вы достигли цели и получили награду!")
        else:
            print("Увы, вы попали в лунку. Попробуйте ещё раз!")
        break

if not done:
    print("Достигнут лимит шагов без достижения цели.")

env.close()  # Закрываем среду

Лаб работа 2

Начнем с создания среды для игры CarRacing-v0, настройки монитора для записи видео и выполнения цикла симуляции со случайными действиями. 
После этого мы сможем сохранить видео файл с анимацией симуляции. 
Вот код для выполнения этой задачи:
import gym
import numpy as np
import cv2
from gym.wrappers.record_video import RecordVideo

# Создание среды для игры CarRacing-v2
env = gym.make("CarRacing-v0")

# Создание монитора для записи видео
video_dir = "/content/video"
env = RecordVideo(env, video_dir, force=True, video_callable=lambda episode_id: True)

# Установка случайного зерна для воспроизводимости
seed = 42
env.seed(seed)
np.random.seed(seed)

# Начало симуляции
observation = env.reset()
done = False

while not done:
    # Генерация случайного действия
    action = env.action_space.sample()

    # Выполнение действия в среде
    t, observation, reward, done, info = env.step(action)

# Завершение симуляции
env.close()

# Загрузка видео файла и отображение его в Colab
video_file = video_dir + "/openaigym.video.%s.video000000.mp4" % env.file_infix
video = open(video_file, "rb").read()
video_path = "/content/video.mp4"
with open(video_path, "wb") as f:
    f.write(video)

video_path

Лаб работа 3

Для того чтобы агент доходил до цели и не падал в яму, мы можем заполнить Q-функцию вручную такими значениями, чтобы агент всегда выбирал действие, которое приводит его к цели. Для этого мы можем установить положительные значения для действий, которые приводят к цели, и отрицательные значения для действий, которые приводят в яму. Остальные значения можно оставить нулевыми. 

Представим Q-функцию в виде словаря, где ключами являются пары (state, action), а значениями - оценки Q-функции для соответствующей пары. В данном случае у нас есть 16 состояний и 4 действия, поэтому у нас будет 64 пары.

Q = {
    (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0,
    (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0,
    (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0,
    (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0,
    (4, 0): 0, (4, 1): 0, (4, 2): 0, (4, 3): 0,
    (5, 0): 0, (5, 1): 0, (5, 2): 0, (5, 3): 0,
    (6, 0): 0, (6, 1): 0, (6, 2): 0, (6, 3): 0,
    (7, 0): 0, (7, 1): 0, (7, 2): 0, (7, 3): 0,
    (8, 0): 0, (8, 1): 0, (8, 2): 0, (8, 3): 0,
    (9, 0): 0, (9, 1): 0, (9, 2): 0, (9, 3): 0,
    (10, 0): 0, (10, 1): 0, (10, 2): 0, (10, 3): 0,
    (11, 0): 0, (11, 1): 0, (11, 2): 0, (11, 3): 0,
    (12, 0): 0, (12, 1): 0, (12, 2): 0, (12, 3): 0,
    (13, 0): 0, (13, 1): 0, (13, 2): 0, (13, 3): 0,
    (14, 0): 0, (14, 1): 0, (14, 2): 0, (14, 3): 0,
    (15, 0): 0, (15, 1): 0, (15, 2): 0, (15, 3): 0,
}
Теперь, чтобы агент использовал эту Q-функцию для выбора действий, мы можем изменить функцию policy, чтобы она выбирала действие с наибольшей оценкой Q-функции для данного состояния.

def policy(state):
    # Получаем действие с наибольшей оценкой Q-функции для данного состояния
    action = max(Q[state], key=Q[state].get)
    return action
Теперь проведем симуляцию с использованием этой Q-функции и посмотрим на результат.

Лаб работа 4

Начнем с реализации табличного Q-Learning алгоритма для задачи Frozen Lake. 

Сначала, нам понадобится импортировать необходимые библиотеки:

import gym
import numpy as np
Затем, мы создадим среду Frozen Lake:

env = gym.make('FrozenLake-v0')
Теперь, нам нужно создать таблицу Q-функции и инициализировать ее случайными значениями:

num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.random.rand(num_states, num_actions)
Теперь, мы определим параметры для обучения:

num_episodes = 10000  # Количество эпизодов обучения
learning_rate = 0.1  # Скорость обучения
discount_factor = 0.99  # Фактор дисконтирования
epsilon = 0.1  # Epsilon для стратегии исследования и использования
Далее, мы начнем обучение агента:

for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Выбор действия с использованием epsilon-greedy стратегии
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Случайное действие
        else:
            action = np.argmax(Q[state, :])  # Действие с наибольшей оценкой Q-функции
        
        # Выполнение выбранного действия
        next_state, reward, done, _ = env.step(action)
        
        # Обновление Q-функции
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]))
        
        state = next_state
После завершения обучения, мы можем использовать обновленную Q-функцию для принятия решений:

state = env.reset()
done = False

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
Это базовая реализация табличного Q-Learning алгоритма для задачи Frozen Lake. Вы можете использовать этот код, чтобы попробовать обучить агента на этой задаче. Удачи!

Лаб работа 5


Начнем с импорта необходимых библиотек:

import gym
import numpy as np
Теперь, создадим среду Frozen Lake:

env = gym.make('FrozenLake-v0')
Затем, определим параметры для обучения:

num_episodes = 10000  # Количество эпизодов обучения
learning_rate = 0.1  # Скорость обучения
discount_factor = 0.99  # Фактор дисконтирования
epsilon = 0.1  # Epsilon для стратегии исследования и использования
Теперь, создадим таблицу Q-функции и инициализируем ее нулями:

num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
Теперь мы можем начать обучение агента:

rewards = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Выбор действия с использованием epsilon-greedy стратегии
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Случайное действие
        else:
            action = np.argmax(Q[state, :])  # Действие с наибольшей оценкой Q-функции
        
        # Выполнение выбранного действия
        next_state, reward, done, _ = env.step(action)
        
        # Обновление Q-функции
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]))
        
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)
После завершения обучения, мы можем проверить результаты, используя обновленную Q-функцию:

state = env.reset()
done = False

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
Теперь, давайте подберем оптимальные значения гиперпараметров. Мы можем использовать цикл for для перебора различных комбинаций значений гиперпараметров и выбрать те, которые дают наилучшие результаты. Например:

best_reward = 0
best_params = {}

for lr in [0.1, 0.2, 0.3]:
    for df in [0.9, 0.95, 0.99]:
        for ep in [0.1, 0.2, 0.3]:
            Q = np.zeros((num_states, num_actions))
            rewards = []
            
            for episode in range(num_episodes):
                # ...
                # Обучение агента
                
            average_reward = np.mean(rewards)
            
            if average_reward > best_reward:
                best_reward = average_reward
                best_params = {'learning_rate': lr, 'discount_factor': df, 'epsilon': ep}
После завершения цикла, вы можете использовать best_params для обучения агента с оптимальными значениями гиперпараметров.

КР1


Начнем с импорта необходимых библиотек:

import gym
import numpy as np
Теперь, создадим среду Taxi:

env = gym.make('Taxi-v3')
Затем, определим параметры для обучения:

num_episodes = 10000  # Количество эпизодов обучения
learning_rate = 0.1  # Скорость обучения
discount_factor = 0.99  # Фактор дисконтирования
epsilon = 0.1  # Epsilon для стратегии исследования и использования
Теперь, создадим таблицу Q-функции и инициализируем ее случайными значениями:

num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
Теперь мы можем начать обучение агента:

rewards = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Выбор действия с использованием epsilon-greedy стратегии
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Случайное действие
        else:
            action = np.argmax(Q[state, :])  # Действие с наибольшей оценкой Q-функции
        
        # Выполнение выбранного действия
        next_state, reward, done, _ = env.step(action)
        
        # Обновление Q-функции
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]))
        
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)
После завершения обучения, мы можем проверить результаты, используя обновленную Q-функцию:

state = env.reset()
done = False

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
