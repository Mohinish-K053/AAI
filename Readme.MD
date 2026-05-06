# Advanced AI Lab Manual — MCA Semester II
### D Y Patil International University, Akurdi, Pune
### School of Computer Science Engineering and Applications (SCSEA)
### Course Code: AID1101

---

## Index

| Sr. No. | Practical | Date |
|---------|-----------|------|
| 1 | Understanding and reading all types of data and data transformation | 2/02/26 |
| 2 | AI based Search algorithms using NumPy & TensorFlow | 16/02/26 |
| 3 | AI for Problem Solving (Tic Tac Toe & TSP) | 2/03/26 |
| 4 | Implementation of AI Generative Model (GAN) | 16/03/26 |
| 5 | Implementation of Deep Generative AI Model (RNN) | 23/03/26 |
| 6 | NLP based Preprocessing of Text Data | 30/03/26 |
| 7 | Twitter data / Neuro marketing Sentiment Analysis | 6/04/26 |
| 8 | Explainable AI: Case Study (SHAP & LIME) | 13/04/26 |

---

## Practical 01 — Understanding and Reading Different Types of Data and Data Transformation

**Student Name:** Mohinish Kashyap | **PRN:** 20250804039
**Date of Experiment:** 2/02/26 | **Date of Submission:** 2/02/26
**Aim:** Understanding and reading different types of data and data transformation
**Objectives:** 1) Image data 2) Audio data
**Software/Tool:** Python / Jupyter / Anaconda / Colab

---

### Theory

**1) Image Data:**
An image is represented by its dimensions (height and width) based on the number of pixels. A pixel takes a specific shade, opacity or color represented as:
- **Grayscale** — integer between 0 to 255
- **RGB** — 3 integers between 0 to 255
- **RGBA** — RGB with an added alpha field for opacity

Five main types of image processing: Visualization, Recognition, Sharpening and restoration, Pattern recognition, Retrieval.

**2) Audio Data:**
Audio data represents analog sounds in digital form. Key characteristics:
- **Time period** — how long a sound lasts (seconds per cycle)
- **Amplitude** — sound intensity in decibels (dB)
- **Frequency** — measured in Hertz (Hz); human hearing range 20–20,000 Hz

Audio file formats: WAV, AIFF, FLAC, MP3

---

### Algorithm

**1A: Basic Image Processing**
```
1. Upload and read the input image.
2. Check if the image is loaded successfully.
3. Convert image from BGR to RGB format.
4. Resize the image to required dimensions.
5. Convert the image to grayscale.
6. Apply Gaussian Blur to reduce noise.
7. Perform edge detection using Canny algorithm.
8. Display all processed images (original, resized, grayscale, blurred, edges).
```

**1B: Frequency Domain Transform (DFT & DCT)**
```
1. Read the input image.
2. Convert the image to grayscale.
DFT Process:
3. Apply 2D Fast Fourier Transform (FFT).
4. Shift zero frequency to center.
5. Compute magnitude spectrum using log scaling.
6. Normalize and convert to colored representation.
DCT Process:
7. Apply 2D Discrete Cosine Transform (DCT).
8. Normalize DCT values for visualization.
9. Convert DCT output to colored image.
10. Display results (original, grayscale, DFT, DCT images).
```

**1C: Audio Feature Extraction using Librosa**
```
1. Import required libraries (librosa, numpy, matplotlib).
2. Load the audio file and obtain audio signal and sample rate.
3. Display basic audio information (sample rate and shape).
Waveform Analysis:
4. Plot the waveform of the audio signal over time.
Amplitude Envelope:
5. Apply pre-emphasis to the audio signal.
6. Compute the amplitude envelope using absolute values.
7. Plot amplitude envelope.
Loudness Calculation:
8. Convert amplitude values to decibels (dB).
9. Plot loudness of the audio signal.
Feature Extraction (MFCC):
10. Extract MFCC features from the audio signal.
11. Visualize MFCC using a spectrogram-like plot.
```

---

### Program Code

**1A: Basic Image Processing**

```python
from google.colab import files
import cv2
import matplotlib.pyplot as plt

uploaded = files.upload()

image = cv2.imread("Untitled.jpeg") #Change the file Name With yours which you will upload

if image is None:
    print("Error: Image not found. Check the file name.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_rgb, (400, 300))

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Resized")
    plt.imshow(resized)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Grayscale")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Blurred")
    plt.imshow(blur, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
```

> **Note:** Upload any `.jpeg` image when prompted. Rename the uploaded file to `Untitled.jpeg` or change the filename in `cv2.imread()`.

---

**1B: Frequency Domain Transform (DFT & DCT)**

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

image = cv2.imread('Untitled.jpeg') # Change the file name you Changed Above

if image is None:
    print("Error: Image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

dft_colored = cv2.applyColorMap(
    cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLORMAP_JET
)

def apply_dct(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

dct_transformed = apply_dct(gray)

dct_normalized = cv2.normalize(dct_transformed, None, 0, 255, cv2.NORM_MINMAX)

dct_colored = cv2.applyColorMap(
    dct_normalized.astype(np.uint8),
    cv2.COLORMAP_JET
)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("DFT (Colored Output)")
plt.imshow(dft_colored)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("DCT (Colored Output)")
plt.imshow(dct_colored)
plt.axis('off')

plt.tight_layout()
plt.show()
```

> **Note:** Run after 1A so `Untitled.jpeg` is already uploaded in Colab.

---

**1C: Audio Feature Extraction using Librosa**

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

uploaded = files.upload()

audio_path = 'WhatsApp Audio 2026-02-09 at 1.13.57 PM(2).mp3' # Change the name with Your Audio File
audio_np_array, sample_rate = librosa.load(audio_path, sr=None)

print("Sample Rate:", sample_rate)
print("Audio Shape:", audio_np_array.shape)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_np_array, sr=sample_rate)
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

amplitude_envelope = np.abs(librosa.effects.preemphasis(audio_np_array))

plt.figure(figsize=(10, 4))
plt.plot(amplitude_envelope)
plt.title('Amplitude Envelope')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

loudness_db = librosa.amplitude_to_db(amplitude_envelope, ref=np.max)

plt.figure(figsize=(10, 4))
plt.plot(loudness_db)
plt.title('Loudness (dB)')
plt.xlabel('Samples')
plt.ylabel('Decibels')
plt.show()

mfccs = librosa.feature.mfcc(y=audio_np_array, sr=sample_rate, n_mfcc=13)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.show()

chroma = librosa.feature.chroma_stft(y=audio_np_array, sr=sample_rate)

plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, sr=sample_rate, x_axis='time')
plt.colorbar()
plt.title('Chroma Features')
plt.show()

spectral_contrast = librosa.feature.spectral_contrast(y=audio_np_array, sr=sample_rate)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spectral_contrast, sr=sample_rate, x_axis='time')
plt.colorbar()
plt.title('Spectral Contrast')
plt.show()

pitches, magnitudes = librosa.piptrack(y=audio_np_array, sr=sample_rate)

pitch_values = np.max(pitches, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(pitch_values)
plt.title('Pitch Values')
plt.xlabel('Frames')
plt.ylabel('Frequency')
plt.show()

spectral_centroid = librosa.feature.spectral_centroid(y=audio_np_array, sr=sample_rate)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_np_array, sr=sample_rate)
plt.plot(spectral_centroid.T, label='Spectral Centroid')
plt.title('Spectral Centroid')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

> **Note:** Upload your `.mp3` audio file to Colab and update `audio_path` with the exact uploaded filename.

---

### Conclusion

In these experiments, different types of signal and data processing techniques were implemented across images and audio. In Part 1A, spatial domain image processing techniques such as resizing, grayscale conversion, blurring, and edge detection were performed to enhance images and extract important features. In Part 1B, frequency domain transformations using DFT and DCT were applied to analyze image information in terms of frequency components. In Part 1C, audio signal processing was carried out using Librosa, where features such as waveform, amplitude envelope, loudness, and MFCC were extracted and visualized.

---

## Practical 02 — AI Based Search Algorithms using NumPy & TensorFlow

**Student Name:** Mohinish Kashyap | **PRN:** 20250804039
**Date of Experiment:** 16/02/26 | **Date of Submission:** 16/03/26
**Aim:** AI based Search algorithms using NumPy & TensorFlow
**Objectives:** a. Informed Search  b. Uninformed Search
**Software/Tool:** Python / Jupyter / Anaconda / Colab

---

### Theory

A search problem consists of a State Space, Start State, Goal State, and Solution (sequence of actions/plan).

**1) Uninformed Search** (Blind/Brute-Force): No domain knowledge. Six types: BFS, DFS, Depth-limited, Iterative deepening DFS, Bidirectional, Uniform cost.

**2) Informed Search** (Heuristic/Directed): Uses heuristic function to estimate closeness to goal. Two types: Greedy Best Fit, A* Search.

**Python Libraries:**
- **NumPy**: Fundamental package for scientific computing; multi-dimensional arrays and matrices.
- **TensorFlow**: Open-source library for numerical computation using data flow graphs; machine learning and deep learning.

---

### Algorithm

**a) Uninformed Search — BFS**
```
BFS(Graph, start, goal)
1. Create an empty queue Q
2. Mark start as visited
3. Enqueue start into Q
4. While Q is not empty:
   a. Dequeue node N from Q
   b. If N == goal:
      Return Success
   c. For each neighbor
      If neighbor not visited:
         Mark neighbor as visited
         Enqueue neighbor
5. Return Failure
```

**a) Uninformed Search — DFS**
```
DFS(Graph, start, goal)
1. Create stack S
2. Push start into S
3. Mark start as visited
4. While S is not empty:
   a. Pop node N from S
   b. If N == goal:
      Return Success
   c. For each neighbor of N:
      If not visited:
         Mark visited
         Push into S
5. Return Failure
```

**b) Informed Search — A***
```
A*(Graph, start, goal)
1. Initialize OPEN list as a priority queue ordered by f(n)
2. Initialize CLOSED list as empty set.
3. For Start node:
   g(Start) = 0
   h(Start) = heuristic(Start, Goal)
   f(Start) = g(Start) + h(Start)
   Insert Start into OPEN
4. While OPEN is not empty:
   a. Remove node n from OPEN having lowest f(n)
   b. If n == Goal:
      Return reconstructed path
   c. Add n to CLOSED
   d. For each neighbor m of n:
      If m in CLOSED:
         Continue to next neighbor
      Compute:
         g(m) = g(n) + cost(n, m)
         h(m) = heuristic(m, Goal)
         f(m) = g(m) + h(m)
      If m not in OPEN:
         Add m to OPEN
      Else if new g(m) < old g(m):
         Update parent of m = n
         Update g(m) and f(m)
5. If OPEN becomes empty:
   Return Failure
```

---

### Program Code

**2a: Uninformed Search — DFS using NumPy**

```python
import numpy as np
import time
import tracemalloc

nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

graph = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0]
])

def dfs_numpy(graph, start_index):
    visited = np.zeros(len(graph), dtype=bool)
    stack = [start_index]
    order = []
    nodes_expanded = 0

    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            order.append(nodes[node])
            nodes_expanded += 1

            for neighbor in range(len(graph)-1, -1, -1):
                if graph[node][neighbor] == 1 and not visited[neighbor]:
                    stack.append(neighbor)

    return order, nodes_expanded

start_time = time.time()
tracemalloc.start()

result, expanded = dfs_numpy(graph, 0)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
end_time = time.time()

print("DFS Order:", result)
print("Nodes Expanded:", expanded)
print("Execution Time:", end_time - start_time, "seconds")
print("Peak Memory Usage:", peak / 10**6, "MB")

print("\nTheoretical Time Complexity: O(V + E)")
print("Theoretical Space Complexity: O(V)")
```

---

**2b: Informed Search — A* using TensorFlow**

```python
import tensorflow as tf
import heapq
import time
import tracemalloc

graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'D': 3, 'E': 1},
    'C': {'F': 5, 'G': 2},
    'D': {'H': 2},
    'E': {'H': 4},
    'F': {},
    'G': {'H': 1},
    'H': {}
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 3,
    'E': 4,
    'F': 6,
    'G': 2,
    'H': 0
}

def a_star_tf(start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))

    g_cost = {node: float('inf') for node in graph}
    g_cost[start] = 0

    parent = {}
    closed_list = set()
    nodes_expanded = 0

    while open_list:
        current = heapq.heappop(open_list)[1]
        nodes_expanded += 1

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1], nodes_expanded

        closed_list.add(current)

        for neighbor in graph[current]:
            if neighbor in closed_list:
                continue

            g_tensor = tf.constant(g_cost[current], dtype=tf.float32)
            edge_tensor = tf.constant(graph[current][neighbor], dtype=tf.float32)
            tentative_g = g_tensor + edge_tensor

            if tentative_g.numpy() < g_cost[neighbor]:
                parent[neighbor] = current
                g_cost[neighbor] = tentative_g.numpy()

                h_tensor = tf.constant(heuristic[neighbor], dtype=tf.float32)
                f_tensor = tentative_g + h_tensor

                heapq.heappush(open_list, (f_tensor.numpy(), neighbor))

    return None, nodes_expanded

start_time = time.time()
tracemalloc.start()

path, expanded_nodes = a_star_tf('A', 'H')

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
end_time = time.time()

print("Shortest Path:", path)
print("Nodes Expanded:", expanded_nodes)
print("Execution Time:", end_time - start_time, "seconds")
print("Peak Memory Usage:", peak / 10**6, "MB")

print("\nTheoretical Time Complexity: O(b^d)")
print("Theoretical Space Complexity: O(b^d)")
```

---

### Conclusion

In these experiments, both A* (A-star) search and Depth First Search (DFS) algorithms were implemented to solve graph traversal and pathfinding problems. DFS, an uninformed search technique, explores nodes deeply without using any heuristic information. The A* algorithm, an informed search technique, uses both actual cost g(n) and heuristic cost h(n) to efficiently find the optimal path. Both algorithms were analyzed in terms of execution time, memory usage, and nodes expanded, demonstrating their practical applications and differences in efficiency and accuracy.

---

## Practical 03 — AI for Problem Solving

**Student Name:** Mohinish Kashyap | **PRN:** 20250804039
**Date of Experiment:** 2/03/26 | **Date of Submission:** 2/03/26
**Aim:** AI for Problem Solving
**Objectives:** a. Tic Tac Toe game  b. Travelling Salesman Problem
**Software/Tool:** Python / Jupyter / Anaconda / Colab

---

### Theory

**1) Tic Tac Toe:** A zero-sum game on a 3x3 grid. Steps: Draw board, Player 1 goes first, Player 2 goes second, alternate until someone wins or no one can win.

**2) Travelling Salesman Problem:** Finding the shortest possible route to multiple destinations and returning to starting point. Solutions: Brute Force, Branch and Bound, Nearest Neighbor, Hungarian Method.

---

### Algorithm

**a) Tic Tac Toe — Minimax**
```
1. Generate the game tree starting from the current state of the game.
   Each node represents a possible game state.
2. Evaluate terminal nodes (end states):
   a. +1 if the AI (maximizing player) wins.
   b. -1 if the human player (minimizing player) wins.
   c.  0 if it's a draw.
3. Backpropagate these values up the tree:
   - For the maximizing player (AI):
     choose the move that leads to the maximum value.
   - For the minimizing player (human):
     choose the move that leads to the minimum value.
4. Repeat this process until the best possible move for the AI is found.
```

**b) Travelling Salesman Problem — Genetic Algorithm**
```
1. Initialize distance matrix for cities.
2. Generate an initial population of random routes.
3. For each generation:
   a. Evaluate fitness (total distance of each route).
   b. Select best two routes (parents).
   c. Apply crossover to create new routes.
   d. Apply mutation with small probability.
   e. Form new population.
4. Repeat until maximum generations are reached.
5. Select the best route with minimum distance.
```

---

### Program Code

**3a: Tic Tac Toe**

```python
import math

def create_board():
    return [' ' for _ in range(9)]

def print_board(board):
    for i in range(3):
        print(' | ' + ' | '.join(board[i*3:(i+1)*3]) + ' |')

def print_positions():
    print("\nBoard Positions:")
    positions = [str(i+1) for i in range(9)]
    for i in range(3):
        print(' | ' + ' | '.join(positions[i*3:(i+1)*3]) + ' |')
    print()

def check_winner(board):
    combos = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    for c in combos:
        if board[c[0]] == board[c[1]] == board[c[2]] and board[c[0]] != ' ':
            return board[c[0]]
    return None

def check_draw(board):
    return ' ' not in board

def minimax(board, is_max):
    winner = check_winner(board)

    if winner == 'O':
        return 1
    if winner == 'X':
        return -1
    if check_draw(board):
        return 0

    if is_max:
        best = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                best = max(best, minimax(board, False))
                board[i] = ' '
        return best
    else:
        best = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                best = min(best, minimax(board, True))
                board[i] = ' '
        return best

def best_move(board):
    best_score = -math.inf
    move = -1
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(board, False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    return move

def player_move(board):
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if move >= 0 and move < 9 and board[move] == ' ':
                board[move] = 'X'
                break
            else:
                print("Invalid move! Try again.")
        except:
            print("Please enter a number between 1 and 9.")

def play_game():
    board = create_board()
    print_positions()
    while True:
        print_board(board)
        player_move(board)

        if check_winner(board) or check_draw(board):
            break

        ai = best_move(board)
        board[ai] = 'O'
        print("\nAI played:")

        if check_winner(board) or check_draw(board):
            break

    print_board(board)

    winner = check_winner(board)
    if winner:
        print(f"\n{winner} wins!")
    else:
        print("\nIt's a draw!")

play_game()
```

---

**3b: Travelling Salesman Problem**

```python
import random

distances = [
    [0, 12, 10, 19, 8],
    [12, 0, 3, 7, 2],
    [10, 3, 0, 6, 20],
    [19, 7, 6, 0, 4],
    [8, 2, 20, 4, 0]
]

num_cities = len(distances)

def total_distance(route):
    dist = 0
    for i in range(len(route)):
        dist += distances[route[i]][route[(i + 1) % num_cities]]
    return dist

def create_route():
    route = list(range(num_cities))
    random.shuffle(route)
    return route

def selection(population):
    population.sort(key=lambda x: total_distance(x))
    return population[:2]

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(num_cities), 2))
    child = [-1]*num_cities
    child[start:end] = parent1[start:end]

    ptr = 0
    for city in parent2:
        if city not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = city
    return child

def mutate(route):
    i, j = random.sample(range(num_cities), 2)
    route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(pop_size=10, generations=100):
    population = [create_route() for _ in range(pop_size)]

    for _ in range(generations):
        parent1, parent2 = selection(population)

        new_population = []
        for _ in range(pop_size):
            child = crossover(parent1, parent2)
            if random.random() < 0.1:
                child = mutate(child)
            new_population.append(child)

        population = new_population

    best = min(population, key=lambda x: total_distance(x))
    return best, total_distance(best)

route, dist = genetic_algorithm()

print("Best Route:", route)
print("Total Distance:", dist)
```

---

### Conclusion

In these experiments, two important AI techniques — Minimax algorithm and Genetic Algorithm — were implemented to solve different types of problems. The Minimax algorithm was used in the Tic Tac Toe game to enable optimal decision-making, resulting in either a win or a draw. The Genetic Algorithm was applied to solve the Travelling Salesman Problem using evolutionary concepts like selection, crossover, and mutation to find a near-optimal solution over multiple generations.

---

## Practical 04 — Implementation of AI Generative Model (GAN)

**Student Name:** Mohinish Kashyap | **PRN:** 20250804039
**Date of Experiment:** 16/03/26 | **Date of Submission:** 16/03/26
**Aim:** Implementation of generative AI model
**Objectives:** a. GAN  b. HMM  c. Auto regressive model
**Software/Tool:** Python / Jupyter / Anaconda / Colab

---

### Theory

**Generative AI** creates new data (images, videos, audio, text, 3D models) by learning patterns from existing data.

**1) GAN:** Has two parts — Generator (learns to generate plausible data, becomes negative training examples for discriminator) and Discriminator (learns to distinguish fake from real, penalizes generator for implausible results).

**2) HMM:** A statistical model dealing with sequences, consisting of Hidden States (unobservable, represent underlying phenomenon) and Observations (observable outcomes associated with each hidden state). First-order Markov chain: next state depends only on current state.

**3) Auto Regressive Model:** Forecasts future behavior based on past behavior data. AR(1) depends on immediately preceding value; AR(2) uses previous two values; AR(0) processes white noise. Least squares method calculates coefficients.

---

### Algorithm — GAN on CIFAR-10

```
1. Import libraries and set device (CPU/GPU).
2. Define hyperparameters (epochs, batch size, learning rate, latent dimension).
3. Load and preprocess CIFAR-10 dataset (normalize images).
4. Build Generator to create fake images from noise.
5. Build Discriminator to classify real and fake images.
6. Initialize loss function (BCE) and optimizers (Adam).
7. For each epoch:
   a. Train Discriminator on real and fake images.
   b. Train Generator to fool the Discriminator.
8. Generate and display images after each epoch.
```

---

### Program Code (GAN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10

scaler = torch.amp.GradScaler('cuda')

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 8, 8)),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1),
        )

    def forward(self, img):
        return self.model(img)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):

        real_images = real_images.to(device, non_blocking=True)

        batch_size = real_images.size(0)
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        z = torch.randn(batch_size, latent_dim, device=device)

        # ------------------
        # Train Discriminator
        # ------------------
        optimizer_D.zero_grad()

        with torch.amp.autocast('cuda'):
            fake_images = generator(z)

            real_loss = loss_fn(discriminator(real_images), valid)
            fake_loss = loss_fn(discriminator(fake_images.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2

        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)

        # ------------------
        # Train Generator
        # ------------------
        optimizer_G.zero_grad()

        with torch.cuda.amp.autocast():
            gen_images = generator(z)
            g_loss = loss_fn(discriminator(gen_images), valid)

        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"D Loss: {d_loss.item():.4f} "
          f"G Loss: {g_loss.item():.4f}")

    if (epoch + 1) % 2 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).cpu()

        grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)

        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis("off")
        plt.title(f"Epoch {epoch+1}")
        plt.show()
```

---

### Conclusion

A Generative Adversarial Network (GAN) was successfully implemented using PyTorch on the CIFAR-10 dataset. The model consists of two neural networks: a Generator, which creates synthetic images from random noise, and a Discriminator, which distinguishes real from generated images. During training, both networks trained simultaneously in an adversarial manner. Over successive epochs, generated images showed gradual improvement in structure and quality. The experiment validates the working principle of GANs and their effectiveness in image generation tasks.

---

## Practical 05 — Implementation of Deep Generative AI Model (RNN)

**Student Name:** Mohinish Kashyap | **PRN:** 20250804039
**Date of Experiment:** 23/03/26 | **Date of Submission:** 23/03/26
**Aim:** Implementation of deep generative AI model
**Objectives:** a. RNN  b. DBM  c. DFM
**Software/Tool:** Python / Jupyter / Anaconda / Colab

---

### Theory

**Deep Generative Models** use deep neural networks to learn the underlying distribution of data to generate novel samples similar to input data but not exactly the same.

**1) RNN (Recurrent Neural Network):** Works better than simple neural networks for sequential data (Time-Series, text). Output from previous step is fed as input to current step. Main feature is its Hidden State (Memory State) which remembers information about a sequence.

**2) DBM (Deep Boltzmann Machine):** Extension of Boltzmann Machine to multiple layers of stochastic binary neurons. Used for unsupervised learning (data generation, feature learning). Neurons within a layer not connected; neurons in adjacent layers fully connected.

**3) DFN/DBN (Deep Belief Network):** Hybrid generative graphical model. Top two layers undirected and symmetric (associative memory); lower layers have directed links. Uses RBMs or autoencoders. No intralayer connections. Hidden units represent features encapsulating data's correlations.

---

### Algorithm — Simple RNN for Sequence Prediction

```
Step 1: Import Libraries
   Import required libraries like torch and torch.nn.
Step 2: Create Dataset
   Prepare sequential input data X (5 time steps each).
   Prepare corresponding output y (next value prediction).
Step 3: Normalize Data
   Scale input and output values by dividing by 50
   to improve training performance.
Step 4: Define RNN Model
   Create class SimpleRNN using nn.Module.
   Add:
      RNN layer (input_size=1, hidden_size=8)
      Fully connected layer to map hidden state to output
   In forward pass:
      Pass input through RNN
      Take last time step output
      Pass through linear layer
Step 5: Define Loss and Optimizer
   Use Mean Squared Error Loss (MSELoss)
   Use Adam optimizer with learning rate 0.01
Step 6: Train the Model
   Loop for 300 epochs:
      Forward pass
      Compute loss
      Backpropagation (loss.backward())
      Update weights (optimizer.step())
Step 7: Test the Model
   Provide new sequence [50,51,52,53,54]
   Normalize input
   Predict next value
   Denormalize output (multiply by 50)
```

---

### Program Code (RNN)

```python
import torch
import torch.nn as nn

# Correct sequence format (batch=4, seq_len=5, input_size=1)
X = torch.tensor([
    [[10],[11],[12],[13],[14]],
    [[20],[21],[22],[23],[24]],
    [[30],[31],[32],[33],[34]],
    [[40],[41],[42],[43],[44]]
], dtype=torch.float32)

y = torch.tensor([[15],[25],[35],[45]], dtype=torch.float32)

# Normalize
X = X / 50.0
y = y / 50.0

class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]   # last timestep
        out = self.fc(out)
        return out

model = SimpleRNN()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(300):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Testing
test = torch.tensor([[[50],[51],[52],[53],[54]]], dtype=torch.float32) / 50.0

prediction = model(test).item() * 50

print("Input: [50,51,52,53,54]")
print("Predicted next number:", prediction)
```

---

### Conclusion

A Recurrent Neural Network (RNN) was implemented using PyTorch to perform sequence prediction. The model was trained on sequential numerical data where each input sequence follows a linear increasing pattern. After training, the RNN successfully learned the underlying pattern and predicted the next value in the new sequence [50, 51, 52, 53, 54] close to 55, showing good generalization capability. Normalization of data improved training stability and convergence. RNNs are effective for handling sequential data and can be applied to time series forecasting, pattern recognition, and prediction problems.

---

## Practical 06 — Implementation of NLP Based Preprocessing of Text Data

**Student Name:** Mohinish Kashyap | **PRN:** 20250804039
**Date of Experiment:** 30/03/26 | **Date of Submission:** 30/03/26
**Aim:** Implementation of NLP based preprocessing of Text data
**Software/Tool:** Python / Jupyter / Anaconda / Colab

---

### Theory

**Natural Language Processing (NLP)** enables computers to understand and process human language. Text data is often unstructured, noisy, and inconsistent. Preprocessing steps include:
- **Tokenization** — splits text into words/sentences
- **Stopword removal** — eliminates common words without significant meaning
- **Stemming** — reduces words to root form by removing suffixes
- **Lemmatization** — converts words to base dictionary forms with better accuracy
- **Bag of Words / TF-IDF** — numerical representations enabling ML models to analyze text

---

### Algorithm

```
1. Start the program.
2. Import required libraries such as NLTK, tokenizer, stopwords, and lemmatizer.
3. Download necessary datasets (stopwords and wordnet) from NLTK.
4. Initialize tools:
   a. Create a tokenizer using TreebankWordTokenizer
   b. Load English stopwords
   c. Initialize WordNet Lemmatizer
5. Input the text data to be processed.
6. Convert the text into lowercase to maintain uniformity.
7. Perform tokenization:
   a. Split the text into individual words (tokens).
8. Remove stopwords and punctuation:
   a. Filter out common words and non-alphanumeric tokens.
9. Apply lemmatization:
   a. Convert each word into its base/root form.
10. Join the processed words to form the final cleaned text.
11. Display the results:
    - Original text
    - Tokens
    - Filtered words
    - Lemmatized words
    - Final cleaned text
12. End the program.
```

---

### Program Code

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

text = "Artificial intelligence is changing the world and creating new opportunities in technology."

text = text.lower()

tokens = tokenizer.tokenize(text)

filtered = [word for word in tokens if word.isalnum() and word not in stop_words]

lemmatized = [lemmatizer.lemmatize(word) for word in filtered]

clean_text = " ".join(lemmatized)

print("Original Text:", text)
print("Tokens:", tokens)
print("After Stopword Removal:", filtered)
print("After Lemmatization:", lemmatized)
print("Final Clean Text:", clean_text)
```

---

### Conclusion

In this practical, text preprocessing was performed using the NLTK library in Python. The input text was processed through lowercasing, tokenization, stopword removal, and lemmatization. The final cleaned text obtained was more meaningful, structured, and suitable for further NLP tasks such as sentiment analysis, text classification, and machine learning. This practical helped in understanding the importance of data cleaning and preprocessing, which is a crucial step in improving the performance and accuracy of NLP models.

---

## Practical 07 — Twitter Data / Neuro Marketing Sentiment Analysis

**Student Name:** Mohinish Kashyap | **PRN:** 20250804039
**Date of Experiment:** 6/04/26 | **Date of Submission:** 6/04/26
**Aim:** Twitter data / Neuro marketing sentiment analysis
**Objectives:** Sentiment Analysis
**Software/Tool:** Python / Jupyter / Anaconda / Colab

---

### Theory

**Sentiment Analysis** is the process of analyzing digital text to determine if the emotional tone is positive, negative, or neutral. Also known as opinion mining, it is an important business intelligence tool.

Benefits: Provide objective insights, Build better products and services, Analyze at scale, Real-time results.

**Working stages:**
- **Preprocessing**: Tokenization, Lemmatization, Stop-word removal
- **Keyword analysis**: NLP assigns sentiment scores (e.g., 10 = satisfaction, 0 = disappointment)

---

### Algorithm

```
1. Start the program.
2. Import required libraries such as NLTK and TextBlob.
3. Download necessary datasets (stopwords and wordnet) from NLTK.
4. Initialize tools:
   a. Create a tokenizer using TreebankWordTokenizer
   b. Load English stopwords
   c. Initialize WordNet Lemmatizer
5. Define input data:
   a. Store multiple sentences in a list for analysis.
6. Define preprocessing function:
   a. Convert text to lowercase
   b. Tokenize the text into words
   c. Remove stopwords and non-alphanumeric words
   d. Apply lemmatization
   e. Join words to form cleaned text
7. Define sentiment analysis function:
   a. Use TextBlob to calculate polarity score
   b. If polarity > 0  → Positive
   c. If polarity < 0  → Negative
   d. If polarity = 0  → Neutral
8. Initialize counters for positive, negative, and neutral sentiments.
9. Process each sentence in the dataset:
   a. Preprocess the sentence
   b. Analyze sentiment using polarity
   c. Classify sentiment
   d. Update respective counters
   e. Display original text, cleaned text, polarity score, and sentiment
10. Repeat the analysis loop (as per program structure if required).
11. Display summary results:
    - Total number of positive, negative, and neutral sentences
12. End the program.
```

---

### Program Code

```python
import nltk
from textblob import TextBlob

nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
tokenizer = TreebankWordTokenizer()

data = [
    "I watched the movie last night and it was a fantastic experience.",
    "The storyline was engaging and the performances by the actors were outstanding.",
    "The visuals and background music added great depth to the film",
    "However, the second half felt a bit slow in some parts."
]

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive 😊"
    elif polarity < 0:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    return polarity, sentiment

def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())

    filtered_tokens = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            filtered_tokens.append(lemma)

    return " ".join(filtered_tokens)

print("Sentiment Analysis Result : \n")
positive_count = 0
negative_count = 0
neutral_count = 0

for i, text in enumerate(data):
    cleaned_text = preprocess(text)
    polarity, sentiment = analyze_sentiment(cleaned_text)

    if "Positive" in sentiment:
        positive_count += 1
    elif "Negative" in sentiment:
        negative_count += 1
    else:
        neutral_count += 1

    print(f"Sentence {i+1}: {text}")
    print(f"Preprocessed: {cleaned_text}")
    print(f"Polarity Score: {round(polarity, 3)}")
    print(f"Sentiment: {sentiment}\n")


print("\n"*3)

print("Summary of the analysis :")
print(f"Positive: {positive_count}")
print(f"Negative: {negative_count}")
print(f"Neutral : {neutral_count}")
```

---

### Conclusion

A sentiment analysis system was successfully implemented using Python libraries NLTK and TextBlob. Text was preprocessed using tokenization, stopword removal, and lemmatization. The TextBlob sentiment analyzer calculated polarity scores classifying sentiments into Positive, Negative, and Neutral categories. The results showed Positive sentences had polarity > 0, Negative < 0, and Neutral = 0. This experiment demonstrates how NLP techniques effectively analyze human emotions from textual data and highlights the importance of preprocessing in improving model accuracy.

---

## Practical 08 — Explainable AI: Case Study

**Student Name:** Mohinish Kashyap | **PRN:** 20250804039
**Date of Experiment:** 13/04/26 | **Date of Submission:** 13/04/26
**Aim:** To implement Explainable AI techniques using SHAP and LIME on the Iris dataset to understand and interpret the predictions of a machine learning model
**Objectives:** To understand the concept of Explainable AI (XAI)
**Software/Tool:** Python / Jupyter / Anaconda / Colab

---

### Case Study

The Iris dataset contains measurements of flower features (sepal length, sepal width, petal length, petal width) for classifying into three species: Setosa, Versicolor, Virginica. A machine learning model (Random Forest) is trained, then Explainable AI techniques are applied. SHAP identifies feature contributions globally and locally; LIME explains individual predictions by approximating the model locally.

---

### Theory

**Explainable AI (XAI)** makes ML models transparent and interpretable. Advanced models act as "black boxes" — XAI helps explain feature contributions and prediction logic, especially important in healthcare and finance.

**SHAP (SHapley Additive Explanations):** Based on game theory, assigns each feature an importance value representing its contribution to the prediction. Provides both global and local interpretability.

**LIME (Local Interpretable Model-agnostic Explanations):** Explains individual predictions by approximating the complex model with a simpler interpretable model in the vicinity of the input.

---

### Algorithm

```
1. Import Libraries
   Import NumPy, SHAP, LIME, Matplotlib, and Scikit-learn.
2. Load Dataset
   Load the Iris dataset and extract features (X) and target labels (y).
3. Split Dataset
   Divide dataset into training and testing sets using train-test split.
4. Train Model
   Train a Random Forest Classifier using training data.
5. Evaluate Model
   Test the model on test data and calculate accuracy.
6. Apply SHAP (Global + Local Explanation)
   - Create SHAP TreeExplainer.
   - Compute SHAP values for test data.
   - Generate summary plot to show global feature importance.
   - Generate force plot to explain a single prediction.
7. Apply LIME (Local Explanation)
   - Create LimeTabularExplainer using training data.
   - Select a test instance.
   - Generate explanation using explain_instance().
8. Display Results
   - Show SHAP plots for feature importance.
   - Display LIME explanation in notebook.
   - Save LIME output as HTML file.
```

---

### Program Code

```python
!pip install shap lime scikit-learn matplotlib

import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

print("\n--- SHAP Explanation ---")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)

shap.initjs()

sample_index = 0
class_index = 0

shap.force_plot(
    explainer.expected_value[class_index],
    shap_values[sample_index, :, class_index],
    X_test[sample_index, :],
    feature_names=feature_names,
    matplotlib=True
)

print("\n--- LIME Explanation ---")

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

i = 0

exp = explainer_lime.explain_instance(
    X_test[i],
    model.predict_proba
)

exp.show_in_notebook(show_table=True)

exp.save_to_file("lime_explanation.html")
```

---

### Conclusion

This experiment demonstrates the importance of Explainable AI (XAI) in understanding machine learning models. A Random Forest classifier was trained on the Iris dataset and achieved good accuracy. Using SHAP, both global and local explanations were obtained — the summary plot highlighted the most important features influencing predictions, and the force plot explained the contribution of each feature for an individual prediction. LIME was used to provide a local explanation for a specific data instance, showing how different features positively or negatively impacted the model's decision. Overall, both SHAP and LIME enhance model transparency and interpretability, helping users trust, analyze, and improve machine learning models in real-world applications.

---

## Course Outcomes

| CO | Description |
|----|-------------|
| CO1 | Implement the concept of problem-solving using AI |
| CO2 | Apply Knowledge representation and Reasoning |
| CO3 | Implement generative and deep generative models |
| CO4 | Keep Abreast the Emerging trends of AI |

---

*Student: Mohinish Kashyap | PRN: 20250804039 | MCA Semester II | DYPIU, Pune | Academic Year: 2023–2024*
