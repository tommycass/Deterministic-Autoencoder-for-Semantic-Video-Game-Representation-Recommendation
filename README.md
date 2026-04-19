# Representational Power of a Deterministic Autoencoder for Video Game Recommendation

**Authors:** Tomás Castro · Nazareno Gonella  
**Institution:** Universidad de San Andrés — AI Engineering  
**Year:** 2025

---

## Overview

This project investigates whether a **Deterministic Autoencoder (DAE)** can learn a compact, semantically meaningful latent space from Steam video game metadata — and whether that representation outperforms raw high-dimensional features for downstream tasks.

We demonstrate that a simple MLP trained on the **32-dimensional latent space** generalizes significantly better than the same MLP trained on the original **918-dimensional feature vector**, and use the learned space to build a fast, semantically coherent recommendation system.

---

## What We Built

- **DAE** with encoder architecture `918 → 256 → 128 → 32` (LeakyReLU + BatchNorm1d), trained on 102,712 Steam games
- **t-SNE visualization** of the latent space revealing genre/style clusters (sports, horror, MMOs, non-gaming software) without supervision
- **MLP comparison**: latent-trained vs. original-feature-trained, evaluated on review count regression as a popularity proxy
- **Cosine similarity recommendation system** operating on latent representations — 0.0001s inference vs. 74s for tag-based filtering

---

## Key Results

| Model | Train MSE | Val MSE | Val R² | Test MSE | Test R² |
|---|---|---|---|---|---|
| MLP on original features | 0.1105 | 1.0599 | -0.265 | 0.4897 | -2.656 |
| **MLP on latent space** | **0.8305** | **0.7582** | **0.087** | **0.2308** | **-0.723** |

The original-feature MLP overfits heavily. The latent-space MLP generalizes substantially better across both validation and test sets.

---

## Recommendation System

Input: game name → DAE encoder → cosine similarity over all 102K games → ranked by review count

**Example (Stardew Valley):**

| Method | Top Recommendation | Time |
|---|---|---|
| Latent Space | Farming Simulator 22, Wild Terra Online | **0.0001s** |
| Tag Filter | My Time At Portia, Potion Craft | 74.46s |

---

## Dataset

Steam games dataset (Kaggle) — 102,712 games, 918 features per game:
- Community tags (453 probabilistic labels)
- Genres (33) and categories (43) as one-hot vectors
- Semantic embeddings from game descriptions via `all-MiniLM-L6-v2` (SentenceTransformers) → 384 dimensions
- Numerical features: price, release year, review count (normalized with StandardScaler)

---

## Stack

`Python` · `PyTorch` · `scikit-learn` · `SentenceTransformers` · `t-SNE` · `NumPy` · `Pandas`

---

## Paper

Full paper available in this repository: [`Paper_DAE_Semantic_Representation`](./Paper_DAE_Semantic_Representation.pdf)

---

---

# Análisis del Poder Representacional de un DAE para Recomendación de Videojuegos

**Autores:** Tomás Castro · Nazareno Gonella  
**Institución:** Universidad de San Andrés — Ingeniería en Inteligencia Artificial  
**Año:** 2025

---

## Descripción

Este proyecto investiga si un **Autoencoder Determinista (DAE)** puede aprender un espacio latente compacto y semánticamente significativo a partir de metadatos de videojuegos de Steam — y si esa representación supera a las características originales de alta dimensionalidad en tareas downstream.

Demostramos que un MLP simple entrenado sobre el **espacio latente de 32 dimensiones** generaliza significativamente mejor que el mismo MLP entrenado sobre el **vector original de 918 características**, y utilizamos el espacio aprendido para construir un sistema de recomendación rápido y semánticamente coherente.

---

## Qué Construimos

- **DAE** con arquitectura de encoder `918 → 256 → 128 → 32` (LeakyReLU + BatchNorm1d), entrenado sobre 102.712 videojuegos de Steam
- **Visualización t-SNE** del espacio latente que revela clusters por género/estilo (deportes, terror, MMOs, software no lúdico) sin supervisión directa
- **Comparación de MLPs**: entrenado sobre espacio latente vs. características originales, evaluados en regresión de cantidad de reseñas como proxy de popularidad
- **Sistema de recomendación por similitud coseno** sobre representaciones latentes — inferencia en 0.0001s vs. 74s del filtrado por tags

---

## Resultados Clave

| Modelo | Train MSE | Val MSE | Val R² | Test MSE | Test R² |
|---|---|---|---|---|---|
| MLP en features originales | 0.1105 | 1.0599 | -0.265 | 0.4897 | -2.656 |
| **MLP en espacio latente** | **0.8305** | **0.7582** | **0.087** | **0.2308** | **-0.723** |

El MLP sobre features originales muestra sobreajuste severo. El MLP sobre espacio latente generaliza significativamente mejor en validación y test.

---

## Sistema de Recomendación

Entrada: nombre del juego → encoder DAE → similitud coseno sobre los 102K juegos → ordenados por cantidad de reseñas

**Ejemplo (Stardew Valley):**

| Método | Top recomendación | Tiempo |
|---|---|---|
| Espacio Latente | Farming Simulator 22, Wild Terra Online | **0.0001s** |
| Filtro de Tags | My Time At Portia, Potion Craft | 74.46s |

---

## Dataset

Dataset de videojuegos de Steam (Kaggle) — 102.712 juegos, 918 features por juego:
- Tags comunitarios (453 etiquetas probabilísticas)
- Géneros (33) y categorías (43) como vectores one-hot
- Embeddings semánticos de descripciones con `all-MiniLM-L6-v2` (SentenceTransformers) → 384 dimensiones
- Features numéricas: precio, año de lanzamiento, cantidad de reseñas (normalizados con StandardScaler)

---

## Stack

`Python` · `PyTorch` · `scikit-learn` · `SentenceTransformers` · `t-SNE` · `NumPy` · `Pandas`

---

## Paper

Paper completo disponible en este repositorio: [`Paper_DAE_Semantic_Representation`](./Paper_DAE_Semantic_Representation.pdf)
