# Améliorations du Modèle de Prédiction Énergétique — Journal de bord

> **Projet** : Prédiction de la consommation énergétique d'Hydro-Québec (IFT6390)  
> **Objectif** : Minimiser le RMSE sur Kaggle  
> **Données** : 3 postes de mesure (A, B, C), consommation horaire en kWh

---

## Table des matières

1. [Résumé des versions](#résumé-des-versions)
2. [Comprendre les données — Contexte essentiel](#comprendre-les-données)
3. [Amélioration 1 — Suppression de la fuite de données (data leakage)](#amélioration-1--suppression-de-la-fuite-de-données)
4. [Amélioration 2 — Encodage du poste (one-hot encoding)](#amélioration-2--encodage-du-poste)
5. [Amélioration 3 — Correction du merge (cross-join → merge exact)](#amélioration-3--correction-du-merge)
6. [Amélioration 4 — Modèles séparés par poste](#amélioration-4--modèles-séparés-par-poste)
7. [Résumé visuel des gains](#résumé-visuel-des-gains)
8. [Glossaire pour débutants](#glossaire-pour-débutants)

---

## Résumé des versions

| Version            | RMSE Kaggle   | R²            | Problème principal                                      | Correction appliquée                       |
| ------------------ | ------------- | ------------- | ------------------------------------------------------- | ------------------------------------------ |
| **v0** (initiale)  | ~94 kWh       | -0.77         | Fuite de données + poste ignoré + merge cassé           | —                                          |
| **v1** (clean)     | ~94 kWh       | -0.77         | Fuite supprimée mais poste toujours ignoré, merge cassé | Suppression des energy lags                |
| **v2** (+ poste)   | ~75.5 kWh     | -0.14         | Poste encodé mais un seul modèle → biais énorme         | One-hot encoding + merge corrigé + fillna  |
| **v3** (per-poste) | **à mesurer** | **à mesurer** | —                                                       | Un modèle Ridge par poste + lags per-poste |

---

## Comprendre les données

Avant de parler des améliorations, il faut comprendre **pourquoi** le problème est difficile.

### Les 3 postes sont très différents

Un « poste » est une station de mesure d'Hydro-Québec. Chaque poste dessert un nombre différent de clients et a une consommation très différente :

| Poste | Train (n)   | Train (moy. kWh) | Test (n)    | Test (moy. kWh) |
| ----- | ----------- | ---------------- | ----------- | --------------- |
| A     | 1 751 (21%) | 82.73            | 474 (27%)   | 50.82           |
| B     | 366 (4%)    | 129.81           | 1 126 (64%) | 72.20           |
| C     | 6 129 (74%) | 259.10           | 154 (9%)    | 269.41          |

**Points clés :**

- Poste C consomme **5× plus** que Poste A.
- Le train est dominé par Poste C (74%), mais le test est dominé par Poste B (64%).
- C'est un déséquilibre massif : le modèle « voit » surtout C pendant l'entraînement mais doit surtout prédire B au test.

### Le décalage saisonnier (distribution shift)

- **Train** : janvier 2022 → janvier 2024 (beaucoup d'hiver, température moyenne 6°C)
- **Test** : février 2024 → juillet 2024 (printemps/été, température moyenne 9.5°C)

L'hiver = chauffage = consommation élevée. L'été = moins de chauffage = consommation plus basse. Le modèle doit **généraliser** vers des températures qu'il a moins vues.

---

## Amélioration 1 — Suppression de la fuite de données

### Le problème : « data leakage » (fuite de données)

**Qu'est-ce que c'est ?**  
Une fuite de données, c'est quand ton modèle utilise, pendant l'entraînement, une information qu'il n'aurait **pas** en conditions réelles.

**Analogie simple :**  
Imagine que tu étudies pour un examen. Si quelqu'un te donne les réponses à l'avance, tu vas avoir 100%. Mais si l'examen change, tu vas échouer car tu n'as rien appris. C'est exactement ça, la fuite de données : le modèle "triche" pendant l'entraînement et échoue sur de nouvelles données.

### Ce qui se passait dans notre code

La v0 du modèle utilisait des **energy lags** (décalages temporels de la consommation) :

```python
# FUITE ! On utilise la consommation passée comme feature
df['energie_lag1'] = df['energie_kwh'].shift(1)    # conso il y a 1h
df['energie_lag24'] = df['energie_kwh'].shift(24)   # conso il y a 24h
df['energie_rolling_mean'] = df['energie_kwh'].rolling(24).mean()
```

**Pourquoi c'est une fuite ?**  
Au moment de faire la prédiction sur Kaggle, on n'a **pas** la consommation passée (`energie_kwh`). Le fichier `energy_test.csv` ne contient pas cette colonne ! Donc le modèle apprend à s'appuyer sur une info qu'il n'aura jamais en production.

**Conséquence :**

- En validation locale : le modèle semble bon (il "voit" la réponse décalée)
- Sur Kaggle (test réel) : les lags sont absents → prédictions catastrophiques

### La correction

On a supprimé **tous** les features basés sur `energie_kwh` :

```python
# AVANT (v0) - FUITE
df['energie_lag1'] = df['energie_kwh'].shift(1)  # ❌

# APRÈS (v1+) - PAS DE FUITE
df['temp_lag1'] = df['temperature_ext'].shift(1)  # ✅ météo historique = toujours disponible
```

Les **weather lags** (décalage de température, humidité, etc.) sont OK car la météo historique est toujours disponible au moment de la prédiction.

### Impact

Supprime la "triche" du modèle. Le RMSE local (train) augmente car le modèle ne peut plus copier la réponse, mais le RMSE Kaggle devient réaliste.

---

## Amélioration 2 — Encodage du poste (one-hot encoding)

### Le problème : le poste est ignoré

La colonne `poste` contient des lettres ('A', 'B', 'C'). Or, scikit-learn ne peut travailler qu'avec des **nombres**. Dans la v0/v1, on faisait :

```python
features = train.select_dtypes(include=[np.number]).columns
```

Cette ligne sélectionne uniquement les colonnes numériques. Résultat : `poste` (type texte) est **silencieusement exclu**.

**Pourquoi c'est grave ?**  
Le modèle voit la température, l'heure, l'humidité... mais ne sait **pas** pour quel poste il prédit. Il produit la **même prédiction** pour les 3 postes au même instant.

**Analogie :**  
C'est comme prédire la facture d'électricité d'un appartement, d'une maison et d'une usine en utilisant la même formule. L'appartement consomme 50 kWh, la maison 72 kWh, l'usine 269 kWh — mais ton modèle prédit ~150 kWh pour tout le monde (la moyenne). Résultat : tout le monde est à côté.

### La correction (v2) : one-hot encoding

Le **one-hot encoding** transforme une variable catégorielle en plusieurs colonnes binaires (0 ou 1) :

| poste | →   | poste_A | poste_B | poste_C |
| ----- | --- | ------- | ------- | ------- |
| A     |     | 1       | 0       | 0       |
| B     |     | 0       | 1       | 0       |
| C     |     | 0       | 0       | 1       |

```python
dummies = pd.get_dummies(df['poste'], prefix='poste')
df = pd.concat([df, dummies], axis=1)
```

Maintenant le modèle Ridge peut apprendre un coefficient différent pour chaque poste :

- `poste_C` a un coefficient positif (grande consommation)
- `poste_A` a un coefficient négatif (petite consommation)

### Limite de cette approche

Dans la v2, les coefficients appris étaient :

- `poste_A` : +10.5
- `poste_C` : -9.9

Mais la vraie différence entre postes est de **~200 kWh**, pas ~20. Un seul modèle Ridge ne peut pas capturer des échelles aussi différentes avec un simple décalage linéaire.

### Impact

RMSE : ~94 → ~75.5 kWh. Amélioration de ~20%, mais insuffisant.

---

## Amélioration 3 — Correction du merge (cross-join → merge exact)

### Le problème : le cross-join

Pour simuler le score Kaggle, on faisait un merge entre nos prédictions et les vraies valeurs :

```python
# AVANT (v0/v1) - BUG
merged = pred_df.merge(gt_df, on='horodatage_local')
```

**Pourquoi c'est faux ?**  
Plusieurs postes partagent le **même** timestamp. Par exemple, à 2024-02-01 05:00:00, il y a une ligne pour Poste B et une pour Poste C. Le merge sur `horodatage_local` seul crée un **produit cartésien** (cross-join) :

| horodatage | poste (pred) | y_pred | poste (gt) | y_true |
| ---------- | ------------ | ------ | ---------- | ------ | --------------------------------------- |
| 05:00      | B            | 28     | B          | 90     |
| 05:00      | B            | 28     | C          | 175    | ← FAUX : pred de B comparée à vrai de C |
| 05:00      | C            | 272    | B          | 90     | ← FAUX : pred de C comparée à vrai de B |
| 05:00      | C            | 272    | C          | 175    |

Résultat : **2 118 lignes au lieu de 1 754**. Toutes les métriques (RMSE, MAE, R²) sont fausses.

**Analogie :**  
C'est comme corriger un examen en mélangeant les copies de différents étudiants : tu compares la note de Marie avec la réponse de Jean. Le résultat ne veut rien dire.

### La correction (v2+)

```python
# APRÈS - CORRECT
merged = pred_df.merge(gt_df, on=['horodatage_local', 'poste'])
```

En ajoutant `poste` comme clé de jointure, chaque prédiction est comparée à la bonne valeur réelle. **1 754 lignes exactement**, comme attendu par Kaggle.

### Impact

Les métriques deviennent fiables. Sans cette correction, on ne pouvait même pas diagnostiquer correctement les autres problèmes.

---

## Amélioration 4 — Modèles séparés par poste

### Le problème résiduel après v2

Même avec le one-hot encoding, un **seul modèle Ridge** apprend :

$$\hat{y} = \beta_0 + \beta_1 \cdot \text{temp} + \beta_2 \cdot \text{poste\_C} + \ldots$$

L'intercept (β₀) était de **215.9** — c'est la moyenne globale du train, dominée par Poste C (74% des données, ~259 kWh).

Le problème est que les **relations entre features et consommation** sont différentes par poste :

- **Poste C** (gros poste) : quand la température baisse de 10°C, la consommation monte de ~100 kWh (beaucoup de thermostats)
- **Poste A** (petit poste) : même baisse de 10°C → la consommation monte de ~20 kWh
- Un seul coefficient `temperature_ext` ne peut pas capturer les deux comportements

**Les biais observés en v2 :**
| Poste | Biais (y_true - y_pred) | Signification |
|-------|------------------------|---------------|
| A | -75.65 | Surestimation massive (+75 kWh) |
| B | +9.29 | Correct |
| C | -169.01 | Surestimation massive (+169 kWh) |

Le modèle unique « moyenne » les trois comportements. Il sur-prédit pour A (petit poste) et sous-ajuste les relations spécifiques de chaque poste.

**Analogie :**  
C'est comme avoir un seul thermostat pour régler la température d'un appartement, d'un bureau et d'un entrepôt. Chaque espace a des besoins différents — il vaut mieux avoir un thermostat par pièce.

### La correction (v3) : un modèle par poste

```python
for poste in ['A', 'B', 'C']:
    train_p = train[train['poste'] == poste]
    ridge_p = RidgeCV(alphas=...).fit(X_train_p, y_train_p)
    models[poste] = ridge_p
```

**Avantages concrets :**

1. **Intercepts adaptés** — Le modèle de Poste A apprend un intercept ~50, celui de C ~269. Plus besoin d'un coefficient `poste_C` qui essaie (et échoue) de corriger un intercept global.

2. **Coefficients adaptés** — La relation température → consommation est apprise séparément. Si Poste C est très sensible au froid et Poste A beaucoup moins, chaque modèle apprend sa propre pente.

3. **Weather lags corrects** — Les shift(1) et shift(24) sont maintenant calculés **au sein de chaque poste**, triés par temps. Avant, un shift(1) pouvait mélanger la température du Poste C à 14h avec celle du Poste B à 14h — c'est du bruit, pas de l'information.

4. **Pas de poste dummies** — Plus besoin de one-hot encoding car chaque modèle ne voit que son propre poste.

### Pourquoi ça marchait mal avant : dilution statistique

En v2, Poste C avait **6 129 lignes** (74% du train) et Poste B seulement **366 lignes** (4%). Le modèle unique optimisait surtout pour C car il domine la loss function (MSE). Les patterns de B étaient « noyés ».

Avec un modèle par poste, Poste B a son propre modèle entraîné sur ses 366 lignes : chaque coefficient est optimisé pour B uniquement.

### Précaution : petit échantillon pour Poste B

366 lignes d'entraînement pour Poste B, c'est peu. C'est pourquoi on utilise Ridge (régularisation) et pas OLS :

- **OLS** minimise l'erreur sans contrainte → risque d'overfitting avec peu de données
- **Ridge** ajoute une pénalité λΣβᵢ² → les coefficients restent petits et le modèle généralise mieux

La force de la régularisation (alpha) est choisie par cross-validation temporelle (`TimeSeriesSplit`) pour chaque poste séparément.

### Impact attendu

Le biais (-75 pour A, -169 pour C) devrait pratiquement disparaître. L'amélioration la plus importante est attendue sur Poste B (64% du test) et Poste A (27% du test).

---

## Résumé visuel des gains

```
v0 (initiale)
├── Problème 1: Energy lags = fuite de données
├── Problème 2: Poste ignoré (colonnes texte filtrées)
├── Problème 3: Merge cross-join (2118 lignes au lieu de 1754)
└── RMSE ≈ 94 kWh (métriques fausses à cause du merge)

v1 (clean)
├── ✅ Suppression des energy lags
├── ❌ Poste toujours ignoré
├── ❌ Merge toujours cassé
└── RMSE ≈ 94 kWh (métriques fausses)

v2 (+ poste + merge fix)
├── ✅ Pas de fuite de données
├── ✅ One-hot encoding du poste
├── ✅ Merge correct sur horodatage + poste
├── ✅ fillna au lieu de dropna (1754/1754 lignes)
├── ❌ Un seul modèle pour 3 postes très différents
└── RMSE ≈ 75.5 kWh

v3 (per-poste models)
├── ✅ Pas de fuite de données
├── ✅ Un modèle Ridge par poste
├── ✅ Weather lags calculés per-poste
├── ✅ Intercepts et coefficients adaptés
└── RMSE = ??? (à mesurer)
```

---

## Glossaire pour débutants

| Terme                   | Explication simple                                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **RMSE**                | Root Mean Squared Error. Mesure l'écart moyen entre prédiction et réalité, en kWh. Plus c'est bas, mieux c'est. Formule : √(1/n × Σ(yᵢ - ŷᵢ)²) |
| **R²**                  | Coefficient de détermination. 1.0 = parfait, 0.0 = aussi bon que prédire la moyenne, négatif = pire que la moyenne.                            |
| **MAE**                 | Mean Absolute Error. Similaire au RMSE mais sans mettre au carré (moins sensible aux gros écarts).                                             |
| **Biais**               | Erreur systématique. Un biais de -75 signifie que le modèle sur-prédit en moyenne de 75 kWh.                                                   |
| **Ridge**               | Régression linéaire avec une pénalité sur la taille des coefficients. Empêche l'overfitting.                                                   |
| **Alpha (λ)**           | Force de la régularisation Ridge. Grand alpha = coefficients petits = modèle simple. Petit alpha = modèle flexible.                            |
| **RidgeCV**             | Ridge avec sélection automatique du meilleur alpha par cross-validation.                                                                       |
| **Cross-validation**    | Technique pour évaluer le modèle en le testant sur des portions de données qu'il n'a pas vues.                                                 |
| **TimeSeriesSplit**     | Cross-validation spéciale pour les séries temporelles : le fold de test est toujours après le fold de train (respecte l'ordre chronologique).  |
| **One-hot encoding**    | Transforme une catégorie (A, B, C) en colonnes binaires (0/1). Permet aux modèles numériques de gérer des catégories.                          |
| **Data leakage**        | Utiliser pendant l'entraînement une information qui ne sera pas disponible au moment de la prédiction.                                         |
| **Cross-join**          | Erreur de merge où chaque ligne d'une table est combinée avec chaque ligne de l'autre, créant des lignes fantômes.                             |
| **Distribution shift**  | Quand les données de test ont une distribution différente de celles de train (ex: entraîné sur l'hiver, testé sur l'été).                      |
| **Intercept**           | La constante β₀ dans la régression. C'est la prédiction quand tous les features sont à zéro.                                                   |
| **Overfitting**         | Le modèle « mémorise » le train au lieu d'apprendre les patterns généraux. Résultat : bon sur le train, mauvais sur le test.                   |
| **Feature engineering** | Créer de nouvelles colonnes à partir des colonnes existantes pour aider le modèle (ex: degrés-jours à partir de la température).               |
| **StandardScaler**      | Normalise les features pour qu'ils aient moyenne=0 et écart-type=1. Important pour Ridge car la pénalité est sensible à l'échelle.             |
| **fillna**              | Remplacer les valeurs manquantes (NaN) par une valeur. Contrairement à `dropna` qui supprime la ligne entière.                                 |
