# Cheat Sheet Entrevue — Projet Énergie IFT6390

---

## Fondamentaux

### Dérivez la solution OLS sur le tableau

Minimiser la perte quadratique :

$$L(w) = \|y - Xw\|^2$$

Gradient : $\nabla_w L = -2X^T(y - Xw) = 0$

$$X^TXw = X^Ty \implies w^* = (X^TX)^{-1}X^Ty$$

---

### Pourquoi division temporelle et non aléatoire?

Les données énergétiques sont **autocorrélées dans le temps** (saisonnalité, tendances). Une division aléatoire laisserait fuiter des données futures dans l'entraînement → scores gonflés artificiellement. La division temporelle simule la vraie mise en production : entraîner sur le passé, prédire le futur.

---

### Que voyez-vous dans vos résidus?

- **Biais systématique par poste :** modèle sur-estime en moyenne (+14 kWh Poste A, +20 kWh Poste B) → corrigé manuellement.
- **Hétéroscédasticité :** erreurs plus grandes aux heures de pointe et en hiver.
- **Structure temporelle résiduelle pour C :** le modèle sous-performe en début de période → corriger avec une fenêtre de récence (9 mois).

---

## Régularisation

### Pourquoi Ridge aide-t-il avec des caractéristiques corrélées?

Des features corrélées rendent $X^TX$ quasi-singulière → coefficients OLS instables et très grands. Ridge ajoute $\lambda I$ :

$$(X^TX + \lambda I)w = X^Ty$$

La matrice devient inversible. Ridge distribue le poids entre features corrélées plutôt que de l'amplifier. Plus $\lambda$ est grand, plus les coefficients sont contractés vers 0.

---

### Comment avez-vous choisi λ?

`RidgeCV` avec **TimeSeriesSplit** (5 folds, respect de l'ordre chronologique). Grille : `np.logspace(-2, 6, 50)`. Métrique : MSE négatif. Le meilleur $\lambda$ est sélectionné automatiquement par validation croisée.

---

### Quel coefficient a été le plus réduit? Pourquoi?

Les features liées aux clients (`clients_connectes`, `tstats_clients`) pour le Poste C. Ces variables extrapolent mal : le parc clients de C passe de 76 à 104 entre train et test, sans précédent dans les données. Les exclure (features `weather_ratio`) a réduit le RMSE de C de ~12 kWh.

---

## Classification

### Quelle cible binaire avez-vous choisie? Justifiez.

`est_pointe` : 1 si l'heure appartient à une période de pointe (6h–9h ou 16h–20h), 0 sinon. Justification :

- Physiquement motivé (deux régimes de consommation distincts)
- Non dérivé de `energie_kwh` → pas de fuite de données
- Robuste et interprétable

---

### Votre classifieur donne P=0.7. Qu'est-ce que cela signifie?

Le modèle estime une probabilité de 70% que cette heure soit une heure de pointe. Ce n'est pas une certitude (≥0.95 le serait). On utilise cette valeur **comme feature continue** dans Ridge — pas comme un label binaire — pour préserver l'incertitude.

---

### Pourquoi utiliser P(pointe) plutôt qu'un indicateur 0/1?

Un indicateur binaire perd l'information de confiance. Avec P=0.7, Ridge peut moduler la prédiction **proportionnellement à la confiance**. Si P=0.51 vs P=0.99, le comportement de Ridge est différent — la nuance améliore les prédictions en périodes de transition (ex : 8h30, 16h15).

---

## Théorie probabiliste

### Expliquez Ridge comme estimation MAP

- **OLS = MLE** sous bruit gaussien : maximiser $p(y|X,w)$ revient à minimiser $\|y-Xw\|^2$
- **Ridge = MAP** avec prior gaussien sur $w$ : $p(w) \propto \exp(-\frac{\lambda}{2}\|w\|^2)$

$$\arg\max_w \; p(w|y,X) = \arg\min_w \; \|y-Xw\|^2 + \lambda\|w\|^2$$

Le prior impose que les coefficients restent petits sauf si les données justifient le contraire.

---

### Pourquoi la régression logistique minimise-t-elle l'entropie croisée?

On modélise $P(y=1|x) = \sigma(w^Tx)$. MLE : maximiser la vraisemblance du jeu d'entraînement :

$$\log \mathcal{L} = \sum_i \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right]$$

**Négatif de cette expression = entropie croisée binaire.** Minimiser la cross-entropy est donc _exactement_ équivalent à maximiser la vraisemblance — ce n'est pas un choix arbitraire.

---

## Versions du modèle

| Version        | RMSE         | Changement                    | Problème résolu                                               |
| -------------- | ------------ | ----------------------------- | ------------------------------------------------------------- |
| v0 initiale    | ~94 kWh      | Baseline naïve                | —                                                             |
| v1 clean       | ~94 kWh      | Suppression energy lags       | **Data leakage** : `energie_kwh` utilisée comme feature       |
| v2 + poste     | ~75.5 kWh    | One-hot encoding + merge fix  | Poste ignoré → même prédiction pour A/B/C                     |
| v3 per-poste   | 66.4 kWh     | Un Ridge par poste            | Intercept global inadapté (dominé par C, 74% des données)     |
| v4 reg. forcée | 63.5 kWh     | Alpha min=1000 pour C         | Overfitting : RidgeCV choisissait alpha=10 (trop faible)      |
| v5 Ridge+KNN   | 50.4 kWh     | KNN k=200 pour C              | Extrapolation linéaire de C vers des températures inconnues   |
| v6a features   | 47.4 kWh     | `weather_ratio` pour A et C   | `clients_connectes` extrapole (25→52 clients pour A)          |
| v6b récence    | 36.0 kWh     | Fenêtre 9 mois pour C         | Dérive temporelle : données 2022 périmées pour test 2024      |
| **v6 final**   | **28.1 kWh** | Bias correction A(-14) B(-20) | Biais systématique : B entraîné sur hiver, testé au printemps |

**Gains cumulés :** 94 → 28 kWh = **-70%** sur le RMSE.  
**Plus grand saut unique :** v6b récence C (-11.4 kWh, 70% de l'erreur était portée par C).

---

## Synthèse

### Parcourez votre modèle complet étape par étape

1. **Feature engineering** (`creer_caracteristiques_v3`) : variables météo, heure, jour, mois, ratio clients/tstats
2. **Classification** : entraîner `LogisticRegression` → générer `P_pointe` comme feature
3. **Ridge par poste** avec features optimales :
   - A : `weather_ratio` (37 feat), toutes les données
   - B : `full` (42 feat), toutes les données
   - C : `weather_ratio` (37 feat), **fenêtre 9 mois** (récence)
4. **Correction de biais** : A → -14 kWh, B → -20 kWh, C → 0
5. **Clip** : `max(prediction, 0)` pour éviter valeurs négatives

RMSE final : **28.13 kWh**, R² = 0.842

---

### Quelle amélioration de R² était la plus importante?

La **fenêtre de récence pour le Poste C** (9 mois vs toutes les données) : -11.39 kWh RMSE. C représente ~70% de l'erreur totale car son test est en fév 2024 seulement, et les 2 ans de données d'entraînement incluent des patterns anciens non représentatifs.

---

### Modifiez ce seuil en direct — que prédisez-vous?

Si on passe le seuil de classification de 0.5 → 0.7 :

- Moins d'heures classées "pointe" → `P_pointe` plus basse pour les heures ambiguës (8h30, 16h15)
- Ridge prédit une consommation **plus basse** sur ces heures de transition
- Les heures clairement en pointe (P=0.95+) ne changent pas
- Net effect : léger sous-estimation aux pointes peu marquées → surveiller les résidus positifs

---
