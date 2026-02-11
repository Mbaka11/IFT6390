# 📚 Notes de Préparation - Entrevue Orale (60%)

## ⚠️ RAPPEL: L'entrevue orale vaut 60% de la note totale!

---

## 🔍 Questions sur les Données et Features Engineering

### Q1: Pourquoi utilise-t-on un encodage cyclique pour les heures et les mois?

**Contexte:**
Dans le dataset, on a:

- Variables simples: `heure` (0-23), `mois` (1-12)
- Variables cycliques: `heure_sin`, `heure_cos`, `mois_sin`, `mois_cos`

**Le problème avec l'encodage numérique simple:**

Si on utilise juste `heure = 0, 1, 2, ..., 23`:

```
Distance entre 0h et 1h:   |1 - 0| = 1
Distance entre 23h et 0h:  |0 - 23| = 23  ❌ FAUX!
```

Le modèle penserait que 23h et 0h sont **très éloignées** alors qu'elles sont séparées d'**une seule heure**!

**La solution: Encodage cyclique avec sin/cos**

On transforme l'heure en coordonnées sur un cercle:

```python
heure_sin = sin(2π × heure/24)
heure_cos = cos(2π × heure/24)
```

**Visualisation mentale:**

```
        12h (midi)
         |
    9h --|-- 15h
         |
    6h --|-- 18h
         |
        0h (minuit)
         ↓
       23h ← Proche de 0h! ✅
```

**Pourquoi sin ET cos (pas juste sin)?**

- Avec **seulement sin**: sin(0°) = sin(360°) = 0 → 0h et 12h auraient la même valeur!
- Avec **sin + cos**: Chaque heure a une combinaison **unique** (x, y) sur le cercle

**Exemple numérique:**

```
0h:  sin=0.00,  cos=1.00
6h:  sin=1.00,  cos=0.00
12h: sin=0.00,  cos=-1.00
18h: sin=-1.00, cos=0.00
23h: sin=0.26,  cos=0.97  ← Proche de 0h ✅
```

**Avantages principaux:**

1. ✅ **Préserve la proximité**: 23h et 0h sont proches dans l'espace des features
2. ✅ **Continuité**: Pas de "saut" artificiel entre 23h et 0h
3. ✅ **Généralisation**: Le modèle apprend que les comportements à 23h peuvent ressembler à ceux de 0h
4. ✅ **Applicabilité**: Fonctionne pour toute variable cyclique (jour de la semaine, mois, saison, angle, etc.)

**Pour l'entrevue, soyez prêt à:**

- ✏️ Dessiner un cercle et placer quelques heures dessus
- 🧮 Expliquer pourquoi on a besoin de DEUX dimensions (sin + cos)
- 💡 Donner un exemple concret: "La consommation à 23h est proche de celle à 0h (nuit)"

**Formule mathématique à connaître:**
$$\text{feature\_sin} = \sin\left(\frac{2\pi \times \text{valeur}}{\text{période}}\right)$$
$$\text{feature\_cos} = \cos\left(\frac{2\pi \times \text{valeur}}{\text{période}}\right)$$

Où période = 24 pour les heures, 12 pour les mois.

---

## 📝 Questions à Préparer pour l'Entrevue

### Fondamentaux

- [ ] Dérivez la solution OLS sur le tableau
- [ ] Pourquoi division temporelle et non aléatoire?
- [ ] Que voyez-vous dans vos résidus?

### Régularisation

- [ ] Pourquoi Ridge aide avec des features corrélées?
- [ ] Comment choisir λ?
- [ ] Quel coefficient a été le plus réduit? Pourquoi?

### Classification

- [ ] Quelle cible binaire avez-vous choisie? Justifiez.
- [ ] Le classifieur donne P=0.7. Signification?
- [ ] Pourquoi utiliser P(pointe) plutôt qu'un indicateur 0/1?

### Théorie probabiliste

- [ ] Expliquez Ridge comme estimation MAP
- [ ] Pourquoi la régression logistique minimise l'entropie croisée?

### Synthèse

- [ ] Parcourez votre modèle complet étape par étape
- [ ] Quelle amélioration de R² était la plus importante?
- [ ] Modifiez ce seuil en direct - prédisez les effets

---

## 📊 Concepts Clés à Maîtriser

_(Cette section sera complétée au fur et à mesure)_

### 1. OLS (Ordinary Least Squares) - MAÎTRISER POUR L'ENTREVUE! ⭐

## 📐 Théorie Mathématique Complète

### Le Problème

On cherche à prédire une variable cible $y$ (consommation énergétique) à partir de caractéristiques $\mathbf{X}$ (température, humidité, etc.).

**Modèle linéaire:**
$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ... + \beta_p x_{ip} + \epsilon_i$$

Ou en notation matricielle:
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Où:

- $\mathbf{y}$ : vecteur de taille $(n, 1)$ - les valeurs cibles
- $\mathbf{X}$ : matrice de taille $(n, p+1)$ - les caractéristiques (avec une colonne de 1 pour l'intercept)
- $\boldsymbol{\beta}$ : vecteur de taille $(p+1, 1)$ - les coefficients à trouver
- $\boldsymbol{\epsilon}$ : vecteur d'erreurs (bruit)

### Objectif: Minimiser l'Erreur Quadratique

On veut trouver $\hat{\boldsymbol{\beta}}$ qui minimise:
$$L(\boldsymbol{\beta}) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

## 📝 Dérivation Étape par Étape (IMPORTANT pour l'entrevue!)

**Étape 1: Écrire la fonction de perte**
$$L(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

**Étape 2: Développer**
$$L(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{X}\boldsymbol{\beta} - \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

Comme $\mathbf{y}^T\mathbf{X}\boldsymbol{\beta}$ est un scalaire, $\mathbf{y}^T\mathbf{X}\boldsymbol{\beta} = \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y}$

$$L(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

**Étape 3: Calculer le gradient**

Rappels d'algèbre linéaire:

- $\frac{\partial}{\partial \boldsymbol{\beta}}(\mathbf{A}\boldsymbol{\beta}) = \mathbf{A}^T$
- $\frac{\partial}{\partial \boldsymbol{\beta}}(\boldsymbol{\beta}^T\mathbf{A}\boldsymbol{\beta}) = 2\mathbf{A}\boldsymbol{\beta}$ (si $\mathbf{A}$ symétrique)

$$\nabla_{\boldsymbol{\beta}} L = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

**Étape 4: Égaler à zéro (condition nécessaire pour un minimum)**
$$-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = 0$$

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

C'est l'**équation normale** !

**Étape 5: Résoudre pour β**

Si $\mathbf{X}^T\mathbf{X}$ est inversible:
$$\boxed{\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

C'est la **solution analytique OLS** ! 🎯

## 🎨 Interprétation Géométrique

### Vision 1: Projection orthogonale

- $\mathbf{y}$ est un vecteur dans $\mathbb{R}^n$
- $\mathbf{X}\boldsymbol{\beta}$ crée un sous-espace de dimension $p$
- OLS trouve la **projection orthogonale** de $\mathbf{y}$ sur ce sous-espace
- Le résidu $\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$ est perpendiculaire au sous-espace

```
        y (vrai)
        |
        |     résidu (erreur)
        |    /
        |   /
        |  /
        | /
        |/
        ŷ = Xβ (prédiction)
    ------------------- (sous-espace généré par X)
```

### Vision 2: Minimisation de distance

OLS trouve le point dans le sous-espace généré par $\mathbf{X}$ le plus proche de $\mathbf{y}$ (au sens de la distance euclidienne).

## 💻 Implémentation Python avec Commentaires Détaillés

```python
import numpy as np

def ols_fit(X, y):
    """
    Calcule les coefficients OLS via la solution analytique.

    Paramètres:
        X : ndarray de forme (n, p) - matrice de caractéristiques SANS colonne de 1
        y : ndarray de forme (n,) - vecteur cible

    Retourne:
        beta : ndarray de forme (p+1,) - coefficients [intercept, coef1, coef2, ...]

    Points clés pour l'entrevue:
    - Pourquoi ajouter une colonne de 1? Pour modéliser l'intercept β₀
    - Pourquoi np.linalg.solve et non l'inverse? Stabilité numérique + efficacité
    - Que faire si X^TX n'est pas inversible? Ridge / régularisation
    """

    # ÉTAPE 1: Ajouter colonne de 1 pour l'intercept
    # X devient (n, p+1) avec X[:, 0] = 1
    n = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n), X])
    # Équivalent: X_with_intercept = np.c_[np.ones(n), X]

    # ÉTAPE 2: Calculer X^T X (matrice de Gram)
    # Forme: (p+1, p+1)
    XTX = X_with_intercept.T @ X_with_intercept

    # ÉTAPE 3: Calculer X^T y
    # Forme: (p+1,)
    XTy = X_with_intercept.T @ y

    # ÉTAPE 4: Résoudre le système X^T X β = X^T y
    # IMPORTANT: On utilise solve() plutôt que inv() pour:
    #   - Stabilité numérique (évite erreurs d'arrondi)
    #   - Efficacité (O(n³) vs O(n³) mais avec meilleure constante)
    #   - Robustesse (gère mieux les matrices mal conditionnées)
    beta = np.linalg.solve(XTX, XTy)

    # Alternative (NON recommandée):
    # beta = np.linalg.inv(XTX) @ XTy  # ❌ Moins stable!

    return beta
    # beta[0] = intercept (β₀)
    # beta[1:] = coefficients des features (β₁, β₂, ..., βₚ)


def ols_predict(X, beta):
    """
    Prédit les valeurs avec les coefficients OLS.

    Paramètres:
        X : ndarray de forme (n, p) - caractéristiques SANS colonne de 1
        beta : ndarray de forme (p+1,) - coefficients [intercept, coef1, ...]

    Retourne:
        y_pred : ndarray de forme (n,) - prédictions

    Points pour l'entrevue:
    - Comment séparer intercept et coefficients? beta[0] vs beta[1:]
    - Forme matricielle: y = X @ w + b, où w=beta[1:] et b=beta[0]
    """

    # MÉTHODE 1: Ajouter colonne de 1 et multiplier
    n = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n), X])
    y_pred = X_with_intercept @ beta

    # MÉTHODE 2 (équivalente): Séparer intercept et coefficients
    # y_pred = beta[0] + X @ beta[1:]

    return y_pred


# ============================================
# EXEMPLE D'UTILISATION AVEC EXPLICATIONS
# ============================================

# Supposons qu'on a:
# - n = 8760 observations (1 an de données horaires)
# - p = 3 features: température, humidité, vitesse_vent

# CHARGEMENT DES DONNÉES
# X_train shape: (8760, 3)
# y_train shape: (8760,)

# 1. ENTRAÎNEMENT
beta_ols = ols_fit(X_train, y_train)
# beta_ols shape: (4,)
# beta_ols[0] = intercept = consommation de base
# beta_ols[1] = coefficient température
# beta_ols[2] = coefficient humidité
# beta_ols[3] = coefficient vitesse_vent

print(f"Intercept (β₀): {beta_ols[0]:.2f} kWh")
print(f"Coefficient température (β₁): {beta_ols[1]:.2f} kWh/°C")
print(f"Coefficient humidité (β₂): {beta_ols[2]:.2f} kWh/%")
print(f"Coefficient vent (β₃): {beta_ols[3]:.2f} kWh/(km/h)")

# INTERPRÉTATION (pour l'entrevue):
# Si β₁ = -5.2, cela signifie:
# "Pour chaque degré de température en plus, la consommation
#  diminue de 5.2 kWh (moins de chauffage)"

# 2. PRÉDICTION
y_pred_train = ols_predict(X_train, beta_ols)
y_pred_test = ols_predict(X_test, beta_ols)

# 3. ÉVALUATION
from sklearn.metrics import r2_score, mean_squared_error

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nPerformance:")
print(f"  R² train: {r2_train:.4f}")
print(f"  R² test: {r2_test:.4f}")
print(f"  RMSE test: {rmse_test:.2f} kWh")

# Pour l'entrevue: Soyez prêt à expliquer R²!
# R² = 0.75 signifie: "Le modèle explique 75% de la variance de y"
# R² = 1.0 → Prédiction parfaite
# R² < 0 → Modèle pire qu'une simple moyenne


# ============================================
# COMPARAISON AVEC SKLEARN (Validation)
# ============================================

from sklearn.linear_model import LinearRegression

model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)

print("\n=== Validation avec sklearn ===")
print(f"Intercept - Vous: {beta_ols[0]:.6f}")
print(f"Intercept - sklearn: {model_sklearn.intercept_:.6f}")
print(f"Coefficients identiques: {np.allclose(beta_ols[1:], model_sklearn.coef_)}")
# Devrait afficher True si implémentation correcte!
```

## ⚠️ Points Critiques pour l'Entrevue

### 1. Pourquoi ajouter une colonne de 1?

**Question type:** "Pourquoi ajoutez-vous np.ones dans votre code?"

**Réponse:**

- Sans intercept: $y = \beta_1 x_1 + \beta_2 x_2$ → la droite passe par l'origine (0, 0)
- Avec intercept: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$ → plus flexible
- La colonne de 1 permet de traiter $\beta_0$ comme les autres coefficients dans le calcul matriciel

### 2. Pourquoi np.linalg.solve plutôt que l'inverse?

**Question type:** "Vous n'utilisez pas l'inverse explicitement, pourquoi?"

**Réponse:**

- `solve(A, b)` résout $Ax = b$ directement via décomposition LU/Cholesky
- `inv(A) @ b` calcule d'abord $A^{-1}$ puis multiplie → 2 opérations coûteuses
- `solve()` est **numériquement plus stable** (moins d'erreurs d'arrondi)
- Exemple: Si $X^TX$ est mal conditionnée, inverse peut échouer

### 3. Quand OLS échoue-t-il?

**Question type:** "Dans quelles situations OLS pose-t-il problème?"

**Réponse:**

1. **$X^TX$ non inversible** (collinéarité parfaite)
   - Exemple: température_celsius et température_fahrenheit
   - Solution: Ridge régularisation
2. **Mal conditionnée** (features très corrélées)
   - Coefficients instables (variance élevée)
   - Solution: Ridge ou PCA
3. **p > n** (plus de features que d'observations)
   - Système sous-déterminé
   - Solution: Ridge ou Lasso

4. **Outliers** (valeurs extrêmes)
   - OLS sensible aux outliers (erreur quadratique)
   - Solution: Régression robuste (Huber loss)

### 4. Complexité computationnelle

**Question type:** "Quelle est la complexité de OLS?"

**Réponse:**

- Calcul de $X^TX$: $O(np^2)$ où n=#observations, p=#features
- Résolution du système: $O(p^3)$
- **Total: $O(np^2 + p^3)$**
- Dominant: $O(np^2)$ si $n >> p$ (cas typique)

### 5. Hypothèses de OLS

**Pour l'entrevue, connaître les hypothèses (mais pas besoin de les vérifier pour ce projet):**

1. **Linéarité**: La relation est linéaire
2. **Indépendance**: Les observations sont indépendantes
3. **Homoscédasticité**: Variance constante des erreurs
4. **Normalité**: Les erreurs suivent une loi normale (pour l'inférence)
5. **Pas de multicolinéarité**: Les features ne sont pas trop corrélées

## 🎯 Checklist pour l'Entrevue OLS

Pratiquez ces exercices:

- [ ] Dériver $\hat{\beta} = (X^TX)^{-1}X^Ty$ au tableau en 5 minutes
- [ ] Expliquer pourquoi on minimise l'erreur quadratique (et non absolue)
- [ ] Dessiner l'interprétation géométrique (projection)
- [ ] Coder ols_fit() de mémoire en 3 minutes
- [ ] Expliquer np.linalg.solve vs inv
- [ ] Interpréter un coefficient: "β₁ = -5.2 signifie..."
- [ ] Expliquer R² à votre grand-mère
- [ ] Donner 3 situations où OLS échoue

### 2. Régression Logistique + Descente de Gradient - MAÎTRISER POUR L'ENTREVUE! ⭐

## 🎯 Pourquoi la Régression Logistique?

**Rappel du contexte:** On veut prédire si une heure donnée sera un **événement de pointe** (1) ou **normale** (0).

- **OLS (Partie 1):** Prédit des valeurs continues (kWh)
- **Régression Logistique (Partie 2):** Prédit des **probabilités** entre 0 et 1

## 📐 Théorie Mathématique Complète

### Le Problème de Classification Binaire

On a:

- $y_i \in \{0, 1\}$ : étiquette binaire (0 = normal, 1 = pointe)
- $\mathbf{x}_i \in \mathbb{R}^p$ : vecteur de caractéristiques (température, heure, etc.)

**Objectif:** Modéliser $P(y=1 | \mathbf{x})$ (probabilité d'être en pointe sachant les features)

### Fonction Sigmoïde (Logistique)

Pour transformer $z \in \mathbb{R}$ en probabilité $p \in [0, 1]$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Propriétés importantes:**

- $\sigma(0) = 0.5$ (point d'inflexion)
- $\lim_{z \to +\infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (dérivée élégante!)

**Visualisation mentale:**

```
p
1.0 |           ___________
    |         /
0.5 |       /    ← Point d'inflexion (z=0)
    |     /
0.0 |___/
    |___|___|___|___|___|___> z
       -5   0   5
```

### Modèle de Régression Logistique

$$z_i = \beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip} = \boldsymbol{\beta}^T \mathbf{x}_i$$

$$P(y_i = 1 | \mathbf{x}_i) = \sigma(z_i) = \frac{1}{1 + e^{-\boldsymbol{\beta}^T \mathbf{x}_i}}$$

**Interprétation:**

- Si $z > 0$ → $p > 0.5$ → Classe 1 (pointe)
- Si $z < 0$ → $p < 0.5$ → Classe 0 (normal)
- Si $z = 0$ → $p = 0.5$ → Frontière de décision

### Fonction de Perte: Entropie Croisée (Cross-Entropy)

**Pourquoi pas MSE (Mean Squared Error)?**

- OLS utilise MSE: $L = \sum (y - \hat{y})^2$
- Avec sigmoïde, MSE → **fonction non-convexe** → pleins de minima locaux ❌

**Entropie Croisée Binaire:**

Pour **une seule observation** $(x_i, y_i)$:

$$\mathcal{L}_i = -\left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

Où $p_i = \sigma(\boldsymbol{\beta}^T \mathbf{x}_i)$

**Intuition:**

- Si $y_i = 1$ : perte = $-\log(p_i)$
  - Si $p_i \to 1$ → perte $\to 0$ ✅ (bonne prédiction)
  - Si $p_i \to 0$ → perte $\to +\infty$ ❌ (très mauvaise prédiction)
- Si $y_i = 0$ : perte = $-\log(1-p_i)$
  - Si $p_i \to 0$ → perte $\to 0$ ✅
  - Si $p_i \to 1$ → perte $\to +\infty$ ❌

**Pour l'ensemble du dataset:**

$$L(\boldsymbol{\beta}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

Où $p_i = \sigma(\boldsymbol{\beta}^T \mathbf{x}_i)$

## 📝 Dérivation du Gradient (IMPORTANT pour l'entrevue!)

**Objectif:** Calculer $\nabla_{\boldsymbol{\beta}} L$ pour la descente de gradient

### Calcul pour une observation

Posons $p = \sigma(z)$ où $z = \boldsymbol{\beta}^T \mathbf{x}$

$$\mathcal{L} = -\left[ y \log(p) + (1-y) \log(1-p) \right]$$

**Étape 1: Dériver par rapport à p**

$$\frac{\partial \mathcal{L}}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}$$

$$= \frac{-y(1-p) + (1-y)p}{p(1-p)} = \frac{p - y}{p(1-p)}$$

**Étape 2: Dériver p par rapport à z (règle de dérivation sigmoïde)**

$$\frac{\partial p}{\partial z} = \frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1 - \sigma(z)) = p(1-p)$$

**Étape 3: Chaîne pour dériver par rapport à z**

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial p} \cdot \frac{\partial p}{\partial z} = \frac{p - y}{p(1-p)} \cdot p(1-p) = p - y$$

**Résultat magique!** ✨ Le gradient simplifie énormément!

**Étape 4: Dériver par rapport à β**

Comme $z = \boldsymbol{\beta}^T \mathbf{x}$, on a $\frac{\partial z}{\partial \boldsymbol{\beta}} = \mathbf{x}$

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \boldsymbol{\beta}} = (p - y) \mathbf{x}$$

### Pour tout le dataset (notation matricielle)

$$\nabla_{\boldsymbol{\beta}} L = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) \mathbf{x}_i = \frac{1}{n} \mathbf{X}^T (\mathbf{p} - \mathbf{y})$$

Où:

- $\mathbf{X}$ : matrice $(n, p+1)$ des caractéristiques
- $\mathbf{p}$ : vecteur $(n,)$ des probabilités prédites $\sigma(\mathbf{X}\boldsymbol{\beta})$
- $\mathbf{y}$ : vecteur $(n,)$ des étiquettes vraies

**Formule finale:**

$$\boxed{\nabla_{\boldsymbol{\beta}} L = \frac{1}{n} \mathbf{X}^T \left(\sigma(\mathbf{X}\boldsymbol{\beta}) - \mathbf{y}\right)}$$

## 🔄 Descente de Gradient

**L'idée:** Pas de solution analytique comme OLS → on itère!

### Algorithme

```
1. Initialiser β ← 0 (ou aléatoire)
2. Pour k = 1 à n_iter:
     a. Calculer les prédictions: p = σ(Xβ)
     b. Calculer le gradient: g = (1/n) X^T (p - y)
     c. Mise à jour: β ← β - α·g
     d. (Optionnel) Calculer et stocker la perte pour suivre convergence
3. Retourner β
```

**Paramètres:**

- $\alpha$ (alpha ou lr) : **taux d'apprentissage** (learning rate)
- $n_{iter}$ : nombre d'itérations

### Choix du Taux d'Apprentissage (Learning Rate)

**Question clé d'entrevue:** "Comment avez-vous choisi le learning rate?"

**Si α trop petit (ex: 0.0001):**

- ✅ Convergence garantie (si fonction convexe)
- ❌ **Très lent** (des milliers d'itérations)
- Graphe loss: descente lisse mais très graduelle

**Si α trop grand (ex: 10.0):**

- ❌ **Divergence!** (oscille de plus en plus)
- ❌ Peut sauter par-dessus le minimum
- Graphe loss: zigzag, montée au lieu de descendre

**Optimal (ex: 0.1 - 1.0 avec normalisation):**

- ✅ Convergence rapide en ~100-500 itérations
- ✅ Stable
- Graphe loss: descente rapide puis plateau

**Conseil pratique:**

1. **Toujours normaliser les features** (StandardScaler) → permet d'utiliser α plus grand
2. Tester plusieurs valeurs: 0.001, 0.01, 0.1, 1.0
3. Tracer la courbe de perte pour vérifier la convergence
4. Si la perte augmente → réduire α

## 💻 Implémentation Python avec Commentaires Détaillés

```python
import numpy as np

# ============================================
# FONCTIONS DE BASE
# ============================================

def sigmoid(z):
    """
    Fonction sigmoïde (logistique).

    Paramètres:
        z : ndarray de n'importe quelle forme

    Retourne:
        sigma(z) : ndarray de même forme, valeurs entre 0 et 1

    Points pour l'entrevue:
    - Pourquoi clip? Pour éviter overflow avec exp(-z) quand z est très négatif
    - exp(-500) ≈ 0, donc σ(500) ≈ 1
    - exp(500) → overflow, mais exp(-(-500)) = exp(500) → problème!
    - Clip z ∈ [-500, 500] garantit stabilité numérique
    """
    # Clip pour stabilité numérique
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(y_true, y_pred_proba):
    """
    Calcule la perte d'entropie croisée binaire.

    Paramètres:
        y_true : ndarray (n,) - étiquettes vraies (0 ou 1)
        y_pred_proba : ndarray (n,) - probabilités prédites P(Y=1)

    Retourne:
        loss : float - perte moyenne

    Points pour l'entrevue:
    - Pourquoi clip les probabilités? log(0) = -∞ → erreur numérique
    - eps = 1e-15 évite log(0) et log(1) exactement
    - Formule: -mean[ y·log(p) + (1-y)·log(1-p) ]
    """
    # Clip pour éviter log(0)
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)

    # Calcul de l'entropie croisée
    loss = -np.mean(
        y_true * np.log(y_pred_proba) +
        (1 - y_true) * np.log(1 - y_pred_proba)
    )

    return loss


def logistic_gradient(X, y, beta):
    """
    Calcule le gradient de la perte d'entropie croisée.

    Paramètres:
        X : ndarray (n, p+1) - caractéristiques AVEC colonne de 1
        y : ndarray (n,) - étiquettes binaires (0 ou 1)
        beta : ndarray (p+1,) - coefficients actuels

    Retourne:
        gradient : ndarray (p+1,) - gradient ∇L

    Formule: ∇L = (1/n) X^T (σ(Xβ) - y)

    Points pour l'entrevue:
    - Pourquoi cette formule simple? Grâce à la dérivée de σ!
    - Interprétation: gradient = moyenne des erreurs pondérées par les features
    - Si p_i > y_i (sur-prédiction) → gradient positif → diminuer β
    """
    n = len(y)

    # Prédictions: p = σ(Xβ)
    z = X @ beta  # Combinaison linéaire
    p = sigmoid(z)  # Probabilités

    # Erreur: p - y
    error = p - y

    # Gradient: (1/n) X^T (p - y)
    gradient = (1/n) * (X.T @ error)

    return gradient


# ============================================
# ENTRAÎNEMENT PAR DESCENTE DE GRADIENT
# ============================================

def logistic_fit_gd(X, y, lr=0.1, n_iter=1000, verbose=False):
    """
    Entraîne la régression logistique par descente de gradient.

    Paramètres:
        X : ndarray (n, p) - caractéristiques SANS colonne de 1
        y : ndarray (n,) - étiquettes binaires (0 ou 1)
        lr : float - taux d'apprentissage (learning rate)
        n_iter : int - nombre d'itérations
        verbose : bool - afficher progression tous les 100 iter

    Retourne:
        beta : ndarray (p+1,) - coefficients optimaux
        losses : list - historique des pertes (pour tracer convergence)

    Points pour l'entrevue:
    - Pourquoi initialiser β à 0? Simple et fonctionne bien (fonction convexe)
    - Alternative: initialisation aléatoire (peu d'impact ici)
    - Critère d'arrêt: nombre d'itérations fixe (pourrait être convergence)
    """
    n, p = X.shape

    # ÉTAPE 1: Ajouter colonne de 1 pour l'intercept
    X_with_intercept = np.column_stack([np.ones(n), X])
    # Shape devient (n, p+1)

    # ÉTAPE 2: Initialiser β à zéro
    beta = np.zeros(p + 1)

    # ÉTAPE 3: Historique des pertes (pour analyser convergence)
    losses = []

    # ÉTAPE 4: Boucle de descente de gradient
    for iteration in range(n_iter):
        # a. Calculer les probabilités actuelles
        z = X_with_intercept @ beta
        p = sigmoid(z)

        # b. Calculer la perte (pour monitoring)
        loss = cross_entropy_loss(y, p)
        losses.append(loss)

        # c. Calculer le gradient
        gradient = logistic_gradient(X_with_intercept, y, beta)

        # d. Mise à jour des paramètres
        beta = beta - lr * gradient
        # β_new = β_old - α·∇L

        # e. Affichage optionnel
        if verbose and (iteration % 100 == 0 or iteration == n_iter - 1):
            print(f"Itération {iteration:4d} | Loss: {loss:.6f}")

    return beta, losses


def logistic_predict_proba(X, beta):
    """
    Retourne les probabilités P(Y=1|X).

    Paramètres:
        X : ndarray (n, p) - caractéristiques SANS colonne de 1
        beta : ndarray (p+1,) - coefficients [intercept, coef1, ...]

    Retourne:
        proba : ndarray (n,) - probabilités entre 0 et 1

    Points pour l'entrevue:
    - Différence avec OLS: on retourne des probabilités, pas des valeurs continues
    - Pour classification: seuil = 0.5 → classe 1 si proba >= 0.5
    - Pourquoi retourner proba et non classe? Plus d'information!
      On peut ajuster le seuil selon le contexte (0.3, 0.5, 0.7...)
    """
    n = X.shape[0]

    # Ajouter colonne de 1
    X_with_intercept = np.column_stack([np.ones(n), X])

    # Calculer z = Xβ
    z = X_with_intercept @ beta

    # Appliquer sigmoïde pour obtenir probabilités
    proba = sigmoid(z)

    return proba


def logistic_predict_class(X, beta, threshold=0.5):
    """
    Retourne les classes prédites (0 ou 1).

    Paramètres:
        threshold : float - seuil de décision (par défaut 0.5)

    Points pour l'entrevue:
    - Pourquoi threshold=0.5? C'est la frontière naturelle (z=0)
    - Peut ajuster selon coût d'erreur:
      • Éviter faux négatifs → threshold = 0.3 (plus sensible)
      • Éviter faux positifs → threshold = 0.7 (plus conservateur)
    """
    proba = logistic_predict_proba(X, beta)
    return (proba >= threshold).astype(int)


# ============================================
# EXEMPLE COMPLET D'UTILISATION
# ============================================

# DONNÉES
# X_train: (8760, 4) - température, heure_sin, heure_cos, weekend
# y_train: (8760,) - événement de pointe (0 ou 1)

# IMPORTANT: Normaliser les features pour la descente de gradient!
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Pourquoi normaliser?
# - Features avec des échelles différentes (température: -20 à 30, humidité: 0 à 100)
# - Gradient descent converge plus vite avec features normalisées
# - Permet d'utiliser un learning rate plus élevé

# 1. ENTRAÎNEMENT
print("=== Entraînement Régression Logistique ===")
beta_log, losses = logistic_fit_gd(
    X_train_scaled,
    y_train,
    lr=0.1,        # Taux d'apprentissage
    n_iter=500,    # 500 itérations
    verbose=True   # Afficher progression
)

# beta_log shape: (5,)  →  [intercept, β_temp, β_hsin, β_hcos, β_weekend]

print(f"\nCoefficients appris:")
print(f"  Intercept (β₀): {beta_log[0]:.4f}")
features_names = ['température', 'heure_sin', 'heure_cos', 'weekend']
for i, name in enumerate(features_names):
    print(f"  β_{name}: {beta_log[i+1]:.4f}")

# INTERPRÉTATION (pour l'entrevue):
# Si β_temp < 0: températures élevées → moins de probabilité de pointe
# Si β_weekend < 0: weekend → moins de probabilité de pointe (consommation plus faible)


# 2. VISUALISER CONVERGENCE
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Itération')
plt.ylabel('Perte (Entropie Croisée)')
plt.title('Convergence de la Descente de Gradient')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Pour l'entrevue: expliquer la forme de la courbe
# - Descente rapide au début (loin du minimum)
# - Plateau ensuite (proche du minimum)
# - Si oscillations → learning rate trop grand


# 3. PRÉDICTIONS
proba_train = logistic_predict_proba(X_train_scaled, beta_log)
proba_test = logistic_predict_proba(X_test_scaled, beta_log)

# Classes (avec seuil 0.5)
y_pred_train = (proba_train >= 0.5).astype(int)
y_pred_test = (proba_test >= 0.5).astype(int)


# 4. ÉVALUATION
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n=== Évaluation ===")
print(f"Accuracy train: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Accuracy test: {accuracy_score(y_test, y_pred_test):.4f}")

print("\nRapport de classification:")
print(classification_report(y_test, y_pred_test,
                          target_names=['Normal', 'Pointe']))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_test)
print("\nMatrice de confusion:")
print(f"                Prédit Normal  Prédit Pointe")
print(f"Vrai Normal          {cm[0,0]:5d}          {cm[0,1]:5d}")
print(f"Vrai Pointe          {cm[1,0]:5d}          {cm[1,1]:5d}")


# 5. COMPARAISON AVEC SKLEARN
from sklearn.linear_model import LogisticRegression

model_sklearn = LogisticRegression()
model_sklearn.fit(X_train_scaled, y_train)

print("\n=== Comparaison avec sklearn ===")
print(f"Intercept - Vous: {beta_log[0]:.6f}")
print(f"Intercept - sklearn: {model_sklearn.intercept_[0]:.6f}")
print(f"Coefficients proches: {np.allclose(beta_log[1:], model_sklearn.coef_[0], atol=0.1)}")
# Note: sklearn utilise optimiseur différent (L-BFGS) → petites différences OK

```

## ⚠️ Points Critiques pour l'Entrevue

### 1. Pourquoi utiliser l'entropie croisée et non MSE?

**Question type:** "Pourquoi pas Mean Squared Error comme pour OLS?"

**Réponse:**

- Avec sigmoïde + MSE → **fonction non-convexe** (plusieurs minima locaux)
- Gradient descent peut rester bloqué dans minimum local ❌
- Entropie croisée + sigmoïde → **fonction convexe** (un seul minimum global) ✅
- Bonus: dérivation mathématique plus élégante (gradient = erreur × features)

### 2. Interpréter la sigmoïde

**Question type:** "Expliquez ce que fait la fonction sigmoïde."

**Réponse:**

- Transforme $z \in (-\infty, +\infty)$ en $p \in (0, 1)$
- Forme en "S" → transition douce entre 0 et 1
- $z = 0$ → $p = 0.5$ → frontière de décision
- Si $z$ très négatif → presque sûr classe 0
- Si $z$ très positif → presque sûr classe 1

**Au tableau:** Dessiner la courbe sigmoïde!

### 3. Pourquoi retourner des probabilités?

**Question type:** "Pourquoi prédire des probabilités plutôt que directement la classe?"

**Réponse:**

1. **Plus d'information**: P=0.51 vs P=0.99 → tous deux classe 1, mais confiance différente!
2. **Flexibilité du seuil**: Peut ajuster selon le contexte
   - Détection fraude: seuil = 0.3 (tolérer faux positifs)
   - Diagnostic médical: seuil = 0.7 (éviter faux positifs)
3. **Utilisable comme feature**: Dans Partie 4, on utilise P(pointe) comme variable!

### 4. Comment choisir le learning rate?

**Question type:** "Vous avez utilisé lr=0.1, pourquoi?"

**Réponse pratique:**

1. Normaliser d'abord les features (StandardScaler)
2. Tester [0.001, 0.01, 0.1, 1.0]
3. Tracer la courbe de perte:
   - Descend bien → bon choix ✅
   - Oscille/diverge → trop grand ❌
   - Plateau trop lent → trop petit ❌
4. Pour ce dataset normalisé: lr=0.1 converge en ~500 iter

### 5. Différences OLS vs Régression Logistique

**Question de synthèse importante!**

| Aspect          | OLS                           | Régression Logistique           |
| --------------- | ----------------------------- | ------------------------------- |
| **Type**        | Régression                    | Classification                  |
| **Cible**       | Continue ($y \in \mathbb{R}$) | Binaire ($y \in \{0,1\}$)       |
| **Prédiction**  | Valeur réelle                 | Probabilité                     |
| **Fonction**    | Linéaire $y = X\beta$         | Sigmoïde $p = \sigma(X\beta)$   |
| **Perte**       | MSE (erreur quadratique)      | Entropie croisée                |
| **Solution**    | Analytique: $(X^TX)^{-1}X^Ty$ | Itérative: descente de gradient |
| **Convergence** | Instantanée (1 calcul)        | Progressive (~500 iter)         |
| **Hypothèse**   | Erreurs normales              | Distribution de Bernoulli       |

## 🎯 Checklist pour l'Entrevue Régression Logistique

Pratiquez ces exercices:

- [ ] Dessiner la fonction sigmoïde au tableau
- [ ] Expliquer pourquoi σ'(z) = σ(z)(1-σ(z))
- [ ] Dériver le gradient de l'entropie croisée (5 étapes)
- [ ] Coder sigmoid() de mémoire
- [ ] Expliquer l'algorithme de descente de gradient en 30 secondes
- [ ] Interpréter un coefficient: "β₁ = -2.3 pour température signifie..."
- [ ] Tracer + commenter une courbe de convergence
- [ ] Expliquer quand utiliser seuil ≠ 0.5
- [ ] Différencier régression vs classification
- [ ] Expliquer pourquoi normaliser les features

---

## PARTIE 3: Ingénierie des Caractéristiques (Feature Engineering) ⭐

### 🎯 Pourquoi l'Ingénierie des Caractéristiques?

**Citation célèbre en ML:**

> "Les algorithmes viennent et vont, mais les features bien conçues restent." - Andrew Ng

**Réalité:**

- Un modèle simple (Ridge) avec **bonnes features** > modèle complexe (Deep Learning) avec features basiques
- **80% du travail en ML** = comprendre les données et créer de bonnes features
- **20% du travail** = choisir et optimiser l'algorithme

**Pour ce projet:**

- OLS/Ridge sont des modèles **linéaires** → pas très flexibles
- Solution: créer des features qui **capturent les patterns non-linéaires**
- Exemple: interaction température × heure capture "il fait froid la nuit"

## 📊 Contexte: Série Temporelle d'Énergie

### Caractéristiques des Données Énergétiques

**Patterns typiques:**

1. **Saisonnalité horaire**: Pointe le matin (7-9h) et soir (17-19h)
2. **Saisonnalité journalière**: Weekend < Semaine
3. **Saisonnalité mensuelle**: Hiver > Été (chauffage)
4. **Dépendance temporelle**: Consommation à t ≈ consommation à t-1
5. **Dépendance météo**: Froid → plus de chauffage

**Défi:**

- Train = hiver (haute consommation)
- Test = printemps/été (basse consommation)
- Modèle doit **généraliser à travers les saisons**!

## 🧰 Types de Caractéristiques à Créer

### 1. Retards (Lags) - Autocorrélation

**Idée:** La consommation passée aide à prédire la consommation future

**Exemples:**

```python
# Retard de 1 heure
df['energie_lag1'] = df['energie_kwh'].shift(1)

# Retard de 24 heures (même heure hier)
df['energie_lag24'] = df['energie_kwh'].shift(24)

# Retard de 168 heures (même heure la semaine dernière)
df['energie_lag168'] = df['energie_kwh'].shift(168)
```

**Intuition:**

- Si consommation à 8h hier = 150 kWh → probable aujourd'hui aussi
- Capture les **patterns qui se répètent** quotidiennement/hebdomadairement

**⚠️ ATTENTION - Fuite de données (Data Leakage):**

**MAUVAIS (fuite):**

```python
# NE PAS FAIRE: utiliser energie_lag1 pour prédire energie_kwh sur test_kaggle
# Problème: Sur Kaggle, on n'a PAS la consommation passée du test!
```

**BON (pas de fuite):**

```python
# OK pour validation locale (on a les vraies valeurs)
# Mais pour Kaggle, il faut soit:
# 1. Ne pas utiliser de lags
# 2. Prédire de façon autorégressive (t → t+1 → t+2)
```

**Pour ce projet:**

- Utilisez les lags pour **améliorer le modèle local** (train/test avec cible)
- Pour Kaggle: soit enlever les lags, soit prédire récursivement

### 2. Statistiques Glissantes (Rolling Statistics)

**Idée:** Moyennes/écarts-types sur une fenêtre temporelle

**Exemples:**

```python
# Moyenne mobile sur 6 heures
df['energie_rolling_mean_6h'] = df['energie_kwh'].rolling(window=6).mean()

# Écart-type mobile sur 24 heures
df['energie_rolling_std_24h'] = df['energie_kwh'].rolling(window=24).std()

# Min/Max sur les 12 dernières heures
df['energie_rolling_min_12h'] = df['energie_kwh'].rolling(window=12).min()
df['energie_rolling_max_12h'] = df['energie_kwh'].rolling(window=12).max()
```

**Intuition:**

- Moyenne mobile = lisse les fluctuations, capture la tendance
- Écart-type mobile = mesure la volatilité/stabilité de la consommation
- Min/Max = détecte les extrêmes récents

**Avantage vs Lags simples:**

- Moins sensible aux outliers ponctuels
- Capture des **tendances locales**

### 3. Interactions entre Variables

**Idée:** Combiner deux features pour capturer des effets conjoints

**Exemples:**

```python
# Température × heure (froid + nuit = beaucoup de chauffage)
df['temp_heure_interaction'] = df['temperature_ext'] * df['heure_cos']

# Température × weekend (comportement différent)
df['temp_weekend'] = df['temperature_ext'] * df['est_weekend']

# Température au carré (effet non-linéaire)
df['temp_squared'] = df['temperature_ext'] ** 2
```

**Intuition:**

- **Linéaire:** $y = \beta_1 \cdot temp + \beta_2 \cdot heure$ → effets séparés
- **Interaction:** $y = \beta_3 \cdot (temp \times heure)$ → effet conjoint!

**Exemple concret:**

- À 20°C, l'heure importe peu (pas de chauffage)
- À -10°C, l'heure importe beaucoup (chauffage la nuit)
- Interaction capture cette **dépendance conditionnelle**

### 4. Transformations Météorologiques

**Degré-jours de chauffage (Heating Degree Days):**

```python
# Si < 18°C, besoin de chauffage
df['degres_jours_chauffage'] = np.maximum(18 - df['temperature_ext'], 0)
```

**Intuition:**

- À 20°C: degré-jours = 0 (pas de chauffage)
- À 10°C: degré-jours = 8 (chauffage modéré)
- À -10°C: degré-jours = 28 (chauffage intense)
- Relation **plus linéaire** avec la consommation que température brute

**Ressentie (Wind Chill):**

```python
# Température ressentie avec le vent
df['temp_ressentie'] = df['temperature_ext'] - 0.5 * df['vitesse_vent']
```

**Humidex (chaleur ressentie):**

```python
# Pour l'été (climatisation)
df['humidex'] = df['temperature_ext'] + 0.5555 * (6.11 * np.exp(5417.7530 * (1/273.16 - 1/(273.15 + df['temperature_ext']))) - 10)
```

### 5. Variables Temporelles Avancées

**Indicateurs de périodes spécifiques:**

```python
# Heures de pointe matin/soir
df['est_pointe_matin'] = ((df['heure'] >= 7) & (df['heure'] <= 9)).astype(int)
df['est_pointe_soir'] = ((df['heure'] >= 17) & (df['heure'] <= 20)).astype(int)

# Saison
df['est_hiver'] = df['mois'].isin([12, 1, 2]).astype(int)
df['est_ete'] = df['mois'].isin([6, 7, 8]).astype(int)
```

**Distance au jour férié:**

```python
# Comportement change quelques jours avant/après les fêtes
# (Nécessite une liste de dates de fêtes)
```

## 💻 Implémentation Python Complète

```python
import numpy as np
import pandas as pd

def creer_caracteristiques(df):
    """
    Crée des caractéristiques supplémentaires pour améliorer la prédiction.

    IMPORTANT pour l'entrevue:
    - Expliquer POURQUOI chaque feature est utile
    - Comprendre quand utiliser lags (attention data leakage!)
    - Savoir interpréter les interactions

    Paramètres:
        df : DataFrame avec colonnes de base (température, heure, etc.)

    Retourne:
        df : DataFrame enrichi avec nouvelles features
    """
    df = df.copy()

    # ============================================
    # 1. RETARDS (LAGS) - Autocorrélation
    # ============================================

    # Lag 1: Consommation il y a 1 heure
    df['energie_lag1'] = df['energie_kwh'].shift(1)

    # Lag 24: Même heure hier (forte corrélation)
    df['energie_lag24'] = df['energie_kwh'].shift(24)

    # Lag 168: Même heure, même jour la semaine dernière
    df['energie_lag168'] = df['energie_kwh'].shift(168)

    # Points pour l'entrevue:
    # - Pourquoi lag24? Consommation à 8h aujourd'hui ≈ 8h hier
    # - Pourquoi lag168? Lundi 8h ≈ lundi précédent 8h
    # - Attention: Ces features créent des NaN au début!


    # ============================================
    # 2. STATISTIQUES GLISSANTES (ROLLING)
    # ============================================

    # Moyenne mobile 6h: Tendance court terme
    df['energie_rolling_mean_6h'] = df['energie_kwh'].rolling(
        window=6,
        min_periods=1  # Évite NaN si < 6 valeurs
    ).mean()

    # Moyenne mobile 24h: Tendance journalière
    df['energie_rolling_mean_24h'] = df['energie_kwh'].rolling(
        window=24,
        min_periods=1
    ).mean()

    # Écart-type mobile 24h: Mesure de volatilité
    df['energie_rolling_std_24h'] = df['energie_kwh'].rolling(
        window=24,
        min_periods=1
    ).std().fillna(0)  # Remplacer NaN par 0

    # Max sur 12h: Détecte pointes récentes
    df['energie_rolling_max_12h'] = df['energie_kwh'].rolling(
        window=12,
        min_periods=1
    ).max()

    # Points pour l'entrevue:
    # - Moyenne lisse le bruit, capture tendance
    # - Std mesure variabilité (stable vs instable)
    # - Max détecte si on sort d'une période de pointe


    # ============================================
    # 3. INTERACTIONS MÉTÉO × TEMPS
    # ============================================

    # Température × heure_cos: Capture "froid la nuit"
    df['temp_heure_cos'] = df['temperature_ext'] * df['heure_cos']

    # Température × heure_sin
    df['temp_heure_sin'] = df['temperature_ext'] * df['heure_sin']

    # Température × weekend: Comportement différent
    df['temp_weekend'] = df['temperature_ext'] * df['est_weekend']

    # Température × mois: Capture saisonnalité
    df['temp_mois_sin'] = df['temperature_ext'] * df['mois_sin']
    df['temp_mois_cos'] = df['temperature_ext'] * df['mois_cos']

    # Points pour l'entrevue:
    # - Pourquoi interaction? Effet de température dépend de l'heure!
    # - Exemple: -10°C à 3h du matin → très haute consommation (chauffage)
    #            -10°C à 14h → moins élevée (soleil, activité)


    # ============================================
    # 4. TRANSFORMATIONS MÉTÉO
    # ============================================

    # Degré-jours de chauffage (seuil 18°C)
    df['degres_jours_chauffage'] = np.maximum(18 - df['temperature_ext'], 0)

    # Degré-jours de climatisation (seuil 22°C)
    df['degres_jours_clim'] = np.maximum(df['temperature_ext'] - 22, 0)

    # Température au carré (non-linéarité)
    df['temp_squared'] = df['temperature_ext'] ** 2

    # Température ressentie avec vent (wind chill simplifié)
    df['temp_ressentie'] = df['temperature_ext'] - 0.5 * df['vitesse_vent']

    # Humidité relative ajustée
    df['humidite_temp'] = df['humidite'] * np.abs(df['temperature_ext']) / 100

    # Points pour l'entrevue:
    # - Degré-jours: relation plus linéaire avec consommation
    # - Temp²: capture accélération de consommation aux extrêmes
    # - Ressentie: le vent augmente la sensation de froid


    # ============================================
    # 5. VARIABLES TEMPORELLES AVANCÉES
    # ============================================

    # Indicateur heures de pointe matin
    df['est_pointe_matin'] = ((df['heure'] >= 7) & (df['heure'] <= 9)).astype(int)

    # Indicateur heures de pointe soir
    df['est_pointe_soir'] = ((df['heure'] >= 17) & (df['heure'] <= 20)).astype(int)

    # Nuit (consommation basse)
    df['est_nuit'] = ((df['heure'] >= 0) & (df['heure'] <= 6)).astype(int)

    # Hiver (haute consommation)
    df['est_hiver'] = df['mois'].isin([12, 1, 2]).astype(int)

    # Été (basse consommation, climatisation possible)
    df['est_ete'] = df['mois'].isin([6, 7, 8]).astype(int)

    # Points pour l'entrevue:
    # - Binning temporel: simplifie les patterns
    # - Capture des "régimes" différents


    # ============================================
    # 6. STATISTIQUES MÉTÉO GLISSANTES
    # ============================================

    # Température moyenne des 3 dernières heures
    df['temp_rolling_mean_3h'] = df['temperature_ext'].rolling(
        window=3,
        min_periods=1
    ).mean()

    # Changement de température (gradient)
    df['temp_diff'] = df['temperature_ext'].diff().fillna(0)

    # Température min/max sur 24h (amplitude thermique)
    df['temp_amplitude_24h'] = (
        df['temperature_ext'].rolling(window=24, min_periods=1).max() -
        df['temperature_ext'].rolling(window=24, min_periods=1).min()
    )

    # Points pour l'entrevue:
    # - Gradient température: chute rapide → plus de chauffage
    # - Amplitude: grande variation → plus énergivore


    # ============================================
    # 7. NOMBRE DE CLIENTS (TRÈS IMPORTANT!)
    # ============================================

    # Si la colonne existe, créer des interactions
    if 'clients_connectes' in df.columns:
        # Clients × température
        df['clients_temp'] = df['clients_connectes'] * df['temperature_ext']

        # Consommation par client (normalisée)
        df['energie_per_client'] = df['energie_kwh'] / (df['clients_connectes'] + 1)

        # Clients × weekend
        df['clients_weekend'] = df['clients_connectes'] * df['est_weekend']

    # Points pour l'entrevue:
    # - clients_connectes est LA variable la plus prédictive!
    # - Plus de clients → plus de consommation (quasi linéaire)
    # - Interactions capturent comportements par client


    # ============================================
    # NETTOYAGE FINAL
    # ============================================

    # IMPORTANT: Les lags et rolling créent des NaN au début
    # Options:
    # 1. Supprimer les lignes avec NaN: df.dropna()
    # 2. Remplir avec 0 ou moyenne: df.fillna(0) ou df.fillna(df.mean())
    # 3. Forward fill: df.fillna(method='ffill')

    # Pour ce projet, on va supprimer (plus sûr)
    # Note: On perd les premières heures de train, mais c'est OK

    return df


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

# Application aux données
train_enrichi = creer_caracteristiques(train)
test_enrichi = creer_caracteristiques(test)

# Supprimer les NaN (dus aux lags/rolling)
train_enrichi = train_enrichi.dropna()
test_enrichi = test_enrichi.dropna()

# Vérifier les nouvelles colonnes
nouvelles_cols = [c for c in train_enrichi.columns if c not in train.columns]
print(f"Nombre de nouvelles features: {len(nouvelles_cols)}")
print(f"\nNouvelles features créées:")
for col in nouvelles_cols:
    print(f"  - {col}")

# Vérifier corrélations avec la cible
correlations = train_enrichi[nouvelles_cols + ['energie_kwh']].corr()['energie_kwh'].sort_values(ascending=False)
print(f"\nTop 10 features par corrélation avec energie_kwh:")
print(correlations.head(10))


# ============================================
# SÉLECTION DES FEATURES POUR LE MODÈLE
# ============================================

# OPTION 1: Prendre toutes les features numériques
features_to_use = [col for col in train_enrichi.columns
                   if col not in ['energie_kwh', 'horodatage_local', 'evenement_pointe']]

# OPTION 2: Sélection manuelle (recommandé pour l'entrevue)
features_to_use = [
    # Météo de base
    'temperature_ext', 'humidite', 'vitesse_vent', 'irradiance_solaire',

    # Temps cyclique
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'jour_semaine_sin', 'jour_semaine_cos',

    # Indicateurs binaires
    'est_weekend', 'est_ferie', 'est_pointe_matin', 'est_pointe_soir',

    # TRÈS IMPORTANT
    'clients_connectes',

    # Lags (attention Kaggle!)
    'energie_lag1', 'energie_lag24',

    # Rolling
    'energie_rolling_mean_6h', 'energie_rolling_mean_24h',

    # Interactions
    'temp_heure_cos', 'temp_weekend',

    # Transformations météo
    'degres_jours_chauffage', 'temp_squared'
]

# Filtrer celles qui existent vraiment
features_disponibles = [f for f in features_to_use if f in train_enrichi.columns]

print(f"\nFeatures sélectionnées: {len(features_disponibles)}")

X_train = train_enrichi[features_disponibles].values
y_train = train_enrichi['energie_kwh'].values
X_test = test_enrichi[features_disponibles].values
y_test = test_enrichi['energie_kwh'].values

# Entraîner un modèle simple pour tester
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

from sklearn.metrics import r2_score
print(f"\nR² avec features enrichies: {r2_score(y_test, model.predict(X_test)):.4f}")
```

## ⚠️ Points Critiques pour l'Entrevue

### 1. Data Leakage avec les Lags

**Question type:** "Vous utilisez energie_lag1, mais comment faire sur Kaggle sans la vraie valeur?"

**Réponse:**
Deux approches:

**Option A: Ne pas utiliser de lags pour Kaggle**

```python
# Features pour train/test local (avec lags)
features_local = [..., 'energie_lag1', 'energie_lag24', ...]

# Features pour Kaggle (sans lags)
features_kaggle = [f for f in features_local if 'lag' not in f]
```

**Option B: Prédiction autorégressive**

```python
# Prédire heure par heure en utilisant prédictions précédentes
predictions = []
for t in range(len(test_kaggle)):
    # Utiliser la prédiction de t-1 comme lag pour prédire t
    X[t, lag1_idx] = predictions[t-1] if t > 0 else last_train_value
    pred = model.predict(X[t])
    predictions.append(pred)
```

**Pour l'entrevue:** Reconnaître le problème montre que vous comprenez!

### 2. Pourquoi créer des Interactions?

**Question type:** "Pourquoi multiplier température et heure au lieu de les utiliser séparément?"

**Réponse:**
Modèle linéaire **sans interaction:**
$$y = \beta_1 \cdot temp + \beta_2 \cdot heure$$
→ L'effet de température est **constant** quelle que soit l'heure

Modèle **avec interaction:**
$$y = \beta_1 \cdot temp + \beta_2 \cdot heure + \beta_3 \cdot (temp \times heure)$$
→ L'effet de température **dépend** de l'heure!

**Exemple concret:**

- Hiver, 3h du matin, -15°C → Chauffage maximal
- Hiver, 14h, -15°C → Chauffage moyen (soleil aide)
- L'interaction capture cette dépendance!

### 3. Décalage Train/Test (Distribution Shift)

**Question type:** "Vos features aident-elles malgré le décalage hiver/été?"

**Réponse:**
Le problème:

- Train = hiver (consommation élevée, chauffage)
- Test = été (consommation basse, climatisation)

**Features qui généralisent MAL:**

- Lags simples (valeurs absolues changent beaucoup)
- Moyennes mobiles (idem)

**Features qui généralisent BIEN:**

- `degres_jours_chauffage`: Relation physique stable
- Interactions météo × temps: Patterns comportementaux persistants
- `clients_connectes`: Normalise la consommation
- Features cycliques (heure_sin/cos): Patterns horaires similaires

**Stratégie:** Privilégier features basées sur **lois physiques** ou **comportements** plutôt que valeurs brutes

### 4. Importance Relative des Features

**Question type:** "Quelle feature a le plus amélioré le modèle?"

**Réponse basée sur l'expérience typique:**

1. **`clients_connectes`** ⭐⭐⭐⭐⭐ (+30% R²)
   - Plus de clients = plus de consommation (quasi linéaire)
2. **`degres_jours_chauffage`** ⭐⭐⭐⭐ (+15% R²)
   - Meilleure que température brute
3. **Lags (lag24, lag168)** ⭐⭐⭐ (+10% R² local)
   - Forte autocorrélation, mais attention leakage Kaggle!
4. **Interactions météo × temps** ⭐⭐ (+5% R²)
   - Capture effets conditionnels
5. **Rolling statistics** ⭐ (+2-3% R²)
   - Lissent le bruit

**Pour l'entrevue:** Connaître l'ordre d'importance montre que vous avez testé!

### 5. Combien de Features Créer?

**Question type:** "Vous avez créé 30 features, est-ce trop?"

**Réponse nuancée:**

**Avantages de beaucoup de features:**

- ✅ Plus d'information pour le modèle
- ✅ Ridge va automatiquement réduire les coefficients peu utiles

**Inconvénients:**

- ❌ Overfitting possible (même avec Ridge)
- ❌ Temps de calcul
- ❌ Plus difficile à interpréter

**Bonne pratique:**

1. Créer beaucoup de features (exploration)
2. Analyser les coefficients Ridge
3. Garder seulement les features importantes
4. **Pour l'entrevue:** Justifier chaque feature gardée!

**Règle empirique:** 10-20 features bien choisies > 100 features aléatoires

## 🎯 Checklist pour l'Entrevue Feature Engineering

- [ ] Expliquer 3 types de features créées (lags, rolling, interactions)
- [ ] Justifier POURQUOI chaque feature est utile (pas juste "j'ai essayé")
- [ ] Identifier le risque de data leakage avec les lags
- [ ] Expliquer pourquoi `degres_jours_chauffage` > `temperature_ext`
- [ ] Donner un exemple concret d'interaction (température × heure)
- [ ] Expliquer comment gérer les NaN des lags/rolling
- [ ] Identifier quelle feature a le plus amélioré R²
- [ ] Expliquer stratégie pour généraliser train (hiver) → test (été)
- [ ] Dessiner graphique: corrélation features avec cible
- [ ] Défendre le nombre de features créées

## 💡 Conseils pour l'Entrevue

1. **Ne pas juste lister les features** → Expliquer le RAISONNEMENT
   - ❌ "J'ai ajouté energie_lag24"
   - ✅ "J'ai ajouté energie_lag24 car la consommation à 8h aujourd'hui ressemble à celle de 8h hier, ce qui capture les habitudes quotidiennes stables"

2. **Préparer des graphiques** montrant l'impact
   - Courbe: R² avec 0, 5, 10, 20 features
   - Heatmap: Corrélations entre features

3. **Anticiper les questions sur le décalage distribution**
   - "Comment votre modèle peut-il prédire l'été s'il n'a vu que l'hiver?"
   - Réponse: Features basées sur comportements/physique, pas valeurs brutes

4. **Savoir quoi enlever si demandé**
   - "Si vous deviez garder seulement 5 features, lesquelles?"
   - Réponse: clients_connectes, degres_jours_chauffage, heure_sin/cos, mois_sin/cos

---

## PARTIE 4: Régression Ridge - MAÎTRISER POUR L'ENTREVUE! ⭐

### 🎯 Le Problème avec OLS

Après avoir créé plein de features (Partie 3), vous avez maintenant:

- **Beaucoup de variables** (20-30 features)
- **Certaines corrélées** entre elles (température et degré-jours, lag1 et lag24)
- Risque d'**overfitting** (modèle trop complexe)

**Symptômes d'overfitting avec OLS:**

- R² train = 0.95, R² test = 0.60 (grand écart!)
- Coefficients très grands en valeur absolue (ex: β₁ = +5000, β₂ = -4998)
- Coefficients instables (changent beaucoup si on ajoute/retire 1 observation)

## 📐 Théorie Mathématique Complète

### Rappel: OLS minimise uniquement l'erreur

$$\hat{\boldsymbol{\beta}}_{OLS} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2$$

**Problème:** Aucune contrainte sur la taille des coefficients!

### Ridge ajoute une Pénalité L2

$$\hat{\boldsymbol{\beta}}_{Ridge} = \arg\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right]$$

Ou en notation matricielle:

$$\hat{\boldsymbol{\beta}}_{Ridge} = \arg\min_{\boldsymbol{\beta}} \left[ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2 \right]$$

**Composantes:**

- $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$ : **Erreur d'ajustement** (comme OLS)
- $\lambda \|\boldsymbol{\beta}\|^2$ : **Pénalité de régularisation** (nouveau!)
- $\lambda \geq 0$ : **Hyperparamètre** contrôlant l'équilibre

### Interprétation du Paramètre λ

**λ = 0:**

- Pas de pénalité
- Ridge = OLS exactement

**λ très petit (ex: 0.01):**

- Pénalité faible
- Ridge ≈ OLS (peu de régularisation)

**λ modéré (ex: 1-100):**

- Équilibre entre ajustement et simplicité
- **Zone optimale généralement**

**λ très grand (ex: 10000):**

- Pénalité dominante
- Tous les coefficients → 0
- Modèle → prédiction par la moyenne

### Solution Analytique

**Dérivation (important pour l'entrevue!):**

Fonction objectif:
$$L(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \lambda \boldsymbol{\beta}^T\boldsymbol{\beta}$$

Développer (comme pour OLS):
$$L(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + \lambda \boldsymbol{\beta}^T\boldsymbol{\beta}$$

Gradient:
$$\nabla_{\boldsymbol{\beta}} L = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + 2\lambda\boldsymbol{\beta}$$

Égaler à zéro:
$$-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + 2\lambda\boldsymbol{\beta} = 0$$

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + \lambda\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

$$(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

**Solution Ridge:**

$$\boxed{\hat{\boldsymbol{\beta}}_{Ridge} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}}$$

**Comparer avec OLS:**

- OLS: $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
- Ridge: $(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
- **Différence:** $+ \lambda \mathbf{I}$ sur la diagonal!

### Avantage: Garantit l'Inversibilité

**Problème avec OLS:**
Si $\mathbf{X}^T\mathbf{X}$ n'est pas inversible (multicolinéarité parfaite):

- Pas de solution unique
- `np.linalg.solve` échoue ou donne résultat instable

**Solution Ridge:**
$\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I}$ est **toujours inversible** si $\lambda > 0$!

- Ajouter $\lambda$ sur la diagonale "stabilise" la matrice
- Solution existe toujours et est unique

## 🔗 Interprétation Bayésienne (Estimation MAP)

**Question clé d'entrevue:** "Quel est le lien entre Ridge et l'estimation MAP?"

### Rappel: Maximum A Posteriori (MAP)

En statistique bayésienne, on cherche:
$$\hat{\boldsymbol{\beta}}_{MAP} = \arg\max_{\boldsymbol{\beta}} P(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X})$$

Par Bayes:
$$P(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) = \frac{P(\mathbf{y} | \mathbf{X}, \boldsymbol{\beta}) \cdot P(\boldsymbol{\beta})}{P(\mathbf{y})}$$

Log-vraisemblance:
$$\log P(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) = \log P(\mathbf{y} | \mathbf{X}, \boldsymbol{\beta}) + \log P(\boldsymbol{\beta}) + \text{cste}$$

### Hypothèses Bayésiennes

**Vraisemblance (likelihood):**
Erreurs gaussiennes: $y_i = \mathbf{x}_i^T\boldsymbol{\beta} + \epsilon_i$, où $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

$$P(\mathbf{y} | \mathbf{X}, \boldsymbol{\beta}) \propto \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2\right)$$

**Prior (a priori):**
Distribution gaussienne centrée: $\boldsymbol{\beta} \sim \mathcal{N}(0, \tau^2\mathbf{I})$

$$P(\boldsymbol{\beta}) \propto \exp\left(-\frac{1}{2\tau^2}\|\boldsymbol{\beta}\|^2\right)$$

**Signification:** On croit a priori que les coefficients sont petits (proches de 0)

### Dérivation MAP = Ridge

$$\log P(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) \propto -\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 - \frac{1}{2\tau^2}\|\boldsymbol{\beta}\|^2$$

Maximiser log-posterior = Minimiser son opposé:

$$\arg\min_{\boldsymbol{\beta}} \left[\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \frac{\sigma^2}{\tau^2}\|\boldsymbol{\beta}\|^2\right]$$

En posant $\lambda = \frac{\sigma^2}{\tau^2}$:

$$\boxed{\hat{\boldsymbol{\beta}}_{MAP} = \hat{\boldsymbol{\beta}}_{Ridge}}$$

**Interprétation Ridge Bayésienne:**

- Ridge = MAP avec prior gaussien centré
- $\lambda$ grand → prior fort (on croit fort que β ≈ 0)
- $\lambda$ petit → prior faible (on fait plus confiance aux données)

**Pour l'entrevue:** Ridge n'est pas juste une "astuce", c'est une estimation bayésienne rigoureuse!

## 📊 Effet sur les Coefficients

### Réduction (Shrinkage) des Coefficients

**Propriété fondamentale:** Ridge **réduit** tous les coefficients vers 0, mais ne les met jamais exactement à 0

**Visualisation:**

```
Coefficient OLS:  |--------●-----------------|  β = 100
Coefficient Ridge:|-----●--------------------|  β = 60
                  0                        150

λ = 0    → β = 100 (OLS)
λ = 1    → β = 60
λ = 10   → β = 25
λ = 100  → β = 5
λ = ∞    → β = 0
```

### Comparaison OLS vs Ridge

| Feature         | Coefficient OLS | Coefficient Ridge (λ=10) | Réduction |
| --------------- | --------------- | ------------------------ | --------- |
| temperature_ext | -8.5            | -6.2                     | 27%       |
| energie_lag1    | 0.85            | 0.55                     | 35%       |
| energie_lag24   | 0.78            | 0.52                     | 33%       |
| temp_heure_cos  | 12.3            | 3.1                      | 75% ⬇️    |
| vitesse_vent    | -0.3            | -0.2                     | 33%       |

**Observation:** Ridge réduit **surtout** les coefficients des features:

- Corrélées avec d'autres (lag1 et lag24)
- Moins importantes (vitesse_vent)
- Instables (interactions)

### Biais-Variance Tradeoff

**OLS (λ = 0):**

- ✅ Pas de biais (estimateur non biaisé)
- ❌ Haute variance (coefficients instables)
- Résultat: **Overfitting** possible

**Ridge (λ > 0):**

- ❌ Légèrement biaisé (coefficients réduits)
- ✅ Basse variance (coefficients stables)
- Résultat: **Meilleure généralisation** sur test!

**Formule mathématique:**
$$\text{Erreur totale} = \text{Biais}^2 + \text{Variance} + \text{Bruit irréductible}$$

Ridge augmente légèrement le biais, mais **réduit beaucoup** la variance → **erreur totale plus faible**!

## 🔍 Choix de λ par Validation Croisée

### Problème

Comment choisir λ optimal? Tester manuellement?

**Mauvaise approche:**

```python
# ❌ NE PAS FAIRE
model = Ridge(alpha=1.0)  # Pourquoi 1.0? Au hasard?
```

**Bonne approche:** Validation croisée pour sélectionner automatiquement!

### Time Series Cross-Validation

**ATTENTION:** Pour séries temporelles, **PAS** de validation croisée aléatoire!

**Mauvais (K-Fold classique):**

```
Train: [████  ████  ████]
Test:  [  ████  ████  ]
```

→ **Fuite d'information:** On utilise le futur pour prédire le passé!

**Bon (TimeSeriesSplit):**

```
Fold 1: Train [████]           Test [█]
Fold 2: Train [████████]       Test [█]
Fold 3: Train [████████████]   Test [█]
Fold 4: Train [████████████████] Test [█]
```

→ **Respecte la chronologie:** Toujours prédire le futur avec le passé

### Implémentation avec RidgeCV

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

# Valeurs de λ à tester (échelle logarithmique)
alphas = [0.01, 0.1, 1, 10, 100, 1000]

# Validation croisée temporelle
tscv = TimeSeriesSplit(n_splits=5)

# RidgeCV teste tous les alphas et sélectionne le meilleur
model_ridge = RidgeCV(alphas=alphas, cv=tscv)
model_ridge.fit(X_train, y_train)

# Meilleur λ trouvé
print(f"λ optimal: {model_ridge.alpha_}")
```

**Points pour l'entrevue:**

1. Pourquoi échelle logarithmique? λ varie sur plusieurs ordres de grandeur (0.01 → 1000)
2. Pourquoi TimeSeriesSplit? Respect chronologie des données
3. Comment RidgeCV choisit? Minimise erreur de validation croisée

## 💻 Implémentation Python Complète

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ============================================
# DONNÉES (après feature engineering)
# ============================================

# Supposons qu'on a 25 features après Partie 3
# X_train: (8000, 25)
# y_train: (8000,)
# X_test: (2000, 25)
# y_test: (2000,)

features_disponibles = [
    'temperature_ext', 'humidite', 'vitesse_vent',
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'est_weekend', 'clients_connectes',
    'energie_lag1', 'energie_lag24',
    'energie_rolling_mean_6h', 'degres_jours_chauffage',
    'temp_heure_cos', 'temp_weekend',
    # ... autres features
]

X_train = train_eng[features_disponibles].values
y_train = train_eng['energie_kwh'].values
X_test = test_eng[features_disponibles].values
y_test = test_eng['energie_kwh'].values

# ============================================
# BASELINE: OLS
# ============================================

print("=" * 60)
print("BASELINE: OLS (Ordinary Least Squares)")
print("=" * 60)

model_ols = LinearRegression()
model_ols.fit(X_train, y_train)

y_pred_ols_train = model_ols.predict(X_train)
y_pred_ols_test = model_ols.predict(X_test)

r2_ols_train = r2_score(y_train, y_pred_ols_train)
r2_ols_test = r2_score(y_test, y_pred_ols_test)
rmse_ols_test = np.sqrt(mean_squared_error(y_test, y_pred_ols_test))

print(f"R² train: {r2_ols_train:.4f}")
print(f"R² test:  {r2_ols_test:.4f}")
print(f"RMSE test: {rmse_ols_test:.2f} kWh")
print(f"Écart train-test: {abs(r2_ols_train - r2_ols_test):.4f}")

# Diagnostique overfitting
if r2_ols_train - r2_ols_test > 0.1:
    print("⚠️  OVERFITTING détecté! (écart train-test > 0.1)")


# ============================================
# RIDGE AVEC λ FIXE
# ============================================

print("\n" + "=" * 60)
print("RIDGE avec λ = 1.0")
print("=" * 60)

model_ridge_fixed = Ridge(alpha=1.0)
model_ridge_fixed.fit(X_train, y_train)

y_pred_ridge_train = model_ridge_fixed.predict(X_train)
y_pred_ridge_test = model_ridge_fixed.predict(X_test)

r2_ridge_train = r2_score(y_train, y_pred_ridge_train)
r2_ridge_test = r2_score(y_test, y_pred_ridge_test)
rmse_ridge_test = np.sqrt(mean_squared_error(y_test, y_pred_ridge_test))

print(f"R² train: {r2_ridge_train:.4f}")
print(f"R² test:  {r2_ridge_test:.4f}")
print(f"RMSE test: {rmse_ridge_test:.2f} kWh")
print(f"Écart train-test: {abs(r2_ridge_train - r2_ridge_test):.4f}")


# ============================================
# RIDGE avec VALIDATION CROISÉE (OPTIMAL)
# ============================================

print("\n" + "=" * 60)
print("RIDGE avec RidgeCV (sélection automatique de λ)")
print("=" * 60)

# Valeurs de λ à tester (échelle log)
alphas = [0.01, 0.1, 1, 10, 100, 1000]

# Time Series Cross-Validation (CRUCIAL pour séries temporelles!)
tscv = TimeSeriesSplit(n_splits=5)

# RidgeCV teste tous les alphas
model_ridge_cv = RidgeCV(alphas=alphas, cv=tscv)
model_ridge_cv.fit(X_train, y_train)

print(f"λ optimal trouvé: {model_ridge_cv.alpha_}")

y_pred_ridgecv_train = model_ridge_cv.predict(X_train)
y_pred_ridgecv_test = model_ridge_cv.predict(X_test)

r2_ridgecv_train = r2_score(y_train, y_pred_ridgecv_train)
r2_ridgecv_test = r2_score(y_test, y_pred_ridgecv_test)
rmse_ridgecv_test = np.sqrt(mean_squared_error(y_test, y_pred_ridgecv_test))

print(f"R² train: {r2_ridgecv_train:.4f}")
print(f"R² test:  {r2_ridgecv_test:.4f}")
print(f"RMSE test: {rmse_ridgecv_test:.2f} kWh")
print(f"Écart train-test: {abs(r2_ridgecv_train - r2_ridgecv_test):.4f}")


# ============================================
# RÉCAPITULATIF COMPARATIF
# ============================================

print("\n" + "=" * 60)
print("RÉCAPITULATIF")
print("=" * 60)

results = pd.DataFrame({
    'Modèle': ['OLS', 'Ridge (λ=1)', f'Ridge (λ={model_ridge_cv.alpha_})'],
    'R² train': [r2_ols_train, r2_ridge_train, r2_ridgecv_train],
    'R² test': [r2_ols_test, r2_ridge_test, r2_ridgecv_test],
    'RMSE test': [rmse_ols_test, rmse_ridge_test, rmse_ridgecv_test],
    'Écart': [abs(r2_ols_train - r2_ols_test),
              abs(r2_ridge_train - r2_ridge_test),
              abs(r2_ridgecv_train - r2_ridgecv_test)]
})

print(results.to_string(index=False))

# Meilleur modèle
best_idx = results['R² test'].idxmax()
print(f"\n🏆 Meilleur modèle: {results.loc[best_idx, 'Modèle']}")


# ============================================
# ANALYSE DES COEFFICIENTS
# ============================================

print("\n" + "=" * 60)
print("COMPARAISON DES COEFFICIENTS OLS vs RIDGE")
print("=" * 60)

# Comparer coefficients
coef_comparison = pd.DataFrame({
    'Feature': features_disponibles,
    'OLS': model_ols.coef_,
    'Ridge': model_ridge_cv.coef_
})

# Calculer réduction (shrinkage)
coef_comparison['Réduction (%)'] = 100 * (
    1 - np.abs(coef_comparison['Ridge']) / (np.abs(coef_comparison['OLS']) + 1e-8)
)

# Trier par réduction
coef_comparison = coef_comparison.sort_values('Réduction (%)', ascending=False)

print(coef_comparison.to_string(index=False))

print("\n📊 Observations:")
print(f"  - Réduction moyenne: {coef_comparison['Réduction (%)'].mean():.1f}%")
print(f"  - Réduction max: {coef_comparison['Réduction (%)'].max():.1f}%")
print(f"  - Feature la plus réduite: {coef_comparison.iloc[0]['Feature']}")


# ============================================
# VISUALISATION: CHEMIN DE RÉGULARISATION
# ============================================

print("\n" + "=" * 60)
print("VISUALISATION: Effet de λ sur les coefficients")
print("=" * 60)

# Tester plusieurs λ
lambdas_test = np.logspace(-2, 4, 50)  # 0.01 à 10000
coefficients_path = []

for lam in lambdas_test:
    model_temp = Ridge(alpha=lam)
    model_temp.fit(X_train, y_train)
    coefficients_path.append(model_temp.coef_)

coefficients_path = np.array(coefficients_path)

# Tracer
plt.figure(figsize=(12, 6))
for i, feature in enumerate(features_disponibles[:10]):  # 10 premières features
    plt.plot(lambdas_test, coefficients_path[:, i], label=feature, linewidth=2)

plt.xscale('log')
plt.xlabel('λ (échelle log)', fontsize=12)
plt.ylabel('Coefficient', fontsize=12)
plt.title('Chemin de Régularisation Ridge', fontsize=14, fontweight='bold')
plt.axvline(model_ridge_cv.alpha_, color='red', linestyle='--',
            linewidth=2, label=f'λ optimal = {model_ridge_cv.alpha_}')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()

# Points pour l'entrevue:
# - Tous les coefficients → 0 quand λ → ∞
# - Mais jamais exactement 0 (différence avec Lasso)
# - Coefficients instables (corrélés) réduits plus vite


# ============================================
# COURBE DE VALIDATION
# ============================================

print("\n" + "=" * 60)
print("COURBE DE VALIDATION: R² vs λ")
print("=" * 60)

r2_train_list = []
r2_test_list = []

for lam in lambdas_test:
    model_temp = Ridge(alpha=lam)
    model_temp.fit(X_train, y_train)

    r2_train_list.append(r2_score(y_train, model_temp.predict(X_train)))
    r2_test_list.append(r2_score(y_test, model_temp.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(lambdas_test, r2_train_list, label='R² train', linewidth=2, color='blue')
plt.plot(lambdas_test, r2_test_list, label='R² test', linewidth=2, color='orange')
plt.axvline(model_ridge_cv.alpha_, color='red', linestyle='--',
            linewidth=2, label=f'λ optimal = {model_ridge_cv.alpha_}')

plt.xscale('log')
plt.xlabel('λ (échelle log)', fontsize=12)
plt.ylabel('R²', fontsize=12)
plt.title('Courbe de Validation Ridge', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Interprétation pour l'entrevue:
# - λ petit → R² train élevé, R² test bas (overfitting)
# - λ optimal → Meilleur compromis
# - λ grand → R² train et test bas (underfitting)
```

## ⚠️ Points Critiques pour l'Entrevue

### 1. Quelle est la différence entre Ridge et OLS?

**Réponse structurée:**

**Mathématiquement:**

- OLS: $\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$
- Ridge: $\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$

**Conceptuellement:**

- OLS: Minimise uniquement l'erreur
- Ridge: Minimise erreur + complexité (taille des coefficients)

**En pratique:**

- OLS: Peut overfitter avec beaucoup de features
- Ridge: Réduit overfitting en "shrinking" coefficients

### 2. Pourquoi Ridge plutôt que Lasso?

**Question piège!** Connaître les différences:

| Aspect                 | Ridge (L2)                  | Lasso (L1)                  |
| ---------------------- | --------------------------- | --------------------------- |
| **Pénalité**           | $\lambda \sum \beta_j^2$    | $\lambda \sum \|\beta_j\|$  |
| **Coefficients**       | Réduits vers 0              | Certains **exactement** 0   |
| **Sélection**          | Non                         | Oui (sélection automatique) |
| **Features corrélées** | Garde toutes, partage poids | Choisit arbitrairement 1    |
| **Solution**           | Analytique                  | Itérative (pas de formule)  |

**Pour ce projet:**

- Ridge préférable car on veut **garder toutes les features** (même corrélées)
- Lasso éliminerait arbitrairement lag1 ou lag24

### 3. Expliquez Ridge comme estimation MAP

**Réponse complète:**

Ridge = Maximum A Posteriori avec prior gaussien:

**Prior:** $\boldsymbol{\beta} \sim \mathcal{N}(0, \tau^2\mathbf{I})$
→ "Je crois a priori que les coefficients sont petits"

**Posterior:** $P(\boldsymbol{\beta}|\mathbf{y}) \propto P(\mathbf{y}|\boldsymbol{\beta}) \cdot P(\boldsymbol{\beta})$

**MAP:** $\arg\max P(\boldsymbol{\beta}|\mathbf{y}) = \arg\min [-\log P(\mathbf{y}|\boldsymbol{\beta}) - \log P(\boldsymbol{\beta})]$

Ce qui donne exactement la formule Ridge avec $\lambda = \frac{\sigma^2}{\tau^2}$

**Interprétation λ:**

- λ grand → Prior fort (on croit vraiment que β ≈ 0)
- λ petit → Prior faible (on fait confiance aux données)

### 4. Pourquoi utiliser TimeSeriesSplit?

**Question type:** "Pourquoi pas KFold classique pour choisir λ?"

**Réponse:**

**Problème avec KFold aléatoire:**

```
Train: [Jan Feb █ Apr █ Jun Jul █]
Test:  [█ █ Mar █ May █ █ Aug]
```

→ On utilise le futur (Juin) pour prédire le passé (Mars)!
→ **Fuite d'information temporelle** → λ sous-optimal

**TimeSeriesSplit respecte chronologie:**

```
Fold 1: Train [Jan Feb Mar] Test [Apr]
Fold 2: Train [Jan Feb Mar Apr May] Test [Jun]
Fold 3: Train [Jan Feb Mar Apr May Jun Jul] Test [Aug]
```

→ Toujours: Entraînement sur passé, test sur futur ✅

### 5. Comment interpréter la réduction des coefficients?

**Question type:** "Le coefficient de temp_heure_cos a été réduit de 75%, qu'est-ce que ça signifie?"

**Réponse:**

**Coefficient réduit beaucoup (>50%) →** Feature probablement:

1. Corrélée avec d'autres (redondance)
2. Peu importante pour la prédiction
3. Instable (varie beaucoup selon échantillon)

**Coefficient réduit peu (<20%) →** Feature probablement:

1. Importante et unique
2. Stable
3. Peu corrélée avec d'autres

**Exemple concret:**

```
energie_lag1:     OLS = 0.85, Ridge = 0.55 (35% réduction)
energie_lag24:    OLS = 0.78, Ridge = 0.52 (33% réduction)
```

→ lag1 et lag24 sont très corrélés → Ridge les réduit tous deux pour éviter redondance

```
clients_connectes: OLS = 12.3, Ridge = 11.8 (4% réduction)
```

→ Variable très importante et peu corrélée → Ridge la garde presque intacte

## 🎯 Checklist pour l'Entrevue Ridge

- [ ] Dériver la solution Ridge: $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
- [ ] Expliquer le rôle de λ (avec 3 cas: 0, modéré, ∞)
- [ ] Montrer lien Ridge = MAP avec prior gaussien
- [ ] Différencier Ridge vs Lasso (tableau comparatif)
- [ ] Justifier TimeSeriesSplit pour séries temporelles
- [ ] Interpréter courbe validation (R² vs λ)
- [ ] Expliquer biais-variance tradeoff
- [ ] Analyser quelles features sont le plus réduites et pourquoi
- [ ] Tracer chemin de régularisation (coefficients vs λ)
- [ ] Expliquer pourquoi Ridge garantit l'inversibilité

---

## PARTIE 5: Modèle à 2 Étages (Classification → Régression) ⭐

### 🎯 L'Idée Centrale

**Observation:** Pas toutes les heures sont équivalentes!

- **Heures de pointe:** Consommation très élevée, patterns différents
- **Heures normales:** Consommation moyenne, plus prévisible

**Stratégie:**

1. **Étage 1 (Classification):** Prédire si l'heure sera en "pointe" (0/1)
2. **Étage 2 (Régression):** Utiliser $P(\text{pointe})$ comme **feature supplémentaire**

## 📐 Pourquoi Utiliser des Probabilités?

### Option 1: Indicateur Binaire (0/1)

```python
# Prédire classe binaire
classe_pred = clf.predict(X)  # [0, 0, 1, 0, 1, ...]
```

**Problème:**

- Perte d'information!
- $P = 0.51$ → classe 1
- $P = 0.99$ → classe 1
- **Mais la confiance est très différente!**

### Option 2: Probabilité Continue ([0, 1])

```python
# Prédire probabilité
proba_pred = clf.predict_proba(X)[:, 1]  # [0.05, 0.23, 0.87, 0.12, 0.98, ...]
```

**Avantages:**

- ✅ Information nuancée (certitude vs incertitude)
- ✅ Variable continue → Ridge peut l'utiliser linéairement
- ✅ $P(\text{pointe})$ élevée → modèle sait que consommation sera haute

**Exemple concret:**

```
Heure 1: P(pointe) = 0.05 → Probablement normal → Prédire ~80 kWh
Heure 2: P(pointe) = 0.50 → Incertain → Prédire ~120 kWh
Heure 3: P(pointe) = 0.95 → Presque sûr pointe → Prédire ~180 kWh
```

$P(\text{pointe})$ devient une **feature informative** pour la régression!

## 🔄 Architecture du Modèle à 2 Étages

```
                    DONNÉES
                       |
        ┌──────────────┴──────────────┐
        |                             |
    ÉTAGE 1                       ÉTAGE 2
Classification                   Régression
        |                             |
Features pour clf:             Features pour reg:
- température                  - température
- heure_sin/cos                - heure_sin/cos
- weekend                      - weekend
- clients_connectes            - clients_connectes
- (PAS de lags!)               - lags
        |                      - rolling stats
        ↓                      - P(pointe) ← NOUVEAU!
        |                             |
Logistic Regression                  ↓
        |                             |
        ↓                             |
  P(pointe) ────────────────────→ Ridge Regression
                                      |
                                      ↓
                                energie_kwh
```

## 💻 Implémentation Python Complète

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================
# ÉTAPE 1: PRÉPARER FEATURES POUR CLASSIFICATION
# ============================================

print("=" * 70)
print("ÉTAGE 1: CLASSIFICATION DES ÉVÉNEMENTS DE POINTE")
print("=" * 70)

# Features pour classifier pointe/normal
# IMPORTANT: Ne PAS utiliser lags d'énergie (ce serait de la triche!)
# On veut prédire la pointe AVANT de connaître la consommation

features_classification = [
    # Météo
    'temperature_ext', 'humidite', 'vitesse_vent', 'irradiance_solaire',

    # Temps (cyclique)
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'jour_semaine_sin', 'jour_semaine_cos',

    # Indicateurs
    'est_weekend', 'est_ferie',

    # TRÈS IMPORTANT
    'clients_connectes',

    # Transformations météo
    'degres_jours_chauffage',

    # Interactions
    'temp_heure_cos', 'temp_weekend'
]

# Vérifier disponibilité
features_clf_dispo = [f for f in features_classification if f in train_eng.columns]

X_train_clf = train_eng[features_clf_dispo].values
y_train_clf = train_eng['evenement_pointe'].values
X_test_clf = test_eng[features_clf_dispo].values
y_test_clf = test_eng['evenement_pointe'].values

print(f"Features pour classification: {len(features_clf_dispo)}")
print(f"Distribution train: {y_train_clf.mean():.1%} pointes")
print(f"Distribution test: {y_test_clf.mean():.1%} pointes")


# ============================================
# ÉTAPE 2: ENTRAÎNER CLASSIFIEUR LOGISTIQUE
# ============================================

# OPTIONNEL mais recommandé: Normaliser les features
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Entraînement
# Note: On peut utiliser sklearn (Partie 2 permettait from scratch)
clf_pointe = LogisticRegression(max_iter=1000, random_state=42)
clf_pointe.fit(X_train_clf_scaled, y_train_clf)

# Évaluation du classifieur
y_pred_clf_train = clf_pointe.predict(X_train_clf_scaled)
y_pred_clf_test = clf_pointe.predict(X_test_clf_scaled)

acc_train = accuracy_score(y_train_clf, y_pred_clf_train)
acc_test = accuracy_score(y_test_clf, y_pred_clf_test)

print(f"\nPerformance classification:")
print(f"  Accuracy train: {acc_train:.4f}")
print(f"  Accuracy test:  {acc_test:.4f}")

print(f"\nRapport de classification (test):")
print(classification_report(y_test_clf, y_pred_clf_test,
                          target_names=['Normal', 'Pointe']))


# ============================================
# ÉTAPE 3: EXTRAIRE PROBABILITÉS P(pointe)
# ============================================

print("\n" + "=" * 70)
print("EXTRACTION DES PROBABILITÉS")
print("=" * 70)

# Probabilités de la classe 1 (pointe)
# predict_proba retourne [P(classe 0), P(classe 1)]
# On veut P(classe 1) → colonne 1
train_eng['P_pointe'] = clf_pointe.predict_proba(X_train_clf_scaled)[:, 1]
test_eng['P_pointe'] = clf_pointe.predict_proba(X_test_clf_scaled)[:, 1]

print(f"Statistiques P(pointe):")
print(f"\nTrain:")
print(f"  Moyenne: {train_eng['P_pointe'].mean():.3f}")
print(f"  Std:     {train_eng['P_pointe'].std():.3f}")
print(f"  Min:     {train_eng['P_pointe'].min():.3f}")
print(f"  Max:     {train_eng['P_pointe'].max():.3f}")

print(f"\nTest:")
print(f"  Moyenne: {test_eng['P_pointe'].mean():.3f}")
print(f"  Std:     {test_eng['P_pointe'].std():.3f}")
print(f"  Min:     {test_eng['P_pointe'].min():.3f}")
print(f"  Max:     {test_eng['P_pointe'].max():.3f}")

# Visualiser distribution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme par classe
axes[0].hist(train_eng[train_eng['evenement_pointe']==0]['P_pointe'],
            bins=50, alpha=0.6, label='Normal', edgecolor='black')
axes[0].hist(train_eng[train_eng['evenement_pointe']==1]['P_pointe'],
            bins=50, alpha=0.6, label='Pointe', edgecolor='black')
axes[0].set_xlabel('P(pointe)', fontsize=11)
axes[0].set_ylabel('Fréquence', fontsize=11)
axes[0].set_title('Distribution de P(pointe) par Classe', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Boxplot par classe
data_boxplot = [
    train_eng[train_eng['evenement_pointe']==0]['P_pointe'],
    train_eng[train_eng['evenement_pointe']==1]['P_pointe']
]
axes[1].boxplot(data_boxplot, labels=['Normal', 'Pointe'])
axes[1].set_ylabel('P(pointe)', fontsize=11)
axes[1].set_title('P(pointe) par Classe Réelle', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Points pour l'entrevue:
# - Bonne séparation? Normal devrait avoir P faible, Pointe P élevé
# - Si chevauchement important → classifieur peu performant


# ============================================
# ÉTAPE 4: RÉGRESSION AVEC P(pointe)
# ============================================

print("\n" + "=" * 70)
print("ÉTAGE 2: RÉGRESSION AVEC P(pointe)")
print("=" * 70)

# Features pour régression = features classiques + P(pointe)
features_regression = features_clf_dispo + [
    # Ajouter features spécifiques régression (lags OK ici)
    'energie_lag1', 'energie_lag24',
    'energie_rolling_mean_6h', 'energie_rolling_mean_24h',
    'temp_squared'
]

# Filtrer celles qui existent
features_reg_dispo = [f for f in features_regression if f in train_eng.columns]

# AJOUTER P(pointe) !
features_final = features_reg_dispo + ['P_pointe']

X_train_final = train_eng[features_final].values
y_train_final = train_eng['energie_kwh'].values
X_test_final = test_eng[features_final].values
y_test_final = test_eng['energie_kwh'].values

print(f"Features totales pour régression: {len(features_final)}")
print(f"  - Features de base: {len(features_reg_dispo)}")
print(f"  - P(pointe): 1")


# ============================================
# COMPARAISON: AVEC vs SANS P(pointe)
# ============================================

# Modèle SANS P(pointe)
X_train_sans = train_eng[features_reg_dispo].values
X_test_sans = test_eng[features_reg_dispo].values

model_sans_p = RidgeCV(alphas=[0.1, 1, 10, 100], cv=TimeSeriesSplit(n_splits=5))
model_sans_p.fit(X_train_sans, y_train_final)
y_pred_sans = model_sans_p.predict(X_test_sans)

r2_sans = r2_score(y_test_final, y_pred_sans)
rmse_sans = np.sqrt(mean_squared_error(y_test_final, y_pred_sans))

print(f"\nRidge SANS P(pointe):")
print(f"  λ optimal: {model_sans_p.alpha_}")
print(f"  R² test:   {r2_sans:.4f}")
print(f"  RMSE test: {rmse_sans:.2f} kWh")


# Modèle AVEC P(pointe)
model_avec_p = RidgeCV(alphas=[0.1, 1, 10, 100], cv=TimeSeriesSplit(n_splits=5))
model_avec_p.fit(X_train_final, y_train_final)
y_pred_avec = model_avec_p.predict(X_test_final)

r2_avec = r2_score(y_test_final, y_pred_avec)
rmse_avec = np.sqrt(mean_squared_error(y_test_final, y_pred_avec))

print(f"\nRidge AVEC P(pointe):")
print(f"  λ optimal: {model_avec_p.alpha_}")
print(f"  R² test:   {r2_avec:.4f}")
print(f"  RMSE test: {rmse_avec:.2f} kWh")


# AMÉLIORATION
amelioration_r2 = r2_avec - r2_sans
amelioration_pct = 100 * amelioration_r2 / (1 - r2_sans)  # % de réduction de l'erreur restante

print(f"\n📈 IMPACT DE P(pointe):")
print(f"  Amélioration R²: +{amelioration_r2:.4f}")
print(f"  Réduction erreur: {amelioration_pct:.1f}%")
print(f"  RMSE réduit de: {rmse_sans - rmse_avec:.2f} kWh")


# ============================================
# ANALYSE DU COEFFICIENT DE P(pointe)
# ============================================

# Coefficient de P(pointe dans le modèle Ridge
idx_p_pointe = features_final.index('P_pointe')
coef_p_pointe = model_avec_p.coef_[idx_p_pointe]

print(f"\n🔍 ANALYSE DU COEFFICIENT P(pointe):")
print(f"  Coefficient: {coef_p_pointe:.2f}")

# Interprétation
if coef_p_pointe > 0:
    print(f"  Interprétation: Augmenter P(pointe) de 0.1 (10 points de %) "
          f"augmente consommation prédite de {coef_p_pointe * 0.1:.1f} kWh")
else:
    print(f"  ⚠️  Coefficient négatif inattendu!")

# Top features par importance (valeur absolue coefficient)
coef_abs = pd.DataFrame({
    'Feature': features_final,
    'Coefficient': model_avec_p.coef_,
    '|Coefficient|': np.abs(model_avec_p.coef_)
}).sort_values('|Coefficient|', ascending=False)

print(f"\nTop 10 features par importance:")
print(coef_abs.head(10).to_string(index=False))


# ============================================
# VISUALISATION: PRÉDICTIONS
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: Réel vs Prédit (SANS P)
axes[0].scatter(y_test_final, y_pred_sans, alpha=0.4, s=10, label='Prédictions')
axes[0].plot([y_test_final.min(), y_test_final.max()],
            [y_test_final.min(), y_test_final.max()],
            'r--', linewidth=2, label='Parfait')
axes[0].set_xlabel('Énergie réelle (kWh)', fontsize=11)
axes[0].set_ylabel('Énergie prédite (kWh)', fontsize=11)
axes[0].set_title(f'SANS P(pointe) - R² = {r2_sans:.4f}',
                 fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Scatter: Réel vs Prédit (AVEC P)
axes[1].scatter(y_test_final, y_pred_avec, alpha=0.4, s=10, label='Prédictions')
axes[1].plot([y_test_final.min(), y_test_final.max()],
            [y_test_final.min(), y_test_final.max()],
            'r--', linewidth=2, label='Parfait')
axes[1].set_xlabel('Énergie réelle (kWh)', fontsize=11)
axes[1].set_ylabel('Énergie prédite (kWh)', fontsize=11)
axes[1].set_title(f'AVEC P(pointe) - R² = {r2_avec:.4f}',
                 fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Points pour l'entrevue:
# - Points devraient être plus proches de la ligne rouge avec P(pointe)
# - Moins de dispersion = meilleures prédictions
```

## ⚠️ Points Critiques pour l'Entrevue

### 1. Pourquoi utiliser P(pointe) et non la classe 0/1?

**Réponse:**

**Classe binaire (0/1):** Perte d'information

- P=0.49 → 0, P=0.51 → 1 : très similaires mais classes différentes!
- P=0.51 → 1, P=0.99 → 1 : très différentes mais même classe!

**Probabilité continue:** Capture la nuance

- P=0.05 → Très sûr normal → Prédire consommation basse
- P=0.50 → Incertain → Prédire consommation intermédiaire
- P=0.95 → Très sûr pointe → Prédire consommation élevée

**Avantage pour Ridge:**

- Ridge est un modèle **linéaire**
- Peut utiliser P(pointe) comme variable continue
- Apprend automatiquement: $\beta_{P\text{pointe}} \times P(\text{pointe})$

### 2. Pourquoi ne pas utiliser lags pour classifier la pointe?

**Question piège:** "Vous utilisez energie_lag1 pour la régression mais pas pour la classification, pourquoi?"

**Réponse:**

**Pour classification (Étage 1):**

- But: Prédire **événement de pointe** (0/1)
- On veut prédire AVANT de connaître la consommation!
- Utiliser lags d'énergie = **tricher** (information du futur)
- Features: seulement météo + temps + clients

**Pour régression (Étage 2):**

- But: Prédire **valeur de consommation** (kWh)
- Les lags aident car consommation actuelle ~ consommation passée
- Ici c'est OK tant qu'on respecte chronologie (pas de fuite Kaggle)

**Exemple concret:**

```
❌ MAL (fuite):
   "Si consommation hier = 180 kWh → probablement pointe aujourd'hui"
   → On utilise une proxy de la cible pour prédire la cible!

✅ BON:
   "Si température = −15°C ET heure = 18h → probablement pointe"
   → On utilise seulement info disponible avant la pointe
```

### 3. Quelle amélioration attendez-vous de P(pointe)?

**Question type:** "Vous ajoutez P(pointe), combien de points de R² espérez-vous gagner?"

**Réponse réaliste basée sur expérience:**

- **Baseline (sans P):** R² ≈ 0.75-0.85
- **Avec P(pointe):** R² ≈ 0.80-0.90
- **Amélioration:** +0.03 à +0.08 points de R²

**Facteurs influençant l'amélioration:**

1. **Performance du classifieur**
   - Accuracy > 0.90 → grosse amélioration
   - Accuracy < 0.70 → petite amélioration (bruit)

2. **Différence consommation pointe/normal**
   - Si pointe = 3× normal → P(pointe) très utile!
   - Si pointe ≈ normal → P(pointe) peu utile

3. **Variables déjà présentes**
   - Si `clients_connectes` déjà là → moins d'amélioration
   - Si seulement météo de base → plus d'amélioration

**Pour l'entrevue:** Dire "J'attends +5% de R² car..." montre que vous avez réfléchi!

### 4. Modèle à 2 étages vs Modèle unique?

**Question:** "Pourquoi ne pas juste ajouter `evenement_pointe` comme feature?"

**Comparaison:**

**Option 1: Ajouter `evenement_pointe` binaire**

```python
features = [..., 'evenement_pointe']  # 0 ou 1
```

→ ❌ Sur test/Kaggle, on ne CONNAÎT PAS la vraie classe!

**Option 2: Prédire puis utiliser P(pointe)** (notre approche)

```python
# Étage 1
P_pointe = clf.predict_proba(X)[:, 1]
# Étage 2
features = [..., P_pointe]
```

→ ✅ On PRÉDIT P(pointe), puis on l'utilise comme feature

**Clé:** On ne triche pas! On prédit une proxy, pas la vraie valeur.

### 5. Comment gérer l'erreur propagée?

**Question avancée:** "L'erreur du classifieur ne se propage-t-elle pas à la régression?"

**Réponse:**

**Oui, il y a propagation d'erreur!**

Si classifieur prédit:

- P(pointe) = 0.80 alors que vraie classe = 0 (faux positif)
  → Régression va sur-prédire la consommation

**Mais:**

1. **Ridge est robuste** aux features bruitées (régularisation)
2. **P(pointe) est probabiliste** (pas binaire) → moins sensible
3. **Amélioration nette** même avec erreur de classification

**Analogie:**

- Météo imparfaite est mieux que pas de météo!
- P(pointe) imparfaite est mieux que rien!

**Trade-off:**

- +Information utile (P distingue patterns pointe/normal)
- -Bruit ajouté (erreurs classification)
- **Net: positif** dans la plupart des cas

## 🎯 Checklist pour l'Entrevue Modèle à 2 Étages

- [ ] Dessiner architecture 2 étages au tableau
- [ ] Expliquer pourquoi probabilité > classe binaire
- [ ] Justifier features différentes pour Étage 1 vs 2
- [ ] Expliquer pourquoi PAS de lags pour classification
- [ ] Calculer amélioration R² due à P(pointe)
- [ ] Interpréter coefficient de P(pointe): "β = 50 signifie..."
- [ ] Comparer histogrammes P(pointe) pour classes 0 et 1
- [ ] Expliquer propagation d'erreur mais bénéfice net
- [ ] Défendre: Pourquoi 2 étages vs modèle unique?
- [ ] Montrer graphiques: Scatter avec/sans P(pointe)

---

## 💡 Astuces pour l'Entrevue

1. **Préparez un brouillon OLS**: Entraînez-vous à dériver $\hat{\beta} = (X^TX)^{-1}X^Ty$ au tableau
2. **Connaissez vos choix**: Pourquoi ces features? Pourquoi cette validation?
3. **Visualisez**: Dessinez les concepts (cercle pour cyclique, graphe loss pour GD)
4. **Soyez honnête**: Si vous ne savez pas, expliquez votre raisonnement
5. **Préparez des exemples**: "Par exemple, pour la température..."

---

## 📈 Suivi de Progression

- [ ] Partie 0: Configuration ✅
- [ ] Partie 1: OLS from scratch
- [ ] Partie 2: Régression logistique & gradient descent
- [ ] Partie 3: Ridge regression
- [ ] Partie 4: Modèle à 2 étages
- [ ] Partie 5: Validation temporelle
- [ ] Partie 6: Modèle final
- [ ] Partie 7: Extension
- [ ] Soumission Kaggle
