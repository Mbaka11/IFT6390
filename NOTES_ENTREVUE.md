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

### 2. Gradient Descent

- TODO: Ajouter algorithme
- TODO: Ajouter choix du learning rate

### 3. Ridge Regression

- TODO: Ajouter lien avec MAP
- TODO: Ajouter effet sur les coefficients

### 4. Régression Logistique

- TODO: Ajouter fonction sigmoïde
- TODO: Ajouter entropie croisée

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
