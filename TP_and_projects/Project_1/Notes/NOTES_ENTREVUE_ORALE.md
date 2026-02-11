# 📚 NOTES POUR L'ENTREVUE ORALE - PROJET ÉNERGIE
## IFT6390 - Fondements de l'Apprentissage Machine

---

## ⚠️ RAPPEL: L'entrevue orale = 60% de la note!

**Ce qu'on attend de vous:**
- Dériver OLS au tableau ✍️
- Expliquer CHAQUE ligne de code que vous avez écrite
- Justifier VOS CHOIX (pourquoi ces features? pourquoi ce λ?)
- Modifier le code en direct et prédire l'effet
- Répondre aux questions théoriques

---

## 📖 PARTIE 1: OLS (Ordinary Least Squares)

### Formule à connaître PAR CŒUR:
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

### Dérivation complète (ESSENTIEL pour l'entrevue!):

**Objectif:** Minimiser l'erreur quadratique moyenne (MSE)

1. **Fonction de coût:**
   $$J(\boldsymbol{\beta}) = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

2. **Développement:**
   $$J(\boldsymbol{\beta}) = \frac{1}{n}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$
   
   $$= \frac{1}{n}(\mathbf{y}^\top\mathbf{y} - \mathbf{y}^\top\mathbf{X}\boldsymbol{\beta} - \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{y} + \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta})$$

3. **Gradient (dérivée par rapport à β):**
   $$\nabla J(\boldsymbol{\beta}) = \frac{1}{n}(-2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta})$$

4. **Mettre le gradient à zéro:**
   $$-2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = 0$$
   
   $$\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}$$

5. **Solution finale:**
   $$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

### Code Python (votre implémentation):
```python
def ols_fit(X, y):
    """
    Implémentation OLS from scratch
    X: matrice de features (n, p)
    y: vecteur cible (n,)
    """
    # Ajouter colonne de 1 pour le biais (intercept)
    X_bias = np.column_stack([np.ones(len(X)), X])
    
    # Calcul de β = (X^T X)^(-1) X^T y
    XtX = X_bias.T @ X_bias
    Xty = X_bias.T @ y
    beta = np.linalg.solve(XtX, Xty)  # Plus stable que inv()
    
    return beta

def ols_predict(X, beta):
    """Prédictions avec le modèle OLS"""
    X_bias = np.column_stack([np.ones(len(X)), X])
    return X_bias @ beta
```

### Questions possibles:
❓ **Pourquoi utiliser `np.linalg.solve()` plutôt que `np.linalg.inv()`?**
- Plus stable numériquement
- Évite d'inverser explicitement la matrice (coûteux et sensible au bruit)

❓ **Quand OLS échoue-t-il?**
- Si X^T X n'est pas inversible (colonnes colinéaires)
- Si p > n (plus de features que d'exemples)

❓ **Pourquoi ajouter une colonne de 1?**
- Pour le terme de biais (intercept) β₀
- Sans ça, la droite passerait par l'origine (0,0)

---

## 📖 PARTIE 2: Régression Logistique

### Fonction sigmoïde:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Propriété importante:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### Entropie croisée (Cross-Entropy Loss):
$$L(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^n [y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

où $\hat{p}_i = \sigma(\mathbf{x}_i^\top\boldsymbol{\beta})$

### Gradient de l'entropie croisée:
$$\nabla L(\boldsymbol{\beta}) = \frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{p}} - \mathbf{y})$$

où $\hat{\mathbf{p}}$ est le vecteur des probabilités prédites.

### Descente de gradient:
```python
def logistic_regression_gd(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Régression logistique avec descente de gradient
    """
    n, p = X.shape
    X_bias = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1)
    
    for iteration in range(n_iterations):
        # Prédictions (probabilités)
        z = X_bias @ beta
        p_hat = 1 / (1 + np.exp(-z))  # sigmoïde
        
        # Gradient
        gradient = (1/n) * X_bias.T @ (p_hat - y)
        
        # Mise à jour
        beta = beta - learning_rate * gradient
    
    return beta
```

### Questions possibles:
❓ **Pourquoi entropie croisée et pas MSE pour classification?**
- MSE n'est pas convexe pour la classification (risque de minima locaux)
- Entropie croisée est bien adaptée aux probabilités (0-1)

❓ **Comment choisir le learning rate?**
- Trop grand → divergence
- Trop petit → convergence lente
- Regarder la courbe de loss

❓ **Que représente P(pointe)=0.7?**
- 70% de probabilité que ce soit un événement de pointe
- C'est une PROBABILITÉ, pas une prédiction binaire

---

## 📖 PARTIE 3: Régularisation Ridge

### Formule Ridge:
$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

### Fonction de coût Ridge:
$$J(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$$

### Effet de λ (lambda):
- **λ = 0** → OLS classique (pas de régularisation)
- **λ petit** → Peu de régularisation
- **λ grand** → Forte régularisation (coefficients → 0)

### Code:
```python
def ridge_fit(X, y, lambda_reg=1.0):
    """Ridge from scratch"""
    X_bias = np.column_stack([np.ones(len(X)), X])
    n, p = X_bias.shape
    
    # Matrice identité (ne pas régulariser le biais!)
    I = np.eye(p)
    I[0, 0] = 0  # Pas de régularisation sur β₀
    
    # Solution Ridge
    XtX = X_bias.T @ X_bias
    beta = np.linalg.solve(XtX + lambda_reg * I, X_bias.T @ y)
    
    return beta
```

### Questions possibles:
❓ **Pourquoi Ridge aide avec des features corrélées?**
- Stabilise l'inversion de X^T X
- Distribue le poids entre features corrélées
- Réduit la variance du modèle

❓ **Ridge = MAP! Expliquez.**
- Ridge = Maximum A Posteriori avec prior gaussien sur β
- Prior: $p(\boldsymbol{\beta}) \sim \mathcal{N}(0, \sigma^2\mathbf{I})$
- Équivaut à ajouter une pénalité L2

❓ **Comment choisir λ?**
- **Validation croisée** (TimeSeriesSplit pour séries temporelles!)
- RidgeCV en sklearn
- Chercher λ qui minimise l'erreur de validation

---

## 📖 PARTIE 4: Division Temporelle (CRUCIAL!)

### ⚠️ INTERDICTION de la validation croisée aléatoire!

**Pourquoi?**
- Les données sont **séries temporelles**
- KFold aléatoire → **fuite d'information** (data leakage)
- On utiliserait le futur pour prédire le passé!

### TimeSeriesSplit:
```
Train: [-------------------]
                Test: [----]

Train: [------------------------]
                        Test: [----]

Train: [-----------------------------]
                                Test: [----]
```

**Principe:** On entraîne toujours sur le passé, on teste sur le futur.

### Dans ce projet:
- **Train:** Hiver 2023-2024
- **Test:** Printemps/Été 2024

**Décalage de distribution:**
- Hiver → consommation plus élevée
- Été → consommation plus faible
- **C'est réaliste!** Le modèle doit généraliser entre saisons.

---

## 📖 PARTIE 5: Modèle à 2 étages

### Architecture:
```
1. Classifieur logistique → P(pointe)
           ↓
2. Régression Ridge avec P(pointe) comme feature
```

### Pourquoi P(pointe) et pas 0/1?

**Mauvaise approche:**
```python
pred_binaire = (p_pointe > 0.5).astype(int)  # Seulement 0 ou 1
```

**Bonne approche:**
```python
# Utiliser la probabilité continue
X_with_proba = np.column_stack([X, p_pointe])  # p_pointe ∈ [0, 1]
```

**Pourquoi?**
- P=0.6 contient PLUS d'info que juste "1"
- P=0.51 vs P=0.99 sont tous deux "pointe" mais très différents!
- Le modèle de régression peut **pondérer** cette information

### Questions possibles:
❓ **Pourquoi 2 étages?**
- Événement de pointe = info cruciale pour consommation
- Mais c'est une variable qu'on ne connaît pas à l'avance
- On doit d'abord la prédire, puis l'utiliser

❓ **Risque de ce modèle?**
- **Propagation d'erreur:** Si le classifieur se trompe, ça affecte la régression
- Solution: améliorer le classifieur en premier!

---

## 📖 CONCEPTS THÉORIQUES AVANCÉS

### Ridge = MAP (Maximum A Posteriori)

**Interprétation probabiliste:**
- **Likelihood:** $p(\mathbf{y}|\mathbf{X}, \boldsymbol{\beta}) \sim \mathcal{N}(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I})$
- **Prior:** $p(\boldsymbol{\beta}) \sim \mathcal{N}(0, \tau^2\mathbf{I})$
- **Posterior:** $p(\boldsymbol{\beta}|\mathbf{X}, \mathbf{y}) \propto p(\mathbf{y}|\mathbf{X}, \boldsymbol{\beta}) \cdot p(\boldsymbol{\beta})$

**MAP = maximiser le posterior:**
$$\max_{\boldsymbol{\beta}} \log p(\boldsymbol{\beta}|\mathbf{X}, \mathbf{y})$$

Équivaut à minimiser:
$$\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$$

où $\lambda = \sigma^2/\tau^2$

### Minimiser entropie croisée

**Pour classification binaire:**
- On veut que $\hat{p}_i$ soit proche de $y_i$ (0 ou 1)
- Entropie croisée pénalise les mauvaises probabilités
- Si $y_i=1$ et $\hat{p}_i=0.9$ → petite perte
- Si $y_i=1$ et $\hat{p}_i=0.1$ → GROSSE perte

**Gradient:** Simple et élégant!
$$\nabla L = \frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{p}} - \mathbf{y})$$

---

## 📖 MÉTRIQUES D'ÉVALUATION

### Pour Régression:

**R² (Coefficient de détermination):**
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- **R² = 1** → Prédiction parfaite
- **R² = 0** → Modèle = moyenne
- **R² < 0** → Pire que la moyenne!

**RMSE (Root Mean Squared Error):**
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$

- En kWh dans ce projet
- Plus petit = mieux
- Interprétable (même unité que y)

### Pour Classification:

**Accuracy:**
$$\text{Accuracy} = \frac{\text{Nombre correct}}{n}$$

**Precision/Recall:**
- **Precision:** Des points prédits "pointe", combien le sont vraiment?
- **Recall:** Des vrais "pointe", combien on en détecte?

---

## 📝 CHECKLIST AVANT L'ENTREVUE

### Je DOIS savoir:
- [ ] Dériver OLS au tableau (de mémoire)
- [ ] Expliquer gradient descent étape par étape
- [ ] Justifier pourquoi TimeSeriesSplit (pas KFold)
- [ ] Expliquer Ridge = MAP avec prior gaussien
- [ ] Différence entre P(pointe) continue vs indicateur 0/1
- [ ] Pourquoi entropie croisée pour classification
- [ ] Comment j'ai choisi mes features
- [ ] Comment j'ai choisi λ (validation croisée)
- [ ] Interpréter les coefficients de mon modèle
- [ ] Expliquer mes résidus (graphique)

### Je DOIS pouvoir:
- [ ] Modifier le code en direct
- [ ] Ajouter/enlever une feature et prédire l'effet
- [ ] Changer λ et expliquer ce qui arrive
- [ ] Changer learning rate et voir l'impact
- [ ] Expliquer chaque ligne de mon code

### Questions pièges possibles:
❓ "Pourquoi pas MSE pour la logistique?"
❓ "Si λ → ∞, que deviennent les coefficients?"
❓ "Quelle feature a le coefficient le plus réduit par Ridge? Pourquoi?"
❓ "P(pointe)=0.7 pour une observation. Que signifie ce chiffre?"
❓ "Votre R² train est 0.95 et test 0.60. Problème?"
❓ "Changez ce seuil de 0.5 à 0.3 - qu'arrive-t-il?"

---

## 🎯 STRATÉGIE POUR L'ENTREVUE

### 1. Soyez confiant et clair
- Parlez lentement
- Utilisez des exemples concrets
- Admettez si vous ne savez pas (mieux que bluffer)

### 2. Justifiez TOUT
- "Pourquoi cette feature?" → "Parce que la température affecte directement la consommation de chauffage..."
- "Pourquoi ce λ?" → "J'ai fait une validation croisée temporelle et λ=10 minimise l'erreur..."

### 3. Montrez votre compréhension
- Connectez la théorie au code
- Expliquez les choix d'implémentation
- Anticipez les questions

### 4. Soyez prêt à modifier le code
- "Que se passe-t-il si on enlève la température?"
- "Changez le learning rate à 0.001"
- Prédisez AVANT d'exécuter!

---

## 📚 RÉVISION PAR CHAPITRE

À remplir au fur et à mesure:

### Chapitre 1: Learning Problem
- Concepts clés: _____________
- Liens avec le projet: _____________

### Chapitre 2: Linear Regression
- Concepts clés: _____________
- Liens avec le projet: _____________

### Chapitre 3: Classification
- Concepts clés: _____________
- Liens avec le projet: _____________

### Chapitre 4: Generalization
- Concepts clés: _____________
- Liens avec le projet: _____________

### Chapitre 5: Probabilistic
- Concepts clés: _____________
- Liens avec le projet: _____________

---

## ✍️ ESPACE POUR VOS NOTES

### Mes choix de features et pourquoi:


### Mes résultats et interprétation:


### Questions que je ne comprends pas encore:


### Points à clarifier avant l'entrevue:


---

**Bonne chance! Vous allez réussir! 🚀**
