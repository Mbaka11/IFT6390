# 📚 Notes de Préparation — Entrevue Devoir 2 : Réseaux de neurones (IFT6390)

> **Mise à jour au fur et à mesure de l'avancement du projet.**
> L'entrevue porte sur la compréhension des choix d'architecture, des hyperparamètres, et des résultats.

---

## 🗂️ Avancement du projet

| Partie                              | Statut       | Notes                                                                              |
| ----------------------------------- | ------------ | ---------------------------------------------------------------------------------- |
| Setup (données, vocab, features)    | ✅ Terminé   | 10 479 train / 2 620 val, vocab = ~50 chars                                        |
| **1.1** MLP fixe + Prédiction 1     | ✅ Terminé   | MSE val=473k, R²=0.12, écart=306k — sous-apprentissage confirmé                    |
| **1.2** Comparaison optimiseurs     | ✅ Terminé   | SGD+mom gagne (R²=0.21), Adam surapprend (R²=0.07), SGD lent (R²=0.13)             |
| **1.3** Deep MLP 5 couches ablation | ✅ Terminé   | ReLU+He gagne (R²=0.26), Sigmoid vanishing (norme≈0), BN/Drop contre-intuitif pire |
| **2.1** LSTM                        | ✅ Terminé   | MSE=562k, R²=-0.04, écart=20k — sous-apprentissage, pire que MLP                   |
| **2.3** Transformeur encodeur       | ✅ Terminé   | MSE=471k, R²=0.13, écart=54k — ≈ MLP, from-scratch = bottleneck                    |
| **3.1** Plongements SMI-TED         | 🔄 Code prêt | Chargement + extraction + PCA/t-SNE coloré par Tc, à exécuter sur Colab            |
| **3.2** Sonde linéaire              | 🔄 Code prêt | Linear(768,1) sur embeddings gelés, 769 params                                     |
| **3.3** Courbe d'efficacité         | 🔄 Code prêt | 4 modèles × 4 fractions (10/25/50/100%), figure centrale                           |

---

# 📝 SECTION 1 — Contenu du projet (étape par étape)

## Setup : Données et exploration

**Objectif** : Charger le CSV, filtrer les molécules avec Tc, séparer train/val.

```python
# On charge le CSV depuis GitHub (~24 600 molécules)
df_raw = pd.read_csv(URL)

# On filtre : seulement les molécules qui ont une valeur de Tc (pas NaN)
# C'est environ 13 100 molécules sur 24 600
df = df_raw[df_raw["Tc"].notna() & df_raw["SMILES"].notna()].copy()

# Séparation 80/20 avec seed=42 pour reproductibilité
# → ~10 480 train, ~2 620 val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
```

**Vocabulaire SMILES** : On extrait tous les caractères uniques des SMILES du dataset (~50 caractères). Ce vocabulaire sera utilisé pour construire les vecteurs de features et aussi pour l'embedding dans LSTM/Transformer.

### Fonctions helper (cellule 7)

```python
from sklearn.metrics import r2_score

def plot_learning_curves(train_losses, val_losses, title=""):
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Entraînement")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Époque")
    plt.ylabel("MSE")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print(f"Écart train-val : {val_losses[-1] - train_losses[-1]:.4f}")


def report_results(train_losses, val_losses, val_pred, y_val, n_params, title=""):
    """Rapport standardisé pour chaque modèle : courbes, MSE val, R² val, paramètres.

    Réutilisée pour MLP, LSTM, Transformeur, SMI-TED+sonde.
    val_pred est passé en argument → agnostique à l'architecture.
    """
    plot_learning_curves(train_losses, val_losses, title=title)
    mse_val = val_losses[-1]
    r2_val = r2_score(y_val, np.array(val_pred).flatten())
    print(f"MSE val        : {mse_val:.2f}")
    print(f"R² val         : {r2_val:.4f}")
    print(f"Nb. paramètres : {n_params:,}")
    return mse_val, r2_val
```

**Pourquoi model-agnostic ?** Chaque architecture (MLP, LSTM, Transformer) calcule `val_pred` différemment. En passant `val_pred` déjà calculé, la fonction fonctionne sans modification pour tous les modèles.

---

## Partie 1.1 — MLP à 2 couches cachées (features fixes)

### Prédiction 1

> Le modèle va présenter un **surapprentissage modéré**, mais sa principale limite sera le **sous-apprentissage**.

**Raisonnement** :

- La représentation **bag-of-characters** (fréquences) détruit l'ordre des atomes : `CCO` (éthanol) et `COC` (diméthyléther) → même vecteur, mais Tc différents.
- Cette perte d'info = le modèle n'a pas accès aux bonnes caractéristiques → sous-apprentissage → MSE val élevée.
- Côté données : ~10 480 exemples > 51 features → ratio favorable → pas de surapprentissage massif.
- Côté capacité : MLP 128×128 sans dropout/BN → ~23k paramètres > 10k exemples → léger surapprentissage (écart train-val non nul).

### Extraction de features : `smiles_to_features`

```python
def smiles_to_features(smiles: str) -> np.ndarray:
    """Convertit un SMILES en vecteur de longueur fixe.

    Exemple: 'CCO' → [2, 0, ..., 1, ..., 3]
             ↑ C apparaît 2 fois   ↑ O 1 fois  ↑ longueur=3

    LIMITATION : 'CCO' et 'COC' donnent le MÊME vecteur !
    L'ordre est complètement perdu (bag-of-characters).
    """
    # Compter la fréquence de chaque caractère du vocabulaire
    counts = np.zeros(len(vocab), dtype=np.float32)
    for c in smiles:
        if c in char_to_idx:
            counts[char_to_idx[c]] += 1
    # Ajouter la longueur totale comme feature supplémentaire
    length = np.array([len(smiles)], dtype=np.float32)
    return np.concatenate([counts, length])  # dim ≈ 51
```

### Code MLP complet (cellule 10)

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# === Reproductibilité ===
torch.manual_seed(SEED)
np.random.seed(SEED)

# === Données en tenseurs PyTorch ===
# unsqueeze(1) : passe y de shape (N,) à (N,1) pour matcher la sortie du MLP (N,1)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Normalisation z-score : μ et σ calculés sur le TRAIN seulement
# → Appliquer la même transformation au val évite le data leakage
# → Les features ont des échelles très différentes (C : 0-50, longueur : 2-300)
#   sans normalisation les gradients sont dominés par les grandes amplitudes
train_mean = X_train_t.mean(dim=0)
train_std = X_train_t.std(dim=0) + 1e-8  # +epsilon pour éviter division par 0
X_train_t = (X_train_t - train_mean) / train_std
X_val_t = (X_val_t - train_mean) / train_std

# DataLoader : mini-lots de 256, mélangés à chaque époque
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

# === MLP 2 couches cachées ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # Couche 1 : input_dim → 128  (51×128 + 128 = 6 656 params)
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # Dérivée = 1 si x>0 → pas de vanishing gradient (contrairement à Sigmoid)
            # Couche 2 : 128 → 128        (128×128 + 128 = 16 512 params)
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Sortie : 128 → 1            (128 + 1 = 129 params)
            nn.Linear(hidden_dim, 1),
            # Pas d'activation finale : régression → valeur continue arbitraire
        )
        # Total ≈ 23 297 paramètres | Pas de dropout/BN : baseline volontairement simple

    def forward(self, x):
        return self.net(x)

# === Boucle d'entraînement ===
def train_model(model, train_loader, X_val_t, y_val_t, lr=1e-3, epochs=100):
    # Adam : lr adaptatif par paramètre, combine momentum + variance → convergence rapide
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # MSE = (1/N)Σ(ŷ-y)² → équivalent MV gaussienne
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()  # Active dropout/BN en mode train (ici sans effet, mais bonne pratique)
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            pred = model(xb)           # Passe avant
            loss = criterion(pred, yb) # Calcul de la perte
            optimizer.zero_grad()      # Réinitialiser les gradients accumulés
            loss.backward()            # Rétropropagation (VJP sur le graphe de calcul)
            optimizer.step()           # Mise à jour des poids
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()   # Désactive dropout/BN (mode inférence)
        with torch.no_grad():  # Pas de gradients → plus rapide, moins de mémoire
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

    return train_losses, val_losses

# === Entraînement ===
input_dim = X_train_t.shape[1]
mlp = MLP(input_dim, hidden_dim=128)
n_params = sum(p.numel() for p in mlp.parameters())

train_losses, val_losses = train_model(mlp, train_loader, X_val_t, y_val_t, lr=1e-3, epochs=100)

# === Résultats ===
mlp.eval()
with torch.no_grad():
    val_pred = mlp(X_val_t).numpy()
mse_val, r2_val = report_results(train_losses, val_losses, val_pred, y_val, n_params,
                                  title="1.1 — MLP à 2 couches cachées")
```

**Choix clés** :

- **ReLU** : gradient = 1 si x>0, pas de saturation → évite le vanishing gradient.
- **Pas de dropout/BatchNorm** : baseline simple, volontairement. Ajouté en 1.3.
- **128 neurones, lr=1e-3, batch=256, 100 époques** : hyperparamètres standard pour Adam sur ce dataset.

### Résultats 1.1

| Métrique        | Valeur   |
| --------------- | -------- |
| MSE val         | 473 457  |
| R² val          | 0.1217   |
| Nb. paramètres  | 24 577   |
| Train MSE       | ~167 000 |
| Écart train-val | 306 462  |

**Analyse** :

- **R² = 0.12 → sous-apprentissage dominant** : le modèle n'explique que 12% de la variance de Tc. Prédiction 1 confirmée — la représentation bag-of-chars perd toute la structure moléculaire.
- **Écart train-val = 306k** (train ≈167k, val ≈473k, ratio ×2.8) : surapprentissage modéré en parallèle, aussi prédit. Le modèle mémorise du bruit sur les features disponibles.
- **Pics dans la courbe d'entraînement** (~époques 30, 60, 90) : instabilité d'Adam due à des mini-lots contenant des molécules atypiques (SMILES très longs ou Tc extrêmes). Pics présents uniquement sur la courbe d'**entraînement** — la validation reste lisse, donc ils sont cosmétiques et non problématiques. `clip_grad_norm_` testé mais trop agressif (R² chutait de 0.12 → 0.07) : retiré.
- **Nb. paramètres = 24 577** : vocab effectif = 60 caractères + 1 longueur = 61 dimensions d'entrée (61×128+128 + ...).

---

## Partie 1.2 — Comparaison d'optimiseurs

### Prédiction 2 (théorie + logique)

**Prédiction** : Adam convergera le plus vite, SGD+momentum en second, SGD pur en dernier. Les trois devraient atteindre une MSE finale similaire car l'architecture et les données sont identiques.

#### Théorie : les trois optimiseurs

**SGD pur** (descente de gradient stochastique) :
$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

- Met à jour dans la direction du gradient instantané du mini-lot.
- Problème : le gradient d'un seul mini-lot est une estimation bruitée du vrai gradient. Sur notre jeu (molécules hétérogènes), la variance d'un mini-lot est élevée → oscillations importantes, convergence lente.
- Le taux d'apprentissage `lr` est global et fixe. Il faut souvent l'ajuster manuellement.

**SGD + momentum** :
$$v_{t+1} = \beta v_t + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \, v_{t+1}$$

- Avec $\beta \approx 0.9$ (valeur par défaut PyTorch), on accumule une moyenne exponentielle des gradients passés.
- Effet physique : "boule qui roule" — dans les directions consistantes le momentum s'accumule, dans les directions bruitées il se moyenne et s'annule. Cela atténue les oscillations et accélère la progression dans les ravins de la surface de perte.
- Converge plus vite que SGD pur, mais demande quand même un bon réglage de `lr`.

**Adam** (Adaptive Moment Estimation) :
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1}^{\text{er}}\text{ moment — direction)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2}^{\text{e}}\text{ moment — variance)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

- Combine momentum (direction via $m_t$) + normalisation par variance (via $v_t$).
- Chaque paramètre a son propre taux d'apprentissage effectif : les paramètres peu mis à jour reçoivent des étapes plus grandes, les paramètres volatils des étapes plus petites.
- Avec $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ (défauts PyTorch), Adam est quasi-insensible au choix de `lr` initial (lr=1e-3 fonctionne bien dans la plupart des cas).
- Résultat : convergence rapide, robuste, peu de réglage.

#### Pourquoi Adam gagne sur ce problème ?

1. **Hétérogénéité des features** : nos 61 features (fréquences de caractères) ont des variances très différentes — certains chars rares (ex. `%`) apparaissent dans très peu de molécules, d'autres (`C`) dans presque toutes. Adam normalise les pas par feature → adapté à cette situation.
2. **Données bruyantes** : les mini-lots de SMILES moléculaires sont hétérogènes (tailles variables, Tc couvrant 100–1000 K). La normalisation de variance d'Adam lisse cette variabilité mieux que SGD.
3. **Pas de réglage lr** : avec `lr=1e-3` fixe pour tout le monde, Adam est dans sa zone de confort par construction, alors que SGD pur risque d'osciller avec un lr trop grand ou stagne avec un lr trop petit.

#### Ce qui devrait se passer dans les courbes

| Optimiseur     | Convergence initiale           | Plateau val         | MSE finale                 |
| -------------- | ------------------------------ | ------------------- | -------------------------- |
| SGD            | Lente, oscillations fortes     | Haut                | Comparable (si lr correct) |
| SGD + momentum | Modérée, oscillations amorties | Moyen               | Comparable                 |
| Adam           | Rapide, descente régulière     | Bas dès ~20 époques | Comparable                 |

> **Si les MSE finales divergent** (ex. SGD ne descend pas) : c'est que lr=1e-3 est sous-optimal pour SGD. Dans l'expérience, on garde `lr=1e-3` pour tous → SGD pourrait rester plus haut.

### Code complet et commenté

```python
import torch
import torch.nn as nn

torch.manual_seed(SEED)
np.random.seed(SEED)

# === Fonction générique d'entraînement ===
# Prend un callable optimizer_fn : params → optimizer
# Cela permet de réutiliser la même boucle pour SGD, momentum, Adam
# sans dupliquer le code — seul l'optimiseur change
def train_with_optimizer(optimizer_fn, epochs=100):
    # On recrée un MLP frais à chaque appel pour garantir la même
    # initialisation des poids pour tous les optimiseurs (fairness)
    model = MLP(input_dim, hidden_dim=128)
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()  # Active le mode entraînement (dropout, BN en mode train)
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()   # Réinitialise les gradients accumulés
            loss.backward()         # Rétropropagation : calcule ∂L/∂θ pour tous les paramètres
            # Seuil large (5.0) : coupe uniquement les gradients explosifs de SGD
            # Adam opère dans une plage normalisée → rarement affecté par ce seuil
            # Sans ce clipping, SGD avec lr=1e-3 diverge sur données hétérogènes → NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()        # Mise à jour θ ← θ - lr * g (ou version adaptée selon l'optimiseur)
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        # Évaluation sur validation à chaque époque (pas de mise à jour des poids)
        model.eval()
        with torch.no_grad():  # Désactive le calcul du graphe de calcul → plus rapide
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)

    return model, train_losses, val_losses

# === Les 3 optimiseurs — même lr=1e-3 pour tous → seule variable = algorithme ===
# SGD : θ_{t+1} = θ_t - lr * g_t
# SGD+mom : v_{t+1} = 0.9*v_t + g_t ; θ_{t+1} = θ_t - lr * v_{t+1}
# Adam : combine 1er moment (direction) + 2e moment (variance adaptative par paramètre)
OPTIMIZERS = [
    ("SGD",            lambda p: torch.optim.SGD(p, lr=1e-3)),
    ("SGD + momentum", lambda p: torch.optim.SGD(p, lr=1e-3, momentum=0.9)),
    ("Adam",           lambda p: torch.optim.Adam(p, lr=1e-3)),
]

# === Entraînement — on fixe le seed juste avant chaque run ===
# Cela garantit que chaque MLP part exactement des mêmes poids initiaux
results_12 = {}
for label, opt_fn in OPTIMIZERS:
    print(f"Entraînement — {label}...")
    torch.manual_seed(SEED)  # Réinitialise le générateur → même init pour chaque modèle
    model_opt, tl, vl = train_with_optimizer(opt_fn)
    results_12[label] = (model_opt, tl, vl)

# === Graphique comparatif : 3 courbes de validation superposées ===
# Objectif : visualiser la vitesse de convergence relative de chaque optimiseur
colors = ["tab:blue", "tab:orange", "tab:green"]
plt.figure(figsize=(9, 4))
for (label, (_, _, vl)), color in zip(results_12.items(), colors):
    plt.plot(vl, label=label, color=color)
plt.xlabel("Époque")
plt.ylabel("MSE de validation")
plt.title("1.2 — Comparaison d'optimiseurs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Rapport individuel (courbe train+val, MSE, R², nb params) via report_results ===
n_params_12 = sum(p.numel() for p in MLP(input_dim, hidden_dim=128).parameters())
for label, (model_opt, tl, vl) in results_12.items():
    model_opt.eval()
    with torch.no_grad():
        vp = model_opt(X_val_t).numpy()
    print(f"\n{'='*45}\n{label}")
    report_results(tl, vl, vp, y_val, n_params_12, title=f"1.2 — {label}")
```

| Optimiseur     | MSE val     | R² val     | Écart train-val |
| -------------- | ----------- | ---------- | --------------- |
| SGD            | 470 453     | 0.1273     | 85 986          |
| SGD + momentum | **426 910** | **0.2080** | 118 375         |
| Adam           | 499 192     | 0.0739     | 251 262         |

**Analyse** :

- **Prédiction de vitesse partiellement confirmée** : Adam et SGD+momentum descendent tous les deux rapidement dans les premières ~20 époques, SGD reste lent pendant les 100 époques. Ce point était bien prédit.

- **Surprise : SGD+momentum gagne en généralisation** (R²=0.21, MSE=427k), pas Adam. Contrairement à la prédiction, Adam finit dernier (R²=0.07). Explication :
  - Adam converge vite **puis surapprend** — son taux d'apprentissage adaptatif lui permet de suivre le bruit de chaque mini-lot. Écart train-val=251k → le modèle Adam mémorise des motifs spécifiques aux mini-lots d'entraînement.
  - SGD+momentum convergence rapide ET stable — le momentum lisse les mises à jour sans sur-adapter, ce qui produit une meilleure régularisation implicite.
  - SGD pur finit avec le plus petit écart train-val (86k) — il n'a pas eu assez d'époques pour surapprendre, ce qui en fait le modèle le moins biaisé mais le plus lent.

- **Impact de clip*grad_norm*(5.0)** : ajouté pour éviter les NaN de SGD. Ce seuil corrompt légèrement les estimées de variance de second moment d'Adam (qui attend des gradients non clippés), ce qui aggrave son instabilité aux époques tardives.

- **Leçon** : dans un régime de données limitées, l'adaptivité d'Adam peut être un défaut — il converge trop précisément vers les données d'entraînement. SGD+momentum offre un meilleur compromis vitesse/généralisation.

---

## Partie 1.3 — Deep MLP : ablation activation/init/BN/dropout

### Objectif

Comparer 4 configurations d'un MLP **profond (5 couches cachées)** pour isoler l'effet de chaque technique de stabilisation/régularisation. On rapporte : MSE val, norme L2 du gradient à la 1ère couche cachée (moyennée sur les mini-lots du dernier epoch), et écart train-val.

### Théorie

- **Config 1 (Sigmoid + défaut)** : Sigmoid sature vers 0 et 1, sa dérivée $\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq 0.25$. En multipliant 5 fois une dérivée $\leq 0.25$, le gradient au 1er layer est $\leq (0.25)^5 \approx 10^{-3}$ → **vanishing gradient**. L'init par défaut de `nn.Linear` (Kaiming uniforme) n'est pas calibrée pour Sigmoid.
- **Config 2 (ReLU + He)** : ReLU a dérivée = 1 si $x>0$ → pas de vanishing. He init $W \sim \mathcal{N}(0, 2/n_{\text{in}})$ compense la "moitié morte" de ReLU → gradients stables à travers les 5 couches.
- **Config 3 (+ BatchNorm)** : BN normalise les activations à chaque couche → $\hat{h} = \frac{h - \mu_B}{\sigma_B}$. Stabilise les distributions internes, permet des lr plus grands, régularise légèrement.
- **Config 4 (+ Dropout 0.3)** : Éteint 30% des neurones aléatoirement à chaque passe → force la redondance, réduit la co-adaptation → meilleure généralisation (écart train-val plus petit).

### Code complet et commenté

```python
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(SEED)
np.random.seed(SEED)

# === MLP profond configurable (5 couches cachées) ===
# Architecture paramétrique : on peut changer activation, init, BN, dropout
# via les arguments du constructeur → même classe pour les 4 configs
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_hidden=5,
                 activation='relu', use_bn=False, dropout_p=0.0, he_init=False):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_hidden):
            # Couche linéaire : in_dim → hidden_dim
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            # BatchNorm AVANT l'activation (convention du papier original)
            # Normalise les pré-activations → distribution centrée → activation efficace
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            # Activation : Sigmoid (config 1) ou ReLU (configs 2-4)
            if activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
            # Dropout APRÈS l'activation
            # Éteint 30% des neurones actifs → régularisation
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = hidden_dim
        # Couche de sortie : hidden_dim → 1 (régression)
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        # Référence à la 1ère couche linéaire pour tracker le gradient
        # On mesure ‖∂L/∂W₁‖₂ pour diagnostiquer le vanishing gradient
        self.first_linear = self.net[0]

        # He init pour les configs ReLU (configs 2-4)
        # W ~ N(0, 2/n_in) → compense la moitié des activations mortes de ReLU
        # Sans He → variance des activations explose ou s'effondre couche par couche
        if he_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# === Entraînement avec tracking de la norme du gradient ===
# On utilise Adam pour toutes les configs (pas de comparaison d'optimiseurs ici)
# Le seul variable est la config architecturale
def train_deep(model, train_loader, X_val_t, y_val_t, lr=1e-3, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    last_epoch_grad_norms = []  # Normes du gradient à la 1ère couche, dernier epoch

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()

            # Norme L2 du gradient à la 1ère couche cachée — dernière époque seulement
            # ‖∂L/∂W₁‖₂ : si petit (~1e-6) → vanishing gradient (Sigmoid)
            #               si ~1.0    → gradient sain (ReLU + He)
            if epoch == epochs - 1:
                grad_norm = model.first_linear.weight.grad.norm(2).item()
                last_epoch_grad_norms.append(grad_norm)

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)

    # Moyenne des normes de gradient sur tous les mini-lots du dernier epoch
    avg_grad_norm = np.mean(last_epoch_grad_norms) if last_epoch_grad_norms else 0.0
    return model, train_losses, val_losses, avg_grad_norm


# === 4 configurations d'ablation ===
# Chaque config ajoute UNE technique par rapport à la précédente
# → on isole l'effet de chaque ajout progressivement
CONFIGS_13 = [
    ("1. Sigmoid + défaut",      dict(activation='sigmoid', use_bn=False, dropout_p=0.0, he_init=False)),
    ("2. ReLU + He",             dict(activation='relu',    use_bn=False, dropout_p=0.0, he_init=True)),
    ("3. ReLU + He + BN",        dict(activation='relu',    use_bn=True,  dropout_p=0.0, he_init=True)),
    ("4. ReLU + He + BN + Drop", dict(activation='relu',    use_bn=True,  dropout_p=0.3, he_init=True)),
]

# === Entraînement des 4 configs (même seed → même init de départ) ===
results_13 = {}
for label, kwargs in CONFIGS_13:
    print(f"\nEntraînement — {label}...")
    torch.manual_seed(SEED)   # Même init de poids pour chaque config
    np.random.seed(SEED)
    model = DeepMLP(input_dim, hidden_dim=128, n_hidden=5, **kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    model, tl, vl, grad_norm = train_deep(model, train_loader, X_val_t, y_val_t, epochs=100)
    results_13[label] = (model, tl, vl, grad_norm, n_params)

# === Graphique comparatif : 4 courbes de validation ===
plt.figure(figsize=(10, 5))
for label, (_, _, vl, _, _) in results_13.items():
    plt.plot(vl, label=label)
plt.xlabel("Époque")
plt.ylabel("MSE de validation")
plt.title("1.3 — Ablation : activation / init / BN / dropout (5 couches cachées)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Tableau récapitulatif ===
print(f"\n{'Config':<28} {'MSE val':>12} {'‖∇W₁‖₂':>12} {'Écart':>12} {'Params':>10}")
print("─" * 78)
for label, (_, tl, vl, gn, np_) in results_13.items():
    ecart = vl[-1] - tl[-1]
    print(f"{label:<28} {vl[-1]:>12,.0f} {gn:>12.6f} {ecart:>12,.0f} {np_:>10,}")

# === Rapports individuels (courbes, MSE, R², nb params) ===
for label, (m, tl, vl, gn, np_) in results_13.items():
    m.eval()
    with torch.no_grad():
        vp = m(X_val_t).numpy()
    print(f"\n{'='*50}\n{label}")
    report_results(tl, vl, vp, y_val, np_, title=f"1.3 — {label}")
    print(f"Norme grad L2 (1ère couche) : {gn:.6f}")
```

### Résultats

| #   | Config                     | MSE val     | R² val     | Norme grad. L2 | Écart train-val | Nb. paramètres |
| --- | -------------------------- | ----------- | ---------- | -------------- | --------------- | -------------- |
| 1   | Sigmoid + défaut           | 621 763     | -0.1534    | 0.000020       | -69 281         | 74 113         |
| 2   | ReLU + He                  | **397 907** | **0.2618** | 400 308        | 376 075         | 74 113         |
| 3   | ReLU + He + BN             | 607 664     | -0.1273    | 224 582        | 576 831         | 75 393         |
| 4   | ReLU + He + BN + Drop(0.3) | 701 691     | -0.3017    | 679 569        | 553 673         | 75 393         |

### Analyse des résultats

- **Config 1 (Sigmoid + défaut) → vanishing confirmé** : Norme = 0.000020 (quasi-nul). Après 5 couches Sigmoid, le gradient arrive à ~0 → la 1ère couche ne s'update plus. R²=-0.15 = pire que prédire la moyenne. Écart train-val _négatif_ (-69k) = le modèle prédit mieux sur val que sur train, ce qui prouve qu'il n'a pas appris du tout (train loss reste élevée aussi). Décroissance quasi-linéaire dans la courbe = stagnation.

- **Config 2 (ReLU + He) → MEILLEURE CONFIG mais explosion de gradient** : R²=0.2618, seule config positive. Mais la norme = **400 308** → explosion massive. Adam absorbe cette instabilité grâce à la normalisation par variance adaptative ($v_t$), ce qui permet quand même la convergence. Surapprentissage important (écart=376k), mais le modèle a réellement appris quelque chose.

- **Config 3 (+ BN) → contre-intuitif : pire que config 2** : BN était censé stabiliser mais R²=-0.13. Explication : sur 61 features tabulaires avec variance hétérogène, BN perturbe les échelles qui encodent de l'information réelle. Les statistiques de mini-lot (moy/var) sur un espace de si petite dimension sont bruyantes. De plus, l'ordre train/val de BN (utilise stats de mini-lot pendant train, mean/var globale à l'inférence) crée un décalage.

- **Config 4 (+ BN + Dropout) → pire de tous** : R²=-0.30. Dropout retire 30% des neurones d'un réseau déjà peu expressif pour des features tabulaires. La combinaison explosion+Dropout+BN produit une instabilité maximale (norme=679k, écart=554k).

- **Leçon** : 5 couches is too deep pour 61 features tabulaires. ReLU+He explose sans régularisation mais Adam compense partiellement. BN et Dropout, conçus pour des réseaux plus larges/profonds ou des images, interagissent mal avec les données tabulaires de faible dimension. **Pour des données tabulaires, 2-3 couches suffisent généralement** (voir résultat 1.1 : 2 couches + Adam = R²=0.12 avec stabilité vs. 5 couches + Adam = R²=0.26 avec explosion).

---

## Dérivations manuscrites — Partie 1

### 1. VJP de $\text{matmul}(W, x)$

Soit $z = Wx$ avec $W \in \mathbb{R}^{m \times n}$ et $x \in \mathbb{R}^{n}$, $z \in \mathbb{R}^{m}$.

On a $z_i = \sum_j W_{ij} x_j$.

**VJP par rapport à $W$** :

On reçoit le vecteur adjoint $\bar{z} = \frac{\partial \mathcal{L}}{\partial z} \in \mathbb{R}^{m}$ (le gradient venant des couches suivantes).

$$\frac{\partial \mathcal{L}}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_{ij}} = \bar{z}_i \cdot x_j$$

Sous forme matricielle :

$$\boxed{\frac{\partial \mathcal{L}}{\partial W} = \bar{z} \, x^\top}$$

C'est un produit extérieur : chaque ligne $i$ de $\partial \mathcal{L}/\partial W$ est $\bar{z}_i$ fois le vecteur $x^\top$.

**VJP par rapport à $x$** :

$$\frac{\partial \mathcal{L}}{\partial x_j} = \sum_i \frac{\partial \mathcal{L}}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_j} = \sum_i \bar{z}_i \cdot W_{ij}$$

Sous forme matricielle :

$$\boxed{\frac{\partial \mathcal{L}}{\partial x} = W^\top \bar{z}}$$

**Interprétation** : Le gradient par rapport aux entrées est obtenu en "remontant" le gradient à travers $W$ transposé. C'est la clé de la rétroprop : on propage $\bar{z}$ vers l'arrière en multipliant par $W^\top$.

### 2. Graphe de calcul du MLP à 2 couches avec VJP annotées

```
Entrée x ∈ ℝⁿ
    │
    ▼
[a₁ = W₁x + b₁]  ──── VJP: ∂L/∂W₁ = δ₁ · x⊤,  ∂L/∂b₁ = δ₁,  ∂L/∂x = W₁⊤ δ₁
    │
    ▼
[h₁ = ReLU(a₁)]  ──── VJP: δ₁ = δ_h₁ ⊙ 𝟙(a₁ > 0)     (dérivée = 1 si a₁>0, 0 sinon)
    │
    ▼
[a₂ = W₂h₁ + b₂]  ──── VJP: ∂L/∂W₂ = δ₂ · h₁⊤,  ∂L/∂b₂ = δ₂,  ∂L/∂h₁ = W₂⊤ δ₂
    │
    ▼
[h₂ = ReLU(a₂)]  ──── VJP: δ₂ = δ_h₂ ⊙ 𝟙(a₂ > 0)
    │
    ▼
[ŷ = W₃h₂ + b₃]  ──── VJP: ∂L/∂W₃ = δ₃ · h₂⊤,  ∂L/∂b₃ = δ₃,  ∂L/∂h₂ = W₃⊤ δ₃
    │
    ▼
[L = (ŷ - y)²]   ──── VJP de MSE: δ₃ = ∂L/∂ŷ = 2(ŷ - y)
```

**Propagation arrière (rétroprop)** — de bas en haut :

1. $\delta_3 = 2(\hat{y} - y)$ — gradient de la MSE par rapport à la prédiction
2. $\frac{\partial \mathcal{L}}{\partial W_3} = \delta_3 \cdot h_2^\top$, $\frac{\partial \mathcal{L}}{\partial h_2} = W_3^\top \delta_3$
3. $\delta_{h_2} = \frac{\partial \mathcal{L}}{\partial h_2}$, $\delta_2 = \delta_{h_2} \odot \mathbb{1}(a_2 > 0)$ — ReLU masque les négatifs
4. $\frac{\partial \mathcal{L}}{\partial W_2} = \delta_2 \cdot h_1^\top$, $\frac{\partial \mathcal{L}}{\partial h_1} = W_2^\top \delta_2$
5. $\delta_{h_1} = \frac{\partial \mathcal{L}}{\partial h_1}$, $\delta_1 = \delta_{h_1} \odot \mathbb{1}(a_1 > 0)$
6. $\frac{\partial \mathcal{L}}{\partial W_1} = \delta_1 \cdot x^\top$

**Observation clé** : à chaque couche, le gradient est multiplié par $W_i^\top$ puis masqué par ReLU. Si les poids sont mal initialisés (trop grands ou trop petits), le gradient explose ou disparaît en traversant les 2 couches. C'est pourquoi He init est important pour les réseaux profonds.

---

## Partie 2.1 — LSTM

### Prédiction 3

> Le LSTM fera **légèrement mieux** que le MLP en validation, mais avec un **surapprentissage plus marqué**.

**Raisonnement** :

- Le LSTM traite les caractères séquentiellement → il peut capturer l'**ordre** des atomes. `CCO` (éthanol) et `COC` (diméthyléther) produisent des séquences différentes, contrairement au bag-of-characters du MLP.
- Mais avec ~10k exemples et des SMILES très variables (2-300+ chars), apprendre des patterns séquentiels **de zéro** est difficile.
- Plus de paramètres (~85k vs ~25k) → risque de mémorisation.
- Gain modéré en R², loin de ce que le transfer learning (Partie 3) apportera.

### Pipeline

```
SMILES → indices (1-based, 0=PAD) → Embedding(32) → pack_padded_sequence → LSTM(128) → h_n[-1] → Linear(1) → Tc
```

### Code complet et commenté

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

torch.manual_seed(SEED)
np.random.seed(SEED)

# === Encodage séquentiel ===
# 0 = PAD, 1..60 = caractères du vocab
# Le PAD est ignoré par pack_padded_sequence → le LSTM ne voit que les vrais caractères
PAD_IDX = 0
char_to_idx_seq = {c: i + 1 for i, c in enumerate(vocab)}
VOCAB_SIZE = len(vocab) + 1  # 61 (60 chars + 1 PAD)

def smiles_to_indices(smiles):
    """'CCO' → tensor([idx_C, idx_C, idx_O])"""
    return torch.tensor([char_to_idx_seq[c] for c in smiles if c in char_to_idx_seq],
                        dtype=torch.long)

# === Dataset : stocke les séquences pré-converties ===
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, targets):
        # Pré-convertir toutes les séquences pour éviter de recalculer à chaque epoch
        self.sequences = [smiles_to_indices(s) for s in smiles_list]
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def collate_smiles(batch):
    """Fonction de collation pour DataLoader :
    - Pad les séquences au max du mini-lot (pas au max global → plus efficace)
    - Retourne aussi les longueurs originales pour pack_padded_sequence
    """
    seqs, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    # pad_sequence : complète les séquences courtes avec PAD_IDX
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.stack(targets)

# === Modèle LSTM ===
class LSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=128, num_layers=1):
        super().__init__()
        # Embedding : indice → vecteur dense de dim 32
        # padding_idx=0 → le vecteur pour PAD est toujours zéro (pas de gradient)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # LSTM : traite la séquence de gauche à droite
        # h_t = f(h_{t-1}, x_t) — état caché accumule le contexte séquentiel
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True)
        # Couche de sortie : état caché final → prédiction scalaire
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)                # (batch, seq_len, 32)
        # pack_padded_sequence : retire les positions PAD du calcul LSTM
        # → le LSTM ne voit que les vrais caractères, pas les zéros de padding
        # enforce_sorted=False : pas besoin de trier par longueur décroissante
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                     batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)        # h_n: (1, batch, 128)
        # h_n[-1] = dernier état caché = résumé de toute la séquence
        return self.fc(h_n[-1])                # (batch, 1)

# === Boucle d'entraînement LSTM ===
# Même structure que train_model/train_with_optimizer mais adapté au format (padded, lengths, targets)
def train_lstm(model, train_loader, val_loader, lr=1e-3, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for padded, lengths, targets in train_loader:
            pred = model(padded, lengths).squeeze(-1)  # (batch,)
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            # Clipping : les RNN sont sujets à l'explosion de gradient
            # (le gradient est multiplié par W_hh à chaque pas de temps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            all_p, all_t = [], []
            for padded, lengths, targets in val_loader:
                pred = model(padded, lengths).squeeze(-1)
                all_p.append(pred)
                all_t.append(targets)
            val_loss = criterion(torch.cat(all_p), torch.cat(all_t)).item()
        val_losses.append(val_loss)

    return model, train_losses, val_losses

# === DataLoaders séquentiels (réutilisés pour Transformer) ===
train_ds = SMILESDataset(train_df["smiles"].values, y_train)
val_ds = SMILESDataset(val_df["smiles"].values, y_val)
train_loader_seq = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_smiles)
val_loader_seq = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_smiles)

# === Entraînement ===
torch.manual_seed(SEED)
np.random.seed(SEED)
lstm_model = LSTMRegressor(VOCAB_SIZE, embed_dim=32, hidden_dim=128, num_layers=1)
n_params_lstm = sum(p.numel() for p in lstm_model.parameters())
print(f"LSTM : {n_params_lstm:,} paramètres")

lstm_model, train_losses_lstm, val_losses_lstm = train_lstm(
    lstm_model, train_loader_seq, val_loader_seq, lr=1e-3, epochs=100
)

# === Résultats ===
lstm_model.eval()
all_preds = []
with torch.no_grad():
    for padded, lengths, targets in val_loader_seq:
        pred = lstm_model(padded, lengths).squeeze(-1)
        all_preds.append(pred)
val_pred_lstm = torch.cat(all_preds).numpy()

mse_lstm, r2_lstm = report_results(
    train_losses_lstm, val_losses_lstm, val_pred_lstm, y_val,
    n_params_lstm, title="2.1 — LSTM"
)
```

**Choix clés** :

- **embed_dim=32** : vocabulaire petit (60 chars) → plongement compact suffisant. Plus grand = plus de paramètres sans bénéfice.
- **hidden_dim=128** : même dimension que le MLP pour permettre une comparaison juste des architectures.
- **pack_padded_sequence** : crucial pour ignorer le padding dans le calcul LSTM. Sans cela, le LSTM traite les zéros de padding comme des vrais tokens → bruit dans l'état caché.
- **h_n[-1]** (pas output) : on utilise l'état caché final, pas les sorties à chaque pas. L'état final est le "résumé" de toute la séquence — c'est l'agrégation la plus simple et standard.
- **clip*grad_norm*(5.0)** : les RNN propagent le gradient à travers le temps — le gradient est multiplié par $W_{hh}$ à chaque pas de temps (BPTT). Sans clipping, explosion exponentielle probable sur les longues séquences.

### Résultats

| Métrique        | Valeur  |
| --------------- | ------- |
| MSE val         | 561 551 |
| R² val          | -0.0417 |
| Nb. paramètres  | 85 025  |
| Écart train-val | 20 055  |

### Analyse

Prédiction 3 carrément fausse — le LSTM fait pire que le MLP. R²≈0 veut dire qu'il prédit à peine mieux que la moyenne, pas "pire que le hasard". L'écart train-val de 20k = zéro surapprentissage, le modèle n'a même pas appris assez pour commencer à mémoriser.

**Pourquoi ça échoue ?** Le bottleneck c'est les données, pas la capacité. L'embedding doit apprendre de zéro ce que chaque caractère veut dire — avec 10k exemples c'est pas assez. Le MLP triche : il voit les fréquences directement. Le LSTM doit redécouvrir cette info en lisant caractère par caractère → demande beaucoup plus de données.

C'est le scénario classique de ch4 (généralisation) : augmenter la complexité n'aide pas quand le bottleneck est la quantité de données. C'est la motivation de la Partie 3 — le transfer learning apporte des représentations déjà apprises sur 91M de molécules.

**Note sur les prédictions fausses** : c'est normal et attendu ! Le prof veut qu'on prédise sur la base de la théorie puis qu'on analyse _pourquoi_ la réalité diffère. La théorie dit "plus de capacité = meilleur fit", mais la pratique montre les limites du compromis biais-variance avec peu de données.

---

## Dérivations manuscrites — Partie 2

### Calcul d'attention à la main (2.2)

**Données** : SMILES `CCO`, $d_{\text{model}} = 4$, $d_k = 2$.

**Plongements** (lignes de $E$) :

$$E = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix}$$

(ligne 1 = C, ligne 2 = C, ligne 3 = O)

#### Étape 1 : Calcul de $Q$, $K$, $V$

$Q = EW_Q$ :

$$Q = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1{\cdot}1 + 0{\cdot}0 + 1{\cdot}1 + 0{\cdot}0 & 1{\cdot}0 + 0{\cdot}1 + 1{\cdot}1 + 0{\cdot}0 \\ \text{idem} & \text{idem} \\ 0{\cdot}1 + 1{\cdot}0 + 0{\cdot}1 + 1{\cdot}0 & 0{\cdot}0 + 1{\cdot}1 + 0{\cdot}1 + 1{\cdot}0 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 2 & 1 \\ 0 & 1 \end{pmatrix}$$

$K = EW_K$ :

$$K = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 2 \\ 0 & 2 \\ 2 & 0 \end{pmatrix}$$

$V = EW_V$ :

$$V = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 2 & 1 \\ 0 & 1 \end{pmatrix}$$

#### Étape 2 : Scores d'attention $S = QK^\top / \sqrt{d_k}$

$$QK^\top = \begin{pmatrix} 2 & 1 \\ 2 & 1 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 0 & 0 & 2 \\ 2 & 2 & 0 \end{pmatrix} = \begin{pmatrix} 2 & 2 & 4 \\ 2 & 2 & 4 \\ 2 & 2 & 0 \end{pmatrix}$$

$$S = \frac{QK^\top}{\sqrt{2}} = \begin{pmatrix} \sqrt{2} & \sqrt{2} & 2\sqrt{2} \\ \sqrt{2} & \sqrt{2} & 2\sqrt{2} \\ \sqrt{2} & \sqrt{2} & 0 \end{pmatrix} \approx \begin{pmatrix} 1.414 & 1.414 & 2.828 \\ 1.414 & 1.414 & 2.828 \\ 1.414 & 1.414 & 0 \end{pmatrix}$$

#### Étape 3 : Softmax (par ligne)

**Lignes 1 et 2** (identiques car $q_C = q_C$) : $s = [\sqrt{2}, \sqrt{2}, 2\sqrt{2}]$

- $e^{\sqrt{2}} \approx 4.113$, $e^{2\sqrt{2}} \approx 16.919$
- Somme = $4.113 + 4.113 + 16.919 = 25.145$
- $A_{1,:} = A_{2,:} = [0.1636, \; 0.1636, \; 0.6727]$

**Ligne 3** (O) : $s = [\sqrt{2}, \sqrt{2}, 0]$

- $e^0 = 1$
- Somme = $4.113 + 4.113 + 1 = 9.226$
- $A_{3,:} = [0.4459, \; 0.4459, \; 0.1084]$

$$A = \text{softmax}(S) \approx \begin{pmatrix} 0.164 & 0.164 & 0.673 \\ 0.164 & 0.164 & 0.673 \\ 0.446 & 0.446 & 0.108 \end{pmatrix}$$

#### Étape 4 : Sortie $O = AV$

$$O = \begin{pmatrix} 0.164 & 0.164 & 0.673 \\ 0.164 & 0.164 & 0.673 \\ 0.446 & 0.446 & 0.108 \end{pmatrix} \begin{pmatrix} 2 & 1 \\ 2 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0.655 & 1.0 \\ 0.655 & 1.0 \\ 1.784 & 1.0 \end{pmatrix}$$

Vérification : la 2e colonne de $V$ est $(1, 1, 1)^\top$ et $\sum_j A_{ij} = 1$ → la 2e colonne de $O$ vaut toujours 1.0. ✓

#### Interprétation

- Les deux atomes C **attendent fortement l'oxygène** (poids 0.67) : le score Q-K est maximal ($2\sqrt{2}$) entre la query de C et la key de O. Cela reflète qu'en chimie, l'oxygène change fortement les propriétés.
- L'oxygène **attend les deux C de manière symétrique** (poids 0.45 chacun) : les deux C ont les mêmes plongements, donc les mêmes keys → scores identiques.
- Les deux C identiques produisent la **même sortie** car ils ont les mêmes queries et voient les mêmes keys/values.

---

## Partie 2.3 — Transformeur encodeur

### Pipeline

```
SMILES → indices (0=PAD) → Embedding(64) → + PE sinusoïdal → TransformerEncoder(2 couches, 4 têtes) → mean pooling masqué → Linear(1) → Tc
```

### Code complet et commenté

```python
import torch
import torch.nn as nn
import numpy as np
import math

torch.manual_seed(SEED)
np.random.seed(SEED)

# === Encodage positionnel sinusoïdal ===
# Injecte l'information de position dans le modèle (le Transformer n'a pas de récurrence)
# PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
# PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer : pas un paramètre apprenable, mais sauvegardé avec le modèle
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# === Transformeur encodeur pour régression ===
class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=512)
        # Chaque couche : MultiHeadAttention + FFN + LayerNorm + résiduel
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        # Tronquer au MAX_LEN=128 pour éviter l'explosion O(n²)
        # Certains SMILES font 300+ chars → matrices 300×300 par tête = très lent
        x = x[:, :MAX_LEN]
        lengths = lengths.clamp(max=MAX_LEN)

        # Masque de padding : True = ignorer cette position
        pad_mask = (x == PAD_IDX)  # (batch, seq_len)
        emb = self.embedding(x)     # (batch, seq_len, d_model)
        emb = self.pos_encoder(emb)
        # Le masque empêche l'attention de regarder les positions PAD
        out = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)

        # Mean pooling masqué : moyenne uniquement sur les positions réelles
        # Différence avec LSTM : LSTM utilise h_n[-1] (dernier état),
        # le Transformer n'a pas d'ordre → on prend la moyenne
        mask = (~pad_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, d_model)
        return self.fc(pooled)


# === Boucle d'entraînement (même format que LSTM) ===
def train_transformer(model, train_loader, val_loader, lr=1e-3, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for padded, lengths, targets in train_loader:
            pred = model(padded, lengths).squeeze(-1)
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            all_p, all_t = [], []
            for padded, lengths, targets in val_loader:
                pred = model(padded, lengths).squeeze(-1)
                all_p.append(pred)
                all_t.append(targets)
            val_loss = criterion(torch.cat(all_p), torch.cat(all_t)).item()
        val_losses.append(val_loss)

    return model, train_losses, val_losses


# === Entraînement ===
torch.manual_seed(SEED)
np.random.seed(SEED)

transformer_model = TransformerRegressor(
    VOCAB_SIZE, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1
)
n_params_tf = sum(p.numel() for p in transformer_model.parameters())
print(f"Transformeur : {n_params_tf:,} paramètres")

transformer_model, train_losses_tf, val_losses_tf = train_transformer(
    transformer_model, train_loader_seq, val_loader_seq, lr=1e-3, epochs=100
)

# === Résultats ===
transformer_model.eval()
all_preds_tf = []
with torch.no_grad():
    for padded, lengths, targets in val_loader_seq:
        pred = transformer_model(padded, lengths).squeeze(-1)
        all_preds_tf.append(pred)
val_pred_tf = torch.cat(all_preds_tf).numpy()

mse_tf, r2_tf = report_results(
    train_losses_tf, val_losses_tf, val_pred_tf, y_val,
    n_params_tf, title="2.3 — Transformeur encodeur"
)
```

**Choix clés** :

- **d_model=64, nhead=4** : chaque tête d'attention a $d_k = 64/4 = 16$ dims. Plus grand → plus de paramètres → risque de surapprentissage sur 10k exemples.
- **num_layers=2** : consigne de l'énoncé. Profondeur minimale pour capturer des interactions multi-niveaux.
- **dim_feedforward=128** : FFN interne de chaque couche. Ratio 2:1 avec d_model (standard = 4:1 dans les gros modèles).
- **dropout=0.1** : régularisation légère. Plus agressif risque de casser l'apprentissage sur peu de données.
- **PE sinusoïdal** (pas appris) : avec <512 positions et <10k exemples, un PE appris surapprend. Le sinusoïdal généralise sans paramètres.
- **Mean pooling masqué** (pas CLS token) : pas de token spécial à apprendre. La moyenne est plus robuste avec peu de données.
- **src_key_padding_mask** : empêche l'attention de regarder les PADs → les poids softmax ne sont calculés que sur les vrais tokens.

### Résultats

| Métrique        | Valeur  |
| --------------- | ------- |
| MSE val         | 470 584 |
| R² val          | 0.1270  |
| Nb. paramètres  | 70 913  |
| Écart train-val | 53 557  |

### Analyse

R²=0.13 ≈ MLP (0.12). Le Transformer from scratch ne fait pas mieux qu'un simple comptage de caractères. Pourquoi ?

1. **Attention ≈ bag-of-chars** : avec 10k exemples, le Transformer n'apprend pas de patterns d'attention chimiquement significatifs. Le mean pooling masqué revient à pondérer les fréquences de caractères — exactement ce que le MLP fait directement via les features fixes.

2. **Trop peu de données pour l'embedding** : comme le LSTM, le Transformer doit apprendre son embedding from scratch. Mais contrairement au LSTM (R²=-0.04), l'attention globale permet au moins de voir tous les caractères simultanément → performance comparable au MLP.

3. **Écart modéré** (53k) : entre le LSTM (20k, sous-apprentissage pur) et le MLP (306k, surapprentissage fort). Le dropout=0.1 + l'attention régularisent légèrement.

4. **Troncation MAX_LEN=64** : impact négligeable car ~85% des SMILES font <50 chars. Les rares molécules tronquées (polymères longs) ne changent pas les statistiques globales.

**Conclusion** : le bottleneck n'est pas l'architecture mais la qualité des représentations. Le transfer learning (Partie 3) devrait résoudre ce problème en fournissant des embeddings pré-entraînés sur 91M de molécules.

---

## Partie 2.4 — Tableau comparatif

| Modèle       | MSE val | R² val | Nb. paramètres | Écart train-val |
| ------------ | ------- | ------ | -------------- | --------------- |
| MLP          | 473 172 | 0.12   | 24 577         | 306 085         |
| LSTM         | 561 551 | -0.04  | 85 025         | 20 055          |
| Transformeur | 470 584 | 0.13   | 70 913         | 53 557          |

---

## Partie 3.1 — Plongements SMI-TED

### Concept

SMI-TED est un modèle de fondation pré-entraîné sur **91M de molécules**. Il encode chaque SMILES en un vecteur de 768 dimensions qui capture la structure chimique. La différence fondamentale avec nos modèles (MLP, LSTM, Transformer) : ceux-ci apprennent les représentations from scratch sur 10k exemples, alors que SMI-TED a déjà vu 91M de molécules → les plongements sont riches et informatifs dès le départ.

### Code complet et commenté

```python
import os, sys, torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# === Chargement de SMI-TED ===
# On clone le dépôt IBM et installe les dépendances
!git clone --depth 1 https://github.com/IBM/materials.git /tmp/materials 2>/dev/null || true
!pip install -q pytorch-fast-transformers torch-optimizer huggingface_hub

SMI_TED_DIR = "/tmp/materials/models/smi_ted/inference/smi_ted_light"
CKPT_FILENAME = "smi-ted-Light_40.pt"  # Nom exact sur HuggingFace
if not os.path.exists(os.path.join(SMI_TED_DIR, CKPT_FILENAME)):
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="ibm-research/materials.smi-ted",
                    filename=CKPT_FILENAME, local_dir=SMI_TED_DIR)

sys.path.insert(0, "/tmp/materials/models/smi_ted/inference")
from load import load_smi_ted

smi_ted = load_smi_ted(folder=SMI_TED_DIR, ckpt_filename=CKPT_FILENAME)
smi_ted.eval()

# === Extraction des plongements (par lots pour éviter OOM) ===
BATCH_EMB = 256

def extract_embeddings(smiles_list):
    """Extrait les plongements SMI-TED (768 dims) par lots."""
    all_emb = []
    for i in range(0, len(smiles_list), BATCH_EMB):
        batch = smiles_list[i:i + BATCH_EMB].tolist()
        with torch.no_grad():
            emb = smi_ted.encode(batch, return_torch=True)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb).numpy()

emb_train = extract_embeddings(train_df["smiles"].values)  # (10479, 768)
emb_val = extract_embeddings(val_df["smiles"].values)      # (2620, 768)

# === Visualisation PCA + t-SNE colorée par Tc ===
emb_all = np.vstack([emb_train, emb_val])
tc_all = np.concatenate([y_train, y_val])

pca = PCA(n_components=2)
emb_2d_pca = pca.fit_transform(emb_all)

# t-SNE sur sous-échantillon (3000 points — t-SNE est O(n²))
N_TSNE = 3000
idx_sub = np.random.choice(len(emb_all), N_TSNE, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, n_iter=1000)
emb_2d_tsne = tsne.fit_transform(emb_all[idx_sub])
```

**Choix clés** :

- **Extraction par lots de 256** : le modèle SMI-TED est lourd en mémoire. Encoder 10k SMILES d'un coup → OOM.
- **PCA sur toutes les données, t-SNE sur 3000** : PCA est linéaire O(n), t-SNE est O(n²) → trop lent sur 13k points.
- **Coloré par Tc** : on veut voir si les plongements séparent naturellement les molécules par température critique. Si oui → une simple couche linéaire suffit (3.2).

### Résultats

_(Compléter après exécution)_

---

## Partie 3.2 — Sonde linéaire

### Concept

Geler SMI-TED (pas de fine-tuning) et entraîner uniquement une couche `Linear(768, 1)` sur les plongements. C'est la manière la plus simple d'exploiter un modèle de fondation — si ça marche bien, c'est que les embeddings sont déjà linéairement séparables par rapport à Tc.

### Code complet et commenté

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(SEED)
np.random.seed(SEED)

# === Préparer les plongements comme tenseurs ===
# emb_train/emb_val viennent de la cellule 3.1 (numpy arrays, shape (N, 768))
emb_train_t = torch.tensor(emb_train, dtype=torch.float32)
emb_val_t = torch.tensor(emb_val, dtype=torch.float32)

# Normalisation z-score (stats du train seulement)
emb_mean = emb_train_t.mean(dim=0)
emb_std = emb_train_t.std(dim=0) + 1e-8
emb_train_norm = (emb_train_t - emb_mean) / emb_std
emb_val_norm = (emb_val_t - emb_mean) / emb_std

train_loader_emb = DataLoader(
    TensorDataset(emb_train_norm, y_train_t),
    batch_size=256, shuffle=True
)

# === Sonde linéaire : une seule couche Linear(768, 1) ===
# 769 paramètres (768 poids + 1 biais) — véritablement minimal
linear_probe = nn.Linear(768, 1)
n_params_probe = sum(p.numel() for p in linear_probe.parameters())

# Entraînement Adam, 100 epochs (même budget que les autres modèles)
optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)
criterion = nn.MSELoss()
train_losses_probe, val_losses_probe = [], []

for epoch in range(100):
    linear_probe.train()
    epoch_loss, n_batches = 0.0, 0
    for xb, yb in train_loader_emb:
        pred = linear_probe(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    train_losses_probe.append(epoch_loss / n_batches)

    linear_probe.eval()
    with torch.no_grad():
        val_pred_probe = linear_probe(emb_val_norm)
        val_loss = criterion(val_pred_probe, y_val_t).item()
    val_losses_probe.append(val_loss)

# === Résultats ===
linear_probe.eval()
with torch.no_grad():
    val_pred_probe = linear_probe(emb_val_norm).numpy()

mse_probe, r2_probe = report_results(
    train_losses_probe, val_losses_probe, val_pred_probe, y_val,
    n_params_probe, title="3.2 — Sonde linéaire (SMI-TED gelé)"
)
```

**Choix clés** :

- **Geler SMI-TED** : on n'entraîne pas le modèle de fondation, on utilise ses embeddings comme features fixes. C'est le "linear probe" standard en transfer learning.
- **Normalisation z-score** : les 768 dimensions ont des échelles différentes. Sans normalisation, les grandes dimensions dominent le gradient.
- **769 paramètres** : le modèle le plus simple possible. Si R² est élevé, c'est que SMI-TED a déjà encodé Tc de manière quasi-linéaire.

### Résultats

_(Compléter après exécution)_

---

## Partie 3.3 — Courbe d'efficacité en échantillons

### Concept

Entraîner les 4 modèles (MLP, LSTM, Transformer, SMI-TED+sonde) sur 10%, 25%, 50%, 100% des données d'entraînement. Le set de validation reste fixe. Objectif : montrer que SMI-TED est plus efficace en données (data-efficient) car ses embeddings sont déjà informatifs.

### Code complet et commenté

```python
import torch
import torch.nn as nn
import numpy as np
import time

torch.manual_seed(SEED)
np.random.seed(SEED)

FRACTIONS = [0.10, 0.25, 0.50, 1.00]

# === Fonctions d'entraînement simplifiées (retourne MSE val) ===
# Chaque fonction recrée le modèle avec SEED fixe, entraîne 100 epochs,
# et retourne la MSE de validation finale.

def train_mlp_quick(X_tr, y_tr, X_v, y_v, epochs=100):
    torch.manual_seed(SEED)
    model = MLP(X_tr.shape[1], hidden_dim=128)
    loader = DataLoader(TensorDataset(X_tr, y_tr.unsqueeze(1)), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return crit(model(X_v), y_v.unsqueeze(1)).item()

def train_lstm_quick(train_smiles, y_tr, val_smiles, y_v, epochs=100):
    torch.manual_seed(SEED)
    tr_ds = SMILESDataset(train_smiles, y_tr)
    v_ds = SMILESDataset(val_smiles, y_v)
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, collate_fn=collate_smiles)
    v_loader = DataLoader(v_ds, batch_size=256, shuffle=False, collate_fn=collate_smiles)
    model = LSTMRegressor(VOCAB_SIZE, embed_dim=32, hidden_dim=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for p, l, t in tr_loader:
            opt.zero_grad()
            crit(model(p, l).squeeze(-1), t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for p, l, t in v_loader:
            preds.append(model(p, l).squeeze(-1)); targs.append(t)
        return crit(torch.cat(preds), torch.cat(targs)).item()

def train_tf_quick(train_smiles, y_tr, val_smiles, y_v, epochs=100):
    torch.manual_seed(SEED)
    tr_ds = SMILESDataset(train_smiles, y_tr)
    v_ds = SMILESDataset(val_smiles, y_v)
    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, collate_fn=collate_smiles_truncated)
    v_loader = DataLoader(v_ds, batch_size=256, shuffle=False, collate_fn=collate_smiles_truncated)
    model = TransformerRegressor(VOCAB_SIZE, d_model=64, nhead=4, num_layers=2,
                                  dim_feedforward=128, dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for p, l, t in tr_loader:
            opt.zero_grad()
            crit(model(p, l).squeeze(-1), t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        preds, targs = [], []
        for p, l, t in v_loader:
            preds.append(model(p, l).squeeze(-1)); targs.append(t)
        return crit(torch.cat(preds), torch.cat(targs)).item()

def train_probe_quick(emb_tr, y_tr, emb_v, y_v, epochs=100):
    torch.manual_seed(SEED)
    mu, sigma = emb_tr.mean(dim=0), emb_tr.std(dim=0) + 1e-8
    emb_tr_n = (emb_tr - mu) / sigma
    emb_v_n = (emb_v - mu) / sigma
    model = nn.Linear(768, 1)
    loader = DataLoader(TensorDataset(emb_tr_n, y_tr.unsqueeze(1)), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return crit(model(emb_v_n), y_v.unsqueeze(1)).item()

# === Boucle : chaque fraction × chaque modèle ===
results_eff = {name: [] for name in ["MLP", "LSTM", "Transformeur", "SMI-TED + sonde"]}

for frac in FRACTIONS:
    n = int(len(train_df) * frac)
    np.random.seed(SEED)
    idx = np.random.choice(len(train_df), n, replace=False)
    idx.sort()

    sub_smiles = train_df["smiles"].values[idx]
    sub_y = y_train[idx]
    sub_X = X_train_t[idx]
    sub_y_t = y_train_t[idx]
    sub_emb = emb_train_t[idx]

    print(f"\nFraction = {frac:.0%} ({n} exemples)")
    t0 = time.time()

    mse = train_mlp_quick(sub_X, sub_y_t.squeeze(), X_val_t, y_val_t.squeeze())
    results_eff["MLP"].append(mse)
    print(f"  MLP           : {mse:,.0f}  ({time.time()-t0:.0f}s)")

    t1 = time.time()
    mse = train_lstm_quick(sub_smiles, sub_y, val_df["smiles"].values, y_val)
    results_eff["LSTM"].append(mse)
    print(f"  LSTM          : {mse:,.0f}  ({time.time()-t1:.0f}s)")

    t1 = time.time()
    mse = train_tf_quick(sub_smiles, sub_y, val_df["smiles"].values, y_val)
    results_eff["Transformeur"].append(mse)
    print(f"  Transformeur  : {mse:,.0f}  ({time.time()-t1:.0f}s)")

    t1 = time.time()
    mse = train_probe_quick(sub_emb, torch.tensor(sub_y, dtype=torch.float32),
                            emb_val_t, torch.tensor(y_val, dtype=torch.float32))
    results_eff["SMI-TED + sonde"].append(mse)
    print(f"  SMI-TED+sonde : {mse:,.0f}  ({time.time()-t1:.0f}s)")

# === Graphique ===
sizes = [int(len(train_df) * f) for f in FRACTIONS]
plt.figure(figsize=(9, 5))
for name, mses in results_eff.items():
    plt.plot(sizes, mses, 'o-', label=name, linewidth=2, markersize=6)
plt.xlabel("Nombre d'exemples d'entraînement")
plt.ylabel("MSE de validation")
plt.title("3.3 — Efficacité en échantillons")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Résultats attendus

- **SMI-TED + sonde** devrait dominer à toutes les fractions, surtout à 10% où les modèles from-scratch n'ont presque rien à apprendre.
- **MLP** et **Transformer** devraient se suivre de près (bag-of-chars).
- **LSTM** sera le pire à toutes les fractions (embedding from scratch trop dur).
- La courbe SMI-TED sera **plate** (peu sensible à la quantité de données) car les embeddings sont déjà informatifs.

---

---

# ❓ SECTION 2 — Questions d'entrevue par section

## Partie 1 — MLP & Optimisation

### Q1 : C'est quoi `smiles_to_features` exactement ?

Un vecteur de longueur fixe = 1 entrée par caractère du vocabulaire (combien de fois `C`, `O`, `(`, `=` etc. apparaissent) + 1 entrée pour la longueur totale du SMILES.

Exemple : `CCO` (éthanol) → `C` apparaît 2 fois, `O` apparaît 1 fois, longueur = 3.

**Limitation** : `CCO` et `COC` (diméthyl éther) donnent le **même vecteur** alors que ce sont des molécules différentes ! L'ordre est ignoré.

---

### Q2 : Pourquoi la MSE et pas une autre métrique ?

- Tc est une valeur continue en Kelvin → problème de **régression**.
- MSE = $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ est la log-vraisemblance négative d'un modèle gaussien.
- Le cours (Ch. 7) montre que minimiser MSE = maximiser la vraisemblance sous hypothèse de bruit gaussien $p(y|x) = \mathcal{N}(\mu(x), \sigma^2)$.
- R² complète le tableau : indique la proportion de variance expliquée (R²=1 parfait, R²=0 = prédire la moyenne, R²<0 = pire que la moyenne).

---

### Q3 : Comment fonctionne la rétropropagation dans un MLP 2 couches ?

Graphe de calcul : $x \to h_1 = \text{ReLU}(W_1 x + b_1) \to h_2 = \text{ReLU}(W_2 h_1 + b_2) \to \hat{y} = W_3 h_2 + b_3 \to \mathcal{L} = (\hat{y} - y)^2$

**VJP de matmul** : Si $z = Wx$, alors $\partial\mathcal{L}/\partial W = (\partial\mathcal{L}/\partial z) \cdot x^\top$ et $\partial\mathcal{L}/\partial x = W^\top \cdot (\partial\mathcal{L}/\partial z)$.

---

### Q4 : Quelle différence entre SGD, Momentum, et Adam ?

| Optimiseur   | Mécanisme                                                                        | Avantage                                                       |
| ------------ | -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **SGD**      | $\theta \leftarrow \theta - \eta g$                                              | Simple, peu de mémoire                                         |
| **Momentum** | Accumule une vitesse $m = \beta m + g$, puis $\theta \leftarrow \theta - \eta m$ | Accélère dans les directions stables, atténue les oscillations |
| **Adam**     | Moment 1 (moyenne) + moment 2 (variance), learning rate adaptatif par paramètre  | Convergence rapide, robuste au choix du lr                     |

**Adam** combine momentum + learning rate adaptatif :
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \quad \theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

### Q5 : Pourquoi Sigmoid + initialisation par défaut cause des problèmes (config 1.3) ?

**Vanishing gradient** : Sigmoid sature à 0 et 1, ses dérivées voisinent zéro. En multipliant les VJP couche par couche, le gradient explose ou disparaît (souvent disparaît).

**Initialisation par défaut** (ex: `torch.nn.Linear` = Kaiming uniforme par défaut) peut ne pas être bien adapté à Sigmoid.

Avec **ReLU + He** : ReLU a dérivée = 1 pour les activations positives, pas de saturation. He initialise $W \sim \mathcal{N}(0, 2/n_{in})$ pour compenser la "moitié morte" de ReLU.

---

### Q6 : À quoi sert BatchNorm ?

Normalise les activations d'un mini-lot : $\hat{h} = \frac{h - \mu_\mathcal{B}}{\sigma_\mathcal{B}}$, puis $y = \gamma \hat{h} + \beta$ (paramètres appris).

- ✅ Stabilise les distributions internes entre couches
- ✅ Permet des learning rates plus grands
- ✅ Agit comme régularisateur léger
- ⚠️ Se comporte différemment à train vs. eval (utilise moyenne/variance globale à l'inférence)

---

### Q7 : À quoi sert Dropout ?

Éteint aléatoirement chaque neurone avec probabilité $p$ pendant l'entraînement. Chaque passe utilise un sous-réseau différent → force une représentation distribuée.

- **Entraînement** : chaque neurone est éteint avec prob $p$, les actifs sont scalés par $1/(1-p)$.
- **Inférence** : tous les neurones actifs (pas de dropout).
- ✅ Réduit la co-adaptation des neurones → régularisation.

---

### Q8 : Pourquoi normaliser les features (z-score) ?

Les features ont des échelles différentes (fréquence de `C` : 0-50 ; longueur : 2-300). Sans normalisation :

- Les gradients sont dominés par les features de grande amplitude
- La convergence est lente et instable
- Le z-score ($\hat{x} = \frac{x-\mu}{\sigma}$) recentre chaque feature à moyenne 0, variance 1
- **Important** : calculer $\mu$ et $\sigma$ sur le **train seulement**, puis les appliquer au val (sinon data leakage)

---

## Partie 2 — LSTM & Transformeur

### Q9 : Pourquoi le LSTM devrait mieux que le MLP sur les SMILES ?

Le MLP ignore l'**ordre** des caractères (bag-of-characters). Le LSTM traite la séquence de gauche à droite, accumulant un état caché qui représente le contexte de la séquence vue jusqu'ici. Pour des molécules où `CC(=O)O` (acide acétique) ≠ `C(=O)(O)C` même composition, l'ordre compte.

**Mais** : LSTM entraîné depuis zéro avec peu de données peut souffrir de sur-apprentissage ou de sous-utilisation de la capacité séquentielle.

---

### Q10 : C'est quoi le padding et le masking, et pourquoi c'est important ?

Les SMILES ont des longueurs variables (min ~2, max ~300+). Pour traiter en batch, on **pad** les séquences courtes avec un token spécial (ex: 0) jusqu'à la longueur max du batch.

Le **masking** assure que les positions paddées sont ignorées dans :

- L'agrégation (moyenne des états LSTM/Transformer) : on ne moyenne que les positions réelles.
- L'attention du Transformer : les positions paddées ne reçoivent pas d'attention.

Sans masking → la moyenne inclut des zéros artificiels → la représentation est biaisée.

---

### Q11 : Comment fonctionne l'encodage positionnel sinusoïdal ?

Les Transformeurs n'ont pas d'ordre intrinsèque (contrairement aux LSTM). On injecte l'information de position avec :
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- Chaque position a un vecteur unique (comme une "empreinte de position")
- Propriété utile : le modèle peut généraliser à des longueurs non vues (interpolation sinusoïdale)

---

### Q12 : Calcul d'attention à la main (CCO) — logique

Données `CCO`, $d_k = 2$ :

1. Calculer $Q = E W_Q$, $K = E W_K$, $V = E W_V$ (matrices $3 \times 2$)
2. Scores : $S = QK^\top / \sqrt{d_k}$ → matrice $3 \times 3$
3. Softmax par ligne : $A = \text{softmax}(S)$
4. Sortie : $O = AV$ → matrice $3 \times 2$

---

## Partie 3 — Transfer Learning (SMI-TED)

### Q13 : C'est quoi SMI-TED et pourquoi c'est utile ?

**SMI-TED** = SMILES-based Transformer Encoder Decoder, pré-entraîné sur **91 millions** de molécules par IBM Research. Produit un plongement de 768 dimensions par molécule.

**Utilité** : En régime de données limitées (~10k), un modèle pré-entraîné apporte une connaissance chimique implicite acquise sur des ordres de grandeur plus de données. La sonde linéaire teste si ces représentations sont déjà "linéairement séparables" pour prédire Tc.

---

### Q14 : C'est quoi une sonde linéaire (linear probe) ?

On **gèle** tous les poids de SMI-TED et on entraîne seulement une couche linéaire $W \in \mathbb{R}^{768 \times 1}$ sur les embeddings extraits.

- ✅ Si la sonde performe bien → les représentations SMI-TED encodent déjà l'information sur Tc
- Si la sonde est médiocre → les représentations sont génériques mais pas spécifiques à Tc

---

### Q15 : Qu'est-ce que la courbe d'efficacité en échantillons révèle ?

On entraîne chaque modèle à 10%, 25%, 50%, 100% des données puis on trace la MSE val.

- Un modèle avec **peu de paramètres** (MLP fixe) se plateau tôt.
- Un modèle avec **transfert learning** (SMI-TED) devrait avoir une bonne performance même à 10%.
- Les modèles séquentiels (LSTM, Transformer) ont besoin de plus de données pour extraire de l'information de l'ordre.
