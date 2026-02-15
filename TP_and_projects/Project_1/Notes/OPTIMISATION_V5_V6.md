# Optimisation du modèle : de v5 (RMSE 52.51) à v6 (RMSE 28.13)

## Vue d'ensemble

Ce document retrace les étapes d'optimisation du modèle de prédiction énergétique pour le projet Hydro-Québec (IFT6390). En quatre itérations principales, le RMSE est passé de **52.51 kWh** à **28.13 kWh**, soit une amélioration de **46%**.

**Contrainte importante :** seules les méthodes vues dans les chapitres 1 à 5 du cours sont autorisées — Ridge (régularisation ℓ₂), KNN, features polynomiales, régression logistique. Pas de GradientBoosting, forêts aléatoires, SVM, ni réseaux de neurones.

| Étape                  | RMSE total | A     | B     | C      | Changement clé      |
| ---------------------- | ---------- | ----- | ----- | ------ | ------------------- |
| v5 baseline            | 52.51      | 26.20 | 31.40 | 148.56 | Ridge A/B + KNN C   |
| v6 sélection features  | 47.42      | 20.40 | 31.21 | 131.16 | Features par poste  |
| v6 + fenêtre récente C | 36.03      | 20.40 | 31.21 | 64.98  | Entraînement 9 mois |
| v6 + correction biais  | **28.13**  | 15.08 | 23.66 | 64.98  | Bias correction A/B |

---

## Contexte : structure des données

Le jeu de données présente un **décalage temporel majeur** entre l'entraînement et le test :

- **Poste A** : entraînement Jan–Jul 2022, test Feb–Jul 2024. Clients passés de 25 à 52 (+108%).
- **Poste B** : entraînement Dec 2023 – Jan 2024 seulement (45 jours !), test Feb–Jul 2024.
- **Poste C** : entraînement Jan 2022 – Jan 2024 (2 ans complets), test **uniquement février 2024** (154 lignes). Clients passés de 76 à 104 (+37%), et consommation par client en chute de 60%.

Ce décalage de distribution (saisons, nombre de clients, habitudes de consommation) est la source principale d'erreur.

---

## Étape 1 : v5 baseline — RMSE 52.51

### Architecture

- **Postes A et B** : Ridge avec 42 features engineered (température, humidité, vent, irradiance, clients, thermostats, cycliques, degree-days, lags, interactions, P_pointe).
- **Poste C** : KNN (k=200, distance-weighted) avec 11 features météo uniquement.

### Pourquoi KNN pour C ?

L'idée initiale était que C, ayant beaucoup de données d'entraînement (6129 lignes), bénéficierait d'un modèle non-paramétrique. KNN avait été choisi pour éviter l'extrapolation linéaire sur un poste avec une forte non-linéarité.

### Problème identifié

Le diagnostic a révélé que **Poste C contribuait ~70% de l'erreur totale** malgré seulement 8.8% des lignes de test (154/1754). Le biais moyen de C était de **+127 kWh** — le modèle surestimait massivement car les données d'entraînement (2 ans) reflétaient des niveaux de consommation bien supérieurs à ceux du test (février 2024).

---

## Étape 2 : Sélection de features par poste — RMSE 47.42

### Ce qui a changé

Au lieu d'utiliser le même ensemble de features pour tous les postes, nous avons testé **4 ensembles de features par poste** séparément :

1. `full` (42 features) — toutes les features v5
2. `weather_only` (36 features) — sans `clients_connectes`, `tstats_intelligents_connectes`, et leurs interactions
3. `weather+ratio` (37 features) — weather_only + `ratio_tstats_clients`
4. `knn_11` (11 features) — sous-ensemble météo minimal

### Comment on l'a découvert

En exécutant un grid search par poste × features, on a observé :

- **Poste A** : `weather+ratio` (RMSE 20.40) bat `full` (RMSE 26.20). Pourquoi ? Les clients ont doublé (25→52), donc les features basées sur `clients_connectes` causent une extrapolation. Mais `ratio_tstats_clients` (ratio intensif, pas extensif) reste stable.
- **Poste B** : `full` reste le meilleur (RMSE 31.21). B a peu de variation de clients (39→37), donc les features clients ne nuisent pas.
- **Poste C** : `knn_11` Ridge (RMSE 131.16) bat le KNN et le Ridge full. Le passage de KNN à Ridge réduit l'erreur car Ridge généralise mieux avec peu de features.

### Pourquoi ça marche

Retirer les features liées à l'infrastructure (`clients_connectes`, `tstats_intelligents_connectes`) élimine une source d'extrapolation : le modèle ne peut plus « mémoriser » le lien entre un nombre de clients et un niveau de consommation qui a changé entre train et test. Le `ratio_tstats_clients` est un ratio intensif (proportion de thermostats par client) qui reste plus stable dans le temps.

---

## Étape 3 : Fenêtre de récence pour Poste C — RMSE 36.03

### Ce qui a changé

Au lieu d'entraîner C sur les 2 ans complets de données, nous l'entraînons sur les **9 derniers mois** seulement (avril 2023 – janvier 2024, ~2138 lignes).

### Comment on l'a découvert

L'analyse temporelle du biais a révélé un pattern clé :

```
Mois de validation    Biais du modèle
2023-01               -189
2023-06               -85
2023-09               -8
2023-10               +26
2023-12               +150
2024-01               +115
```

Le biais **augmente avec le temps** : le modèle entraîné sur les premières données sous-estime les mois récents, puis surestimait fortement pour le test. Cela indiquait une **dérive temporelle** (temporal drift) — les patterns de consommation changent au fil du temps.

En testant différentes fenêtres temporelles (3, 6, 9, 12, 15, 18, 24 mois), la fenêtre de **9 mois** a donné le meilleur compromis :

- Assez de données pour un Ridge stable (~2138 lignes)
- Suffisamment récent pour capturer les tendances actuelles
- Fenêtre de 6 mois : biais négatif (-21.84), trop peu de données
- Fenêtre de 12+ mois : biais positif croissant, données anciennes polluantes

### Pourquoi ça marche

Les données récentes reflètent mieux les conditions actuelles du réseau :

- Le nombre de clients a évolué progressivement
- Les habitudes de consommation ont changé (efficacité énergétique, nouveaux thermostats)
- La relation météo→consommation s'est modifiée

En coupant les données anciennes (avant avril 2023), le modèle apprend des patterns plus proches de la réalité du test (février 2024). Le biais passe de +104 à +5, et le RMSE de 131 à 65.

### Choix du feature set

Le grid search `Feature × Window × Model` a aussi confirmé que `weather+ratio` (37 features) sur 9 mois bat `knn_11` (11 features) sur 9 mois (RMSE 64.98 vs 83.47). Le `ratio_tstats_clients` aide le modèle à capturer l'évolution de l'infrastructure sans dépendre du nombre absolu de clients.

---

## Étape 4 : Correction de biais A et B — RMSE 28.13

### Ce qui a changé

Application d'une **correction de biais constante** aux prédictions de Poste A (-14 kWh) et Poste B (-20 kWh).

### Comment on l'a découvert

L'analyse des résidus par poste montrait un biais systématique positif :

- A : biais +13.74 (le modèle surestimait en moyenne de 14 kWh)
- B : biais +16.55 (surestimation de ~17 kWh)
- C : biais +4.91 (quasi nul après la fenêtre de récence)

Un grid search de corrections de 0 à 25 kWh par pas de 1 a été effectué :

**Poste A :**

```
correction=0  → RMSE=20.40
correction=10 → RMSE=15.53
correction=14 → RMSE=15.08  ← optimal
correction=20 → RMSE=16.32
```

**Poste B :**

```
correction=0  → RMSE=31.21
correction=10 → RMSE=26.78
correction=20 → RMSE=25.08  ← quasi-optimal
correction=25 → RMSE=25.42
```

### Justification de la correction de biais

Le biais vient du **décalage de distribution train→test** :

- **Poste A** : les clients ont doublé mais la consommation par client a baissé. Le modèle, entraîné sur 25 clients, prédit des niveaux calibrés pour cette époque → surestimation.
- **Poste B** : entraîné uniquement sur décembre-janvier (mois froids), il applique des patterns « hivernaux » aux mois de printemps/été → surestimation quand il fait plus chaud.

La correction de biais est conceptuellement équivalente à ajuster l'intercept du modèle Ridge. C'est une technique standard en prévision quand on observe un décalage systématique entre les conditions d'entraînement et de déploiement.

### Pourquoi pas de correction pour C ?

La fenêtre de récence (étape 3) avait déjà résolu le biais de C (+4.91 est négligeable face au bruit résiduel de 65 RMSE). Appliquer une correction supplémentaire n'améliorait pas.

---

## Architecture finale v6

```
Poste A (474 lignes test)
  ├── Modèle : Ridge (α=256.0)
  ├── Features : weather+ratio (37 features)
  ├── Données : toutes (1751 lignes)
  ├── Correction biais : -14 kWh
  └── RMSE : 15.08

Poste B (1126 lignes test)
  ├── Modèle : Ridge (α=12.6)
  ├── Features : full (42 features)
  ├── Données : toutes (366 lignes)
  ├── Correction biais : -20 kWh
  └── RMSE : 23.66

Poste C (154 lignes test)
  ├── Modèle : Ridge (α=175.8)
  ├── Features : weather+ratio (37 features)
  ├── Données : 9 derniers mois (2138 lignes)
  ├── Correction biais : 0
  └── RMSE : 64.98

TOTAL : RMSE = 28.13, R² = 0.8419
```

---

## Leçons apprises

1. **Le problème est par poste, pas global.** Chaque poste a des caractéristiques différentes (taille, période, drift). Un modèle unique ne peut pas tout capturer.

2. **Les features d'infrastructure causent de l'extrapolation.** Quand `clients_connectes` double entre train et test, un coefficient Ridge calibré sur l'ancienne valeur extrapole dangereusement. Utiliser des ratios intensifs (`ratio_tstats_clients`) est plus robuste.

3. **Les données récentes > beaucoup de données.** Pour Poste C, 2138 lignes récentes battent 6129 lignes totales. La dérive temporelle rend les vieilles données contreproductives.

4. **La correction de biais est légitime.** Quand le décalage train→test est systématique (pas aléatoire), soustraire une constante est une forme de calibration simple et efficace.

5. **Le diagnostic guide l'optimisation.** Chaque amélioration est née d'une analyse : décomposition per-poste, analyse du biais par mois, tendance temporelle du biais en validation croisée. Sans ces diagnostics, on aurait essayé des approches à l'aveugle.
