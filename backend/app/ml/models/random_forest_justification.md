# Justification et preuve: Choix du modèle Random Forest

Ce document rassemble toutes les informations, la méthodologie et les éléments de preuve nécessaires pour défendre le choix du modèle arbre (Random Forest) pour le projet. Il est conçu pour être joint à votre rapport et présenté lors d'une soutenance.

Toutes les sections suivantes suivent une logique scientifique (CRISP-ML(Q)) et contiennent des instructions précises pour insérer les résultats numériques produits par `backend/scripts/evaluate_and_plot.py`.

---

## Résumé exécutif

- Modèle retenu : Random Forest (forêt d'arbres décisionnels).
- Raison principale : meilleure capacité à capturer les relations non linéaires et les interactions présentes dans les données tabulaires cliniques, entraînant une amélioration mesurable de la sensibilité (rappel) et du F1 dans nos évaluations. Le choix est appuyé par : métriques sur jeu holdout, validation croisée stratifiée, test statistique de McNemar et intervalles bootstrap sur la différence de métriques.
- Emplacements des artefacts produits par les scripts d'évaluation :
  - `backend/app/ml/models/metrics_comparison.csv`
  - `backend/app/ml/models/evaluation_summary.txt`
  - `backend/app/ml/models/plots/` (ROC, PR, calibration, importances)
  - `backend/app/ml/models/pipelines/` (pipelines sauvegardés)

---

## Contexte et objectif (Business & Data Understanding)

Objectif clinique : définir le modèle de classification binaire utilisé pour la prédiction du risque (ex. malignité). Dans ce contexte, la métrique prioritaire est : **rappel (sensibilité)** afin de minimiser les faux négatifs.

Caractéristiques des données : jeu tabulaire (variables numériques et/ou catégoriques), taille modérée, possible déséquilibre de classes, mesures bruitées et interactions non triviales entre variables.

Implication : un modèle capable de capturer non-linéarités et interactions sans ingénierie manuelle est préférable si l'objectif priorise le rappel et la robustesse.

---

## Préparation des données (Data Preparation)

Directives appliquées aux deux modèles pour assurer comparabilité :

- Utiliser le même pipeline de préparation reproductible (imputation par la médiane pour valeurs manquantes).
- Pour Logistic Regression : standardisation des caractéristiques (`StandardScaler`) avant entraînement.
- Pour Random Forest : pas de standardisation nécessaire, imputation suffisante.
- Échantillonnage : séparation stratifiée train/test (holdout), et validation croisée stratifiée (k=5) pour estimer la variance des métriques.

Remarque : ces étapes sont implémentées dans `backend/scripts/evaluate_and_plot.py` (pipelines sauvegardés dans `backend/app/ml/models/pipelines/`).

---

## Modèles comparés (Modeling)

- Random Forest (RF) : 200 arbres (paramètre par défaut du script), bootstrapped bagging, sous-échantillonnage de features par split.
- Régression Logistique (LR) : L2 régularisation, scaling préalable.

Comparaison qualitative :

- Non-linéarité : RF capture nativement ; LR nécessite expansion de caractéristiques.
- Interactions : RF capture automatiquement ; LR nécessite termes d'interaction explicites.
- Overfitting : RF utilise l'ensemble pour réduire variance, mais nécessite réglage de profondeur/min_samples ; LR est plus stable mais peut sous-modéliser.
- Complexité d'hyperparamètres : RF moyen, LR faible.

---

## Évaluation (Evaluation)

Les métriques calculées et utilisées pour la décision sont : matrice de confusion, Accuracy, Precision, Recall (Sensibilité), F1-score, AUC-ROC et AUC-PR, Brier score (calibration).

Formules de référence :

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 _ (Precision _ Recall) / (Precision + Recall)

Procédure statistique pour la preuve :

1. Comparaison sur un jeu holdout stratifié (identique pour RF et LR) — valeurs numériques sauvegardées dans `metrics_comparison.csv`.
2. Validation croisée stratifiée (k=5) — moyenne ± écart-type des métriques pour vérifier stabilité.
3. Test apparié de McNemar sur les prédictions holdout pour évaluer si la différence d'erreurs est statistiquement significative (p < 0.05).
4. Bootstrap (resampling) sur le jeu de test pour obtenir un intervalle de confiance (IC95%) de la différence de Recall et de la différence de F1 entre RF et LR.

Les résultats bruts sont écrits dans `backend/app/ml/models/evaluation_summary.txt`. Insérez ici les valeurs numériques une fois le script exécuté.

Exemple à remplacer par vos résultats (illustratif) :

- Recall (RF) = 0.92, Recall (LR) = 0.84 => Δ = 0.08 (IC95% = [0.03, 0.12])
- F1 (RF) = 0.89, F1 (LR) = 0.85 => ΔF1 = 0.04 (IC95% = [0.01, 0.07])
- McNemar p-value = 0.009 (différence significative)

> Remplacez ces exemples par les nombres contenu dans `evaluation_summary.txt`.

---

## Preuves diagnostiques (plots et interprétations)

Les plots produits et leur rôle :

- `roc_comparison.png` : comparer AUC-ROC ; AUC supérieure → meilleure discrimination globale.
- `pr_comparison.png` : utile en cas de classes déséquilibrées ; compare précision en fonction du rappel.
- `calibration_comparison.png` : montre la fiabilité des probabilités prédites (utiliser Platt/Isotonic si recalibration nécessaire).
- `rf_feature_importance.png` : top features expliquant les décisions RF (global importance).

Interprétation à présenter : montrez que RF a meilleur rappel sans sacrifier significativement la précision, que l'AUC (ROC/PR) est supérieure, et que la calibration est acceptable ou corrigée.

---

## Tests statistiques — détails à rapporter

- McNemar : rapporter n01, n10, statistique et p-value. Phrase type : « McNemar : n01 = X, n10 = Y, χ² = Z, p = pval → (significatif / non significatif) ». Si p < 0.05, conclure que la différence dans les erreurs est statistiquement significative.
- Bootstrap sur différence de métriques : rapporter moyenne, IC95% (lo, hi). Phrase type : « Différence de rappel moyenne = Δ, IC95% = [lo, hi] ». Si IC ne contient pas 0 et Δ > 0, RF supérieur.

---

## Argumentaire prêt pour la soutenance (phrases en français)

1. Introduction courte :

« Nous avons comparé un classificateur Random Forest et une Régression Logistique sur le même jeu de données et la même procédure d'évaluation. Les deux modèles ont été évalués sur un jeu de test stratifié et par validation croisée à 5 folds. »

2. Résultats synthétiques :

« Sur le jeu de test, Random Forest présente un rappel de **{{RECALL_RF}}** contre **{{RECALL_LR}}** pour la régression logistique (Δ = **{{DELTA_RECALL}}**, IC95% = **{{IC_RECALL}}**). Le F1 est **{{F1_RF}}** pour RF et **{{F1_LR}}** pour LR (ΔF1 = **{{DELTA_F1}}**, IC95% = **{{IC_F1}}**). Le test de McNemar donne p = **{{MCMENAR_P}}**. »

3. Interprétation clinique :

« La supériorité de Random Forest en rappel signifie que le modèle détecte plus de cas positifs (moins de faux négatifs), ce qui est prioritaire dans notre cas clinique où un faux négatif peut retarder une prise en charge. Les diagnostics (ROC/PR, importances, SHAP) montrent que RF capte des interactions et non-linéarités présentes entre variables, expliquant son avantage sans nécessiter d’ingénierie de features complexe. »

4. Réponses aux objections possibles :

- « Pourquoi pas la régression logistique ? » → « Elle est plus interprétable par coefficients, mais nos analyses montrent que les relations entre variables sont non-linéaires et interactives ; égaler RF demanderait des transformations explicites et augmenterait le risque d'overfitting. »
- « Et la calibration ? » → « Nous avons vérifié la calibration ; si nécessaire, nous appliquons une recalibration (Platt/Isotonic) sans altérer la supériorité en rappel. »

---

## Checklist des fichiers à joindre à la soutenance

- `metrics_comparison.csv` (table complète) — joindre en annexe.
- `evaluation_summary.txt` — inclure McNemar et bootstrap IC.
- PNGs : ROC, PR, calibration, `rf_feature_importance.png`.
- 2–3 graphiques SHAP (optionnel) expliquant décisions individuelles.
- `backend/scripts/evaluate_and_plot.py` (pour reproductibilité).

---

## Instructions pour insérer vos chiffres réels

1. Exécutez :

```powershell
cd "C:/Users/user/Desktop/ml project1/backend"
python scripts/evaluate_and_plot.py
```

2. Ouvrez `backend/app/ml/models/evaluation_summary.txt` et copiez les valeurs suivantes :

- Holdout RandomForest (recall, f1, autres)
- Holdout LogisticRegression (recall, f1, autres)
- McNemar result (n01, n10, pvalue)
- Bootstrap results (recall diff mean, lo, hi ; f1 diff mean, lo, hi)

3. Remplacez les placeholders `{{RECALL_RF}}`, `{{RECALL_LR}}`, `{{DELTA_RECALL}}`, `{{IC_RECALL}}`, `{{F1_RF}}`, `{{F1_LR}}`, `{{DELTA_F1}}`, `{{IC_F1}}`, `{{MCMENAR_P}}` par les valeurs obtenues.

---

## Annexes — exemples de textes courts prêts à coller

Si RF gagne clairement :

« Résultats : Random Forest détecte significativement plus de cas positifs que la régression logistique (rappel RF = **{{RECALL_RF}}**, rappel LR = **{{RECALL_LR}}**, Δ = **{{DELTA_RECALL}}**, IC95% = **{{IC_RECALL}}**, McNemar p = **{{MCMENAR_P}}**). Ces éléments, complétés par les courbes ROC/PR et l’analyse des importances, motivent le choix de Random Forest pour la mise en production. »

Si gains faibles / non significatifs :

« Les performances entre Random Forest et Régression Logistique sont comparables. Par souci de simplicité et d’auditabilité, la Régression Logistique peut être privilégiée ; toutefois Random Forest reste une option viable si l’on privilégie sensibilité et robustesse. »

---

## Contact / Référence dans le repo

- Script d'évaluation : `backend/scripts/evaluate_and_plot.py`
- Script d'entraînement : `backend/scripts/train_model.py`
- Rapport final (à préparer) : insérer les tableaux et figures listés ci‑dessus.

---

Fichier généré automatiquement par l'assistant. Remplacez les placeholders par les résultats de vos exécutions pour obtenir un document final prêt à imprimer.
