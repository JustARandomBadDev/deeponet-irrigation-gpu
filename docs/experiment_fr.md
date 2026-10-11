# Carnet d'expérimentation — Prédiction de l'humidité des sols et accélération GPU

English version: [experiment_en.md](experiment_en.md).

Ce carnet reconstruit l'expérience dans son ordre logique, depuis les données
agronomiques jusqu'aux optimisations GPU. Il ne décrit donc pas seulement le
système final : il explique pourquoi chaque expérience a été menée, ce qu'elle
a montré et quelle décision en a découlé.

Les valeurs présentées ici proviennent en priorité des résultats
machine-readable figés dans `results/`, puis des fichiers de benchmark finaux et
de la documentation courante. Les anciens résultats restent utiles pour retracer
l'histoire, mais ne remplacent pas les mesures finales lorsqu'une expérience a
été répétée avec un protocole plus complet.

## 1. Objectif et question de recherche

L'irrigation de précision cherche à apporter de l'eau au bon endroit et au bon
moment. Une prévision de l'humidité du sol peut aider à anticiper une baisse
d'humidité, à éviter un apport inutile ou à mieux planifier l'usage de l'eau.
Dans ce projet, l'objectif prédictif est d'estimer l'humidité future à plusieurs
horizons, de 1 à 24 heures, à partir des dernières 24 heures de mesures.

Le projet partait également d'un intérêt pour **DeepONet**, une architecture de
réseau neuronal conçue pour apprendre des opérateurs : au lieu d'associer
simplement un vecteur à une sortie, elle représente une fonction historique et
l'interroge en un point, ici l'horizon de prévision. Il n'était cependant pas
supposé que cette architecture serait automatiquement plus précise qu'un réseau
plus simple.

Deux questions distinctes ont donc guidé le travail :

1. **Qualité prédictive :** DeepONet améliore-t-il la prévision de l'humidité du
   sol par rapport à une règle simple et à un perceptron multicouche ?
2. **Performance de calcul :** dans quelles conditions l'inférence GPU devient-
   elle utile pour ce DeepONet ?

Cette séparation est essentielle. Un modèle peut être intéressant pour étudier
une architecture d'opérateur et son déploiement GPU sans être le meilleur
prédicteur de l'expérience.

## 2. Dataset Arnesano

### 2.1 Données retenues

Le projet utilise la version 2 du jeu de données *Soil Moisture, Irrigation
Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation Field
Trial*, publié sur Mendeley Data sous le
[DOI 10.17632/c837v6p8ph.2](https://doi.org/10.17632/c837v6p8ph.2). Il décrit une
exploitation instrumentée à Arnesano, dans les Pouilles en Italie, avec mesures
de sol, télémétrie d'irrigation et données météorologiques.

Le secteur 4 a été retenu. Avec la même politique conservatrice de nettoyage,
il fournissait le plus grand nombre d'origines valides pour tous les horizons
parmi les secteurs 1, 2 et 4. Le secteur 3 ne fournissait pratiquement aucune
fenêtre historique complète de 24 heures à cause d'une longue fin de signal
plate ; le secteur 5 a été exclu à cause d'une incohérence documentée sur les
volumes d'eau.

Chaque instant du maillage contient cinq variables, dans cet ordre :

1. humidité du sol ;
2. volume d'eau appliqué ;
3. pluie ;
4. température de l'air ;
5. humidité de l'air.

Le maillage est régulier, avec un pas de 10 minutes. Une histoire de 24 heures
contient donc 144 observations et forme un tenseur de taille `[144, 5]`. Les
horizons étudiés sont 1, 3, 6, 12 et 24 heures. La **cible**, parfois appelée
*ground truth*, est la mesure d'humidité effectivement enregistrée à l'horizon
considéré ; il s'agit d'une référence expérimentale fournie par le capteur, pas
d'une mesure parfaite de toute l'eau physiquement présente dans le sol.

### 2.2 Nettoyage et prévention des fuites temporelles

Le tableau « merged » fourni par les auteurs a été préféré au tableau déjà
prétraité, car il conserve la grille complète de 10 minutes. Les principales
décisions sont les suivantes :

- contrôle de la plage physique de l'humidité, écrêtage à 100 et retrait des
  pics manifestes ;
- interpolation seulement si une lacune intérieure complète ne dépasse pas
  20 minutes ;
- rejet de toute fenêtre touchant une lacune plus longue ;
- suppression des fins de série correspondant à un capteur bloqué ;
- décalage du volume d'eau d'un pas, afin que l'entrée à l'instant `t` ne
  contienne qu'un intervalle d'irrigation déjà achevé ;
- aucune météo ni irrigation future dans les entrées.

Les découpages apprentissage, validation et test sont chronologiques et
contigus, selon des proportions temporelles 70/15/15. Les histoires ou cibles
qui traversent une frontière sont purgées. Cela produit 24 156 exemples
d'apprentissage, 4 235 de validation et 5 097 de test. Toutes les statistiques
de normalisation sont ajustées uniquement sur les lignes utilisées par les
histoires d'apprentissage.

La distinction est la suivante : l'**apprentissage** ajuste les poids ; la
**validation** choisit les variantes de modèle ; le **test** n'est consulté
qu'une fois la configuration figée. Cette chronologie évite qu'une information
du futur influence indirectement les décisions de modélisation.

### 2.3 Une série difficile malgré son apparente stabilité

La série contient de nombreux plateaux. Sur le test, la cible est exactement
inchangée par rapport à l'origine dans 94,75 % des cas à 1 heure, 87,49 % à
3 heures, 82,06 % à 6 heures, 75,52 % à 12 heures et 66,45 % à 24 heures.
Une prédiction qui ne bouge jamais est donc déjà très compétitive à court terme.

En parallèle, une forte dérive chronologique est visible : la moyenne des
cibles vaut 78,15 sur l'apprentissage, 60,51 sur la validation et 55,68 sur le
test. L'expérience combine ainsi deux difficultés : beaucoup de changements
nuls, mais une distribution absolue de l'humidité qui se déplace fortement au
cours du temps.

## 3. Baseline : persistance

La **persistance** est la règle de prévision la plus simple : elle suppose que
l'humidité future sera égale à l'humidité courante.

```text
ŷ(t + h) = y(t)
```

Cette baseline est scientifiquement importante. Si le sol varie lentement, un
réseau neuronal doit apporter plus qu'une petite correction aléatoire pour la
dépasser. Elle évite aussi de présenter comme utile un modèle complexe qui ferait
moins bien que « ne rien changer ».

Trois métriques sont utilisées :

- la **MAE** est la moyenne des erreurs absolues ;
- la **RMSE** est la racine de l'erreur quadratique moyenne et pénalise davantage
  les grandes erreurs ;
- le **R²** vaut 1 pour une prédiction parfaite, 0 pour une référence fondée sur
  la moyenne, et peut devenir négatif quand le modèle est moins bon que cette
  référence moyenne.

| Jeu        |    MAE |   RMSE |      R² |
| ---------- | -----: | -----: | ------: |
| Validation | 0,2916 | 1,2174 |  0,2472 |
| Test       | 0,2743 | 1,0489 | -0,1073 |

Le faible MAE de persistance est cohérent avec les plateaux. Sur les 4 164
échantillons de test considérés « presque inchangés », son erreur est exactement
nulle. En revanche, sa RMSE atteint 2,4516 sur les 933 échantillons qui changent
réellement. La RMSE globale est donc un meilleur guide que la seule MAE pour
évaluer les transitions rares mais importantes.

## 4. Premières expériences MLP

Un **MLP** (*multilayer perceptron*, ou perceptron multicouche) est un réseau
composé de couches linéaires et de fonctions non linéaires. Le premier modèle
aplatissait les 144 × 5 valeurs historiques, ajoutait l'horizon normalisé, puis
appliquait l'architecture suivante :

```text
721 → 128 → ReLU → 64 → ReLU → 1
```

`ReLU` conserve les valeurs positives et remplace les valeurs négatives par
zéro ; cette non-linéarité permet au réseau de représenter autre chose qu'une
simple régression linéaire. Ce premier MLP prédisait directement l'humidité
future absolue.

| Modèle direct |    MAE |   RMSE |      R² |
| ------------- | -----: | -----: | ------: |
| Validation    | 0,6994 | 1,1110 |  0,3731 |
| Test          | 1,7122 | 1,9633 | -2,8795 |

La validation semblait encourageante en RMSE, mais le test s'effondrait. Un
audit du pipeline n'a trouvé ni erreur de dénormalisation, ni mauvais index de
cible, ni fuite de variable future. Le diagnostic a plutôt mis en évidence la
dérive temporelle des valeurs absolues. Le biais moyen, défini comme
`prédiction - cible`, passait de -0,2333 en validation à -1,6207 en test et
atteignait -2,0143 à l'horizon 24 heures.

**Observation :** le modèle apprenait une valeur absolue fortement liée au
régime d'humidité de la période d'apprentissage.

**Hypothèse :** lui demander uniquement le changement futur devrait être plus
stable lorsque le niveau absolu se déplace.

## 5. Prédiction résiduelle et MLP final

### 5.1 Changer la question apprise

La cible résiduelle est la variation entre l'origine et l'horizon :

```text
Δy = y(t + h) - y(t)
ŷ(t + h) = y(t) + Δŷ
```

Avec cette formulation, prédire `Δŷ = 0` revient exactement à la persistance.
Le réseau apprend donc la question plus ciblée : « de combien l'humidité va-t-
elle changer ? » plutôt que « quelle valeur absolue aura-t-elle ? ».

Une première normalisation résiduelle soustrayait la moyenne d'apprentissage.
Or une sortie réseau nulle correspondait alors à un changement physique de
-0,336 plutôt qu'à la persistance. La formulation finale divise seulement par
l'écart-type résiduel d'apprentissage, 4,435837, sans retirer la moyenne :

```text
cible réseau = Δy / 4,435837
```

La dernière couche est initialisée à zéro. Au début de l'apprentissage, le
modèle reproduit donc exactement la persistance. Cette petite décision fournit
un a priori physiquement cohérent sans modifier les entrées.

### 5.2 Expériences successives

Le premier MLP résiduel centré atteignait une RMSE de validation de 1,1755. Le
passage à l'échelle centrée sur la persistance, avec MSE — l'erreur quadratique
moyenne — l'a abaissée à 0,9460. Une architecture un peu plus large,
`256 → 128 → 64`, a atteint 0,9347.

Deux hypothèses supplémentaires ont été testées puis rejetées :

- la perte de Huber, plus robuste aux grands écarts, a donné 1,1796 ;
- une pondération modérée des changements dans l'apprentissage a donné 1,0449.

Ces résultats indiquent que les grands changements de l'apprentissage ne
devaient pas être davantage accentués. La validation et le test sont restés
naturels : aucune répartition temporelle n'a été modifiée.

La dernière passe a évalué des caractéristiques temporelles simples, calculées
exclusivement avant ou à l'origine de prévision. Trois tendances d'humidité ont
été retenues : humidité courante moins humidité 1, 3 et 6 heures auparavant.
Elles ont amélioré la RMSE de validation de 0,9347 à 0,8975, soit environ 4 %.
Les cumuls récents d'irrigation et de pluie, seuls ou combinés, n'ont pas apporté
de gain comparable.

### 5.3 Modèle MLP figé

Le MLP final reçoit 720 valeurs historiques, un horizon et trois tendances :

```text
724 → 256 → 128 → 64 → 1
```

Il utilise ReLU, MSE, Adam, un taux d'apprentissage de `1e-3`, des lots de 512
et la graine 42. Une graine fixe l'initialisation pseudo-aléatoire et permet de
tester si un résultat dépend d'un tirage chanceux. La sélection a été faite
uniquement sur validation. Trois graines ont donné une RMSE de validation
moyenne de 0,9244 ± 0,0193 ; la graine 42, déjà choisie par validation, atteint
0,8975.

Sur le test final, ce MLP atteint une RMSE de 0,9478 et un R² de 0,0958. Un
ancien MLP sans tendances a obtenu fortuitement 0,9310 sur test, mais seulement
0,9347 sur validation. Il n'a pas remplacé le modèle officiel : choisir après
lecture du test aurait constitué une fuite méthodologique.

## 6. Expérience DeepONet

### 6.1 Principe

DeepONet est une architecture proposée pour apprendre des applications entre
fonctions, appelées opérateurs. Dans notre cas, l'historique de 24 heures joue le
rôle de la fonction d'entrée et l'horizon est le point où cette fonction est
interrogée. La formulation suit le principe original de DeepONet décrit par
[Lu et al.](https://doi.org/10.1038/s42256-021-00302-5).

```text
Historique 24 h + tendances
             ↓
        réseau Branch
             ↓
        vecteur de taille 64
             \
              produit scalaire → variation prédite → humidité future
             /
        vecteur de taille 64
             ↑
         réseau Trunk
             ↑
       horizon / 24
```

Le réseau **Branch** encode l'état historique. Le réseau **Trunk** encode la
coordonnée de requête, ici l'horizon. Leur produit terme à terme est sommé pour
produire un unique résidu normalisé. Cette structure, et non une concaténation
suivie d'un autre grand réseau, définit le DeepONet testé.

### 6.2 Recherche bornée et modèle retenu

Le premier modèle sur histoire brute obtenait une RMSE de validation de 1,1143.
Réduire le taux d'apprentissage ou la taille du Trunk ne l'a pas amélioré. Une
unique ablation ajoutant les trois tendances historiques du MLP a réduit la
RMSE à 1,0292 ; ce gain d'environ 7,6 % justifiait leur conservation.

Le modèle figé contient :

- Branch : `723 → 256 → 128 → 64` ;
- Trunk : `1 → 64 → 64 → 64` ;
- dimension latente : 64 ;
- produit scalaire et biais opérateur ;
- reconstruction résiduelle avec l'écart-type 4,435837 ;
- 234 945 paramètres entraînables.

Six entraînements DeepONet au total ont été réalisés. La configuration finale
a été sélectionnée sur validation avec la graine 42. Sa RMSE sur trois graines
vaut 1,1202 ± 0,1025, ce qui révèle une sensibilité non négligeable à
l'initialisation.

### 6.3 Résultat scientifique

Le DeepONet atteint une RMSE de 1,0292 en validation et 1,2112 sur le test. Le
MLP figé atteint respectivement 0,8975 et 0,9478. Le résultat important est donc
sans ambiguïté : **le MLP final prédit mieux que le DeepONet testé, globalement
et à chacun des cinq horizons de test**.

DeepONet a néanmoins été conservé comme cible de déploiement. Il est le modèle
d'opérateur au cœur de la question technologique et permet d'étudier séparément
les coûts d'une structure Branch/Trunk sur CPU et GPU. Conserver ce modèle ne
revient pas à prétendre qu'il est plus précis.

## 7. Comparaison prédictive

| Modèle                 | Validation MAE | Validation RMSE | Validation R² | Test MAE |  Test RMSE |    Test R² |
| ---------------------- | -------------: | --------------: | ------------: | -------: | ---------: | ---------: |
| Persistance            |         0,2916 |          1,2174 |        0,2472 |   0,2743 |     1,0489 |    -0,1073 |
| MLP résiduel figé      |         0,3598 |      **0,8975** |    **0,5909** |   0,5678 | **0,9478** | **0,0958** |
| DeepONet résiduel figé |         0,5304 |          1,0292 |        0,4620 |   0,8390 |     1,2112 |    -0,4767 |

La persistance conserve la meilleure MAE parce qu'elle est parfaite sur les
plateaux. La RMSE raconte une histoire différente : le MLP réduit mieux les
grosses erreurs lors des changements. DeepONet reste meilleur que la persistance
sur le test à 1 et 3 heures, mais devient légèrement moins bon à 6 heures et
nettement moins bon à 12 et 24 heures.

| Horizon test | Persistance RMSE |   MLP RMSE | DeepONet RMSE |
| -----------: | ---------------: | ---------: | ------------: |
|          1 h |           0,9557 | **0,7717** |        0,8377 |
|          3 h |           1,0007 | **0,8337** |        0,8877 |
|          6 h |           1,0427 | **0,9028** |        1,0488 |
|         12 h |           1,0920 | **1,0023** |        1,4396 |
|         24 h |       **1,1577** |     1,2052 |        1,6930 |

La figure suivante montre l'horizon 24 heures. La vérité terrain est la mesure
future ; la persistance reporte la mesure courante ; les deux réseaux ajoutent
leur variation prédite. Elle rend visibles les longues zones plates et les
transitions rares qui dominent la RMSE.

![Prédictions finales à l'horizon 24 h](../results/deeponet_final_test_predictions_24h.png)

## 8. Gel de l'expérience de déploiement

Après la sélection sur validation et l'unique évaluation finale, le checkpoint
`models/deeponet_reference.pt` a été figé. Un checkpoint `.pt` contient les
poids PyTorch et les métadonnées nécessaires pour reconstruire exactement le
modèle. Le gel empêche que la phase de déploiement dérive vers une nouvelle
phase de réglage prédictif.

Le modèle a ensuite été exporté vers **ONNX**, un format portable qui décrit un
graphe de calcul indépendamment de PyTorch. Le graphe reçoit :

- `branch_input [batch, 723]` ;
- `horizon [batch, 1]` ;
- `current_moisture [batch, 1]` ;

et renvoie directement `prediction [batch, 1]` en unités physiques. Le produit
Branch/Trunk, la remise à l'échelle du résidu et l'ajout de l'humidité courante
sont inclus dans ONNX.

Dix cas déterministes — deux pour chacun des cinq horizons — servent de
**vecteurs dorés**. Ils fixent les entrées et les sorties attendues pour comparer
les moteurs d'inférence. L'écart maximal PyTorch/ONNX Runtime est
`7,6294 × 10⁻⁶`, sous la tolérance FP32 de `10⁻⁴`.

La commande suivante reconstruit et vérifie ces artefacts :

```bash
uv sync --project python
uv run --project python setup-onnx
```

`setup-onnx` **n'entraîne jamais le modèle**. Il part du checkpoint figé,
contrôle le contrat, réexporte ONNX, vérifie l'équivalence numérique et
rafraîchit les vecteurs dorés Python et C++. Le modèle ONNX étant lui-même
versionné, l'utilisateur C++ n'a besoin ni de Python, ni de PyTorch, ni du jeu
de données pour exécuter l'inférence.

## 9. C++ et ONNX Runtime

La chaîne de déploiement est volontairement courte :

```text
DeepONet figé
      ↓
     ONNX
      ↓
ONNX Runtime C++
   ↙       ↘
 CPU       CUDA
```

L'**inférence** est l'utilisation d'un réseau déjà entraîné pour produire une
prédiction, sans modifier ses poids. **ONNX Runtime** est un moteur qui charge
le graphe ONNX, optimise cette inférence et délègue ses opérations à un
fournisseur d'exécution. Le
`CPUExecutionProvider` exécute sur le processeur. Le
`CUDAExecutionProvider` place les opérations compatibles sur un GPU NVIDIA via
CUDA, la plateforme de calcul parallèle de NVIDIA.

Le projet C++ utilise ONNX Runtime 1.30.0, acquis sous forme d'archive binaire
officielle CUDA 13 par CMake. Il valide au chargement les noms, types et formes
des quatre tenseurs. Les lots sont dynamiques : la dimension `batch` peut changer
sans modifier le modèle.

Les mêmes vecteurs dorés sont exécutés sur CPU et CUDA. L'écart maximal du
runtime C++ FP32 par rapport aux valeurs attendues est `3,8147 × 10⁻⁶`. Cette
étape vérifie la correction numérique avant toute mesure de vitesse ; un moteur
rapide mais différent du modèle figé serait sans intérêt expérimental.

## 10. Premiers benchmarks CPU contre CUDA

### 10.1 Protocole

Un **batch**, ou lot, est le nombre d'échantillons traités par un appel au
modèle. Un lot de 1 représente une requête isolée. Un lot de 4096 traite 4096
prévisions ensemble et peut mieux remplir les unités de calcul parallèles du
GPU.

La **latence** est le temps nécessaire pour terminer un appel ; plus elle est
faible, mieux c'est. Le **débit** est le nombre d'échantillons traités par
seconde ; il augmente souvent avec le lot, même si la latence totale du lot
augmente. Ces deux critères répondent à des usages différents : interaction
temps réel pour la latence, traitement massif pour le débit.

Les mesures finales ont été réalisées sur une NVIDIA GeForce RTX 2060 et un
Intel Core i5-9600K, avec CUDA 13.4, ONNX Runtime 1.30.0 et une compilation
Release. Chaque combinaison utilise 100 itérations d'échauffement, puis 1000
itérations mesurées, répétées trois fois. Les dix entrées réelles des vecteurs
dorés sont répétées cycliquement pour former les grands lots. Le chargement du
modèle et la création de session sont hors chronométrage.

La voie CUDA normale part de données en mémoire CPU et rend un résultat visible
sur CPU ; elle inclut donc les transferts gérés par ONNX Runtime.

| Batch |    CPU, ms |   CUDA, ms | Débit CPU, éch./s | Débit CUDA, éch./s | Accélération CUDA/CPU |
| ----: | ---------: | ---------: | ----------------: | -----------------: | --------------------: |
|     1 | **0,0181** |     0,1201 |        **55 175** |              8 326 |                0,151× |
|     8 | **0,0259** |     0,1166 |       **309 043** |             68 625 |                0,222× |
|    32 | **0,0586** |     0,1278 |       **546 108** |            250 455 |                0,459× |
|   128 |     0,1552 | **0,1449** |           824 815 |        **883 223** |                1,071× |
|   256 |     0,2689 | **0,2048** |           952 148 |      **1 249 784** |                1,313× |
|   512 |     0,4859 | **0,3012** |         1 053 681 |      **1 700 110** |                1,613× |
|  1024 |     0,9388 | **0,4848** |         1 090 780 |      **2 112 079** |                1,936× |
|  2048 |     1,9312 | **0,8553** |         1 060 499 |      **2 394 545** |                2,258× |
|  4096 |     4,1299 | **1,6125** |           991 782 |      **2 540 168** |            **2,561×** |

Le GPU n'est donc pas « toujours plus rapide ». Le CPU gagne clairement aux
lots 1, 8 et 32. Le croisement observé se situe entre 32 et 128 ; les points
testés ne permettent pas de lui attribuer une valeur plus précise. À 4096, les
grandes multiplications matricielles utilisent suffisamment le GPU pour obtenir
2,56 fois la vitesse du CPU en latence de lot.

## 11. Expérience I/O Binding

La mémoire principale du CPU, souvent appelée mémoire **hôte**, est distincte
de la mémoire du GPU, ou mémoire **device/VRAM**. Une entrée doit généralement
être copiée de l'hôte vers le GPU (**H2D**) et le résultat du GPU vers l'hôte
(**D2H**).

L'**I/O Binding** d'ONNX Runtime permet de lier explicitement des buffers aux
entrées et sorties d'une session. Cela évite certaines allocations ou copies
implicites, mais ne supprime pas magiquement les transferts exigés par le contrat
de l'application.

Dans le premier essai, les entrées naissaient toujours sur CPU et la sortie
devait y revenir. Les copies H2D et D2H restaient donc incluses. Cette voie
atteint 0,1221 ms contre 0,1201 ms pour CUDA normal au batch 1, 0,1749 contre
0,1449 ms au batch 128, et 1,6433 contre 1,6125 ms au batch 4096. Elle est plus
lente à chacun des neuf lots testés.

**Conclusion :** l'I/O Binding hôte-vers-hôte n'a pas fourni de gain stable. Il
ne doit pas être présenté comme un interrupteur générique d'accélération.

## 12. Profilage avec Nsight Systems

Le benchmark répond à la question « combien de temps ? », mais pas à la
question « pourquoi ? ». Un **profileur** observe les appels CPU, les copies et
les noyaux GPU afin de localiser les coûts. Nsight Systems, l'outil de
[profilage système de NVIDIA](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
a été utilisé sur des traces courtes aux batchs 1 et 4096.

Un **kernel** CUDA est une fonction lancée sur le GPU. Son lancement possède un
coût fixe côté CPU/GPU, même si le calcul interne est minuscule. Le DeepONet
CUDA normal déclenche environ 21 kernels par inférence. À petit batch, beaucoup
de ces kernels sont très courts : leur orchestration pèse davantage que le
calcul utile. À batch 4096, les grandes **GEMM** — multiplications générales de
matrices qui composent les couches linéaires — dominent davantage le temps GPU.
Les transferts H2D deviennent également visibles dans le coût total.

Les temps Nsight ne sont pas utilisés comme résultats de benchmark. La collecte
d'une trace perturbe l'exécution ; elle sert ici uniquement de preuve
diagnostique pour choisir les expériences suivantes.

## 13. Optimisation 1 — CUDA Graphs

### 13.1 Hypothèse

Puisque la petite charge est dominée par des lancements répétés, regrouper une
séquence déjà connue devrait réduire l'orchestration. Un **CUDA Graph** capture
une suite stable d'opérations CUDA puis la rejoue avec un unique lancement de
graphe. Les contraintes d'ONNX Runtime exigent des formes et adresses stables ;
une session est donc créée pour chaque batch testé. La capture a lieu avant la
boucle mesurée, conformément au support CUDA Graph du
[CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview).

### 13.2 Résultats

| Batch | CUDA normal, ms | CUDA Graph, ms | Gain contre CUDA normal |
| ----: | --------------: | -------------: | ----------------------: |
|     1 |          0,1201 |     **0,0535** |              **2,244×** |
|     8 |          0,1166 |     **0,0607** |              **1,921×** |
|    32 |          0,1278 |     **0,0861** |              **1,483×** |
|   128 |          0,1449 |     **0,1373** |                  1,056× |
|   256 |      **0,2048** |         0,2083 |                  0,983× |
|   512 |          0,3012 |     **0,3006** |                  1,002× |
|  4096 |          1,6125 |     **1,5939** |                  1,012× |

Le résultat correspond à l'hypothèse : le gain est fort lorsque le coût de
lancement domine, puis devient neutre lorsque les grandes GEMM dominent. La
trace contient un `cudaGraphLaunch` par rejeu au lieu des lancements individuels
répétés. Les copies restent présentes, ce qui explique que le graphe ne résolve
pas le coût de transfert des grands lots.

À batch 1, CUDA Graph améliore CUDA de 2,24×, mais reste à 0,0535 ms contre
0,0181 ms pour le CPU. Réduire un surcoût GPU ne suffit donc pas à garantir la
meilleure latence absolue pour une requête isolée.

## 14. Optimisation 2 — I/O Binding avec données résidentes sur GPU

Cette expérience change volontairement le contrat mesuré.

```text
Contrat hôte-à-hôte : CPU → GPU → inférence → CPU

Contrat GPU-résident : données déjà sur GPU
                              ↓
                          inférence
                              ↓
                      résultat laissé sur GPU
```

Les buffers sont alloués et les entrées sont transférées avant le chronométrage.
Pendant les itérations mesurées, il n'y a ni copie H2D, ni copie D2H, ni
allocation. Une seule copie finale sert à vérifier la correction. Les traces
Nsight confirment que les copies observées appartiennent à la préparation et à
la lecture finale, pas aux itérations mesurées.

À batch 4096, cette voie atteint **0,5734 ms**, soit **7,14 millions
d'échantillons par seconde**. Elle est **2,812× plus rapide** que CUDA normal à
1,6125 ms. Le gain augmente avec le batch : 1,19× à batch 1, 1,31× à 128,
1,95× à 512 et 2,81× à 4096.

Ce résultat ne représente pas la latence d'une application ordinaire qui reçoit
et rend des données CPU. Il mesure le potentiel du modèle lorsqu'il est intégré
à une chaîne plus large où l'étape précédente produit déjà les entrées sur GPU
et où l'étape suivante consomme la sortie sur GPU.

## 15. Optimisation 3 — FP16 et Tensor Cores

Le modèle canonique utilise des nombres **FP32**, codés sur 32 bits. Le mode
expérimental **FP16** utilise 16 bits pour les poids et opérations internes, tout
en conservant des entrées et sorties FP32. La représentation est moins précise,
mais déplace moins de données et peut activer les **Tensor Cores**, unités
spécialisées dans les multiplications matricielles de précision réduite.

Le modèle FP32 canonique n'a pas été modifié et aucun réentraînement n'a eu
lieu. Un second fichier ONNX est produit par une conversion déterministe.

| Batch | CUDA FP32, ms | CUDA FP16, ms | Accélération FP16 |
| ----: | ------------: | ------------: | ----------------: |
|     1 |    **0,1201** |        0,1269 |            0,947× |
|     8 |    **0,1166** |        0,1336 |            0,872× |
|    32 |    **0,1278** |        0,1411 |            0,905× |
|   128 |    **0,1449** |        0,1502 |            0,965× |
|   256 |        0,2048 |    **0,1786** |            1,147× |
|   512 |        0,3012 |    **0,2762** |            1,090× |
|  1024 |        0,4848 |    **0,4372** |            1,109× |
|  2048 |        0,8553 |    **0,7762** |            1,102× |
|  4096 |        1,6125 |    **1,4400** |        **1,120×** |

FP16 est donc nuisible aux petits lots, où ses conversions et coûts fixes ne
sont pas amortis. À partir de 256, le gain mesuré est de 9 à 15 %. À batch 4096,
le débit passe de 2,54 à 2,84 millions d'échantillons par seconde.

L'écart absolu maximal par rapport à la référence FP32 vaut 0,013729 unité
d'humidité ; l'écart moyen à batch 4096 vaut 0,007320. Ces valeurs proviennent
des cas dorés répétés et ne constituent pas une nouvelle évaluation prédictive
sur tout le test. Nsight identifie des kernels `tensorop_f16_s1688`, preuve que
des GEMM éligibles aux Tensor Cores ont bien été sélectionnées. FP16 reste donc
une option expérimentale, utile pour le grand batch si cette différence
numérique est acceptable, et non un nouveau modèle par défaut.

## 16. Pourquoi aucun kernel CUDA sur mesure ?

Après le profilage, une fusion CUDA personnalisée a été envisagée. Le graphe
ONNX FP32 contient six `Gemm`, quatre `Relu`, deux `Mul`, deux `Add`, un
`ReduceSum` et une constante. Les biais sont déjà intégrés aux opérations
`Gemm`, et les multiplications matricielles importantes sont déléguées à des
kernels GPU optimisés. Les opérations ReLU et élémentaires restantes sont
visibles sous forme de petits kernels séparés.

Écrire une multiplication matricielle à la main aurait dupliqué un travail déjà
hautement optimisé. Fusionner seulement les petites opérations aurait nécessité
un opérateur ONNX Runtime personnalisé ou un chemin d'inférence parallèle, avec
un coût de maintenance et de validation disproportionné. Surtout, CUDA Graphs
avait déjà supprimé l'essentiel du surcoût de lancement aux petits lots.

La décision expérimentale a donc été **de ne pas écrire de kernel CUDA sur
mesure**. Il s'agit d'une conclusion d'ingénierie fondée sur les profils, pas
d'une tâche laissée inachevée.

## 17. Synthèse finale

| Expérience              | Problème visé                                    | Résultat observé                                           | Décision                            |
| ----------------------- | ------------------------------------------------ | ---------------------------------------------------------- | ----------------------------------- |
| Persistance             | Établir une référence minimale                   | Très forte sur les plateaux ; RMSE test 1,0489             | Référence conservée                 |
| MLP direct              | Apprendre l'humidité absolue                     | RMSE test 1,9633 et fort biais temporel                    | Abandonné                           |
| MLP résiduel            | Résister au changement de niveau absolu          | RMSE test 0,9478 pour le modèle sélectionné sur validation | Meilleur prédicteur testé           |
| DeepONet                | Tester une formulation d'opérateur multi-horizon | RMSE test 1,2112, moins bon que le MLP                     | Conservé comme cible d'étude GPU    |
| CUDA normal             | Accélérer l'inférence                            | Plus lent à petit batch, 2,561× le CPU à 4096              | Utile pour charges groupées         |
| I/O Binding hôte-à-hôte | Contrôler buffers et transferts                  | Plus lent à tous les batchs                                | Résultat négatif conservé           |
| CUDA Graphs             | Réduire le coût des lancements                   | 2,244× CUDA à batch 1 ; neutre à grand batch               | Utile pour petits lots GPU          |
| I/O GPU-résident        | Retirer les transferts du chemin mesuré          | 0,5734 ms et 7,14 M éch./s à 4096                          | Utile seulement dans une chaîne GPU |
| FP16                    | Exploiter précision réduite et Tensor Cores      | Plus lent jusqu'à 128 ; 1,120× à 4096                      | Option grand batch, non canonique   |
| Fusion personnalisée    | Réduire les petits kernels restants              | Opportunité jugée trop faible au regard du coût            | Aucun kernel ajouté                 |

### Conclusion prédictive

Le MLP résiduel est le meilleur prédicteur testé. Sa structure plus simple et
ses tendances explicites sont mieux adaptées à ce jeu chronologique que le
DeepONet évalué. Le DeepONet n'est ni meilleur que le MLP, ni meilleur que la
persistance globale sur le test.

### Conclusion GPU

L'intérêt du GPU dépend de la granularité. Le CPU minimise la latence pour les
petits lots. Le GPU prend l'avantage entre les lots testés 32 et 128, puis son
débit augmente jusqu'aux grands lots. La localisation des données compte autant
que le calcul : supprimer les transferts dans une vraie chaîne GPU-résidente
offre un gain bien plus grand que l'I/O Binding hôte-à-hôte.

### Conclusion d'ingénierie

Le profilage a été plus utile que l'application aveugle d'optimisations. Il a
justifié CUDA Graphs pour le petit batch, montré le potentiel d'une chaîne
GPU-résidente, limité l'intérêt de FP16 aux grands lots et évité un développement
CUDA sur mesure sans bénéfice démontré.

## 18. Limites

- **Périmètre agronomique :** l'étude principale porte sur un secteur et une
  période de support exploitable relativement courte. Elle ne démontre pas une
  généralisation à d'autres sols, cultures ou années.
- **Dérive temporelle :** les moyennes d'humidité changent fortement entre les
  trois périodes. Le résiduel réduit ce problème sans le faire disparaître.
- **Plateaux :** la forte proportion de cibles inchangées favorise la
  persistance et rend MAE et RMSE parfois contradictoires.
- **Portée du résultat DeepONet :** six entraînements bornés ont été réalisés.
  Le résultat ne prouve pas que tous les opérateurs neuronaux sont moins bons
  qu'un MLP ; il décrit cette architecture et ce jeu de données.
- **Stabilité aléatoire :** DeepONet présente une dispersion entre graines plus
  élevée que souhaité.
- **Matériel :** les performances ont été mesurées sur une seule configuration
  principale, une RTX 2060 et un i5-9600K. Elles ne se transposent pas
  automatiquement à d'autres générations.
- **Lots artificiellement agrandis :** au-delà de dix exemples, le benchmark
  répète cycliquement les cas dorés. Le coût de calcul est valide pour mesurer
  le runtime, mais ne représente pas une arrivée naturelle de 4096 mesures
  indépendantes.
- **Contrat GPU-résident :** ses chiffres excluent intentionnellement les
  transferts et ne sont pas comparables à une application hôte-à-hôte complète.
- **Précision réduite :** FP16 modifie légèrement les prédictions. Son effet n'a
  pas été réévalué sur l'ensemble du test prédictif.
- **Reproductibilité historique :** les scripts et journaux de sélection sont
  présents, mais tous les checkpoints intermédiaires ne font pas partie d'une
  commande unique rejouant l'histoire. L'expérience de déploiement figée est,
  elle, reproductible depuis un clone propre.

## 19. Reproductibilité

Le chemin de déploiement principal part directement du modèle ONNX versionné :

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/deeponet_golden_test --cpu
./build/deeponet_golden_test --cuda
./build/deeponet_golden_test --cuda-iobinding
./build/deeponet_benchmark --provider all --warmup 100 --iterations 1000 --repetitions 3
```

La reconstruction des artefacts ONNX depuis le checkpoint figé est séparée :

```bash
uv sync --project python
uv run --project python setup-onnx
```

Elle ne télécharge pas le dataset et n'entraîne rien. Le contrat ONNX, les
artefacts figés, les prérequis et les commandes reproductibles sont regroupés
dans [la documentation d'implémentation](implementation.md).

## 20. Conclusion

Cette expérience ne conduit ni à « DeepONet est meilleur », ni à « le GPU est
toujours meilleur ». Le modèle prédictif le plus convaincant est le MLP
résiduel, avec une RMSE de test de 0,9478 contre 1,2112 pour DeepONet. DeepONet
reste néanmoins une cible pertinente pour étudier le déploiement d'un opérateur
neuronal multi-horizon.

Pour ce petit réseau, le CPU offre la meilleure latence aux faibles batchs. Le
GPU devient utile lorsque suffisamment de prédictions sont regroupées : CUDA
normal est 2,56× plus rapide que le CPU à batch 4096. CUDA Graphs réduit le
surcoût d'orchestration et accélère CUDA de 2,24× à batch 1, sans toutefois
battre le CPU à cet endroit. Une chaîne réellement GPU-résidente révèle un
potentiel de 7,14 millions d'échantillons par seconde à batch 4096, tandis que
FP16 apporte un gain plus modeste de 1,12× avec une petite différence numérique.

La conclusion la plus générale est méthodologique : la formulation résiduelle a
été choisie à partir du comportement des données, puis les optimisations GPU ont
été choisies à partir du profilage. Cette démarche a produit des gains mesurés
et a aussi permis de rejeter proprement des idées — I/O Binding hôte-à-hôte et
kernel CUDA personnalisé — lorsqu'elles n'étaient pas justifiées.

## Références essentielles

- Dataset Arnesano v2 : [Mendeley Data, DOI 10.17632/c837v6p8ph.2](https://doi.org/10.17632/c837v6p8ph.2).
- Lu et al., *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, [Nature Machine Intelligence, 2021](https://doi.org/10.1038/s42256-021-00302-5).
- [ONNX Runtime — CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
- [ONNX Runtime — I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html).
- [NVIDIA Nsight Systems — User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
- [NVIDIA CUDA C++ Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).
