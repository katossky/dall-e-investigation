# Titre avec un peu d'humour

> Note pour mes co-auteurs: je pense qu'il est pertinent d'utiliser directement R Studio pour la rédaction (qui permet d'écrire du code Python, de toute façon), principalement pour bénéficier du très puissant Book Down (livres avec R Markdown) et d'une intégration très facile avec Github. Hackmd reste utilisable pour des retouches ponctuelles. Arthur
> 
> **Ressource pour écrire avec Markdown:**
> - https://bookdown.org/yihui/rmarkdown/




## 1. <titre qui résume notre étude par ex. : "Mini Dall-E comprend les couleurs et les images, mais pas l'espace">

## 2. Dall-E: un modèle _text-to-image_ à trois composantes et à trois temps

### 2.1 Les trois composantes

#### 2.1.1 Encodeur

L'encodeur d'un modèle est la partie qui convertit les entrées (texte et images dans notre cas) en une suite de vecteurs homogènes utilisable pour la modélisation statistique. Tous les modèles étudiées utilisent un encodeur de texte^[ainsi qu'un _tokenizer_] externe fixé qu'ils n'essaient même pas d'adapter au contexte.

|----|----|----|


L'encodeur et le décodeur (partie suivante) sont souvent entraînés simmultanément et le détail est présenté à la section 3.

#### 2.1.2 Décodeur

#### 2.1.3 Discriminateur (CLIP)

Le modèle CLIP est un modèle de réseaux de neurones qui a été développé afin d'associer à une image la description la plus pertinente parmis un ensemble proposé. Le but de ce modèle est d'apprendre à effectuer cette association de manière non spécifique (zero-shot learning), contrairement aux réseaux de neurones entrainés spécifiquement à la reconnaissance d'un champ en particulier (e.g. réseau de neurone dédiée à la reconnaissance de la race d'un chien, entrainée sur ce champ particulier). Afin d'être le plus généraliste possible, le modèle a été entraîné sur 400 millions de paires image/texte.

Ce modèle généraliste s'appuie fortement sur les forces du NLP (notamment l'apprentissage supervisé du langage), plutôt que de construire explicitement un modèle entrainé à choisir spécifiquement une description parmi plusieurs. En effet, au lieu d'entraîner un modèle sur un ensemble de textes et images directement en entrée du modèle, [?il manque pas un bout de phrase ici?] avec correction itérative des poids. Le modèle est entrainé sur les espaces condensées des images et des textes (embedding space), données par des encodeurs, pour calculer des indices de similarité et de dissimilarité (contrastive pre-training).  Cet apprentissage de la notion de semblable/dissemblable permet ainsi de pouvoir correctement prédire sur d'autres jeux de données que ceux utilisés dans la phase d'entraînement. La qualité des encodeurs est alors primordiale puisqu'ils regissent en amont la représentation vectorielle. Ainsi, la qualité du modèle repose sur la qualité des espaces condensées.
Pour la phase d'entraînement, le modèle reçoit par paquet NxN paires texte-image avec Nx1 paires correctes et NxN-1 paires autres. Son but est alors de maximiser la similarité entre les paires correctes et de minimiser celle entre les paires incorrectes. 

Lors de la phase d'inférence, on fournit à CLIP un ensemble de textes et une image, et  CLIP calcule les indices de similarité à partir des données fournies par les encodeurs et des poids entrainés du modèle. Il en résulte une distribution de nombres sommant à 1, et le texte recevant le score le plus élevé est considéré comme la prédiction du modèle. Il est intéressant de noter que la manière dont les différents textes sont proposées à CLIP à son importance: on obtiendra ainsi de meilleurs résultats en intégrant les noms des différentes catégories à une phrase. Par exemple, pour prédire le type d'animal sur une photo, on préférera proposer comme descriptions "une photo de: ..." et remplacer les "..." par chat, chien, éléphant, plutôt qu'utiliser directement les noms des animaux comme texte.

Les auteurs du modèle notent que malgré la grande généralité du modèle, ses performances restent médiocres sur des données complétement hors du jeu de données d'entraînement. CLIP ne résoud donc pas réellement le manque de capacité de généralisation de ce type de modèle, mais le contourne en utilisant un type d'apprentissage plus généraliste et une très grosse quantité de données. 
[...]

### 2.2 Les trois temps

#### 2.2.1 Entraînement de l'encodeur d'images

L'auto-encodeur de Dall-E ...

Celui de Dall-E mini ...

Quant à notre auto-encodeur ...

Le tableau suivant résume les différences entre les trois modèles:

|   | Dall-E | Dall-E Mini | Notre modèle |
|---|---|----|---|
| Auto-encodeur | dVAE |----|---|
| Jeu de données utilisé | ---- |----|---|
| Taille du jeu d'entraînement | ---- |----|---|
| Résolution des images orginales | 256×256 |----|---|
| Résolution des images compressées | 32×32 |----|---|

Sources: "Stage 1. We train a discrete variational autoen- coder (dVAE)1 to compress each 256×256 RGB image into a 32 × 32 grid of image tokens, each element of which can assume 8192 possible values."


#### 2.2.2 Entraînement du générateur d'images

#### 2.2.3 Génération d'images en production

### 2.3 Les données

## 3. Mini Dall-E et Dall-E: le jeu des 7 différences 

## 4. Procédure de test des images générées par mini Dall-E avec CLIP

### 4.0 Procédure générale

#### 4.0.1 La procédure utilisée par les auteurs de Dall-E

#### 4.0.2 Notre procédure

### 4.1 Données utilisées pour les tests

### 4.2 Test de CLIP, utilisé pour le benchmark

Le modèle CLIP est utilisé pour évaluer la pertinence entre une description et une image. Pour cela, CLIP doit être en mesure de pleinement comprendre la description et également d'avoir une analyse précise de l'image. Pour pouvoir évaluer le modèle DALL-E, il est nécessaire de s'assurer de la capacité du modèle CLIP à noter et à ordonner plusieurs images par rapport à une description selon le degré de pertinence. Pour cela, nous avons étudié le modèle CLIP d'un point de vue qualitatif puis d'un point de vue quantitatif.

#### 4.2.1 Evaluation manuelle d'un jeu d'image tiré de MSCOCO

* Evaluation de plusieurs descriptions pour une image

Pour évaluer le modèle CLIP, il est nécessaire de constituer une base d'images et de descriptions. Pour cela, nous partons du jeu de données [MSCOCO](https://cocodataset.org/#home) et nous tirons aléatoire 100 images pour procéder à une évaluation en trois étapes : 
1.  **(Génération des descriptions)**: Pour ces 100 images, nous choississons manuellement une description parmi les 5 proposés et nous constituons deux descriptions supplémentaires, une correspondant moyennement à l'image et une autre sans rapport à l'image. 
2.  **(Evaluation manuelle croisée)**: Nous évaluons manuellement la pertinence de chacune des descriptions par rapport à l'image. Pour des raisons d'objectivité, l'évaluateur d'une paire description-image est une personne différence de cette qui a généré la paire.
3.  **(Evaluation par CLIP)** Nous évaluons ensuite par le modèle CLIP l'ensemble des paires description-image.

* Evaluation de plusieurs images pour une description ?
1. [Generation par mini DALL-E]
2. [Classement manuel]
3. [Classement par CLIP]

### 4.3 ...


#### Sensibilité des résultats à la formulation des _prompts_

Mini Dall-E est extrêmement sensible à la formulation des _prompts_. Faire précéder les prompts de "a picture of" permet de grandement améliorer les résultats.


[**EXEMPLE ]

## Annexes

### A.1 Fonctionnement du modèle CLIP

> Contenu déplacé dans le corps du rapport. Section à conserver? Par exemple pour parler de l'**entraînement** de CLIP, qui n'a pas ça place dans le rapport principal? [name=Arthur Katossky]



### Bibliographie


#### CLIP

* **Article scientifique:** "Learning Transferable Visual Models From Natural Language Supervision" (2021),  https://arxiv.org/abs/2103.00020
* **Article de blog:** "CLIP: Connecting
Text and Images" (2021), https://openai.com/blog/clip/


#### Dall-E

- **Article scientifique:** "Zero-Shot Text-to-Image Generation" (2021), https://arxiv.org/abs/2102.12092
- **Article de blog:** "DALL·E: Creating
Images from Text" (2021), https://openai.com/blog/dall-e

#### Dall-E mini

- **Rapport:** "Dall-E Mini. Generate images from a text prompt in this interactive report: DALL·E on a smaller architecture." (2021), https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy
- Dall-E mini utilise *Taming Transformers* (https://github.com/CompVis/taming-transformers) comme VQ-GAN

#### MS COCO (2017)

- **Site:** https://cocodataset.org
