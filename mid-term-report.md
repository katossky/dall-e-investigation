# Note de mi-parcours <br>Projet de statistiques appliquées

_**Consignes** Cette note d’étape ne doit pas être une version préliminaire du rapport mais plutôt un point sur le travail déjà fait, ce que vous envisagez de faire, les difficultés rencontrées et anticipées, les pistes prévues pour les résoudre, et quelques résultats préliminaires. C’est un document de 3-4 pages maximum, sans annexe, qui permet à votre correspondant ENSAE de voir si vous êtes sur les bons rails._

## 0. Résumé du projet et de son avancement

Notre projet de statistiques appliquées consiste à étudier le modèle _DALL-E_, un modèle "text-to-image" visant à produire une image pertinente à partir d'une description textuelle. 
Ce projet peut schématiquement être scindé en 3 étapes: 
1. Comprendre la structure de ce modèle complexe de machine learning
2. Analyser les images produites par ce modèle afin d'estimer sa pertinence 
3. Entraîner une version "mini" du modèle sur un jeu de données réduit

Les trois premiers mois ont été principalement consacrés à la première partie et nous sommes actuellement en train de travailler en paralèlle sur les 2 suivantes.

### Tableau des travaux réalisés
| Numéro | Travaux réalisés | Résultats |
| -------- | -------- | -------- |
| 1     | revue scientifique du modèle     | Présentations     |
| 2     | prise en main de l'implémentation mini-dalle et génération d'images à partir de cette dernière     | Notebook + 6000 images générées     |
| 3     | premières analyses qualitatives et quantitatives sur la qualité des images générées     | Présentations     |
| 4     | début d'implémentation d'un modèle adhoc sur le thème des paysages, construction d'un jeu de données     | Code     |



### Tableau des risques et des difficultés
| Numéro | Difficulté | Rencontrée | Anticipée | Résolue | Solution |
| --------| -------- | -------- | -------- | -------- | -------- |
| 1 | Complexité cognitive à appréhender le modèle    | X     | X     | X     | Lectures et présentations croisées    |
| 2 |  Production de résultats réguliers dans des délais contraints    |   X   |   X  |  X    |  Mise au point d'une réunion chaque semaine en interne pour le suivi et la coordination    |
| 3 | Complexité à produire des images à partir d'un modèle déjà existant     |   X   |  X   |  X    |  Lecture en mobprogramming de l'implémentation et mise en place d'un notebook exemple   |
| 4 |  Complexité à produire un modèle adhoc     |  X    |  X    |  (en cours)    | Remontée d'information et sollicitations dès que blocage des encadrants     | 
| 5 | Coordination des travaux     |  X    |  X    |  X    | Mise en place d'un dépot git commun et réunion hebdomadaire      |
| 6 |      |      |      |      |      |
| 7 |      |      |      |      |      |
| 8 |      |      |      |      |      |
| 9 |      |      |      |      |      |
| 10 |      |      |      |      |      |


### Tableau des travaux restants à faire
| Numéro | Travaux restants à faire | Statut |
| -------- | -------- | -------- |
| 1     | raffinements sur la génération d'image à la demande des tuteurs     | En cours     |
| 2     | analyse colorimétrique des images générées     | En cours     |
| 3     | finir l'implémentation du modèle, l'entraîner et analyser les résultats     | En cours     |
| 4     | produire le rapport final   | En cours     |


## 1. Travaux effectués

### 1.A Compréhension du modèle

Le début du projet a été de lire des articles (scientifiques ou blogposts) afin de comprendre le modèle DALL-E, son architecture et sa décomposition en différents éléments.
Hormis la publication de l'article scientifique de DALL-E, le modèle DALL-E n'est pas entièrement public et il existe des zones d'ombre dans l'article scientifique. En effet, seule la partie auto-encodeur des images a été publiée (openai/DALL-E) avec à la fois le code source et le modèle entrainé disponible. Ni le modèle entrainé, ni le code source du transformer n'est disponible au grand public. OpenAI propose une API pour tester le modèle entrainé uniquement à un nombre restreint de beta-testeurs.
Ainsi pour approfondir nos connaissances, nous nous sommes donc également appuyés sur les implémentations publiques d'une version mini du modèle (dalle-mini : https://github.com/borisdayma/dalle-mini, DALLE-pytorch : https://github.com/lucidrains/DALLE-pytorch) par d'autres personnes et organismes. 


Cette partie vise à décrire très succintement le fonctionnement de DALL-E.

[**mettre un petit schéma**]

Le modèle DALL-E peut être décomposé en 2 briques principales:
* Un **auto-encodeur** permettant de projetter les images dans un espace de dimensions plus réduites. Un auto-encodeur est un réseau de neurones particulier constitué d'un *encodeur* et d'un *décodeur*. L'encodeur projette l'image en entrée dans un espace de faible dimension, et le décodeur tente de reconstruire l'image originelle à partir de cette représentation compressée. On peut calculer une perte entre l'image reconstruite et l'image d'origine et ainsi utiliser la *backpropagation* pour ajuster les paramètres du modèle. Utiliser un auto-encodeur va permettre de travailler sur l'espace latent, dans lequel les images sont représentées par des tokens image et non pas par l'ensemble des pixels, rendant la phase d'apprentissage et la phase d'inférence plus simple et générique. Le modèle DALL-E utilise une version particulière d'autoencodeur, appelée dVAE. [expliquer le DVAE en une phrase]
* Un **transformer** qui apprend à produire de manière itérative une image à partir d'une description textuelle. La description textuelle est également encodée par un encoder (différent de celui utilisé pour les images) sous la forme de tokens. Lors de la phase d'inférence, le transformer reçoit les tokens de texte, et génère un token d'image à partir de ceux-ci. Ce token image est réinjecté dans le transformer qui produit le suivant, et ainsi de suite. Finalement, le transformer a produit tous les tokens images. On peut alors utiliser le décodeur de l'auto-encodeur pour recréer une image à partir de ces tokens. 


Pour le modèle dalle-mini:
* La phase d'apprentissage consiste à entraîner le decoder du transformer sur un jeu de données constitué des pairs (caption,image) à l'aide d'un encodeur-image exogène déjà entrainée (VQGAN) et d'un encodeur-texte exogène déjà entrainée (BART). Le résultat final est un decoder-texte-vers-image entrainée qui produit des tokens images, qui peut être traduite en image à l'aide du decodeur-image exogène.

* La phase d'inférence permet après l'apprentissage, de générer des images à partir des captions. Pour cela, les captions en entrée sont données à l'encodeur-texte qui produit une séquence de tokens-texte. Ces tokens-texte sont données en entrée du decodeur-texte-vers-image, renvoyant plusieurs séquences de tokens-image. Ces séquences de tokens-image sont traduites en image à l'aide du decodeur-image exoègne.

En outre, il est nécessaire de pouvoir mesurer la qualité des images générées par rapport à la caption. Cette mesure nous ait fourni par le modèle CLIP. En effet, ce modèle permet de donner un score à la paire d'une image et d'un texte la décrivant, et peut donc être utilisé pour évaluer la qualité des sorties de DALL-E. 


### 1.1 Auto-formation

Pour pouvoir pleinement appréhender le modèle, nous avons dû nous former à des concepts qui ne seront abordés qu'à partir de la troisième année de l'ENSAE, à savoir le deep learning, le NLP et les méthodes de computer vision.

- réseaux de neurones (Youtube, tutoriel pytorch)
- réseaux de neurones récurrents, convolutifs
- transformer
- tutoriel HuggingFace

### 1.2 Lecture des articles

En parallèle des auto-formations pour la remise à niveau, la première mission demandée par les tuteurs a été d'expliquer le modèle DALL-E à partir des différents documents de la bibliographie : 

[**completer/mettre l'ensemble des papiers bibliographie des tuteurs**]
- mécanisme d'attention
- architecture _transformer_
- principe de la quantization
- _generative adversarial network_ (GAN)
- _variationnal auto-encoders_
- "Zero-Shot Text-to-Image Generation" (2021) qui introduit le modèle **Dall-E**
- “Learning Transferable Visual Models From Natural Language Supervision” (2021) qui introduit le modèle **CLIP**
- "Dall-E Mini. Generate images from a text prompt in this interactive report: DALL·E on a smaller architecture." (2021) qui décrit **Dall-E Mini**

[**mettre l'ensemble des papiers bibliographie perso**]

La lecture des articles et l'auto-formation nous a permis de restituer à la fois le fonctionnement de mini-DALLE et en partie le fonctionnement de DALL-E sous formes de deux séances de présentations aux tuteurs. [**mettre le lien ves slides**] 

### 1.B. Exploration

Après la compréhension du modèle, une deuxième mission a été de générer des images à partir de mini-dalle et d'évaluer leur pertinence. Pour cela, il a fallu utiliser plusieurs jeux de données, produire un notebook permettant la génération d'image à partir des modèles publics préentrainé, générer les images et analyser quantitativement et qualitivativement ces dernières par rapport à l'image de référence (golden image - image gold).

### 1.B.1 Les données

Nous avons utilisés deux jeux de données (image-caption) disponibles au public, MS COCO et Flickr30k

- MS COCO (Microsoft Common Objects in Context) est un jeu de données contenant 328 000 images et une multitude d'annotations dont des descriptifs textuels.
- Flickr30k est un jeu de données contenant 31 000 images de Flickr, comportant pour chaque image 5 descriptifs textuels différents, issus d'un processus manuel.

Pour accéder facilement aux jeux de données, nous avons utilisé la librairie fiftyone pour MSCOCO et également des annotations disponibles sur le site (https://cocodataset.org/#download). Pour Flickr30k, nous sommes partis du découpage d'Andrej Karpathy et de Li Fei-Fei  (https://cs.stanford.edu/people/karpathy/deepimagesent/) pour accéder aux annotations et au découpage entraînement-test-validation.

Dans un premier temps, nous avons étudié un premier jeu de données à partir de MSCOCO de 100 images. Nous avons étudié la pertinence des descriptifs proposés et avons généré deux descriptifs alternatifs. A parti des trois descriptifs par image, nous avons regardé la pertinence de CLIP sur la capacité à rendre compte de la pertinence image-caption.
Puis, nous avons utilisé la caption d'origine pour la génération d'image.

Dans un second temps, nous avons utilisé le jeu de test de 1000 images-caption de Flickr30k pour la génération d'image.

#### 1.B.1 Générer des images avec Dall-E mini

A partir d'une caption pour chaque image de nos jeux de données, nous avons générer 10 images pour chaque descriptif de MSCOCO, soit 1000 images générées et 5 images pour chaque descriptif de Flickr30k, soit 5000 images générées. Ces images ont été générées chacun en un week-end.


#### 1.B.2 Évaluer manuellement les images 

En plus de la génération d'image à partir des descriptifs des jeux de données, nous avons essayé la génération d'image à partir de nos propres descriptifs pour voir quels sont les cas où mini-dalle arrivait à générer une image subjectivement correcte par rapport à la description demandée.

[**Résultats à mettre ici et à refaire ressortir visuellement - mettre des images**]

* forme
* texture
* couleur

#### 1.B.3 Évaluer les images avec CLIP

A partir du descriptif et des images générées et de l'image d'origine, CLIP nous fournit un score pour chaque pair image-descriptif


[**Résultats à mettre ici et à refaire ressortir visuellement - mettre des images**]



### 1.C. Modélisation

Nous nous apprêtons à produire une variante de Dall-E mini basée sur l'architecture de dalle-mini en utilisant les données de MS-COCO en se restreignant aux images de paysages (montagne, plage, plaine, etc.).

Pour cela, nous avons extrait à partir des descriptifs de MS-COCO, les images correspondantes aux paysages, en filtrant par mot-clé. Nous obtenons un jeu de données de plus de 10 000 images à affiner par la suite.



## 2. Difficultés rencontrées


## 3. Travaux restants à faire

