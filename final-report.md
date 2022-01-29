# Titre avec un peu d'humour

> Note pour mes co-auteurs: je pense qu'il est pertinent d'utiliser directement R Studio pour la rédaction (qui permet d'écrire du code Python, de toute façon), principalement pour bénéficier du très puissant Book Down (livres avec R Markdown) et d'une intégration très facile avec Github. Hackmd reste utilisable pour des retouches ponctuelles. Arthur
> 
> **Ressource pour écrire avec Markdown:**
> - https://bookdown.org/yihui/rmarkdown/



## 1. <titre qui résume notre étude par ex. : "Mini Dall-E comprend les couleurs et les images, mais pas l'espace">

**Résumé / Abstract?**

> Est-ce qu'on mettrait pas un petit glossaire aussi, vu les abbréviations fleuries? Léo

## 2. Dall-E: un modèle (_text-to-image_ à trois composantes et à trois temps)

Le modèle Dall-E peut être divisé en 2 briques et un élément de soutien:
* Un auto-encodeur, dont l'encodeur va permettre de compresser les images dans un espace latent, le décodeur n'étant utilisé que pendant la phase d'apprentisage
* Un *transformer* qui apprend à produire de manière itérative une image à partir d'une description textuelle
* CLIP, un autre modèle ne faisant pas directement partie de DALL-E, mais qui peut être utilisé pour évaluer la qualité des images produites

L'idée générale du modèle DALL-E est la suivante: chaque image peut être encodée dans un espace latent de dimension plus faible, et chaque texte peut également être encodé. Après ces deux encodages, on obtient une séquence de *tokens* - ceux de texte puis ceux de l'image - il est alors possible d'entraîner un *transformer* à modéliser de manière autorégressive la chaîne de token. 
Ainsi, au moment de l'inférence, on transformer le texte d'entrée en tokens de texte à l'aide de l'encodeur de texte, puis on le passe dans le transformer. Celui-ci va prédire le premier token image à partir des tokens de texte, puis le deuxième à l'aide de tous les tokens précédents et ainsi de suite. Pour finir, le décodeur de l'encodeur d'image est utilisé pour "dessiner" une image à partir des tokens images. 

# 2.1 Auto-encodeur

L'auto-encodeur est utilisé afin de comprimer les images. Un auto-encodeur est un réseau de neurones en deux parties
* L'encodeur compresse l'image d'origine dans un espace latent de dimension plus faible. Cette opération de réduction de dimension est conceptuellement similaire à l'analyse en composantes principales (ACP).
* Le décodeur décompresse l'image, c'est-à-dire qu'il s'efforce de recréer l'image d'origine à partir de sa version compressée.

Un tel modèle est entraîné par descente de gradient (backpropagation) afin que l'image en sortie du décodeur soit la plus semblable possible à celle en entrée de l'encodeur. On calcule une *loss* (perte) en fonction des différences entre l'image d'entrée et celle de sortie, qui est minimisée au cours de l'apprentissage de ce réseau de neurones. 

Une fois qu'un tel réseau de neurone est fonctionnel, l'idée va être de compresser les images en utilisant uniquement l'encodeur et de travailler sur la version compressée, qui est de dimension plus faible que l'image d'origine. Puisqu'à partir de cette version, le décodeur a été capable de regénérer toute l'image pendant la phase d'apprentissage, on suppose que la compression contient les informations importantes concernant l'image.

Les auto-encodeurs basiques ne sont cependant pas idéaux, et des versions plus complexes ont été développées.

Les **VAE**, *variational auto-encoder*, encodent les images non pas comme un point dans l'espace latent mais comme une distribution sur cet espace latent. A partir de cette distribution, il est possible d'échantillonner un point, de reconstruire l'image, puis de calculer la loss. Ce niveau de complexité supplémentaire permet à l'espace latent d'avoir de meilleurs propriétés.

Les **VQ-VAE**, *Vector Quantised-Variational Autoencoder*, introduisent un *codebook*, qui est une liste discrète de vecteurs dans l'espace latent. Cette liste est apprise par le modèle pendant la phase d'entraînement. Les différents vecteurs utilisés pour représenter l'image dans l'espace latent doivent appartenir au codebook. Dans un premier temps, on compresse l'image en utilisant des vecteurs proches de ceux du codebook, puis on remplace chacun d'entre eux par le vecteur le plus proche dans le codebook.


Le modèle DALL-E utilise un **dVAE**, *discrete variational auto-encoder*. Ce type d'auto-encodeur utilise également un codebook. Mais cette fois,  chaque image est encodée sous la forme d'un ensemble de distributions sur les différents vecteurs du codebook. Le problème de ce principe est que la discrétisation empêche les opérations différentielles nécessaires à la backpropagation. 2 astuces permettent de contourner ce problème: l'utilisation de la Gumbel softmax relaxation et le fait qu'il est permis aux vecteurs de vivre dans l'enveloppe convexe des vecteurs du codebook [**a expliquer / développer**]. 

DALL-E mini utilise quand à lui un **VQGAN**, *description*. Ce type d'auto-encodeur utilise comme encodeur et décodeur des **CNN**, *convolutional neural network* ou réseaux de neurones convolutionnels, le premier compressant l'image dans un espace latent en utilisant seulement les vecteurs d'un codebook discret et le deuxième tentant de reconstruire à partir de cette représentation l'image d'origine. [**!Compléter l'explication sur les VQGAN, il manque la partie expliquant qu'on fait intervenir des transformer!**]

# 2.2 Transformer

Les transformer ou les modèles seq2seq sont performants pour prédire des séquences. Mais il n'est pas possible de les utiliser avec tous les pixesl d'une image car cela demanderait trop de capacités de calculs. C'est pourquoi on utilise un autoencodeur afin de travailler dans un espace de dimension plus faible.


# 2.3 Discrimateur

*CLIP* signifie "Contrastive Language Pre-Training". 
Le modèle CLIP est un modèle de réseaux de neurones qui a été développé afin d'associer à une image la description la plus pertinente parmis un ensemble proposé. Le but de ce modèle est d'apprendre à effectuer cette association de manière non spécifique (zero-shot learning), contrairement aux réseaux de neurones entrainés à la reconnaissance d'un champ en particulier (e.g. réseau de neurone dédiée à la reconnaissance de la race d'un chien, entrainée sur ce champ particulier). Afin d'être le plus généraliste possible, le modèle a été entraîné sur 400 millions de paires image/texte. 

Ce modèle généraliste s'appuie fortement sur les forces du *NLP* (Natural Language Processing) et notamment sur l'apprentissage supervisé du langage, plutôt que de construire explicitement un modèle entrainé à choisir spécifiquement une description parmi plusieurs. 

En effet, l'entrée du modèle n'est pas directemenr un ensemble de textes et d'images. A la place, le texte en entrée passe par un encodeur de texte, et l'image par un encodeur d'image. Les vecteurs dans l'espace condensé obtenu sont alors intégrés dans une matrice, avec le vecteur de chaque image et de chaque texte.

Pour la phase d'entraînement, le modèle reçoit par paquet NxN paires texte-image avec Nx1 paires correctes et NxN-1 paires autres. Le modèle calcule alors l'indice de similarité entre chaque paire texte-image(dans l'embedding space). Son but est de maximiser la similarité entre les paires correctes et de minimiser celle entre les paires incorrectes. 

Cet apprentissage de la notion de semblable/dissemblable permet ainsi de pouvoir correctement prédire sur d'autres jeux de données que ceux utilisés dans la phase d'entraînement. La qualité des encodeurs est alors primordiale puisqu'ils regissent en amont la représentation vectorielle; cela revient à dire que la qualité du modèle repose sur celle des espaces condensées.

Lors de la phase d'inférence, on fournit à CLIP un ensemble de textes et une image, et  CLIP calcule les indices de similarité à partir des données fournies par les encodeurs et des poids entrainés du modèle. Les valeurs obtenus peuvent être normalisées afin d'obtenir une distribution de probabilités sur les textes possibles, le texte recevant le score le plus élevé est considéré comme la prédiction du modèle. Il est intéressant de noter que la manière dont les différents textes sont proposées à CLIP à son importance: on obtiendra ainsi de meilleurs résultats en intégrant les noms des différentes catégories à une phrase. Par exemple, pour prédire le type d'animal sur une photo, on préférera proposer comme descriptions "une photo de: ..." et remplacer les "..." par chat, chien, éléphant, plutôt qu'utiliser directement les noms des animaux comme texte. 

CLIP peut servir à d'autres tâches que la classification: il est possible de ne pas normaliser les scores et de se contenter de la valeur "brute" corespondant au produit scalaire des embeddings [?à traduire par vecteur latent?]. 



Les auteurs du modèle notent que malgré la grande généralité du modèle, ses performances restent médiocres sur des données complétement hors du jeu de données d'entraînement. CLIP ne résoud donc pas réellement le manque de capacité de généralisation de ce type de modèle, mais le contourne en utilisant un type d'apprentissage plus généraliste et une très grosse quantité de données. 
[...]




> Je laisse dessous la structure précédente du texte,



### 2.1 Les trois composantes

#### 2.1.1 Encodeur

L'encodeur d'un modèle est la partie qui convertit les entrées (texte et images dans notre cas) en une suite de vecteurs homogènes utilisable pour la modélisation statistique. Tous les modèles étudiées utilisent un encodeur de texte^[_ainsi qu'un tokenizer_] externe fixé qu'ils n'essaient même pas d'adapter au contexte.

|----|----|----|


L'encodeur et le décodeur (partie suivante) sont souvent entraînés simultanément et le détail est présenté à la section 3.

#### 2.1.2 Décodeur

#### 2.1.3 Discriminateur (CLIP)



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
- **Article de blog:** "How is it so good ?(DALL-E Explained Pt.2)", https://ml.berkeley.edu/blog/posts/dalle2/
#### Dall-E mini

- **Rapport:** "Dall-E Mini. Generate images from a text prompt in this interactive report: DALL·E on a smaller architecture." (2021), https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy
- Dall-E mini utilise *Taming Transformers* (https://github.com/CompVis/taming-transformers) comme VQ-GAN. Voir aussi l'article scientifique présentant le VQGAN: "Taming transformers for high-resolution image synthesis (2021)": https://compvis.github.io/taming-transformers/

#### MS COCO (2017)

- **Site:** https://cocodataset.org

#### Bases sur les réseaux de neurones et le machine learning
- **Série de vidéos youtube**: Neural networks, https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi  
- **Article de blog:** "First neural network for beginners ", https://towardsdatascience.com/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf
- **Article de blog:** Understanding variational auto-encoders, https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73