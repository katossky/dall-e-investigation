# Titre avec un peu d'humour

> Note pour mes co-auteurs: je pense qu'il est pertinent d'utiliser directement R Studio pour la rédaction (qui permet d'écrire du code Python, de toute façon), principalement pour bénéficier du très puissant Book Down (livres avec R Markdown) et d'une intégration très facile avec Github. Hackmd reste utilisable pour des retouches ponctuelles. 
> 
> **Ressource pour écrire avec Markdown:**
> - https://bookdown.org/yihui/rmarkdown/
> [name=Arthur]


## 0. 

### 0.1 Glossaire

* DL: Deep Learning = Apprentissage profond
* NN: Neural Network = Réseau de neurones 
* RNN : Recurrent Neural Network = Réseau de neurones récurrent
* GPT: Generative Pre-Trained Model
* VAE: Variational Auto-Encoder = Auto-encodeur variationnel
* VQ-VAE: Vector-Quantised Variational Auto-Encoder
* NLP: Natural Language Processing
* GAN: Generative Adversarial Network
* NMT: Natural Machine Translation
* LSTM: Long-Short Term Memory
* MLE: Maxium Likelihood Estimator
* AGI: Artificial general intelligence
* GRU: Gated recurrent unit
* CNN: Convolutionnal neural network
* VQGAN: 
* BERT:Bidirectionnal Encoder Representation from Transformers
* BART: 
* OCR: Optical Character Recognition



### 0.2 Anglicismes

Embedding
Input 
Output

## 1. <titre qui résume notre étude par ex. : "Mini Dall-E comprend les couleurs et les images, mais pas l'espace">

**Résumé / Abstract?**

## 2. Dall-E: un modèle (_text-to-image_ à trois composantes et à trois temps)

Le modèle Dall-E peut être divisé en 2 briques et un élément de soutien:
* Un auto-encodeur, dont l'encodeur va permettre de compresser les images dans un espace latent, le décodeur n'étant utilisé que pendant la phase d'apprentisage
* Un *transformer* qui apprend à produire de manière itérative une image à partir d'une description textuelle
* CLIP, un autre modèle ne faisant pas directement partie de DALL-E, mais qui peut être utilisé pour évaluer la qualité des images produites

L'idée générale du modèle DALL-E est la suivante: chaque image peut être encodée dans un espace latent de dimension plus faible, et chaque texte peut également être encodé. Après ces deux encodages, on obtient une séquence de *tokens* - ceux de texte puis ceux de l'image - il est alors possible d'entraîner un *transformer* à modéliser de manière autorégressive la chaîne de token. 
Ainsi, au moment de l'inférence, on transformer le texte d'entrée en tokens de texte à l'aide de l'encodeur de texte, puis on le passe dans le transformer. Celui-ci va prédire le premier token image à partir des tokens de texte, puis le deuxième à l'aide de tous les tokens précédents et ainsi de suite. Pour finir, le décodeur de l'encodeur d'image est utilisé pour "dessiner" une image à partir des tokens images [^ref_art_dal] [^ref_berk_dal] [^ref_opai_dal]

# 2.1 Auto-encodeur

L'auto-encodeur est utilisé afin de comprimer les images. Un auto-encodeur est un réseau de neurones en deux parties
* L'encodeur compresse l'image d'origine dans un espace latent de dimension plus faible. Cette opération de réduction de dimension est conceptuellement similaire à l'analyse en composantes principales (ACP).
* Le décodeur décompresse l'image, c'est-à-dire qu'il s'efforce de recréer l'image d'origine à partir de sa version compressée.

Un tel modèle est entraîné par descente de gradient (backpropagation) afin que l'image en sortie du décodeur soit la plus semblable possible à celle en entrée de l'encodeur. On calcule une *loss* (perte) en fonction des différences entre l'image d'entrée et celle de sortie, qui est minimisée au cours de l'apprentissage de ce réseau de neurones. 

Une fois qu'un tel réseau de neurone est fonctionnel, l'idée va être de compresser les images en utilisant uniquement l'encodeur et de travailler sur la version compressée, qui est de dimension plus faible que l'image d'origine. Puisqu'à partir de cette version, le décodeur a été capable de regénérer toute l'image pendant la phase d'apprentissage, on suppose que la compression contient les informations importantes concernant l'image.

Les auto-encodeurs basiques ne sont cependant pas idéaux, et des versions plus complexes ont été développées.

Les **VAE**, *variational auto-encoder*, encodent les images non pas comme un point dans l'espace latent mais comme une distribution sur cet espace latent. A partir de cette distribution, il est possible d'échantillonner un point, de reconstruire l'image, puis de calculer la loss. Ce niveau de complexité supplémentaire permet à l'espace latent d'avoir de meilleurs propriétés [^refvae].

Les **VQ-VAE**, *Vector Quantised-Variational Autoencoder*, introduisent un *codebook*, qui est une liste discrète de vecteurs dans l'espace latent. Cette liste est apprise par le modèle pendant la phase d'entraînement. Les différents vecteurs utilisés pour représenter l'image dans l'espace latent doivent appartenir au codebook. Dans un premier temps, on compresse l'image en utilisant des vecteurs proches de ceux du codebook, puis on remplace chacun d'entre eux par le vecteur le plus proche dans le codebook.


Le modèle DALL-E utilise un **dVAE**, *discrete variational auto-encoder*. Ce type d'auto-encodeur utilise également un codebook. Mais cette fois,  chaque image est encodée sous la forme d'un ensemble de distributions sur les différents vecteurs du codebook. Le problème de ce principe est que la discrétisation empêche les opérations différentielles nécessaires à la backpropagation. 2 astuces permettent de contourner ce problème: l'utilisation de la Gumbel softmax relaxation et le fait qu'il est permis aux vecteurs de vivre dans l'enveloppe convexe des vecteurs du codebook [**a expliquer / développer**]. 

DALL-E mini utilise quand à lui un **VQGAN**, *description* [^refvqgan], [^ref_dal_mini], [^ref_dal_mini_vqgan]. Ce type d'auto-encodeur utilise comme encodeur et décodeur des **CNN**, *convolutional neural network* ou réseaux de neurones convolutionnels, le premier compressant l'image dans un espace latent en utilisant seulement les vecteurs d'un codebook discret et le deuxième tentant de reconstruire à partir de cette représentation l'image d'origine. [**!Compléter l'explication sur les VQGAN, il manque la partie expliquant qu'on fait intervenir des transformer!**]

# 2.2 Transformer

Les transformer ou les modèles seq2seq sont performants pour prédire des séquences. Mais il n'est pas possible de les utiliser avec tous les pixesl d'une image car cela demanderait trop de capacités de calculs. C'est pourquoi on utilise un autoencodeur afin de travailler dans un espace de dimension plus faible.


# 2.3 Discrimateur

*CLIP* signifie "Contrastive Language Pre-Training". 
Le modèle CLIP est un modèle de réseaux de neurones qui a été développé afin d'associer à une image la description la plus pertinente parmis un ensemble proposé. Le but de ce modèle est d'apprendre à effectuer cette association de manière non spécifique (zero-shot learning), contrairement aux réseaux de neurones entrainés à la reconnaissance d'un champ en particulier (e.g. réseau de neurone dédiée à la reconnaissance de la race d'un chien, entrainée sur ce champ particulier). Afin d'être le plus généraliste possible, le modèle a été entraîné sur 400 millions de paires image/texte [^ref_art_clip] [^ref_bpost_clip]

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

L'encodeur d'un modèle est la partie qui convertit les entrées (texte et images dans notre cas) en une suite de vecteurs homogènes utilisable pour la modélisation statistique. Tous les modèles étudiées utilisent un encodeur de texte (ainsi qu'un tokenizer)
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

### A.2 The illustrated transformer [^bpost_ilutransfo] 

**Vision large de la structure du transformer**

Cet article présente l'architecture des transformers, en prenant comme exemple de tâche la traduction d'une phrase. 
Un des gros avantages des transformeurs est qu'ils permettent la parallélisation des calculs (très utile dans la mesure où les limites des capacités de calcul sont souvent un problème en ML). 

Un transformer est composé d'un encodeur et d'un décodeur, mais ici, chaque encodeur est en fait une pile d'encodeurs et de même pour les décodeurs.
![](https://i.imgur.com/i8od1gO.png)

Chaque encodeur a la même structure, mais des poids différents, ils contiennent 2 sous-unités: 
![](https://i.imgur.com/jrmdFwL.png)
La couche de *self-attention* permet à l'encodeur de "regarder" d'autres mots lorsqu'il encode un mot. La couche de *feed forward* est la même pour chaque encodeur.

![](https://i.imgur.com/6Z75T22.png)

Les décodeurs incluent une couche supplémentaire. Cette couche aide le decoder à se concentrer sur des parties importantes de la séquence en entrée. 

**Fonctionnement du transformer**
Chaque mot en entrée est transformé en vecteur en utilisant un algorithme **d'embedding**. 

La *self-attention* permet au transformer de s'intéresser à d'autres parties de la phrase, liées au mot en cours de traduction. C'est un processus basé sur le calcul matriciel:
1. Pour chacun des vecteurs en entrées, on crée 3 vecteurs, appelées *key* (**K**), *query*(**Q**) et *value* (**V**). Ces vecteurs sont calculées comme le produit des embeddings des entrées par des matrices (**W<sup>K</sup>**, **W<sup>Q</sup>** et **W<sup>V</sup>**), dont les valeurs sont ajustées pendant l'entraînement du modèle. 
2. Ensuite, on calcule le score de tous les mots par rapport à un autre mot, le premier par exemple. Pour ce faire, on multiplie le vecteur *query* du mot cible par le vecteur *key* de chacun des autres mots
3. On divise par 8 (la racine de la dimension du vecteur *key*, ici 64)
4. A ce stade des calculs, on a une valeur par mot, par rapport au mot cible. On applique alors un softmax qui fait que la somme des valeurs vaut 1 et qu'elles sont toutes positives. 
5. On calcule ensuite la somme des vecteurs *values* pondérée par le coefficient obtenu pour chaque mot à l'étape précédente.

**On obtient finalement le résultat de la couche de self-attention pour le premier mot**

Au lieu de faire ce processus itérativement - chaque mot prenant à tour de rôle le rôle du mot "cible" - on peut le faire beaucoup plus rapidemnet en utilisant le produit matriciel.

![](https://i.imgur.com/R3ZTyQ1.png)

Les vecteurs embeddings sont empilés les uns au-dessus des autres pour former la matrice X (dont chaque ligne correspond à un embedding, c'est-à-dire, d'une certaine façon, à un mot). On multiplie alors X par chacune des 3 matrices de poids et on obtient les matrices query, key et values.

Les étapes 2 à 5 peuvent alors être condensés par un calcul matriciel, comme visible sur la figure suivante:

![](https://i.imgur.com/6P6H6D6.png)

Cette idée d'attention peut être complexifiée sous la forme de *mlti-headed attention*. L'idée générale est d'avoir k exemplaires des matrices key, query et values au lieu d'un seul. On obtiendra autant de matrices Z qu'on a de têtes (c'est-à-dire d'exemplaire de chacune des matrices), qu'on concatène en une grosse matrice. 

D'autre part, pour donner au modèle une idée de la position entre les différents mots dans la phrase, on peut ajouter un *encoding positionnel*, c'est-à-dire qu'on ajoute à l'embedding de chaque mot une valeur correspondant à un motif permettant de savoir sa position.

Chaque sous-couche dans chaque encodeur a aussi une connexion résiduelle suivie d'une couche de normalisation. Cela veut simplement dire que la sortie de la couche de self-attention est ajoutée à l'input originel, puis que l'on effectue une normalisaiton 

> Je ne sais pas coment on fait cette normalisation, apparemment cela se réfère à l'article: https://arxiv.org/abs/1607.06450
>[name=leo]


Lorsque l'on n'est plus sur l'encodeur tout en bas de la pile, l'input originel ajouté est celui de l'encodeur en question, et pas l'input tout en bas de la pile d'encodeurs.

L'encodeur tout en haut de la pile d'encodeurs produit des matrices d'attentions K et V qui vont pouvori être utilisées par chacun des décodeurs dans la couche d'attention encodeur-décodeur. L'encodeur produit un mot à la fois, qui est alors réintroduit dans le décodeur du bas de la pile. 
Les couches de self attention des décodeurs fonctionnent un peu différemment. D'une part, ils ne peuvent pas regarder les positions futures (ce qui est logique, étant donné que ces mots ne sont pas encore connues, mais ce qui n'était pas le cas pendant la phase d'entraînement.) D'autre part, les matrices de queries sont créées à partir du décodeur de dessous dans la pile, alors ques les matrices keys et values viennent toujours de l'encodeur en haut de sa pile. 

Tout à la fin du transformer se trouvent un réseau de neurone entièrement connecté (appelé *fully connected neural network*) qui projette la sortie de l'encodeur du haut dans un espace de la longueur du vocabulaire. Ainsi, à chaque mot du vocabulaire est associé un score, qui est transformé en probabilités par un softmax. Finalement, le mot associé à la probabilité la plus élevée est choisi. Une autre possibilité est de conserver les deux mots ayant les plus grandes probabilités, de les réinjecter dans la pile d'encodeur et de conserver celui associé à la perte la moins élevée.

La fonction de perte demande de calculer la différence entre deux distributions de probabilité, ce qui fait intervenir la cross-entropie et la divergence de Kullback-Leibler.

### A3. A beginer guide on reccurent neural network [^bpostRNNbeg]

Les RNN (pour *reccurrent neural network* ou réseau de neurones récurrent), utilise une seule *cellule* de calculs. Au lieu de lui injecter d'un coup toute l'entrée, cela fait élément par élément. Après l'introduction de chaque élément, le RNN fait des calculs et met à jour son *hidden state*, qui sera réintroduit dans le réseau à l'étape suivante, en même temps que l'élément suivant. 

Les calculs de chaque étape prennent donc en compte d'une part le nouvel input mais aussi le "contexte", sous la forme du *hidden state*, qui contient une partie de l'information des étapes passées.

Selon le but du réseau, on peut vouloir garder uniquement l'ouput final, ou conserver également les ouputs intermédiaires. 

![](https://i.imgur.com/k7sQyOC.png)

### A.4 Attention mechanism [^bpost_att_mech]

Il est possible de distinguer 2 types d'attentions:
* *la self attention* s'intéresse aux liens à l'intérieur de l'input
* l'attention générale se concentre sur les interdépendances entre l'input et l'output

Ces méchanismes d'attention sont surtout utilisés dans les modèles de NLP. Cet article s'intéresse plus précisemment à l'attention dans les modèles seq2seq et à la self-attention. 

![](https://i.imgur.com/grExiFr.png)

Le graphique précédent présente l'attention dans les modèles seq2seq (avec un RNN). Un tel modèle sans attention a du mal à gérer les longues séquences parce que le décodeur a uniquement accès au dernier hidden state. Lorsqu'il y a de l'attention, le décodeur peut cette fois accéder aux hidden state de chaque élément de l'input. 

Comme représenté par le schéma ci-dessous, il existe différents méchanismes d'atention: Bahdanau et Luong.
![](https://i.imgur.com/psH0Xiy.png)

Les étapes de calcul
1. Calculer les hidden state de l'encodeur (pour tout input)


Représentation graphique des calculs (pas totalement comprise):
![](https://i.imgur.com/4JrL0u7.png)

> Je mets certaines mais pas toutes de mes notes sur cet article, notamment parce qu'il parle de RNNs (et que ce n'est pas le cas dans les transformer) et que je l'ai trouvé compliqué et pas assez intéressant pour que je m'y replonge
> [name=Léo]

### A.5 Understanding VAE [^bpost_unVAE]

Un VAE est une sorte d'autoencodeur dont la partie encodage est régularisée de manière à avoir de bonnes propriétés, et notamment afin que l'on puisse générer des données à partir de l'espace latent. Le terme variationnel vient du fait que la régularisation a un lien avec la méthode d'inférence variationnelle en statistiques

**Réduction de dimension, ACP et autoencodeurs**

La réduction de dimension consiste à réduire le nombre de caractéristiques permettant de réduire des données, par sélection ou extraction (création de nouvelles caractéristiques à partir des anciennes). 

Dans un auto-encodeur, l'encodeur crée de nouvelles caractéristiques et le décodeur fait l'inverse, donc on fait en quelque sorte de compression de données, de l'espace initial dans l'espace d'encodage aussi appelé *latent space*:

![](https://i.imgur.com/s07iIvP.png)

L'objectif peut alors être de trouver le meilleur couple d'encodeur décodeur dans une famille, cad celui minimisant l'erreur de reconstruction. Dans le cas d'un réseau de neurones, on peut chercher à atteindre cet objectif via la backpropagation de l'erreur. 

Lorsque l'on fait de l'ACP, c'est plus ou moins le but: on cherche des approximations linéaires permettant de minimiser la distance euclidienne entre les points au début et leur projection. On le fait en prenant les n eigen vectors associés au n eigen values les plus élevées. 

**VAE**
Dans un auto-encodeur classique, on risque d'avoir de l'overfitting menant à de mauvaises propriétés de l'espace latent. Cela empêche notamment de prendre un point dans l'espace latent afin de générer des données similaires à celle en entrée du modèle. 

On va donc imposer une régularisaiton pendant l'entraînement pour avoir une certaine régularité de l'espace latent. Au lieu d'encoder chaque point de données par un point dans l'espace latent, on les encode comme une **distribution** sur l'espace latent. On peut alors échantillonner un point depuis cette distribution, décoder, calculer une erreur de reconstruciton et backpropager l'erreur. La distribution sur l'espace latent peut être décrite par un vecteur moyenne et une matrice de covariance. La *loss* utilisée est une combinaison d'un terme de reconstruction rendant le processus de compression/ décompression efficace et d'une terme de régularisation permettant une certaine régularité de l'espace latent. 

On veut deux propriétés essentielles pour l'espace latent:
1. Continuité: 2 points proches dans l'espace latent devrait également être proches dans la réalité
2. Complétude: un point issu de la distributio nest "meaningful" une fois décodé. 
3. 
On atteint ces objectifs en forcant la distribution sur l'espace latent à ^tre proche d'une loi normale, ce qui a tendance à faire augmenter l'erreur de reconstuction

### A.6 World level translation English to Marathi Neural machine translation using encoder-decoder model [^bpost_wltran]

On s'est rendus compte que les RNNs étaient très bon pour le NLP. 
Les modèles seq2seq utilisent une architecture avec un encodeur et un décodeur, les 2 étant des LSTM (Long-Short Term-Memory, RNN bons pour la mémoire à long-terme).

L'encodeur lit l'input et le résume en un *internal state vectors*, d'une part l'état de la cellule c et d'autre part l'état interne caché h. L'encodeur produit également des outputs mais on ne les utilise pas puisqu'on ne commence la traduction qu'après avoir lu toute la phrase. Les états hk et ck (pour la traduction d'une phrase de k mots) produit à la fin de la lecture de la phrase par l'encodeur sont appelés les encodings de l'input car ils en sont un résumé (de la totalité de la séquence). Le décodeur prend ces vecteurs comme input et génère la séquence de sortie. 

Les états initiaux du décodeur sont les états finaux de l'encodeur. A chaque étape on donne au décodeur le vrai output et non pas l'output prévu par l'étape précédente, ce *teaching forcing* permet d'accélérer l'apprentissage. Puis on calcule une loss et on utilise la backpropagation. 

A *inference time* le décodeur recoit les états de l'encodeur dépendant de l'input. Puis, à chaque étape suivante, il recoit à la fois l'output prévu à l'étape précédente et les états internes calculées à l'étape précédente. Quand le décodeur produit "STOP", la raduction s'arrête. 

Processus d'inférence:
![](https://i.imgur.com/1Yn1H7V.png)

Processus d'entraînement:
![](https://i.imgur.com/TI7gM7v.png)


### A.7 Understanding LSTM [^bpost_unlstm]

### A.8 The unreasonnable effectiveness of RNNs [^bpost_efffeRNN]

### A.9 Transformer model (1/2) [^vid_transf1/2]

[^vid_transf1/2]: **Vidéo**, Transformer Model (1/2): Attention Layers ,*https://www.youtube.com/watch?v=FC8PziPmxnQ*

[^bpost_efffeRNN]: **Article de blog**, The Unreasonable Effectiveness of Recurrent Neural Networks,*http://karpathy.github.io/2015/05/21/rnn-effectiveness/*

[^bpost_unlstm]: **Article de blog**, Understanding LSTM Networks, *http://colah.github.io/posts/2015-08-Understanding-LSTMs/*

[^bpost_wltran]: **Article de blog**, Word Level English to Marathi Neural Machine Translation using Encoder-Decoder Model, *https://towardsdatascience.com/word-level-english-to-marathi-neural-machine-translation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7*

[^bpost_unVAE]: **Article de blog**, Understanding Variational Autoencoders (VAEs), *https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73*

[^bpost_att_mech]: **Article de blog**, Attention mechanism *https://blog.floydhub.com/attention-mechanism/*

[^bpostRNNbeg]: **Article de blog**, A beginner's guide on reccurrent neural network with PyTorch, *https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/*

[^bpost_ilutransfo]: **Article de blog**: The illustrated transformer, *http://jalammar.github.io/illustrated-transformer/*



### Bibliographie

> J'ai rajouté des citations via numéro, pour qu'on sache qu'elle citation va où. Par conséquent, la partie bibliographie en "view" n'a aucun sens, mais elle est toujours pertinente pour organiser les sources dans la partie "edit". [name=Léo]

#### CLIP

* [^ref_art_clip]:**Article scientifique:** "Learning Transferable Visual Models From Natural Language Supervision" (2021),  https://arxiv.org/abs/2103.00020
* [^ref_bpost_clip]: **Article de blog:** "CLIP: Connecting
Text and Images" (2021), https://openai.com/blog/clip/


#### Dall-E

- [^ref_art_dal]: **Article scientifique:** "Zero-Shot Text-to-Image Generation" (2021), https://arxiv.org/abs/2102.12092
- [^ref_opai_dal]: **Article de blog:** "DALL·E: Creating
Images from Text" (2021), https://openai.com/blog/dall-e
- [^ref_berk_dal]: **Article de blog:** "How is it so good ?(DALL-E Explained Pt.2)", https://ml.berkeley.edu/blog/posts/dalle2/
- 
#### Dall-E mini

- [^ref_dal_mini]:**Rapport:** "Dall-E Mini. Generate images from a text prompt in this interactive report: DALL·E on a smaller architecture." (2021), https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy
- [^ref_dal_mini_vqgan]: Dall-E mini utilise *Taming Transformers* (https://github.com/CompVis/taming-transformers) comme VQ-GAN. 
- [^refvqgan]: **Article scientifique** présentant le VQGAN: "Taming transformers for high-resolution image synthesis (2021)": https://compvis.github.io/taming-transformers/

#### MS COCO (2017)

- **Site:** https://cocodataset.org

#### Bases sur les réseaux de neurones et le machine learning
- **Série de vidéos youtube**: Neural networks, https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi  
- **Article de blog:** "First neural network for beginners ", https://towardsdatascience.com/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf
- [^refvae]: **Article de blog:** Understanding variational auto-encoders, https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73