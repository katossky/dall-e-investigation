---

slideOptions:
  transition: fade
  theme: white
  
---

# Présentation

Note: chaque partie prend 2-3 min.

----

## 1. Réseaux de neurones 

----

### 1.1 Du perceptron au perceptron multi-couche

----

#### Le perceptron : définition

* un neurone est une entité qui prend des signaux en entrée et qui renvoit un signal en sortie
* avec n entrées, est une fonction de $R^n*R^n*R$ dans $R$  $f(X,w,b)= \sigma(w'X+b)$
    * $X$ les signaux en entrée dans $R^n$
    * $w$ le vecteur des poids accordés à chaque signaux
    * $b$ le biais d'activation du neurone
    * $\sigma$ une fonction continue de $R$ dans $R$

----

#### Le perceptron : intérêt

* un neurone peut être entrainé (choix de $w$,$b$) pour apprendre un comportement (apprentissage supervisé)

----

#### Le perceptron : représentation graphique


* **source : https://towardsdatascience.com/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf**
![perceptron](https://miro.medium.com/max/1302/1*UA30b0mJUPYoPvN8yJr2iQ.jpeg)

----

#### Le perceptron : exemples

* prendre deux booléens et donner l'union
* prendre une image 1000*1000 pixels et dire si oui ou non c'est un chien
* prendre des mots et savoir si c'est écrit en français ou non
* besoin de classification 1= oui 0= a priori non, autre= ?
* e.g. pour les signaux, simplification dans $[0,1]$ avec comme but pour le signal de sortie d'être dans $\{0,1\}$

----

#### Le perceptron : Union

* prendre deux booléens et donner l'union
    *  $X1=0, X2=0 => Y_{expected}=0$
    *  $X1=0, X2=1 => Y_{expected}=1$
    *  $X1=1, X2=0 => Y_{expected}=1$
    *  $X1=1, X2=1 => Y_{expected}=1$
*  $w1=w2=1; b=1/2 ;\sigma=\mathbb{1}(wX-b>0)$
*  ou $w1=w2=10; b=5 ;\sigma=\mathbb{1}(wX-b>0)$

----

#### Le perceptron : Recap et limites

* entrée $X$ donnée, $(w , b)$ paramètres du neurone, $\sigma$ fonction d'activation choisie
* on peut trouver $(w,b)$ pour obtenir $Y$ à partir de $X$  pour des données linéairement séparables
* problématiques de l'overfitting et de l'interprétabilité ?

----

#### L'union fait la force : le perceptron multicouche

* les autres exemples énoncés précédemment demandent à combiner plusieurs neurones en couches pour être plus précis = Perceptron multi-couches
* **level up** : peut également prédire une variable catégorielle (binaire -> valeurs discrètes finies) =  nombre de neurones en couche de sortie

----

#### L'union fait la force : le perceptron multicouche

* possibilité de mettre plusieurs couches de neurones intermédiaires = gain ?
* **théorème d'approximation universelle** : un réseau de neurones avec au moins une couche intermédiaire peut approximer n'importe quelle fonction sur un compact de $R^n$ ( linéairement séparables => données quelconques)


----

#### Le perceptron multicouche : représentation graphique


* **source : https://towardsdatascience.com/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf**
![MLP](https://miro.medium.com/max/1000/1*v1ohAG82xmU6WGsG2hoE8g.png)


----

#### Le perceptron multicouche : exemples 

* reconnaître l'écriture d'un chiffre et donner son chiffre
* régression logistique dans le cas non linéaire

----

#### Le perceptron multicouche : apprentissage

* fonction de coût/perte $L(W,B)$ : écart quadratique moyenne, valeur absolue moyenne
* but = **minimiser les erreurs** => apprendre => corriger les poids/biais
* $min_{W,B} L(W,B)$ 
* algorithme de descente du gradient

----

#### Le perceptron multicouche : descente du gradient

* calcul du gradient sur la fonction de perte
* correction des poids et biais des neurones
* **minimas locaux**

----

#### Le perceptron multicouche : limites

* dans le cas d'une image, prend les pixels en entrée de manière indépendante. Certains pixels dépendent de ses voisins (spatialement ou temporellement)
* sens du signal unidirectionnel

----

### 1.2 Vers des modèles avancées et l'au-delà

* CNN : Réseau de neurones entièrement connectés / Convolution / Déconvolution
* Réseau de neurones récurrents

----

#### Réseau de neurones convolutifs : but

* But : Réduction de dimensions - passer de $n*m$ dimensions à $d$ dimensions  ($d << n*m$)
* en rajoutant la notion de dépendance spatio-temporelle entre neurones

----

#### Réseau de neurones convolutifs : opérations

* Opération de convolution : ajouter l'information des voisins dans le neurone (ajouter de la dépendance)
* Opération de pooling : regrouper et réduire la dimension (perte d'informations redondantes ou non)
* Opération de correction
* Connexion à un réseau de neurones entièrement connectés

----

#### Réseau de neurones convolutifs : convolution

* **source : https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53**
![convolution](https://miro.medium.com/max/500/1*GcI7G-JLAQiEoCON7xFbhg.gif)

----

#### Réseau de neurones convolutifs : pooling

* **source : https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53**
![pooling](https://miro.medium.com/max/500/1*KQIEqhxzICU7thjaQBfPBQ.png)

----

#### Déconvolution

* But : reconstituer l'information en dimension $n*m$ à partir de l'information à $d$ dimensions
* il y a eu de la perte => reconstituer parfaitement n'est pas possible

----

#### Déconvolution : méthode

* Il suffit d'appliquer les opérations transposées de la convolution
* jeux sur les paramètres de striding et de padding
    * [A guide to convolution arithmetic for deep learning, Vincent Dumoulin, Francesco Visin
](https://arxiv.org/pdf/1603.07285.pdf)

----

#### Réseaux de neurones récurrents : Définition

* possibilité d'avoir des cycles dans le graphe des neurones
* but : travailler avec une mémoire, un contexte

----

#### Réseaux de neurones récurrents : représentation graphique

* **source : https://towardsdatascience.com/four-common-types-of-neural-network-layers-c0d3bb2a966c**
![RNN](https://miro.medium.com/max/333/1*evR7fkBJLxYD4mcbB-4YTw.png)

----

#### Réseaux de neurones récurrents : limites

* Le temps de calcul est rédhibitoire 
* vanishing gradient : utilisation de la fonction tanh
* multiplication de grandes décimales proches de 0
* besoin d'avoir l'intégralité de la séquence pour produire une prédiction

---

## 2. Attentions et transformeurs

----

### 2.1 Mécanismes d'attention

Pour modéliser les dépendence de long terme, les réseaux récurrents ont trois inconvénients majeurs:

1. le temps de calcul est redhibitoire
2. impossible de paralléliser
3. la portée de la récurrence est, en pratique, limitée

Les trois problèmes sont résolus par les mécanismes d'attention, qui par la suite ont empiriquement trouvé la démonstration de leur force dans les architectures dites "transformer".

<small>Il existe plusieurs mécanisme d'attention, mais nous ne parlons ici que de la _self-attention_.</small>

----

#### Principe de base

L'attention fonctionne pour une suite de $N$ observations dépendantes les unes des autres $(\mathbf{x}_n)_{n=1..N} \in (\mathbb{R}^k)^N$, modélisées de façon auto-régressives, où $k$ représente le nombre de variables observées.

<small>La technique, initialement développée pour le texte peut être utilisé pour d'autres domaines. Dans le cas du texte, il faut dans un premier temps convertir la suite de caractère en _tokens_, puis "plonger" chaque _token_ dans un espace de dimension $k$ appelé _plongement_ et appris de façon indépendante par un auto-encodeur.</small>

----

#### Principe de base

En entrée du réseau de neurones classique, on remplace $\mathbf{x}_j \in \mathbb{R}^k$ par $\mathbf{z}_j \in \mathbb{R}^{k'}$, qui est une combinaison non-linéaire des $(\mathbf{x}_i)$.

<small>J'ai encore du mal à traduire ce qui se passe réellement en français classique.</small>

----

#### Écriture matricielle 1/2

![](https://i.imgur.com/5xLEOCy.png =x600)

----

#### Écriture matricielle 2/2

![](https://i.imgur.com/XEOcnDg.png)

----

#### Avantages

* Le mécanisme d'attention comprends de nombreuses étapes purement linéaires, qui peuvent être délégués à des unités de calcul spécialisée (GPU). Lesmatrices $W^Q$, $W^K$ et $W^V$ peuvent donc être apprises facilement.
* Par ailleurs, le réseau de neurones "branché" sur les $\mathbf{z}_n$ est un réseau _feed-forward_ très simple, et donc les propagations avant et arrière peuvent être calculées en parallèle, contrairement à un réseau récurrent.
* La portée des récureences est potentiellement infinie.

----

#### Avantages

_Cherry on the cake_, le mécanisme d'attention sur des images peut en pratique "apprendre" le principe de la convolution!

<small>Le coût de calcul ne devient-il pas très grand pour un très grand $N$? Si et justement il s'agit dun _sparse transformer_.</small>

----

### 2.2 Attention multi-tête

Même si ce mécanisme d'attention est multidimensionnels (puisque $\mathbf{Q}$ et $\mathbf{K}$ ont plusieurs colonnes), l'information transmise via $\mathbf{z}_j$ est limitée.

Des auteurs ont alors proposé une architecture "multi-tête", avec non pas un mais plusieurs mécanismes d'attention.

<small>À ce stade il n'est pas clair pour moi pourquoi augmenter le nombre de tête est plus utile qu'augmenter la taille de $\mathbf{Q}$ et $\mathbf{K}$...</small>

----

![](https://i.imgur.com/NYG9UUR.png)

----

### 2.3 Le transformeur originel

Les mécanismes d'attention ont eu tellement de succès que des architecture basée quasi-uniquement sur ce principe ont rapidement vu le jour. Le plus célèbre est celui de l'article _Attention is all you need_ (2017) qui intorduit l'architecture "Transformer".

----


![](https://i.imgur.com/LtAA26h.png)

----

![](https://i.imgur.com/N5ptNLh.png)

----

#### Bas les masques !

Les mécanismes d'attention fonctionnent de la même façon dans le décodeur que dans l'encodeur **à l'exception notable** que des masques (i.e. des opérateurs qui passe certaines éléments de la matrix d'attention à $-\infty$) empêchent le décodeur de porter attention à des tokens arrivant _après_ dans l'ordre de la phrase.

Dans le cas d'images, ces opérateurs peuvent empêcher certaines parties du décodeurs de prendre en compte des pixels trop éloignés.

----

### 2.4 Des transformeurs partout

L'architecturee initiale d'un transformeur était composée d'un encodeur et d'un décodeur, chacun constitué de 6 modules. Un module consistait en un sous-module d'attention (_self-attention_) et un sous-module de récurrence (_feed-forward network_). Chacune des composantes étant intéressante en elle-même, certaines équipes ont privilégié l'encodeur (BERT) ou le décodeur (GPT-2 et 3).

Dans Dall-E, c'est le décodeur qui est utilisé, comme dans BERT. L'encodeur, en effet, est lui basé sur les auto-encodeurs discrets.

---

## 3. Auto-encodeurs discrets (dVAE)

### Une partie clé du modèle DALL-E

* Les images en entrée sont de trop grandes dimensions pour être analysées par un transformer (256\*256). On veut les compresser en perdant le moins d'informations possibles. 
* C'est ce que fait le dVAE, qui les transforme en une grille de 32\*32.

----

### Auto-encodeur (AE)
![VAE](https://miro.medium.com/max/2000/1*UdOybs9wOe3zW8vDAfj9VA@2x.png)

----

* Réseau de neurones en 2 parties
    * 1. Encodeur projette l'image dans une plus faible dimension
    * 2. Décodeur reconstruit l'image
* On entraîne le réseau en comparant l'image d'entrée et celle de sortie

<small> **source: https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73** </small>

----

### Auto-encodeur variationnel discret

* VAE = régulariser l'entraînement de l'encodeur pour assurer certaines propriétés de l'espace latent.
* **Objectif** : avoir la continuité et la complétude.

----

* Dans notre cas (dVAE) :  
    * Chaque entrée est encodée comme un ensemble de distributions sur les vecteurs du "codebook" (8192 vecteurs ici).
    * Puis, on applique une relaxation (la *Gumbel Softmax relaxation*) pour approximer de manière continue notre modèle discret et ainsi permettre la backpropagation.

---

## 4. Modèles de langue : le cas GPT-2

**source : https://jalammar.github.io/illustrated-gpt2/**

![](https://i.imgur.com/mD1LX47.png)

----

* GPT-2 prend en entrée une suite de mots 
* Génère le mot suivant
* Ajoute le mot à la liste des inputs pour générer la prochaine suggestion (auto-régression)
* Decoder-Only

----


### Fonctionnement du modèle : Gestion des inputs 


![](https://i.imgur.com/ZxiWIvL.png)

----

### Vectorisation des tokens ou mots

* Chaque mot ou token est vectorisé selon l'embedding matrix correspondante (matrice de vocabulaire)
* La vectorisation rend compte de la signification du mot
* Plus l'embedding size est grande plus la signification est conservée
* On réalise aussi un encodage du positionnement du mot dans la séquence

----

### Fonctionnement du modèle : Gestion de la dépendance et de l'attention


![](https://i.imgur.com/DM9W6gi.png)


----

Dépendance entre les mots d'une même séquence : 
* Prendre en compte le contexte de la phrése est d'une importance majeure (rôle d'un pronom)
* Avec GPT-2 L'interprétation de chaque token ne change pas avec l'ajout d'un token suivant
* L'interprétation des tokens précédents est utilisée pour déterminer le/les tokens suivants 
* Trio Requête-Clé-Valeur : pour chaque nouvelle requête, on attribue un score à toutes les clés existantes
* On peut réaliser une somme pondétée par les scores de chaque token pour déterminer un score de self-attention.

----

### Fonctionnement du modèle : Rendu des outputs


![](https://i.imgur.com/AIhqm9v.png)


----

Le vecteur à la sortie du bloc transformeur décodeur 
* Est multiplié à la matrice d'embedding 
* Cela donne la pertinence de chaque token de la table de vocabulaire (en probabilité)
* On peut choisir le token avec le meilleure score 
* ... Ou choisir un token parmi les n meilleurs candidats. 

---


## 5. Dall-E 


DALL-E est un modèle entrainé par OpenAI qui :
* prend en entrée un texte (et une image)
* génère en sortie des images correspondant aux entrées


----

DALL-E permet générer de nouvelles images selon une description et éventuellement d'une image initiale :

* en modifiant des caractéristiques d'une image existante (attribut d'un objet ; vue ou perspective)
* en composant une nouvelle image
* tout en comprenant le contexte et des connaissances ou référence du monde (géographique ; temporelle)
* "auto-apprendre de nouveaux concepts"

----


DALL-E est un modèle de transformation du langage (texte et image) :
* peut prendre en entrée 1280 tokens
* utilise GPT-3
* génère 512 images (CLIP filtre et donne les 32 meilleurs)


----

DALL-E est un modèle de deep learning mélant :
* NLP avec GPT-3
* Computer Vision en utilisant et en générant des images


----

* DALL-E peut être décomposée en 2 sous-parties:
    * Un dVAE compressant les images
    * Un transformeur auto-régressif 

----

### 5.1 Partie encodage d'image (dVAE)

* Compresse les images de 256\*256 pixels à 32\*32 tokens
* Réseaux de neurones convolutionnels
* Paramétrisation fine de l'encodeur et du décodeur

----

### 5.2 Partie génération de séquences (transformer)

* Le transformeur de Dall-E est un décodeur uniquement, à la manière de GPT-3.
* À partir de la concaténation des tokens de texte (vectorisés par plongement^[1:Il n'est pas clair dans à ce stade d'où vient le plongement utilisé. Il n'a pas l'air d'être appris par Dall-E.]) et des 1024 tokens d'image...
* ... le transformer prédit le prochain latent visuel en se basasnt sur les tokens de text et les latents visuels précédents

----

### 5.3 Évaluation et re-classement

* La qualité des images est jugée par des humains
* Les images générées par le tranformer sont réévaluées par CLIP, un modèle qui juge la pertinence des association texte-image
* Augmenter le nombre de résultats générés par Dall-E avant le re-classement par CLIP augmente grandement la qualité des images générées (telle qu'évaluée par des humains)

----

![](https://i.imgur.com/Ka3I0Ju.png)

----

### 5.3 Données et entraînement

* Dall-E possède 12 milliards de paramètres mais est largement régularisé
* Dall-E est entraîné sur ensemble de 250 millions d'images + +titre, récupérées sur Internet
* De nombreuses procédures d'optimisation numériques sont décrite dans l'article (utilisation d'une précision 16 bits, distribution des calculs, calcul sur carte graphique notammen)
