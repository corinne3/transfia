# TransfIA — Rapport des tests et manuel utilisateur

> **Ce document** : tout ce qui a été testé sur TransfIA, ce qui a échoué, pourquoi, ce qui a été corrigé et ce qu'on en retient ;
> puis la comparaison chiffrée des modèles ; puis le **manuel utilisateur** de l'interface web.
> **Source** : `docs/JOURNAL_TESTS.md` (chaque ligne y a un numéro, ex. **[4.4]**, repris ici) et les bancs d'essai `data/bench/`.
> **Auteure** : Corinne Bergès (conçu et mesuré par l'auteure, codé avec l'aide d'un assistant IA). **État** : 31 octobre 2026.

---

## Sommaire

- **Partie A — Rapport des tests**
  1. [Résumé en une page](#1-résumé-en-une-page)
  2. [Cadre et méthode](#2-cadre-et-méthode)
  3. [Résultats brique par brique](#3-résultats-brique-par-brique)
  4. [Banc d'essai : petit modèle vs gros modèles](#4-banc-dessai--petit-modèle-vs-gros-modèles)
  5. [Les 12 leçons transverses](#5-les-12-leçons-transverses)
  6. [Limites connues et pistes](#6-limites-connues-et-pistes)
- **Partie B — Manuel utilisateur de l'interface**
  7. [Démarrer l'interface](#7-démarrer-linterface)
  8. [Les 7 onglets, un par un](#8-les-7-onglets-un-par-un)
  9. [Lire les signaux de confiance](#9-lire-les-signaux-de-confiance)
  10. [Problèmes fréquents](#10-problèmes-fréquents)
- [Annexe : rejouer les tests](#annexe--rejouer-les-tests)

---

# Partie A — Rapport des tests

## 1. Résumé en une page

**Ce qu'est TransfIA** : un assistant qui transforme des articles scientifiques en élevage (INRAE *Productions Animales*, licence CC BY) en réponses sourcées et en contenus pédagogiques (QCM, fiches, glossaires, cas pratiques). 100 % open source, hors ligne, sur un PC Windows **sans GPU, 8 Go de RAM**.

**Les chiffres clés**

| Mesure | Résultat | Où |
|---|---|---|
| Questions hors corpus inventées (hallucinations) | **0** (petit et 3B) | banc [B.1], [B.3] |
| Bonnes questions refusées à tort | **0** | idem |
| Fidélité des réponses (phrases soutenues par leurs sources) | **1,00** (1,5B) ; 0,94 (3B) | idem |
| Recherche : bonne source dans les 5 premiers (hybride) | 0,97 | [7.7] |
| Latence d'une réponse (médiane, CPU) | 8 s (RAM libre) à 20 s (RAM saturée) ; 45 s avec le 3B | [9.15], [B.2], [B.3] |
| Énergie du banc complet | 12,8 Wh (1,5B) ; 14,9 Wh (3B) | [B.1], [B.3] |
| Latence avec Gemini 3.1 Flash-Lite (même code, sans carte) | 8,6 s (p50) contre 23,8 s en local sur les mêmes questions ; 0 hallucination hors corpus | [G.2], [G.3] |
| Fine-tuning QLoRA (QCM conformes) | 0/4 → 3/4, avec 1,18 % des paramètres entraînés | [L.5], [L.8] |
| Tests automatiques | 146 réussis | `pytest` |

**Les 5 conclusions à retenir**

1. **Un petit modèle (1,5 Md de paramètres) suffit pour un RAG fiable**, à condition de l'entourer de garde-fous mesurés : refus hors corpus, vérification de chaque phrase, blocage des réponses non soutenues.
2. **« Valide » n'est pas « juste »** : un JSON parfait peut contenir une mauvaise réponse de QCM ; un glossaire bien formé peut définir de mémoire. Il faut des contrôles de fond, écrits en code.
3. **Un modèle plus gros (3B) réussit mieux les tâches de capacité** (agent 0/3 → 3/3 autonome, QCM 3/9 → 7/9) **mais n'est pas plus fidèle** (0,94 vs 1,00) et il est 2,2 fois plus lent. Les garde-fous restent nécessaires.
4. **La contrainte matérielle décide du modèle** : sur 8 Go de RAM, le 4B est inutilisable (un appel > 15 min). Le 3B est le plafond de cette machine.
5. **Le fine-tuning apprend un comportement (le format), pas des faits** ; il crée un « contrat de prompt » à respecter.

---

## 2. Cadre et méthode

### 2.1 La machine et les données

- **PC** : Windows 11, processeur seul, **8,4 Go de RAM** (souvent moins de 1 Go libre avec un navigateur ouvert).
- **Corpus** : 10 articles INRAE *Productions Animales* 2026 ; 796 passages au départ, **694 après nettoyage** (bibliographies retirées [8.4]).
- **Modèles locaux (Ollama)** :
  - **qwen2.5:1.5b** (Apache 2.0) : le modèle de référence ;
  - **qwen2.5:3b** (*Qwen Research License*, usage non commercial) : même famille, pour comparer la **taille** seule ;
  - **qwen3:4b** (Apache 2.0) : la version « gros modèle open source » et le professeur du fine-tuning ;
  - embeddings **multilingual-e5-small**, reranker **mMiniLM** (cross-encoder).

### 2.2 Comment on a testé

- **Une brique à la fois**, dans l'ordre du projet (installation → recherche → RAG → génération → agents → évaluation → ML → interface → bonus).
- **Chaque test est consigné** dans le journal : *ce qu'on a lancé → ce qu'on a vu → la cause → la correction → la leçon*.
- **Après chaque correction**, on rejoue : une correction peut en casser une autre (c'est arrivé : [7.9], [9.3]).
- **Tests automatiques** (`pytest`, 146) : chaque défaut corrigé reçoit un test, pour qu'il ne revienne pas (**non-régression**).
- **Banc d'essai** (`transfia bench`) : les mêmes scénarios rejoués à l'identique pour comparer des modèles ; cache désactivé (vraies durées) ; conditions de la machine notées.
- **Versions** : chaque livraison s'appelle « majN » ; le numéro est affiché dans l'interface et par `transfia doctor` [9.18].

### 2.3 Comment lire les mesures

- **Fidélité** : part des phrases de la réponse qui sont soutenues par le passage qu'elles citent.
- **Refus corrects** : sur les questions hors corpus, le système doit refuser ; sur les autres, il doit répondre.
- **p50 / p95** : la médiane et le « presque pire cas » (95 % des réponses sont plus rapides). Le p95 compte pour l'utilisateur.
- **Petits jeux** : 10 à 18 questions → une différence de quelques points n'est **pas** significative ; on lit des tendances [7.3].

---

## 3. Résultats brique par brique

### 3.1 Installation et ingestion [1.1 à 1.3]

- **Ce qu'on a testé** : `transfia doctor`, les tests, la collecte et le découpage des PDF.
- **Ce qui a échoué** :
  - le streaming cassait le suivi des traces [1.1] ;
  - le code contactait Internet (Hugging Face) alors que le projet devait être hors ligne [1.2].
- **Corrections** : téléchargement des modèles une fois pour toutes dans `models/`, puis mode hors ligne.
- **Leçon** : « 100 % open source » ne veut pas dire « 100 % hors ligne » : ça s'organise.

### 3.2 Recherche et RAG (F4, F5) [3.1 à 4.6, 6.8, 6.9, 7.9, 9.10]

- **Ce qu'on a testé** : 3 modes de recherche (mots-clés BM25, sens par embeddings, hybride + reranker), puis la réponse sourcée.
- **Ce qui a échoué, et pourquoi** :
  - **7,5 min par question** : trop de modèles chargés en même temps pour 8 Go → le PC écrivait sur le disque (*swap*) [4.1] → routage vers le gros modèle rendu **optionnel**.
  - **Aucune citation** : le 1,5B oubliait les [n] [4.2] → un exemple dans le prompt (*few-shot*) + **attribution des citations par le code** après coup.
  - **Réponse inventée affichée** (grippe aviaire) [4.4] → **blocage** des réponses dont aucune phrase n'est soutenue.
  - **171 s par réponse** [4.6] : reformulation inutile, contexte trop long, reranker sur 20 passages → **50 s**, reranker de 51 s à 1,1 s.
  - **Refus à tort** pour « Que dit le corpus sur le colostrum ? » [6.8] : la tournure de conversation polluait la recherche → **nettoyage des tournures** avant la recherche.
  - **Le score du reranker n'est pas une probabilité** [6.9] (« le colostrum » −3,6 ; « colostrum » −2,6) → refus à **double critère** : score bas **et** mots de la question absents des passages.
  - **2 hallucinations hors corpus** révélées par l'évaluation (grippe aviaire, autruche) [7.9] → **ancrage sur le sujet** : mots de la question inconnus du corpus → refus.
  - **Mammites** [9.10] : le mot existe dans le corpus, mais dans aucun passage retenu → réponse sur la viande « validée » → **ancrage sur les termes rares**.
- **Résultat final** : 0 hallucination hors corpus, 0 refus à tort, fidélité 1,0 [7.10], [9.15].
- **Leçons** :
  - un garde-fou **de sortie** est indispensable, pas seulement d'entrée ;
  - une vérification par « mots en commun » ne vérifie pas le **sujet** ;
  - **mesurer avant de corriger** (la cause du refus à tort était mesurable : le score).

### 3.3 Génération de contenus pédagogiques (F7) [5.1 à 5.4, 9.5, 9.12]

- **Ce qu'on a testé** : QCM, glossaires, fiches, cas pratiques, en JSON validé par un schéma.
- **Ce qui a échoué, et pourquoi** :
  - **QCM au JSON parfait mais aux réponses fausses** (2 sur 3) [5.1] : le modèle choisissait la lettre **avant** d'écrire l'explication → explication placée avant la lettre dans le schéma + **contrôle de cohérence** (l'explication désigne-t-elle bien le bon choix ?).
  - **Questions ambiguës ou en double** [5.2] → détection + **boucle de réparation** (le code critique, le modèle corrige, on ne garde que si c'est mieux).
  - **Réparations ratées** [5.3] → les questions fausses sont **retirées** (un QCM faux enseigne l'erreur).
  - **Glossaire défini de mémoire** [9.5] : 15 termes « cités », dont plusieurs absents du passage → chaque terme doit être **dans** sa source, sinon avertissement.
- **Leçon** : **valide ≠ juste**. Le schéma garantit la forme ; seul un contrôle de fond garantit le contenu.

### 3.4 Agents et MCP (F8) [6.1 à 6.16, 9.6, 9.7, 9.13, 9.14, 9.17, 9.19]

- **Ce qu'on a testé** : un agent qui choisit lui-même ses outils (recherche, génération…), en mode natif (*tool calling*) et en mode texte (ReAct) ; le même agent via le protocole **MCP** ; une mémoire long terme ; un mode « planifier puis exécuter ».
- **Ce qui a échoué, et pourquoi** :
  - **L'agent répondait de mémoire**, faux (« bactéries de l'estomac » au lieu d'archées du rumen) [6.1], malgré la consigne → **garde-fou d'ancrage**, puis **plan B** : le code lance lui-même la recherche [6.2], dans un **contexte neuf** pour ne pas recopier l'erreur [6.3].
  - **Mode texte** [6.5], [9.7], [9.14] : mauvais outil, arguments faux, 8 fois le même appel, « observation » inventée par le modèle → messages d'erreur lisibles, **réparation des arguments**, **détection de boucle**, tout ce que le modèle écrit après « Observation : » est coupé.
  - **MCP** [6.7] : outils trop fins → outil de haut niveau `ask_corpus`.
  - **L'agent réécrivait la fiche** déjà vérifiée [9.13] → `return_direct` : la sortie de l'outil est rendue telle quelle [9.19].
  - **Planificateur** [6.13 à 6.16] : livrable creux (« [Rédiger ici] ») → vérification de la couverture **avant** de planifier, preuves communes, contrôle qualité, sources écrites par le code.
- **Leçons** :
  - **une consigne n'est pas un contrôle** ;
  - **seul le code a le droit d'écrire une observation** ;
  - avec un petit modèle, **un workflow guidé vaut mieux qu'un agent autonome** [6.16] (confirmé par le banc : 0/3 autonome en 1,5B, 3/3 en 3B).

### 3.5 Évaluation et sobriété (F11, F14) [7.1 à 7.10]

- **Ce qu'on a testé** : jeux de questions générés (pseudo-questions et questions reformulées par LLM), 8 questions **hors corpus**, métriques de recherche et de réponse, latence, énergie, dimensionnement mémoire.
- **Ce qui a trompé, et pourquoi** :
  - **0,0 partout** [7.1] : le jeu de test visait d'anciens articles → contrôle de compatibilité du jeu avec l'index.
  - **BM25 « gagnant »** [7.2] : les pseudo-questions recopient le texte → biais lexical ; **le jeu de test choisit le gagnant**.
  - **Refus à 1,0 trompeur** [7.4] : aucune question hors corpus → ajout de 8 questions hors corpus [7.8].
  - **Moyennes mensongères** [7.8] : on mélangeait réponses et refus (quasi gratuits) → matrice de confusion du refus, coûts calculés sur les seules réponses.
  - **Latence de 0,8 s** [7.10] : réponses lues dans le cache → cache désactivé pendant l'évaluation.
- **Leçons** :
  - **valider l'instrument avant de mesurer** ;
  - **une moyenne sur deux populations ment** ;
  - **un cache fausse les mesures de performance**.

### 3.6 ML classique, clustering, graphe (F3, F6) [8.1 à 8.9]

- **Classification par thème** [8.1, 8.2] : F1 de 0,91 grâce à une **fuite** (des passages d'un même article en entraînement ET en test ; mots appris : « aubrac, 1994 »). Validation croisée **groupée par article** : 0,99 → 0,85 pour TF-IDF. **Une fuite peut faire choisir le mauvais modèle.**
- **Clustering** [8.4, 8.5] : un groupe = bibliographies, un autre = « br » (balises HTML) → il a servi d'**audit des données** ; après nettoyage, 8 vrais sujets.
- **Graphe de connaissances** [8.6 à 8.9] : 60 entités, 485 liens ; la recherche locale renvoyait tout le graphe ; la synthèse globale répondait « rien » → corrigé grâce à la **trace** de l'étape *map*.
- **Leçon** : le non supervisé est un excellent outil de contrôle qualité des données.

### 3.7 Interface web (F13) [9.1 à 9.19]

- **Ce qu'on a testé** : les 7 onglets, sur de vraies questions.
- **Ce qui a échoué, et pourquoi** :
  - **Réponses justes bloquées** [9.3] : le chemin « streaming » de l'interface n'avait pas l'attribution des citations → **régression** introduite par une correction (maj34) ; d'où la factorisation des contrôles [9.2].
  - **Colonnes de recherche identiques** [9.4] ; **graphe coupé** [9.8] ; **tableaux d'évaluation illisibles** [9.9] (impossible de savoir quel tableau correspond à quelle manipulation) → date, jeu, note libre (`--note`).
  - **Ancien code exécuté** [9.18] : le serveur tournait sur une version périmée sans qu'on puisse le voir → **numéro de version affiché**.
- **Leçons** :
  - deux chemins de code dupliqués finissent par diverger ;
  - **rendre visible la version déployée** (comme un tag d'image en production) ;
  - une mesure sans contexte est inexploitable.

### 3.8 Bonus F9 : atelier multi-agents [F9.1 à F9.6]

- **Ce qu'on a testé** : rédacteur → relecteur → vulgarisateur, en machine à états.
- **Ce qui a échoué, et pourquoi** :
  - **3 tours « à vide »** [F9.1] : la boucle suivait le drapeau « valide » du LLM → routage sur les **problèmes réellement ouverts**.
  - **Relecteur complaisant** [F9.2], puis **trop sévère** [F9.6] : un juge de 1,5 Md se trompe dans les deux sens → relecteur **automatique** (règles) en plus du LLM.
  - **La dernière étape déformait** (« les vaches Charolaise ont des colostrums faibles ») [F9.3] → **contrôle final** (chiffres perdus, phrases sans appui) et **élagage** des phrases toujours non soutenues.
- **Leçon** : toute étape qui modifie le texte doit être contrôlée, surtout la dernière.

### 3.9 Bonus F10 : fine-tuning [L.1 à L.11]

- **Ce qu'on a fait** :
  - 40 QCM produits par qwen3:4b (« professeur »), filtrés : 6 rejetés comme ambigus ;
  - entraînement **QLoRA** sur Colab (GPU gratuit) : 1,18 % des paramètres, 1 min 05 ;
  - export **GGUF** 0,99 Go, utilisé localement dans Ollama (`transfia-qcm`).
- **Résultats** :
  - QCM conformes **0/4 → 3/4** [L.8] ;
  - mais une bonne réponse incohérente avec son explication [L.8] ;
  - **1er essai local pire que le modèle de base** [L.10] : le modèle était appelé avec un autre prompt que celui de l'entraînement (**décalage de distribution**) → **contrat de prompt** ;
  - après correction : sur le sujet et cohérent, mais pédagogie faible [L.11].
- **Leçons** :
  - **le fine-tuning apprend un comportement, pas la vérité** ;
  - un fine-tuning crée un **contrat d'interface** ;
  - **vérifier la licence de chaque taille de modèle** (le 3B n'a pas la même que le 1,5B) [L.1].

---

## 4. Banc d'essai : petit modèle vs gros modèles

### 4.1 Le protocole

- **Même commande, même jeu, mêmes tâches** : `transfia bench`, 5 scénarios.
  - **RAG** : 17 questions, dont des questions hors corpus.
  - **Quiz** : 3 thèmes × 3 questions (méthane entérique, colostrum du veau, fourrages des chèvres).
  - **Glossaire** : colostrum.
  - **Agent** : 3 tâches.
  - **Multi-agents** : colostrum, grand public.
- **Conditions** : cache coupé, PC sur secteur ; depuis maj47, la RAM libre, la version d'Ollama et le temps de chargement du modèle sont notés.

### 4.2 Les résultats

| Mesure | **1,5B** (qwen2.5:1.5b) | **3B** (qwen2.5:3b) | **4B** (qwen3:4b) |
|---|---|---|---|
| RAG : fidélité | **1,00** | 0,94 | ✘ (délai dépassé) |
| RAG : refus corrects | 1,00 | 1,00 | ✘ |
| RAG : hallucinations hors corpus / refus à tort | 0 / 0 | 0 / 0 | ✘ |
| RAG : latence p50 / p95 | 20 s / 41 s | 45 s / 98 s | > 15 min par appel |
| Quiz : questions gardées | 3 / 9 (1 thème en erreur) | **7 / 9** | non mesuré |
| Quiz : défauts détectés | 2 doublons, 1 réparation ratée | 2 ambiguës, 1 lettre corrigée, 1 réparation ratée | — |
| Glossaire : termes absents de leur source | 1 | 1 | ✘ |
| Agent : tâches menées seul | 0 / 3 (plan B ×3) | **3 / 3** | non mesuré |
| Multi-agents | 3 tours, élagage, 1 alerte | ✘ délai de 300 s dépassé | ✘ |
| Durée / énergie du banc | 31 min / 12,8 Wh | 36 min / 14,9 Wh | arrêté après 2 h 30 |

### 4.3 Ce qu'on en conclut

- **Plus gros = meilleur sur les tâches de capacité** : utiliser des outils seul, produire des QCM corrects.
- **Plus gros ≠ plus fidèle** : 0,94 contre 1,00. La fidélité vient surtout des **garde-fous** et du contexte, pas de la taille.
- **Les garde-fous restent nécessaires** avec le 3B : ils servent moins souvent (moins de questions retirées, plus aucun plan B), mais ils servent (QCM ambigus, lettre fausse, terme inventé).
- **Le coût** : le 3B est 2,2 fois plus lent par réponse ; +16 % d'énergie sur le banc.
- **Le 4B ne tient pas sur la machine** : 0,8 Go de RAM libre au départ ; le modèle (2,5 Go) plus les embeddings débordent sur le disque ; un seul appel dépasse 15 min. La réflexion de qwen3 était bien coupée (vérifié par le banc), donc ce n'est pas la cause.
- **Décision d'architecte** :
  - **1,5B** par défaut (rapide, fidèle) ;
  - **3B** à la demande pour la génération de contenus et l'agent (`generation_model`, routage) ;
  - au-delà, il faut **16 Go de RAM**, un GPU, ou **le cloud** (même code avec Gemini : `config/gcp.yaml`, chapitre 13 du dossier).
- **En une phrase** : « J'ai mesuré la taille du modèle contre la qualité, la latence et l'énergie, avec le même banc. La contrainte matérielle a tranché. »

### 4.5 Le même code dans le cloud : Gemini (sans carte bancaire)

- **Ce qu'on a fait** : `config/gcp.yaml`, clé d'API Google AI Studio (niveau gratuit, sans compte de facturation), modèle **gemini-3.1-flash-lite** via LiteLLM. Seul le LLM change : index Qdrant, embeddings e5, reranker et garde-fous restent locaux.
- **Hors corpus** : « Quelle est la capitale de l'Australie ? » → **refusée sans appeler Gemini** (aucun passage ne contient « capitale ») : le garde-fou d'entrée économise aussi des appels payants.
- **Banc RAG sur 10 questions** (5 du corpus, 5 hors corpus, pour rester dans le quota gratuit) :

| Mesure | 1,5B local | 3B local | **Gemini 3.1 Flash-Lite** |
|---|---|---|---|
| Questions | 17 | 17 | 10 ⚠️ échantillon |
| Fidélité | 1,00 | 0,94 | 0,96 |
| Refus corrects / hallucinations / refus à tort | 1,00 / 0 / 0 | 1,00 / 0 / 0 | 1,00 / 0 / 0 |
| Latence p50 / p95 | 20 s / 41 s | 45 s / 98 s | **8,6 s / 28 s** |
| Énergie mesurée sur le PC | 12,8 Wh (banc complet) | 14,9 Wh | 0,63 Wh (RAG seul) |

- **Lecture** :
  - **les garde-fous tiennent avec un 3ᵉ modèle** : 0 hallucination, 0 refus à tort ;
  - **Gemini est 2,8 fois plus rapide** que le 1,5B local à échantillon égal (calcul sur les serveurs de Google ; restent sur le PC la recherche, le reranker et la vérification) ;
  - **fidélité un peu plus basse** : Gemini **reformule** davantage (« déterminante pour sa survie ») et le vérificateur compare des **mots**, pas le sens → des phrases justes sont jugées « non soutenues ». Changer de modèle change aussi ce que mesurent les garde-fous : **il faut ré-évaluer à chaque changement de modèle** ;
  - ⚠️ **l'énergie de 0,63 Wh est celle du PC seulement** : la consommation des serveurs de Google n'est pas mesurée. Comparer « local vs cloud » sur l'énergie demanderait les données du fournisseur ;
  - **Comparaison stricte, mêmes 10 questions** [G.3] :

| Mesure (10 questions identiques) | 1,5B local | Gemini 3.1 Flash-Lite |
|---|---|---|
| Fidélité | **1,00** | 0,96 |
| Refus corrects / hallucinations / refus à tort | 1,00 / 0 / 0 | 1,00 / 0 / 0 |
| Latence p50 / p95 | 23,8 s / 47,3 s | **8,6 s / 28,4 s** |
| Durée / énergie mesurée sur le PC | 2,6 min / 1,08 Wh | 1,5 min / 0,63 Wh (+ serveurs Google, non mesurés) |

  - ⚠️ 10 questions : tendance, pas preuve statistique.
- **Coût** : 0 € (niveau gratuit, quota journalier par projet). Contrepartie : conditions d'usage (voir le chapitre 13 du dossier fonctionnel) et pas de garantie de service.
- **En une phrase** : « Grâce à l'architecture hexagonale (LiteLLM + configuration), le passage du local à Gemini s'est fait sans changer le code ; le même banc a montré les mêmes garanties de refus, une latence divisée par 2,8 à échantillon égal, et une limite de mon vérificateur lexical face à un modèle qui reformule. »

### 4.4 Ce que le banc lui-même nous a appris [B.2, B.4, B.5]

- **RAM saturée = latence doublée** (20 s au lieu de 8 s) : une latence se mesure dans des conditions fixées.
- **Un tableau ne doit jamais afficher une panne comme un résultat** : le 1ᵉʳ tableau du 4B montrait « 0/9 » au lieu de « en erreur » → corrigé.
- **Un banc long doit sauvegarder au fil de l'eau** : interrompu, le banc 4B a perdu ses mesures → depuis maj48, le fichier est réécrit après chaque scénario [B.6].

---

## 5. Les 12 leçons transverses

| # | Leçon | Exemples |
|---|---|---|
| 1 | **Valide ≠ juste** : un schéma garantit la forme, pas le fond | QCM [5.1], glossaire [9.5], fine-tuning [L.8] |
| 2 | **Une consigne n'est pas un contrôle** : ce qui compte est vérifié par le code | agent [6.1], relecteur [F9.1] |
| 3 | **Garde-fou de sortie**, pas seulement d'entrée | blocage [4.4], [9.16] |
| 4 | **Mesurer avant de corriger** | score du reranker [6.8], trace *map* [8.8] |
| 5 | **Valider l'instrument de mesure** | jeu incompatible [7.1], biais lexical [7.2] |
| 6 | **Une moyenne sur deux populations ment** | refus vs réponses [7.8] |
| 7 | **Un cache fausse les performances** | [5.4], [7.10] |
| 8 | **Une correction peut en casser une autre** → rejouer tout | [7.9], [9.3] |
| 9 | **Deux chemins dupliqués divergent** → factoriser | streaming [9.2] |
| 10 | **Rendre visible la version qui tourne** | [9.18] |
| 11 | **Petit modèle : workflow guidé > agent autonome** ; plus gros modèle = plus d'autonomie, pas plus de fidélité | [6.16], banc [B.3] |
| 12 | **La contrainte matérielle et la licence décident du modèle** | swap [4.1], 4B [B.4], licence 3B [L.1] |

---

## 6. Limites connues et pistes

- **Jeux d'évaluation petits** (17 questions) : tendances, pas de preuve statistique → constituer un **jeu doré relu par un humain**.
- **Seuil de refus réglé à la main** → le **calibrer** sur un jeu annoté (régression logistique de Platt).
- **Corpus de 10 articles** ; 2 thèmes n'ont qu'un article → classification peu évaluable.
- **Juges LLM peu fiables** (1,5B) → mesurer leur accord avec un humain (`eval agreement`).
- **Reranker** : + 0,9 s pour un gain non démontré sur peu de questions → à trancher sur un jeu plus grand.
- **Pédagogie des QCM** : le fine-tuning n'a pas appris à faire de bons distracteurs → plus d'exemples relus par un enseignant.
- **Cloud** : le même code tourne avec Gemini (sans carte) [G.1, G.2] ; le déploiement Cloud Run / BigQuery est conçu mais non réalisé.
- **Vérificateur lexical** : pénalise les modèles qui reformulent (Gemini) → passer à une vérification **sémantique** (embeddings ou modèle NLI d'implication).

---

# Partie B — Manuel utilisateur de l'interface

> **À qui s'adresse cette partie** : un enseignant ou un formateur en élevage, sans compétence technique.
> **Ce que fait l'interface** : poser des questions au corpus scientifique et obtenir des réponses **sourcées et vérifiées** ; produire des contenus pédagogiques ; explorer le corpus.

## 7. Démarrer l'interface

1. **Ouvrir Ollama** (l'application qui fait tourner le modèle de langue sur le PC).
2. **Lancer l'API**, dans un premier terminal PowerShell :
   ```powershell
   cd C:\dev\transfia
   .venv\Scripts\activate
   transfia serve
   ```
3. **Lancer l'interface**, dans un second terminal :
   ```powershell
   cd C:\dev\transfia\frontend
   npm run dev
   ```
4. **Ouvrir** <http://localhost:5173> dans le navigateur.
5. **Vérifier l'en-tête** (en haut à droite) : un point vert, puis `modèle · nombre de passages · config · version`, ex. `qwen2.5:1.5b · 694 passages · config local · maj47`.
   - Point rouge « API injoignable » → l'étape 2 n'est pas lancée ou s'est arrêtée.
   - Version différente de la dernière livraison → l'ancien serveur tourne encore : l'arrêter (Ctrl+C) et le relancer.

**Conseils** :
- Fermer les applications lourdes (navigateur avec beaucoup d'onglets, Teams) : avec 8 Go de RAM, cela peut **diviser par deux** les temps de réponse.
- Le **1er appel** est plus lent (chargement du modèle en mémoire).
- Pendant que l'interface tourne, ne pas lancer de commande `transfia` qui touche à l'index dans un autre terminal (la base Qdrant n'accepte qu'un programme à la fois).

## 8. Les 7 onglets, un par un

### 8.1 Questions sourcées (onglet par défaut)

- **À quoi ça sert** : poser une question en français sur le corpus et obtenir une réponse **citée** ([1], [2]…).
- **Comment faire** :
  1. Écrire la question dans « Pose une question sur le corpus… ».
  2. *(Facultatif)* Choisir une **espèce** dans la liste (« Toutes espèces » par défaut) pour ne chercher que dans les articles de cette espèce.
  3. Valider. La réponse s'écrit au fil de l'eau ; les étapes (recherche, vérification) s'affichent au-dessus.
- **Ce qu'on voit** :
  - la **réponse**, avec ses numéros de citation ;
  - « **Vérification : x/y phrases soutenues par leurs sources** » ; les phrases non soutenues sont listées en rouge ;
  - les **sources** numérotées, avec titre, lien (DOI) et l'**extrait** exact utilisé ;
  - 👍 / 👎 : donner son avis (il est enregistré et sert à améliorer l'évaluation).
- **Cas normal de refus** : si la question sort du corpus (« Quelle est la capitale de l'Australie ? ») ou si aucun passage ne traite vraiment du sujet, l'assistant **refuse** au lieu d'inventer. C'est voulu.
- **Bonnes questions** : précises et autonomes, ex. « Pourquoi le colostrum est-il indispensable au veau ? », « Quels leviers réduisent le méthane entérique ? ». On peut enchaîner : « Et chez les chèvres ? » (l'historique est pris en compte).

### 8.2 Recherche

- **À quoi ça sert** : voir **quels passages** le système trouve, sans génération de texte. Utile pour préparer un cours ou comprendre une réponse.
- **Comment faire** : taper des mots-clés (ex. « émissions de méthane, sevrage des porcelets »).
- **Ce qu'on voit** : 3 colonnes côte à côte, une par méthode :
  - **BM25 (lexical)** : les passages qui contiennent les mêmes mots ;
  - **Dense (embeddings)** : les passages qui ont le même **sens**, même avec d'autres mots ;
  - **Hybride + rerank** : la combinaison des deux, reclassée par un modèle plus fin. C'est celle qu'utilise l'onglet Questions.
- **Lecture** : si les colonnes diffèrent, c'est normal : chaque méthode a ses forces. Le score affiché sert à comparer les passages **d'une même colonne**.

### 8.3 Générateur

- **À quoi ça sert** : produire un **contenu pédagogique** à partir du corpus.
- **Comment faire** :
  1. Choisir le **type** : QCM, Fiche de synthèse, Glossaire, Cas pratique.
  2. Choisir le **niveau** : Bac pro, BTS, Master, Grand public.
  3. Écrire le **thème** (ex. « le colostrum chez le veau »).
  4. Cliquer **Générer** (compter 1 à 5 minutes sur CPU).
- **Ce qu'on voit** :
  - le contenu mis en forme, avec ses sources regroupées par article ;
  - des **avertissements ⚠️** si un contrôle a trouvé un défaut : lettre de QCM corrigée, question ambiguë ou retirée, terme absent de sa source, élément peu soutenu ;
  - **Copier le Markdown** : pour coller le contenu dans un document de cours.
- **À savoir** : moins de questions que demandé = des questions **fausses ont été retirées** (un QCM faux enseigne l'erreur). **Relire avant usage** reste indispensable : les avertissements aident, ils ne remplacent pas l'enseignant.

### 8.4 Assistant

- **À quoi ça sert** : formuler une demande libre ; l'assistant **choisit lui-même ses outils** (recherche, génération, glossaire…).
- **Comment faire** :
  1. Choisir le mode : **Tool calling natif** (recommandé) ou **ReAct textuel** (démonstration pédagogique du raisonnement « pensée → action → observation »).
  2. Écrire la demande, ex. « Prépare 3 questions de QCM niveau bac pro sur le sevrage des porcelets ».
- **Ce qu'on voit** : une **frise** des étapes : chaque appel d'outil, ce qu'il a renvoyé (« Observation »), puis le résultat final ✅.
- **À savoir** : avec le petit modèle, l'assistant passe souvent par le **plan B** (le système lance lui-même la recherche) ; c'est un filet de sécurité, pas une panne. Un contenu produit par un outil est rendu **tel quel**, sans être réécrit.

### 8.5 Graphe

- **À quoi ça sert** : voir les **notions** du corpus (espèces, gaz, pratiques…) et leurs liens.
- **Comment faire** : cliquer sur un nœud pour **mettre en évidence ses voisins** ; recliquer pour tout réafficher.
- **Lecture** : la taille d'un nœud indique son importance ; les couleurs, les **communautés** (groupes de notions souvent citées ensemble).

### 8.6 Carte des thèmes

- **À quoi ça sert** : une carte de tous les passages, regroupés automatiquement par sujet.
- **Ce qu'on voit** : des points colorés (un point = un passage), la liste des thèmes avec leurs mots caractéristiques, et le **score silhouette** par nombre de groupes (*k*).
- **Lecture** : un score silhouette faible (≈ 0,1) veut dire que les sujets **se chevauchent** : c'est normal pour un corpus où l'alimentation, l'environnement et la santé se croisent.

### 8.7 Évaluation

- **À quoi ça sert** : consulter les **rapports de mesure** de la qualité (produits en ligne de commande par `transfia eval`).
- **Ce qu'on voit** : un rapport par carte, avec sa **date**, son type (recherche ou RAG), le **jeu de questions**, la configuration et la **note** saisie au lancement (`--note "…"`).
  - Rapport **recherche** : Hit@5, MRR, nDCG@5, latence, par mode.
  - Rapport **RAG** : fidélité, bonne source citée, refus corrects, latence p50/p95, énergie par question.
  - ⚠️ « cache LLM actif » : les latences de ce rapport ne sont pas représentatives.
- **Astuce** : toujours ajouter une note au lancement (`transfia eval rag --note "après ajout de 5 articles"`) pour savoir à quoi correspond chaque tableau.

## 9. Lire les signaux de confiance

| Signal | Où | Signification | Que faire |
|---|---|---|---|
| Citation [n] | Questions, Générateur | la phrase s'appuie sur la source n | cliquer / lire l'extrait pour vérifier |
| « x/y phrases soutenues » | Questions | part des phrases vérifiées dans leur source | si x < y : lire les phrases en rouge avec prudence |
| Réponse bloquée | Questions | aucune phrase soutenue → texte remplacé par un refus | reformuler, ou la question sort du corpus |
| Refus | Questions | hors corpus ou sujet absent des passages | normal ; vérifier dans l'onglet Recherche |
| ⚠️ lettre corrigée | Générateur (QCM) | l'explication désignait un autre choix : le code a corrigé | relire la question |
| ⚠️ ambiguë / retirée | Générateur (QCM) | plusieurs bonnes réponses possibles / question fausse supprimée | normal ; en demander d'autres si besoin |
| ⚠️ terme absent de sa source | Générateur (glossaire) | définition probablement donnée de mémoire | vérifier ou supprimer le terme |
| « plan B » | Assistant | le modèle n'a pas utilisé ses outils : le système a cherché pour lui | normal avec le petit modèle |

## 10. Problèmes fréquents

| Symptôme | Cause probable | Solution |
|---|---|---|
| « API injoignable » | `transfia serve` non lancé ou arrêté | le relancer (section 7) |
| Réponses très lentes (> 1 min) | RAM saturée | fermer navigateur / Teams ; garder le petit modèle |
| Toujours le même résultat au Générateur | cache des réponses | normal pour une demande identique ; changer le thème ou le niveau |
| « Storage folder … already accessed » | une commande `transfia` lancée pendant que le serveur tourne | arrêter le serveur, lancer la commande, relancer le serveur |
| Version affichée ancienne | ancien serveur encore actif | Ctrl+C dans le terminal du serveur, puis `transfia serve` |
| Graphe vide | graphe non construit | `transfia graph build` |
| Carte des thèmes ou questions sans réponse | index vide | `transfia ingest` |

---

## Annexe : rejouer les tests

```powershell
transfia doctor                                        # version, modèles, RAM
pytest                                                 # 146 tests automatiques
transfia eval rag --note "ma note"                     # évaluation RAG (cache coupé)
transfia eval show                                     # détail : refus à tort, réponses à tort
transfia bench --label petit                           # banc complet, petit modèle (≈ 30 min)
transfia --config config/local-3b.yaml bench --label 3b
transfia --config config/gcp.yaml bench --label gemini --only rag --max-questions 10   # Gemini, sans carte
$f = Get-ChildItem data\bench\*.json | % FullName; transfia bench compare @f --out data\bench\comparaison.md
```

- **Journal détaillé** : `docs/JOURNAL_TESTS.md` ; **dossier fonctionnel** : `docs/dossier-fonctionnel-transfia.md`.
