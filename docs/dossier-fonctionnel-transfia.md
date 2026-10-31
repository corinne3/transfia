# TransfIA — Dossier fonctionnel et technique

> **Nom provisoire.** Application open source qui transforme des synthèses de recherche en élevage en contenus pédagogiques sourcés.
> Domaine : transfert des résultats de la recherche vers la formation agricole.
> Auteure : Corinne Bergès — repo public GitHub.

---

## Sommaire

0. Cadre et contraintes
1. Utilisateurs et besoins
2. Architecture d'ensemble
3. Fonctionnalités (F1 → F13) : besoin, concepts IA, outils, choix, explications
4. Focus : tools vs skills vs MCP
5. Focus : le prompt engineering traité comme du code
6. Récapitulatif de la stack
7. Structure du repo
8. Plan sur 4 semaines
11. Travailler sereinement avec l'IA : dimensionnement et frugalité, évaluation (compléments), cadre légal et éthique, sécurité, fiabilité, humain dans la boucle, checklist
12. TransfIA dans le cloud : architecture portable, équivalences Azure / AWS / GCP, ce qui est gratuit
13. Implémentation sur Google Cloud (GCP)
14. Implémentation sur Azure (alternative gratuite)

---

## 0. Cadre et contraintes

- **Objectif principal : apprendre en profondeur.** Chaque concept IA actuel est implémenté au moins une fois, même si c'est un peu artificiel.
- **100 % open source** : code, modèles (open-weights) et données.
- **Matériel : PC sans GPU, sans crédits API payants.**
  - Inférence LLM en local, sur CPU, avec des modèles quantifiés (3 à 8B paramètres).
  - Le seul moment où l'on sort du CPU : le fine-tuning, sur un GPU gratuit (Colab ou Kaggle).
- **Données** : revue *INRAE Productions Animales*, articles 2020+ sous licence **CC BY 4.0**.
  - On ne commit pas les PDF : on commit le script qui les télécharge.
  - Attribution (auteurs, titre, DOI) systématique dans chaque contenu généré.
- **Temps** : environ 40 h (1 h à 2 h par jour pendant 1 mois).
- **Principe pédagogique : « from scratch d'abord, puis la lib ».** On code soi-même la version simple pour comprendre, puis on passe à l'outil standard pour comparer.

---

## 1. Utilisateurs et besoins

| Persona | Besoin principal | Fonctionnalités clés |
|---|---|---|
| **Formateur** (lycée agricole, BTS) | Préparer un cours à jour à partir de la recherche, sans lire 30 pages | F5, F7, F8 |
| **Enseignant-chercheur** (master) | Retrouver vite les études sur un sujet, voir les liens entre thèmes | F4, F6 |
| **Apprenant** | Poser des questions, s'entraîner avec des quiz | F5, F7 |
| **Éleveur / grand public** | Comprendre un résultat de recherche en langage simple | F5, F9 |
| **Équipe projet (toi)** | Mesurer la qualité, surveiller, améliorer | F11, F12 |

**Besoins utilisateurs (user stories) :**
- « Je veux **chercher** des études sur la santé des veaux et voir les plus pertinentes. » → F4
- « Je veux **poser une question** et avoir une réponse **avec les sources** exactes. » → F5
- « Je veux **voir la carte** des thèmes et des liens entre concepts. » → F6
- « Je veux **générer une fiche, un quiz ou un cas pratique** adapté au niveau BTS. » → F7
- « Je veux qu'un assistant me **construise une séquence de cours complète** sur un thème. » → F8
- « Je veux qu'un contenu soit **relu scientifiquement et vulgarisé** avant de l'utiliser. » → F9
- « Je veux être sûr que l'outil **n'invente rien**. » → F5, F11, F12

---

## 2. Architecture d'ensemble

```
                        ┌─────────────────────────────────────┐
                        │   Front React (TypeScript, Vite)     │
                        │   chat streaming · recherche · graphe│
                        └───────────────┬─────────────────────┘
                                        │ HTTP + SSE (streaming)
                        ┌───────────────▼─────────────────────┐
                        │         API FastAPI (Python)         │
                        └───┬───────────┬───────────┬─────────┘
                            │           │           │
              ┌─────────────▼──┐  ┌─────▼──────┐  ┌─▼──────────────────┐
              │  RAG / Search  │  │  Agents    │  │  Génération        │
              │  hybride       │  │  tools +   │  │  pédagogique       │
              │  + rerank      │  │  skills    │  │  (structured out.) │
              └──────┬─────────┘  └─────┬──────┘  └─────────┬──────────┘
                     │                  │ MCP               │
       ┌─────────────▼──────────────────▼───────────────────▼─────────┐
       │  Stockage : Qdrant (vecteurs) · BM25 · graphe NetworkX ·      │
       │            SQLite (métadonnées, résultats d'éval)             │
       └─────────────▲─────────────────────────────────────────────────┘
                     │
       ┌─────────────┴────────────────────────────────────────────────┐
       │ Pipeline d'ingestion (batch) : collecte → parsing → chunking  │
       │ → NER → classification → embeddings → indexation → graphe     │
       └───────────────────────────────────────────────────────────────┘

   Transverse : Ollama (LLM local) · Phoenix (traces) · garde-fous · banc d'éval
```

---

## 3. Fonctionnalités

> Format de chaque fiche :
> **Besoin** → **Ce que fait l'appli** → **Concepts IA** → **Outils possibles** → **Choix** → **Comment ça marche** → **Pièges** → **À retenir**

---

### F1 — Collecte du corpus et catalogue

**Besoin** : disposer d'un corpus propre, traçable et réutilisable légalement.

**Ce que fait l'appli :**
- Récupère la liste des articles de la revue (titre, auteurs, date, DOI, résumé, mots-clés, licence).
- Filtre : année ≥ 2020 **et** licence CC BY.
- Télécharge les PDF dans `data/raw/` (ignoré par git).
- Stocke le catalogue dans SQLite.

**Concepts** : data engineering, provenance et licences des données, idempotence d'un pipeline.

**Outils possibles :**
- Sources de métadonnées : OAI-PMH (protocole standard des revues OJS), API OpenAlex, API Crossref.
- HTTP : `httpx`, `requests`.
- Stockage : SQLite, DuckDB, Parquet.

**Choix** : OAI-PMH ou OpenAlex (à vérifier en semaine 0) + `httpx` + SQLite.

**Comment ça marche :**
- OAI-PMH expose les métadonnées via des requêtes du type `?verb=ListRecords&metadataPrefix=oai_dc`, avec pagination par `resumptionToken`.
- Chaque article reçoit un identifiant stable (le DOI). Relancer le script ne télécharge que ce qui manque : c'est l'**idempotence**.

**Pièges :**
- Ne pas surcharger le serveur : délai entre requêtes et `User-Agent` explicite.
- Vérifier la licence **article par article**, pas seulement au niveau de la revue.

**À retenir :**
- « Garbage in, garbage out » : la qualité et la traçabilité des données conditionnent tout le reste.
- La licence et la provenance font partie de la gouvernance des données.

---

### F2 — Ingestion multimodale des documents

**Besoin** : exploiter tout le contenu des articles, y compris les tableaux et les figures, pas seulement le texte.

**Ce que fait l'appli :**
- Convertit chaque PDF en document structuré : titres, sections, paragraphes, tableaux, figures, légendes, numéros de page.
- Extrait les tableaux sous forme exploitable (Markdown ou DataFrame).
- Génère une **description textuelle des figures** avec un modèle vision-langage (VLM), pour qu'elles deviennent cherchables.
- OCR si une page est scannée ou si du texte est contenu dans une image.

**Concepts** : document parsing, layout analysis, OCR, VLM, multimodalité.

**Outils possibles :**
- Parsing : **Docling**, Unstructured, Marker, PyMuPDF, pdfplumber, MinerU.
- OCR : Tesseract, EasyOCR, RapidOCR (intégrés dans Docling).
- VLM : SmolVLM, Moondream, Qwen-VL (petites versions), Granite Vision.

**Choix** : **Docling** (open source, fonctionne sur CPU, gère tableaux et structure) + un **petit VLM via Ollama** pour les figures (seulement sur un échantillon, car c'est lent sur CPU).

**Comment ça marche :**
- **Layout analysis** : un modèle de détection découpe chaque page en zones (titre, texte, tableau, figure).
- **Table structure recognition** : un second modèle reconstruit les lignes et colonnes du tableau.
- **OCR** : reconnaissance de caractères sur les zones image. Inutile pour les PDF « natifs », dont le texte est déjà encodé.
- **VLM** : un encodeur d'images (type ViT) transforme la figure en tokens visuels, injectés dans le LLM, qui génère une description.

**Pièges :**
- Ordre de lecture faux sur les mises en page en 2 colonnes.
- Le VLM peut **halluciner des valeurs** lues sur un graphique : il faut marquer ces descriptions comme « générées ».
- Le temps CPU : on traite d'abord quelques dizaines d'articles.
- **Cas vécu — le clustering révèle la saleté des données** : sur 8 clusters, l'un était fait de **bibliographie** (« of, and, https, org, doi ») et un autre portait « br » (balises `<br>` des tableaux). Ces passages polluaient aussi la recherche (une référence anglaise remontée en 3e résultat sur « colostrum »). Parades à l'ingestion : section « Références » écartée, passages à forte densité d'années/DOI/initiales écartés (`looks_like_references`), `<br>` nettoyés ; et à la ré-indexation, les anciens chunks de l'article sont **supprimés** (sinon des orphelins restent dans l'index). Leçon : **le non supervisé est un excellent outil d'audit des données** ; « garbage in, garbage out ».

**À retenir :**
- Différence OCR / parsing / VLM : lire des caractères / comprendre la structure / comprendre le sens visuel.
- Multimodal = plusieurs types d'entrées (texte, image, audio) projetés dans un espace commun.

---

### F3 — Enrichissement : chunking, entités, classification, thèmes

**Besoin** : rendre le corpus navigable (filtres par espèce, par thème) et préparer la recherche.

**Ce que fait l'appli :**
1. **Chunking** : découpe chaque article en morceaux de taille adaptée à la recherche.
2. **NER (reconnaissance d'entités)** : repère les espèces, races, pathologies, molécules, nutriments.
3. **Classification thématique** : attribue à chaque article un thème (nutrition, santé, génétique, bien-être, environnement, économie…).
4. **Clustering** : découvre automatiquement les grands sujets du corpus et les visualise sur une carte 2D.

**Concepts** : chunking, NER, **machine learning classique** (classification supervisée, métriques), apprentissage non supervisé (clustering, réduction de dimension), zero-shot.

**Outils possibles :**
- Chunking : from scratch, LangChain text splitters, LlamaIndex node parsers, chunker de Docling.
- NER : spaCy (modèle français), **GLiNER** (NER zero-shot, léger), LLM avec structured output.
- Classification : **scikit-learn** (TF-IDF + régression logistique), embeddings + kNN, LLM zero-shot.
- Clustering : **UMAP + HDBSCAN**, KMeans, BERTopic.

**Choix** :
- Chunking : from scratch (taille fixe), puis le chunker de Docling basé sur la structure, puis sémantique, avec comparaison.
- NER : **GLiNER** (fonctionne sur CPU, labels définis librement), comparé au LLM.
- Classification : **comparaison de 3 approches** (voir ci-dessous).
- Clustering : UMAP + HDBSCAN sur les embeddings.

**Comment ça marche :**
- **Chunking**
  - *Taille fixe* : N tokens avec un chevauchement (overlap) pour ne pas couper une idée en deux.
  - *Structurel* : on découpe selon les sections du document.
  - *Sémantique* : on coupe là où la similarité entre phrases consécutives chute.
  - Chaque chunk garde ses **métadonnées** : article, section, page, DOI.
- **NER avec GLiNER** : un petit encodeur compare les fragments de texte aux labels fournis (« race bovine », « pathologie »…). Aucun ré-entraînement n'est nécessaire, c'est du **zero-shot**.
- **Classification : 3 approches comparées**
  1. *Baseline ML classique* : TF-IDF (poids d'un mot = fréquent dans le doc mais rare dans le corpus) + régression logistique.
  2. *Embeddings + kNN ou régression logistique* : on remplace TF-IDF par des vecteurs sémantiques.
  3. *LLM zero-shot* : on demande au LLM de choisir parmi une liste de thèmes.
  - Étiquettes d'entraînement : quelques dizaines d'articles annotés (par le métier, toi, ou à partir des mots-clés auteurs, ce qu'on appelle de la *weak supervision*).
  - Évaluation : **cross-validation**, précision, rappel, F1, matrice de confusion.
- **Clustering**
  - UMAP réduit les vecteurs (ex. 768 dimensions) à 2D en préservant les voisinages.
  - HDBSCAN regroupe les points denses et laisse les points isolés en « bruit ».
  - Un LLM nomme chaque cluster à partir de ses articles représentatifs.

**Pièges :**
- Chunks trop petits = perte de contexte ; trop gros = bruit et coût.
- Classes déséquilibrées : la métrique *accuracy* devient trompeuse, il faut regarder le F1 macro.
- **Data leakage** : ne jamais ajuster le TF-IDF sur le jeu de test.
- **Cas vécu — fuite par GROUPE** : F1 macro 0,91, accuracy 0,96… mais les « mots discriminants » du thème *économie* étaient « aubrac, 1994, vab » : des mots d'UN article, pas du thème. Cause : l'étiquette est le thème de l'article (≈ 80 passages par article) et la CV mélange les passages → des passages du même article sont en train ET en test → le modèle apprend à **reconnaître l'article**. Parade : **StratifiedGroupKFold** (groupe = article) : un article est entièrement en train ou en test. On affiche les scores **à périmètre égal** (mêmes classes) : l'écart mesure alors la fuite, et elle seule (1re version : comparaison 4 thèmes vs 2 thèmes, corrigée). Conséquence : un thème représenté par un seul article ne peut pas être évalué honnêtement (il faut au moins 2 articles par thème). Même piège ailleurs : patients en médecine, clients, séries temporelles (on ne teste pas sur le passé).
- **Cas vécu — silhouette faible** : silhouette de 0,10 à 0,14 pour k = 2 à 8, croissante jusqu'au bord de la plage. Lecture : **pas de structure en groupes nets** (au-dessus de 0,5 = net ; vers 0,1 = flou), et un meilleur k au bord de la plage n'est pas un « vrai » k. Les clusters sont à lire comme des tendances, pas comme des thèmes.

**À retenir :**
- Précision = parmi ce que j'ai prédit positif, combien sont vrais. Rappel = parmi les vrais positifs, combien j'en ai trouvés. F1 = moyenne harmonique des deux.
- Biais-variance, overfitting, cross-validation, régularisation (L1/L2) : tout est illustrable avec la baseline TF-IDF.
- Supervisé (on a les étiquettes) vs non supervisé (on découvre une structure).
- Une baseline simple d'abord : elle bat souvent des approches plus « modernes » sur peu de données.

---

### F4 — Moteur de recherche dans le corpus

**Besoin** : trouver les passages les plus pertinents, que la question emploie les mêmes mots que l'article ou non.

**Ce que fait l'appli :**
- Barre de recherche avec filtres (espèce, thème, année).
- Résultats classés, avec passage surligné, article, page et lien DOI.

**Concepts** : embeddings, similarité cosinus, recherche lexicale (BM25), recherche dense, **recherche hybride**, fusion de rangs (RRF), **reranking** (cross-encoder), index vectoriel (ANN), filtres par métadonnées.

**Outils possibles :**
- Embeddings : sentence-transformers avec **multilingual-e5** (small/base), **BGE-M3**, nomic-embed.
- BM25 : `rank_bm25`, `bm25s`, Tantivy, Elasticsearch/OpenSearch.
- Base vectorielle : **Qdrant**, LanceDB, Chroma, FAISS, pgvector, Weaviate, Milvus.
- Reranker : **bge-reranker-v2-m3**, cross-encoders mMARCO multilingues.

**Choix** :
- Embeddings : **multilingual-e5-small** (rapide sur CPU), comparé à **BGE-M3** (meilleur, plus lourd).
- BM25 : **bm25s**.
- Base vectorielle : **Qdrant en mode local** (sans serveur, gère filtres et hybride).
- Reranker : **bge-reranker-v2-m3**, seulement sur le top 20.

**Comment ça marche :**
- **Embedding** : un encodeur transforme un texte en vecteur. Deux textes au sens proche ont des vecteurs proches.
- **Similarité cosinus** : `cos(a, b) = a·b / (‖a‖ ‖b‖)`, compris entre −1 et 1. Codée d'abord **à la main en NumPy**.
- **BM25** : score lexical. Un mot compte plus s'il est rare dans le corpus (IDF), avec saturation de la fréquence et normalisation par la longueur du document. Excellent sur les termes exacts (noms de races, molécules).
- **Recherche dense** : excellente sur les paraphrases et synonymes, faible sur les termes rares ou codes exacts.
- **Hybride + RRF** : on lance les deux recherches puis on fusionne : `score(d) = Σ 1 / (k + rang(d))`, avec k ≈ 60. Pas besoin de normaliser des scores hétérogènes.
- **Reranking** :
  - *Bi-encoder* (embeddings) : question et document encodés **séparément**. Rapide, mais approximatif.
  - *Cross-encoder* (reranker) : question et document lus **ensemble**. Précis, mais lent. Donc on l'applique sur peu de candidats.
- **ANN (approximate nearest neighbors)** : au-delà de quelques milliers de vecteurs, on ne compare pas tout. Qdrant utilise HNSW, un graphe multi-niveaux de voisins.

**Pièges :**
- Les modèles e5 attendent les préfixes `query:` et `passage:`. Les oublier dégrade la qualité.
- Il faut re-indexer si on change de modèle d'embedding : les espaces vectoriels ne sont pas compatibles.
- **Cas vécu — les tournures de conversation polluent la recherche** : « Que dit le corpus sur le colostrum ? » → meilleur score du reranker **-5,7** (passage sur l'alimentation INRA, hors sujet → refus par le seuil -3) ; « colostrum » → **-2,6** (le bon passage). Les mots « que dit le corpus » ne figurent dans aucun passage : c'est du **bruit** pour l'embedding et le cross-encoder. Parade : `retrieval/query.py` retire ces tournures **pour la recherche seulement** (le LLM garde la question d'origine). Leçon double : nettoyer la requête, et un **seuil calibré sur 3 questions est fragile** → à recalibrer sur un vrai jeu d'éval (brique 7).
  - **Suite** : après nettoyage, -5,7 → -3,6 (bon passage en 1er), mais toujours sous -3. Et « colostrum » = -2,6 contre « le colostrum » = -3,6 : **un score de cross-encoder n'est pas une probabilité**, il varie avec la forme de la question. Parade : **double critère** de refus : score bas **ET** mots porteurs de la question absents des 3 meilleurs passages. « colostrum » est dans le passage → réponse ; « grippe aviaire » n'est nulle part → refus maintenu. (Alternative plus rigoureuse : **calibrer** les scores en probabilités, par régression logistique de Platt, sur un jeu annoté.)

**À retenir :**
- Lexical vs sémantique : complémentaires, d'où l'hybride.
- Pipeline classique en 2 étages : *retrieval* (rappel, rapide) puis *reranking* (précision, lent).
- Métriques de recherche : Recall@k, MRR, nDCG.

---

### F5 — Questions-réponses sourcées (RAG avancé)

**Besoin** : poser une question en langage naturel et obtenir une réponse fiable, avec les sources exactes.

**Ce que fait l'appli :**
- Chat avec réponse en streaming.
- Chaque phrase de la réponse renvoie à une citation : `[Auteur, année, p. X]` cliquable.
- Si le corpus ne contient pas la réponse : « Je ne trouve pas cette information dans le corpus. »

**Concepts** : RAG, query rewriting, multi-query, **HyDE**, context window, citations, grounding, contrôle des hallucinations (NLI), streaming.

**Outils possibles :**
- Orchestration : from scratch, LlamaIndex, LangChain, Haystack.
- LLM local : **Ollama**, llama.cpp, vLLM (GPU, donc exclu ici).
- Vérification : modèles NLI multilingues (ex. mDeBERTa-XNLI), LLM-as-judge.

**Choix** : **RAG codé from scratch** (pour tout comprendre), puis une réimplémentation avec LlamaIndex pour comparer. LLM : un **modèle 3 à 4B quantifié** via Ollama (familles Qwen, Gemma, Llama, Mistral : la version exacte sera choisie au démarrage).

**Comment ça marche :**
1. **Query rewriting** : le LLM reformule la question (corrige, complète avec l'historique du chat).
2. **Multi-query** : il génère 3 variantes de la question pour élargir le rappel.
3. **HyDE (Hypothetical Document Embeddings)** : le LLM rédige une *réponse hypothétique*, et on cherche avec l'embedding de cette réponse. Elle ressemble davantage aux passages du corpus que la question elle-même.
4. **Retrieval** hybride + reranking (F4).
5. **Construction du prompt** : instructions + passages numérotés + question. Consigne : « Réponds uniquement à partir des passages, cite [n] pour chaque affirmation. »
6. **Génération en streaming** : les tokens sont envoyés au front au fur et à mesure.
7. **Vérification** : chaque phrase est confrontée à son passage cité par un modèle **NLI** (le passage implique-t-il la phrase ?). Les phrases non soutenues sont signalées.

**Pièges :**
- *Lost in the middle* : les LLM exploitent moins bien les infos placées au milieu d'un long contexte. Il faut limiter et bien ordonner les passages.
- Petits modèles : ils suivent moins bien les consignes de citation, d'où l'utilité de la vérification automatique.
- HyDE peut **dériver** si le LLM imagine une réponse fausse : c'est à comparer avec et sans, dans le banc d'éval.

- **Cas vécu — le SUJET n'était pas vérifié** : « Quels sont les facteurs de risque des mammites ? » → passages sur la viande transformée → « les facteurs de risque des mammites comprennent les crustacés, les viandes transformées et la cuisson à haute température [4] », validée 1/1 : les mots de la phrase sont bien dans le passage [4]… sauf « mammite ». Parade : **ancrage sur les termes rares** de la question (≤ 5 % des passages : ce sont eux qui disent de quoi on parle) : absents de tous les passages retenus → refus ; présents dans une phrase mais pas dans le passage cité → phrase non soutenue. Leçon : un recouvrement de mots ne vérifie pas le sujet ; un juge NLI ou LLM le ferait mieux, à un coût plus élevé.

**À retenir :**
- RAG = injecter de la connaissance au moment de l'inférence, sans ré-entraîner. On peut mettre la base à jour, et les réponses sont traçables.
- Hallucination = sortie plausible mais non fondée. Les leviers : grounding, citations, vérification, droit de dire « je ne sais pas ».
- RAG vs fine-tuning : RAG pour la **connaissance**, fine-tuning pour le **comportement et le format**.

---

### F6 — Explorateur de connaissances (GraphRAG)

**Besoin** : voir comment les concepts sont reliés (pathologie ↔ espèce ↔ facteur de risque ↔ étude) et répondre aux questions « globales ».

**Ce que fait l'appli :**
- Graphe interactif : nœuds = concepts et articles ; arêtes = relations (« affecte », « réduit », « étudié dans »).
- Questions transversales : « Quels sont les grands enjeux de bien-être animal dans le corpus ? »

**Concepts** : knowledge graph, extraction de triplets (sujet, relation, objet) par LLM, entity resolution, détection de communautés, **GraphRAG** (local et global).

**Outils possibles :**
- Graphe : **NetworkX**, Neo4j, Memgraph.
- Frameworks : Microsoft GraphRAG, LightRAG, LlamaIndex PropertyGraph.
- Visualisation : Cytoscape.js, Sigma.js, react-force-graph.

**Choix** : **NetworkX** (léger, en mémoire) + extraction de triplets par LLM en structured output + **algorithme de Leiden ou Louvain** pour les communautés + **react-force-graph** pour l'affichage.

**Comment ça marche :**
1. Pour chaque chunk, le LLM extrait des triplets selon un schéma Pydantic : `{"sujet": "...", "relation": "...", "objet": "..."}`.
2. **Entity resolution** : on fusionne les doublons (« vache laitière » = « vaches laitières ») via la similarité d'embeddings et des règles.
3. **Communautés** : un algorithme regroupe les nœuds fortement connectés. Le LLM résume chaque communauté.
4. **Recherche locale** : on part des entités de la question et on explore leurs voisins.
5. **Recherche globale** : on interroge les résumés de communautés (méthode *map-reduce*), utile pour les questions de synthèse que le RAG classique gère mal.

**Pièges :**
- L'extraction est coûteuse sur CPU : on la limite à un sous-corpus.
- Graphe bruité si les relations ne sont pas normalisées : on fixe une liste fermée de types de relations.
- **Cas vécu — graphe** : 60 entités, 485 liens, 4 communautés cohérentes (climat/génétique ; races bovines ; santé/qualité ; alimentation) ✔. Deux bugs : (1) la recherche **locale** « méthane » renvoyait presque tout le graphe : on prenait les arêtes des VOISINS (méthane est relié à 50 nœuds sur 60) → on ne garde que les arêtes du nœud lui-même ; (2) la recherche **globale** répondait « pas d'éléments » : à l'étape *map*, le LLM ne recevait qu'une liste de mots par communauté → on lui donne aussi les **passages** qui fondent la communauté (c'est ce que fait GraphRAG avec ses « rapports de communauté »). 2e essai : la recherche locale est corrigée ✔, la globale répond encore « pas d'éléments ». Hypothèse : avec l'échappatoire « réponds RIEN si… », un petit modèle **prend la sortie facile**. Parade : pas d'échappatoire quand il y a des extraits (c'est l'étape *reduce* qui trie), et **trace de l'étape map** affichée (combien d'extraits, gardé ou écarté) pour diagnostiquer au lieu de deviner. **3e essai** : hypothèse confirmée, 4 communautés sur 4 gardées, synthèse sur tout le corpus (3-NOP, vaccination, génomique, consanguinité, rationnement) ✔ ; mais réponse coupée par la limite de tokens → synthèse bornée à 5 points et coupure signalée (`done_reason = "length"`). Limite : liens de **co-occurrence** seulement (« apparaissent ensemble », pas « cause » ou « réduit ») ; `--llm` extrait de vraies relations, mais lentement.

**À retenir :**
- RAG classique = bon pour « trouver un passage ». GraphRAG = bon pour les relations multi-sauts et les questions globales.
- Coût : une indexation plus chère en échange de meilleures réponses de synthèse.

---

### F7 — Générateur de contenus pédagogiques

**Besoin** : produire en un clic une fiche, un quiz ou un cas pratique adapté à un niveau donné, à partir d'un ou plusieurs articles.

**Ce que fait l'appli :**
- Choix : type de contenu (fiche synthèse, QCM, cas pratique, glossaire) × niveau (grand public, bac pro, BTS, master).
- Sortie **structurée**, validée, puis affichée proprement et exportable (Markdown, JSON).

**Concepts** : **structured outputs**, JSON schema, constrained decoding, few-shot prompting, prompt templates, **skills** (savoir-faire chargé à la demande), adaptation au niveau de lecture.

**Outils possibles :**
- Validation : **Pydantic**.
- Sortie structurée : mode `format` (JSON schema) d'Ollama, **Instructor**, Outlines, PydanticAI.
- Templates : **Jinja2**.

**Choix** : **Pydantic + JSON schema natif d'Ollama**, templates **Jinja2**, et chaque type de contenu est une **skill** (voir section 4).

**Comment ça marche :**
- **Schéma Pydantic** pour chaque type, par exemple pour un QCM :
  ```python
  class Question(BaseModel):
      enonce: str
      choix: list[str]          # 4 choix
      bonne_reponse: int        # index 0-3
      explication: str
      source: Citation          # article + page
  class Quiz(BaseModel):
      niveau: Literal["grand_public", "bac_pro", "bts", "master"]
      questions: list[Question]
  ```
- **Constrained decoding** : à chaque token, le moteur interdit ceux qui rendraient le JSON invalide vis-à-vis du schéma. La sortie est donc toujours parsable.
- **Validation + retry** : si une règle métier échoue (ex. `bonne_reponse` hors limites), on renvoie l'erreur au LLM pour correction.
- **Few-shot** : 1 ou 2 exemples de bons contenus dans le prompt (idéalement fournis par le métier).
- **Adaptation au niveau** : consignes de longueur de phrase et de vocabulaire, avec une mesure de lisibilité (score de Flesch adapté au français) pour vérifier.

**Pièges :**
- Un JSON valide n'est pas forcément un contenu juste : la vérification de fond se fait en F9 et F11.
- Forcer un schéma trop complexe dégrade la qualité des petits modèles : on garde des schémas simples.
- **Cas vécu** (qwen2.5:1.5b, QCM « méthane entérique ») : JSON parfait, mais 2 questions sur 3 avec la mauvaise lettre (explication de B, réponse marquée C). Parades ajoutées :
  - **ordre des champs** : `explication` AVANT `bonne_reponse` dans le schéma. Le LLM écrit de gauche à droite : il raisonne, puis l'index découle du raisonnement (*reason before answer*) ;
  - **contrôle de cohérence** (`generation/consistency.py`) : on compare l'explication à chaque proposition ; si une autre colle nettement mieux → index corrigé + avertissement ; si deux collent autant → « question ambiguë » ;
  - **sources** : seules les sources citées sont listées, regroupées par article.
  - **2e essai** : Q1 corrigée (A → D), mais Q2 et Q3 ambiguës, et Q3 avec les mêmes propositions que Q2. On ne sait pas corriger ça seul → **boucle de réparation** : on renvoie au LLM son quiz + la liste précise des défauts, il réécrit, on re-contrôle et on garde la version seulement si elle est **meilleure**. Même schéma qu'un relecteur humain (critique → correction), pour 1 appel LLM de plus, et seulement en cas de défaut.
  - **3e essai** : la réparation tentée n'a rien amélioré avec le 1,5B (« pas mieux » → version d'origine gardée). Deux leviers :
    - **dernier recours** : les questions encore défectueuses sont **retirées** (un QCM faux *enseigne* l'erreur : mieux vaut 1 bonne question que 3 douteuses) ;
    - **routage par tâche** : `llm.generation_model` (ou `--model qwen2.5:3b`). Une réponse au chat doit être rapide ; un quiz se prépare à l'avance, on peut se payer un plus gros modèle. Frugalité ≠ toujours le plus petit modèle : le **bon** modèle pour **chaque** tâche.
  - **Cache** : même prompt = même réponse relue dans le cache SQLite (d'où un résultat identique d'un essai à l'autre). `--no-cache` pour redemander au LLM.

**À retenir :**
- Zero-shot / few-shot / chain-of-thought : les 3 techniques de prompting de base.
- Structured output = rendre une sortie de LLM fiable pour du code en aval.
- Valide ≠ juste : le schéma garantit la FORME, pas le FOND. Le fond se contrôle à part (règle métier, heuristique, juge, humain).

---

### F8 — Assistant « ingénieur pédagogique » (agent)

**Besoin** : « Construis-moi une séquence de 3 séances sur le bien-être des porcs, niveau BTS. » La tâche est multi-étapes : l'assistant doit chercher, planifier, rédiger et assembler.

**Ce que fait l'appli :**
- L'agent **planifie** (objectifs → séances → contenus), **appelle des outils** (recherche, lecture d'article, génération de quiz…), **charge les skills** nécessaires et assemble le résultat.
- Le front affiche chaque étape en direct : « Je cherche… », « Je lis l'article X… », « Je rédige la séance 2… ».

**Concepts** : agent, boucle **ReAct** (raisonner → agir → observer), **tool calling**, **skills**, planification, mémoire (court et long terme), **MCP**, human-in-the-loop.

**Outils possibles :**
- Frameworks d'agents : from scratch, **LangGraph**, smolagents, PydanticAI, LlamaIndex Agents, CrewAI, AutoGen.
- MCP : **SDK Python officiel** (FastMCP).

**Choix** :
1. **Boucle d'agent from scratch** (~100 lignes) avec le tool calling natif d'Ollama.
2. Réécriture avec **LangGraph** (graphe d'états, checkpoints, interruption humaine).
3. Les outils du corpus exposés via un **serveur MCP**.

**Comment ça marche :**
- **La boucle d'agent :**
  ```
  messages = [system, user]
  tant que pas fini :
      réponse = LLM(messages, tools=schémas_des_outils)
      si réponse contient des tool_calls :
          pour chaque appel : exécuter l'outil, ajouter le résultat aux messages
      sinon : réponse finale → fin
  ```
- **Tools disponibles** (exemples) : `search_corpus(query, filtres)`, `read_article(doi, section)`, `get_entities(theme)`, `generate_content(type, niveau, sources)`, `load_skill(nom)`.
- **Planification** : l'agent rédige d'abord un plan structuré (Pydantic), puis l'exécute étape par étape (pattern *plan-and-execute*).
- **Mémoire** :
  - court terme = l'historique des messages ;
  - long terme = préférences enregistrées (ex. « ce formateur veut des cas pratiques bovins »).
- **Human-in-the-loop** : LangGraph met l'exécution en pause pour faire valider le plan avant la rédaction.
- **Garde-fous** : nombre maximal d'itérations et timeout par outil.

**Pièges :**
- Les petits modèles font des appels d'outils mal formés : il faut valider les arguments et renvoyer l'erreur au modèle.
- Boucles infinies : prévoir un nombre max d'étapes.
- Trop d'outils = le modèle se trompe d'outil. On reste sous ~10 outils, avec des descriptions très claires.
- **Cas vécu — l'agent qui n'utilise pas ses outils** (qwen2.5:1.5b) : « D'où vient le méthane des ruminants ? » → réponse directe, ZÉRO appel d'outil, « de mémoire » : « des bactéries de l'estomac ». Faux : ce sont des **archées méthanogènes**, dans le **rumen**. Le prompt l'interdisait déjà (« Ne réponds jamais de mémoire ») : **une consigne n'est pas un contrôle**. Parade (`agents/react.py`) : **garde-fou d'ancrage** :
  - réponse finale sans avoir appelé un outil de connaissance (`search_corpus`, `read_article`…) → renvoyée UNE fois à l'agent : « cherche d'abord » ;
  - s'il insiste → réponse marquée `ungrounded` et affichée ⚠️ « non sourcée ».
  - (Les API cloud offrent `tool_choice="required"` pour forcer un appel d'outil ; Ollama ne l'a pas : on le fait dans la boucle.)
  - **2e essai** : après le rappel, « Je vais vérifier la source de la question. »… et toujours aucun appel. Le modèle *dit* qu'il va agir sans *agir*. **Plan B** : le code lance lui-même `search_corpus(question)` et donne les passages à l'agent (affiché ℹ️ « plan B »). L'agent **redevient un workflow** : c'est la **dégradation gracieuse**. Leçon : sous ~3 Md de paramètres, le tool calling est peu fiable ; pour un petit modèle, un workflow (RAG) bat un agent.
  - **3e essai** : la recherche forcée trouve le bon article (Gondret 2026)… et le modèle **recopie quand même** sa réponse de mémoire. Cause : **ancrage sur l'historique** : la conversation contient déjà 2 fois sa propre réponse ; un petit modèle préfère rester cohérent avec lui-même plutôt que lire les passages. Parade : le plan B repart d'un **contexte neuf** et passe la main au **pipeline RAG complet** (F5 : citations, vérification, droit au refus). Leçon : en cas d'échec, ne pas « insister » dans la même conversation, **réinitialiser le contexte**.
- **Cas vécu — ReAct en mode texte** (qwen2.5:1.5b) : pour « D'où vient le méthane ? », l'agent appelle `load_skill` (mauvais outil), avec `{"skill": ...}` au lieu de `{"name": ...}` (mauvais argument), puis répète 8 fois le même appel malgré l'avertissement (boucle), et sa « réponse finale » est un reste de format `Thought/Action`. Parades :
  - **message d'erreur lisible** : la signature attendue + un exemple, au lieu d'un dump Pydantic illisible pour un petit modèle ;
  - **tolérance aux arguments** : 1 argument obligatoire manquant + 1 argument inconnu → renommé automatiquement, et le modèle en est informé (« note : 'skill' interprété comme 'name' ») ;
  - **détection de boucle** : 2 appels répétés bloqués → arrêt (`stuck`), au lieu de brûler les étapes restantes ;
  - **plan B aussi après une boucle** ou le max d'étapes, si aucune source n'a été consultée ;
  - **nettoyage** de la réponse finale (lignes `Thought/Action` retirées).
  - **2e essai** : l'argument mal nommé est réparé ✔, la boucle est stoppée ✔, le plan B donne une réponse juste et sourcée ✔. Nouveaux défauts vus : valeurs presque justes (`"cas-pratique"`, `"bac pro"`) → **normalisation des enums** (casse, accents, tiret/espace/_) ; et **dérive de tâche** : on pose une question de fait, l'agent se met à produire un cas pratique. La dérive, elle, ne se répare pas par le code : c'est la limite du modèle (le plan B la rattrape).
- **Cas vécu — MCP** : l'agent **découvre** 4 outils qu'il ne connaissait pas ✔. Mais le plan B côté client n'avait que `search_corpus` (le pipeline RAG est dans l'autre processus) : réponse sans citation, non vérifiée, qui dérive (lésions podiales, porcs). Parade : un outil MCP **haut niveau** `ask_corpus` (pipeline RAG complet côté serveur). **Granularité des outils** : bas niveau = le client fait le travail (il doit être bon) ; haut niveau = le serveur garantit la qualité (idéal pour un client faible). Un bon serveur MCP propose les deux.
- **Cas vécu — la mémoire perdue au passage de relais** : préférence enregistrée « cas pratiques en élevage bovin » ; l'agent échoue → plan B → pipeline RAG… qui ne reçoit PAS la mémoire (injectée seulement dans le prompt de l'agent). Réponse sourcée, mais pas un cas pratique. Parade : la mémoire **suit le relais** : `rag.answer(..., preferences=...)`, et l'outil MCP `ask_corpus` accepte `preferences`. Règle d'architecture : **tout contexte utilisateur doit se propager à chaque composant** qui produit la réponse (agent, workflow de secours, serveur distant). Les préférences jouent sur la FORME, jamais sur les faits (ceux-ci viennent des passages).
- **Cas vécu — plan-and-execute** (« séquence de 2 séances sur le bien-être des poules pondeuses ») : plan de 6 étapes validé par l'humain ✔ ; refus du plan ✔ ; mais livrable **creux** (« [Rédiger ici] » partout, Sources vides), une erreur de fond (« viande de poule » pour des pondeuses), et aucune progression affichée. Parades (`agents/planner.py`) :
  - **contrôle de couverture AVANT de planifier** : le corpus parle-t-il du sujet ? Sinon arrêt immédiat, 0 appel LLM (vérifier la faisabilité avant de dépenser) ;
  - **base de preuves commune** : les passages trouvés sont donnés à chaque étape et à la synthèse (les étapes n'ont plus à « chercher » elles-mêmes, ce que le 1,5B ne sait pas faire) ;
  - **contrôle qualité du livrable** : textes à compléter détectés, absence de citations signalée ;
  - **Sources construites par le code** à partir des [n] réellement cités (le LLM ne peut ni en inventer, ni en oublier) ;
  - **progression affichée** étape par étape (observabilité).
  - **2e essai** : « poules pondeuses » → plan non lancé ✔ (vrai négatif : le corpus n'en parle pas). Mais « réduction du méthane entérique » → plan non lancé ✘ (**faux négatif** : un article entier traite du sujet). Cause : « Construis une séquence de 2 séances sur… » est une **consigne**, pas un sujet ; ces mots ne sont dans aucun passage. Parade : `clean_query` retire aussi les consignes de tâche (on ne cherche que le sujet). Leçon : **tout contrôle automatique a des faux positifs ET des faux négatifs** ; il faut tester les deux cas (un sujet absent ET un sujet présent).
  - **3e essai** (méthane, couvert) : 6 étapes affichées ✔, mais les agents d'étape (avec outils) relancent des recherches maladroites et concluent « je n'ai pas trouvé »… alors que les passages sont DANS leur prompt. Livrable vide ; le contrôle qualité le signale ✔ (« aucune citation »). Parade : **exécutant simple** : quand la base de preuves existe, chaque étape = **1 appel LLM** avec les passages (sans outils). Leçon : **le bon niveau d'autonomie** : on ne donne des outils que si l'information n'est pas déjà là ; sinon l'outil devient une distraction (et 5 appels au lieu de 1).
  - **4e essai** : étapes et livrable enfin SUR le sujet (méthodes de prédiction lait/fèces, microbiote du rumen, sélection) ✔, mais sans citation (signalé ⚠️) et sans vraie structure de séquence (liste répétitive de 10 points). **Plafond du modèle** : synthétiser 6 résultats en un livrable structuré et cité dépasse 1,5 Md de paramètres. Les contrôles du code sont tous au vert et rendent l'échec visible. Suite logique : routage par tâche (gros modèle pour la synthèse), moins d'étapes, ou le générateur structuré de F7.
- **Bilan F8 (à raconter)** : 6 échecs réels d'un agent sur petit modèle (réponse de mémoire, mauvais outil, mauvais arguments, boucle, dérive de tâche, synthèse non citée) → 6 garde-fous (ancrage, plan B vers workflow, réparation d'arguments, détection de boucle, contrôle de couverture, contrôle qualité). Conclusion : **avec un petit modèle, un workflow structuré bat un agent autonome**, et les garde-fous rendent l'échec visible au lieu de silencieux.

**À retenir :**
- Agent = LLM + outils + boucle + (mémoire, planification).
- Workflow (étapes fixées par le code) vs agent (le LLM décide des étapes). Un workflow est plus fiable, un agent plus flexible.

---

### F9 — Atelier multi-agents : rédacteur, relecteur, vulgarisateur

**Besoin** : un contenu relu avant usage, comme dans une vraie équipe éditoriale.

**Ce que fait l'appli :**
- **Rédacteur** : produit un premier jet à partir des sources.
- **Relecteur scientifique** : vérifie chaque affirmation contre les sources et liste les problèmes.
- **Vulgarisateur** : adapte le texte validé au niveau cible.
- Boucle de correction : rédacteur ↔ relecteur jusqu'à validation, ou 3 tours maximum.

**Concepts** : systèmes multi-agents, rôles, orchestration (superviseur, séquentiel), pattern **reflection / critique**, LLM-as-judge interne, diversité de modèles.

**Outils possibles** : LangGraph, CrewAI, AutoGen, smolagents, from scratch.

**Choix** : **LangGraph** (graphe explicite : rédaction → relecture → condition → correction ou vulgarisation). Le relecteur utilise un **modèle d'une autre famille** que le rédacteur, pour limiter les angles morts partagés.

**Comment ça marche :**
- Un **état partagé** (brouillon, sources, liste de problèmes, compteur de tours) circule entre des nœuds.
- Une **arête conditionnelle** décide de la suite : s'il reste des problèmes et moins de 3 tours, retour au rédacteur ; sinon, passage au vulgarisateur.
- Le relecteur répond en structured output : `{"affirmation", "statut": "soutenue|non_soutenue|contredite", "source"}`.

**Pièges :**
- Le coût et la latence se multiplient sur CPU : on travaille sur des contenus courts.
- Les agents peuvent « tomber d'accord » sur une erreur : c'est pour cela que la vérification NLI (F5) et les humains (F11) restent nécessaires.

**À retenir :**
- Multi-agents = spécialisation des rôles + vérification croisée. Ce n'est pas toujours mieux qu'un seul agent bien prompté : il faut le mesurer.

---

### F10 — Modèle spécialisé élevage (fine-tuning)

**Besoin** (un peu artificiel, assumé) : un petit modèle qui maîtrise mieux le vocabulaire et le **format** des contenus pédagogiques.

**Ce que fait l'appli :**
- Fine-tune un petit modèle (0,5 à 3B) pour générer des QCM au bon format et au bon ton.
- Compare 3 configurations dans le banc d'éval : petit modèle seul, petit modèle + RAG, petit modèle fine-tuné + RAG.

**Concepts** : transfer learning, **SFT** (supervised fine-tuning), **LoRA**, **QLoRA**, quantification, distillation, données synthétiques, overfitting, *catastrophic forgetting*.

**Outils possibles** : **Unsloth**, Hugging Face PEFT + TRL, Axolotl, LLaMA-Factory. Export : **GGUF** (llama.cpp), puis import dans Ollama.

**Choix** : **Unsloth sur Colab ou Kaggle (GPU gratuit)**, export GGUF, puis inférence locale sur ton CPU via Ollama.

**Comment ça marche :**
- **Dataset** : quelques centaines d'exemples (extrait d'article → QCM), générés par un plus gros modèle puis filtrés (c'est de la *distillation*). On garde un jeu de validation séparé.
- **LoRA** : au lieu de modifier une grande matrice de poids W, on apprend une petite correction `ΔW = B·A` de rang r (ex. 8 ou 16). On n'entraîne alors qu'environ 1 % des paramètres.
- **QLoRA** : le modèle de base est chargé en 4 bits (moins de mémoire GPU), et on entraîne les adaptateurs LoRA par-dessus.
- **Quantification** pour l'inférence : poids en 4 ou 5 bits (ex. Q4_K_M). Le modèle est environ 4 fois plus petit, avec une petite perte de qualité.
- On surveille la **loss** de validation pour détecter l'overfitting.

**Pièges :**
- Le fine-tuning injecte mal de **nouveaux faits** et peut augmenter les hallucinations. Il sert à apprendre un **comportement**, pas des connaissances.
- *Catastrophic forgetting* : le modèle perd des capacités générales s'il est trop entraîné.
- Licence : vérifier celle du modèle de base et celle du modèle générateur des données synthétiques.
- **Cas vécu — le fine-tuning en vrai** : 40 QCM distillés de qwen3:4b (Apache 2.0 ; 6 rejetés par le filtre qualité), QLoRA sur Colab T4 gratuit (1,18 % des paramètres, 1 min 05), validation 1,63 → 1,40. Mesure sans schéma imposé : QCM conforme **0/4 (base) → 3/4 (fine-tuné)**… mais un QCM fine-tuné garde une lettre incohérente avec son explication : **le fine-tuning apprend la FORME, pas la justesse**. Puis, intégré à TransfIA avec le prompt habituel, le modèle a recopié l'exemple de style : **décalage de distribution**. Parade : **contrat de prompt** (même consigne et même format qu'à l'entraînement, consigne définie à UN seul endroit).

**À retenir :**
- Pré-entraînement → SFT → alignement (RLHF, DPO). Savoir expliquer chaque étape en une phrase.
- PEFT = adapter un modèle en entraînant peu de paramètres.
- Prompting → RAG → fine-tuning : du moins cher au plus cher, à essayer dans cet ordre.

---

### F11 — Banc d'évaluation

**Besoin** : savoir objectivement si une modification (chunking, modèle, prompt) **améliore ou dégrade** la qualité.

**Ce que fait l'appli :**
- Un **jeu de test** versionné : questions + réponse attendue + passage source.
- Une commande `make eval` qui lance toutes les configurations et produit un tableau comparatif.
- Une page d'annotation humaine (le métier note les sorties).

**Concepts** : golden dataset, données synthétiques, métriques de recherche (Recall@k, MRR, nDCG), métriques RAG (**faithfulness**, answer relevance, context precision et recall), **LLM-as-judge**, jury de modèles, **accord inter-annotateurs**, calibration juge ↔ humain, tests de non-régression.

**Outils possibles** : **Ragas**, DeepEval, TruLens, promptfoo, Phoenix Evals, from scratch.

**Choix** : **métriques codées from scratch d'abord**, puis **Ragas** pour comparer. **Jury de 2 à 3 modèles** de familles différentes via Ollama. Annotation humaine dans une page simple du front.

**Comment ça marche :**
- **Jeu de test** :
  - partie synthétique : le LLM génère une question à partir d'un passage, donc la réponse est vérifiable dans le texte ;
  - partie humaine : 10 à 50 questions écrites par le métier.
- **Métriques de recherche** : la recherche a-t-elle remonté le passage attendu dans le top k ?
- **Faithfulness** : on découpe la réponse en affirmations et on vérifie que chacune est soutenue par le contexte.
- **LLM-as-judge** : une grille explicite (exactitude 1-5, niveau adapté, utilité), avec un raisonnement avant la note.
- **Calibration** : on compare les notes du jury aux notes humaines (corrélation, kappa de Cohen). On sait alors jusqu'où faire confiance au juge automatique.

**Pièges :**
- Biais des juges : préférence pour les réponses longues, pour l'ordre de présentation, pour leur propre famille de modèle.
- Jeu de test trop facile ou trop proche des données de mise au point (c'est de l'*overfitting* sur le jeu de test).
- **Cas vécu — un jeu d'éval pour un autre corpus** : `eval retrieval` → 0,0 partout pour les 4 configurations. Ce n'est pas la recherche qui est nulle : le jeu de référence avait été écrit pour les articles d'exemple, et l'index contient les 10 vrais articles INRAE. Les scores mesuraient le **décalage**, pas la qualité. Parade : contrôle de **couverture du jeu** avant toute évaluation (part des questions dont la source attendue est indexée ; < 50 % → arrêt et explication). Leçon : **valider l'instrument de mesure avant de mesurer**.
- **Cas vécu — une moyenne sur deux populations ment** : avec 8 questions hors corpus, les refus (sans appel au LLM, ~0 token) font baisser la latence et les tokens moyens (1 338 contre 2 220). Parade : coût mesuré sur les seules questions **répondues**, et **matrice de confusion du refus** : refus à tort (utilisateur frustré) ≠ réponse à tort hors corpus (**hallucination**, plus grave). Une seule métrique « refus corrects = 0,83 » cache ces deux erreurs de coût très différent. Ce qu'elle cachait en vrai : 1 « refus à tort » qui était un bon refus (question générée « non autonome » : « …dans ce passage ? ») et **2 hallucinations** (grippe aviaire : « le traitement est un mélange d'ensilage de maïs et de ray-grass » ; ration d'autruche). Cause : le nettoyage de l'index (brique 8) avait fait remonter le score au-dessus du seuil, et le vérificateur lexical acceptait une phrase faite de mots du passage + le sujet de la question. Parade : **ancrage sur le sujet** : un mot de la question absent de TOUT le corpus (« grippe », « aviaire », « autruche ») → refus si le passage n'est pas franchement pertinent, et toute phrase qui le contient est non soutenue. Leçon : **une amélioration peut en casser une autre** ; on rejoue toute l'évaluation après chaque changement (test de non-régression). Résultat après correction : 0 hallucination hors corpus, 0 refus à tort. Mais latence 0,8 s au lieu de 22 s : les réponses venaient du **cache LLM** (mêmes prompts qu'au run précédent). Parade : une évaluation **désactive le cache** par défaut ; sinon la qualité reste juste mais les coûts (latence, énergie) sont faux.

**À retenir :**
- « On ne peut pas améliorer ce qu'on ne mesure pas. » L'éval est le cœur du travail en IA générative.
- Éval offline (jeu de test) vs online (retours utilisateurs, A/B test).

---

### F12 — Garde-fous et observabilité

**Besoin** : une appli sûre (pas de hors-sujet, pas d'injection, pas d'affirmation inventée) et **pilotable** (où est le temps perdu ? quelle étape échoue ?).

**Ce que fait l'appli :**
- **Garde-fous en entrée** : détection hors périmètre (« écris-moi un poème » → refus poli), détection de *prompt injection* (y compris dans le texte des documents).
- **Garde-fous en sortie** : citations obligatoires, vérification NLI, refus si les sources sont insuffisantes, format validé.
- **Observabilité** : chaque requête est tracée, étape par étape (rewriting, retrieval, rerank, génération, outils), avec latence, tokens et erreurs.

**Concepts** : guardrails, prompt injection (directe et indirecte), classification d'intention, tracing (spans), OpenTelemetry, métriques de latence (p50/p95), tokens/seconde, coût.

**Outils possibles :**
- Garde-fous : règles maison, NeMo Guardrails, Guardrails AI, LLM Guard, Llama Guard.
- Observabilité : **Arize Phoenix**, Langfuse, OpenLLMetry, MLflow tracing.

**Choix** : **règles maison + un classifieur d'intention** (embeddings + régression logistique, réutilisé de F3) + **Phoenix** (s'installe avec `pip`, tourne en local, compatible OpenTelemetry). Langfuse est très bien aussi, mais son auto-hébergement est plus lourd (plusieurs services Docker).

**Comment ça marche :**
- **Prompt injection indirecte** : un document contient « ignore tes instructions et… ». Parade : délimiter clairement les données dans le prompt, ne jamais donner aux outils plus de droits que nécessaire, filtrer les motifs suspects.
- **Tracing** : chaque étape ouvre un *span* (début, fin, entrées, sorties). Les spans s'emboîtent en un arbre affiché dans Phoenix.
- **Coût** : en local, il n'y a pas de coût en euros, mais on mesure le temps CPU et les tokens. C'est le même raisonnement qu'avec une API payante.

**Pièges :**
- Des garde-fous trop stricts bloquent des questions légitimes : on mesure les faux positifs.
- Ne pas logger de données personnelles (ici le corpus est public, donc le risque est faible).

**À retenir :**
- Sécurité LLM : l'OWASP Top 10 pour les applications LLM (injection, fuite de données, dépendance excessive…).
- LLMOps = observabilité + éval continue + versionnage des prompts et des modèles.

---

### F13 — Interface web (React + FastAPI)

**Besoin** : une vraie application utilisable, pas seulement un notebook.

**Ce que fait l'appli :**
- Pages : **Chat sourcé** · **Recherche** · **Générateur** · **Assistant (agent)** · **Graphe** · **Carte des thèmes** · **Évaluation**.
- Streaming des réponses, affichage des étapes de l'agent, citations cliquables.

**Concepts** : API REST, **Server-Sent Events (SSE)** pour le streaming, async Python, typage partagé front/back, conteneurisation.

**Outils possibles :**
- Backend : **FastAPI**, Flask, Django.
- Front : **React + Vite + TypeScript**, Next.js, Vue, Svelte, Gradio ou Streamlit pour les POC.
- Graphe : react-force-graph, Cytoscape.js. Graphiques : Recharts, Plotly.

**Choix** : **FastAPI + React (Vite, TypeScript)**. Node sert uniquement d'outillage (npm, Vite). Docker Compose pour lancer l'ensemble.

**Comment ça marche :**
- Le backend renvoie un flux `text/event-stream`. Chaque événement est soit un token, soit une étape d'agent, soit une citation.
- Le front lit le flux et met à jour l'affichage au fil de l'eau.
- FastAPI génère automatiquement le schéma OpenAPI, d'où l'on peut générer les types TypeScript du front.

**À retenir :**
- Le streaming améliore la **latence perçue** (temps jusqu'au premier token, TTFT) même si le temps total est inchangé.

---

### F14 — Tableau de bord sobriété (dimensionnement et frugalité)

**Besoin** : choisir la configuration **la plus légère qui atteint la qualité voulue**, et le prouver avec des chiffres.

**Ce que fait l'appli :**
- Pour chaque configuration testée (modèle, quantification, nombre de chunks, avec ou sans reranker, avec ou sans agent…), elle mesure :
  - la **qualité** (score du banc d'éval F11) ;
  - la **latence** (TTFT, p50 et p95) ;
  - les **tokens** en entrée et en sortie ;
  - la **mémoire** utilisée (RAM) ;
  - l'**énergie** consommée et l'**empreinte carbone** estimée.
- Elle affiche un graphique **qualité vs coût** et met en évidence le **front de Pareto** : les configurations qu'aucune autre ne bat à la fois en qualité et en coût.
- Elle intègre un **routeur** : le petit modèle répond par défaut, et la question ne passe au plus gros que si c'est nécessaire (question complexe, ou confiance faible).

**Concepts** : dimensionnement (mémoire, débit, latence), frugalité, quantification, routing et cascade de modèles, cache, mesure d'énergie, optimisation multi-critère.

**Outils possibles :**
- Énergie : **CodeCarbon**, EcoLogits (estimation pour les LLM appelés par API), Scaphandre.
- Mesures système : `psutil`, `time.perf_counter`, métriques renvoyées par Ollama (tokens/s, durées).
- Cache : dictionnaire ou SQLite (cache exact), cache sémantique basé sur les embeddings.

**Choix** : **CodeCarbon** + métriques Ollama + `psutil`, résultats stockés avec ceux du banc d'éval, graphique dans le front.

**Comment ça marche :** voir la section 11.1 pour les formules de dimensionnement.

**À retenir :**
- « Le meilleur modèle » n'existe pas : il existe le **plus petit modèle suffisant** pour un besoin et un niveau de qualité donnés.
- La qualité se mesure **toujours avec** le coût, la latence et l'énergie.

---

## 4. Focus : tools vs skills vs MCP

C'est l'un des sujets les plus actuels. On les implémente tous les trois dans TransfIA.

### Tool (outil)

- **Définition** : une **fonction exécutable** que le LLM peut demander d'appeler. Elle **agit** ou **renvoie des données**.
- Le LLM reçoit pour chaque outil : un nom, une description, un JSON schema des paramètres.
- Le LLM ne l'exécute pas lui-même : il produit une **demande d'appel** structurée, que **ton code** exécute, puis tu lui renvoies le résultat.
- Exemple : `search_corpus(query="mammites", espece="bovin", top_k=5)`.

```python
tool_schema = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "Cherche des passages dans les articles du corpus. À utiliser pour toute question factuelle sur l'élevage.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "espece": {"type": "string", "enum": ["bovin", "porcin", "ovin", "caprin", "volaille", "poisson"]},
                "top_k": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }
}
```

### Skill (compétence)

- **Définition** : un **savoir-faire packagé** : instructions, exemples, templates, éventuellement des scripts. Il **change la manière** dont le modèle réalise une tâche.
- Format (convention popularisée par les *Agent Skills*) : un dossier par skill, avec un `SKILL.md` qui contient un en-tête `name` + `description`, puis le corps des instructions.
- **Progressive disclosure** (chargement progressif), le principe clé :
  1. **Niveau 1** : seules les descriptions courtes de toutes les skills sont dans le prompt (peu de tokens).
  2. **Niveau 2** : quand une skill est pertinente, le modèle charge son `SKILL.md` complet (via un tool `load_skill`).
  3. **Niveau 3** : la skill peut pointer vers des fichiers annexes (exemples, grilles) chargés seulement si nécessaire.
- Exemple dans TransfIA :
  ```
  skills/
    quiz-qcm/            SKILL.md · exemples.json · grille-qualite.md
    fiche-bac-pro/       SKILL.md · template.md.j2
    cas-pratique/        SKILL.md · exemples.md
    vulgarisation/       SKILL.md
    relecture-scientifique/ SKILL.md
  ```
  ```markdown
  ---
  name: quiz-qcm
  description: Créer un QCM pédagogique sourcé à partir d'articles. À utiliser quand l'utilisateur demande un quiz ou des questions d'entraînement.
  ---
  # Créer un QCM
  1. Chercher les passages avec search_corpus.
  2. Une question = une notion clé, 4 choix, 1 seule bonne réponse.
  3. Distracteurs plausibles (erreurs fréquentes), jamais absurdes.
  4. Explication courte + citation obligatoire.
  5. Adapter le vocabulaire au niveau (voir grille-qualite.md).
  ```

### MCP (Model Context Protocol)

- **Définition** : un **protocole standard** pour exposer des tools (et des ressources et des prompts) d'un **serveur** vers n'importe quel **client** LLM compatible.
- 3 primitives :
  - **tools** : des actions à appeler ;
  - **resources** : des données à lire (ex. un article) ;
  - **prompts** : des modèles de prompts réutilisables.
- Dans TransfIA : le serveur `transfia-mcp` expose `search_corpus`, `read_article`, `get_entities`. Ton agent l'utilise, **mais aussi n'importe quel autre client MCP**.

### Tableau comparatif

| | Tool | Skill | MCP |
|---|---|---|---|
| Nature | Fonction (code) | Instructions et savoir-faire (texte + annexes) | Protocole de transport |
| Répond à | « Que peut **faire** le modèle ? » | « **Comment** bien faire telle tâche ? » | « **Comment brancher** des outils partout ? » |
| Effet | Action ou données | Change le comportement | Interopérabilité |
| Coût en contexte | Schéma toujours présent | Description seule, puis chargement à la demande | Dépend des outils exposés |
| Exemple TransfIA | `search_corpus` | `quiz-qcm` | serveur `transfia-mcp` |

> **En une phrase** : les **tools** donnent des **mains** au modèle, les **skills** lui donnent du **métier**, et **MCP** est la **prise standard** pour brancher les outils.

---

## 5. Focus : le prompt engineering traité comme du code

**Concepts** : prompts versionnés, templates, tests automatisés, non-régression, optimisation automatique de prompts.

- **Prompts dans des fichiers** (`prompts/*.j2`), jamais en dur dans le code. Versionnés dans git.
- **Tests** : pour chaque prompt, des cas d'entrée et des assertions :
  - format valide (schéma Pydantic) ;
  - citations présentes ;
  - refus attendu sur une question hors sujet ;
  - note minimale au LLM-as-judge.
- **Outils possibles** : pytest (choix principal), **promptfoo** (outil Node en ligne de commande, très utilisé pour comparer prompts et modèles), DSPy (optimisation automatique des prompts).
- **Choix** : **pytest** pour les tests, **promptfoo** pour comparer plusieurs prompts ou modèles côte à côte, et un essai de **DSPy** en bonus pour découvrir l'optimisation automatique.
- **Techniques implémentées et comparées** : zero-shot, few-shot, chain-of-thought, rôle (system prompt), délimiteurs pour les données, self-consistency (plusieurs réponses + vote).

**À retenir :**
- Un prompt est un artefact de production : il se versionne, se teste et se surveille.
- Changer de modèle = refaire passer les tests de prompts.

---

## 6. Récapitulatif de la stack

| Brique | Choix (open source, CPU) | Alternatives |
|---|---|---|
| Langage | Python 3.12 + TypeScript | — |
| LLM local | **Ollama** + modèles 3-8B quantifiés (Qwen, Gemma, Llama, Mistral) | llama.cpp, LM Studio |
| VLM | Petit VLM via Ollama | SmolVLM, Moondream |
| Parsing PDF | **Docling** | Unstructured, Marker, PyMuPDF |
| NER | **GLiNER** | spaCy, LLM |
| ML classique | **scikit-learn**, UMAP, HDBSCAN | BERTopic |
| Embeddings | **multilingual-e5-small**, BGE-M3 | nomic-embed |
| Lexical | **bm25s** | rank_bm25, Tantivy |
| Base vectorielle | **Qdrant (local)** | LanceDB, Chroma, FAISS |
| Reranker | **bge-reranker-v2-m3** | cross-encoder mMARCO |
| Graphe | **NetworkX** | Neo4j, LightRAG, MS GraphRAG |
| Structured output | **Pydantic** + JSON schema Ollama | Instructor, Outlines |
| Agents | **From scratch → LangGraph** | smolagents, PydanticAI, CrewAI |
| MCP | **SDK Python officiel** | — |
| Éval | **From scratch → Ragas** | DeepEval, TruLens |
| Tests de prompts | **pytest + promptfoo** | DSPy |
| Observabilité | **Phoenix** | Langfuse, MLflow |
| Fine-tuning | **Unsloth (Colab/Kaggle)** → GGUF | PEFT/TRL, Axolotl |
| Backend | **FastAPI** | Flask |
| Front | **React + Vite + TS** | Gradio, Streamlit |
| Données | **SQLite** | DuckDB |
| Déploiement | **Docker Compose** | Hugging Face Spaces |
| Mesure d'énergie (F14) | **CodeCarbon** + métriques Ollama + psutil | EcoLogits, Scaphandre |

> Les versions exactes des modèles et des librairies seront vérifiées au démarrage : l'écosystème évolue chaque mois.

---

## 7. Structure du repo

```
transfia/
├── README.md                 # pitch, démo, installation, licences
├── LICENSE                   # MIT ou Apache 2.0
├── docker-compose.yml
├── data/                     # ignoré par git (sauf catalogue d'exemple)
├── docs/
│   ├── concepts/             # une note par concept : quoi, pourquoi, observé
│   └── decisions/            # ADR : choix techniques et justification
├── backend/
│   ├── ingestion/            # F1, F2, F3
│   ├── retrieval/            # F4
│   ├── rag/                  # F5
│   ├── graph/                # F6
│   ├── generation/           # F7
│   ├── agents/               # F8, F9
│   ├── mcp_server/           # serveur MCP
│   ├── guardrails/           # F12
│   ├── api/                  # FastAPI
│   └── observability/
├── skills/                   # une skill = un dossier avec SKILL.md
├── prompts/                  # templates Jinja2 versionnés
├── eval/                     # jeu de test, métriques, rapports (F11)
├── finetuning/               # notebooks Colab (F10)
├── notebooks/                # explorations from scratch
├── frontend/                 # React + Vite + TS
└── tests/                    # pytest (code + prompts)
```

### 7.1 Structure réellement implémentée (écart avec le plan ci-dessus)

À l'implémentation, le dossier `backend/` est devenu **`src/transfia/`**. Les briques sont les mêmes, seul l'emplacement change.

**Pourquoi** : c'est la convention standard d'un paquet Python (« src layout »).
- Le projet s'installe avec `pip install -e .` et fournit la commande `transfia`.
- Les tests vérifient le code **installé**, pas des fichiers « qui marchent parce qu'on est dans le bon dossier ».
- Le nom du paquet (`transfia`) apparaît dans les imports : `from transfia.rag.pipeline import RAGPipeline`.

**Correspondance plan → code** :

| Prévu dans le plan | Réalisé | Fonctionnalité |
|---|---|---|
| `backend/ingestion/` | `src/transfia/ingestion/` | F1, F2, F3 |
| `backend/retrieval/` | `src/transfia/retrieval/` + `src/transfia/embeddings/` | F4 |
| `backend/rag/` | `src/transfia/rag/` | F5 |
| `backend/graph/` | `src/transfia/graph/` | F6 |
| `backend/generation/` | `src/transfia/generation/` + `src/transfia/skills_runtime/` | F7 |
| `backend/agents/` | `src/transfia/agents/` | F8, F9 |
| `backend/mcp_server/` | `src/transfia/mcp_server/` | MCP |
| `backend/guardrails/` | `src/transfia/guardrails/` | F12 |
| `backend/api/` | `src/transfia/api/` | F13 |
| `backend/observability/` | `src/transfia/observability/` | F12, F14 |
| `eval/` | `src/transfia/evaluation/` (code) + `data/eval/` (rapports) | F11 |
| `notebooks/` | `src/transfia/foundations/` (modules exécutables) | Fondations |
| — (non prévu) | `src/transfia/llm/` (clients LLM, cache, routeur) | F5, F14 |
| — (non prévu) | `src/transfia/ml/` (classification, clustering) | F3 |
| — (non prévu) | `src/transfia/sobriety/` (dimensionnement, Pareto) | F14 |
| — (non prévu) | `src/transfia/core/` (modèles, ports, prompts) | Socle |
| — (non prévu) | `src/transfia/cli.py`, `config.py`, `container.py` | Socle |
| — (non prévu) | `config/` (YAML local, light, test, GCP, Azure) | Ch. 12 |
| — (non prévu) | `models/` (copie locale des modèles, non commitée) | Hors ligne |

`prompts/`, `skills/`, `frontend/`, `finetuning/`, `tests/`, `docs/` sont restés à la racine, comme prévu.
Le parcours des fichiers commande par commande est décrit dans `docs/PARCOURS_DU_CODE.md`.

---

## 8. Plan sur 4 semaines (~40 h)

| Semaine | Fonctionnalités | Concepts phares |
|---|---|---|
| **0** | F1, structure du repo | Data engineering, licences |
| **1** | F2, F3, F4 | Parsing, OCR/VLM, chunking, NER, ML classique, embeddings, hybride, rerank |
| **2** | F5, F11 (première version), F12 (traces) | RAG, HyDE, citations, éval, LLM-as-judge, observabilité |
| **3** | F7, F8, F9, MCP, skills | Structured outputs, tool calling, skills, agents, multi-agents |
| **4** | F6, F10, F13, finalisation | GraphRAG, LoRA, React streaming, README, démo |

**Ordre de priorité si le temps manque** : F1 → F4 → F5 → F11 → F8 → F7 → F13 → F9 → F12 → F3 → F6 → F2 (VLM) → F10.

---

## 11. Travailler sereinement avec l'IA

> Ce qui fait la différence entre une démo et un système fiable. Chaque point est appliqué dans TransfIA.

### 11.1 Dimensionnement et frugalité

**L'échelle de frugalité : toujours se poser les questions dans cet ordre.**
1. A-t-on vraiment besoin d'IA ? Une règle, une requête SQL, une regex suffisent-elles ?
2. Le **ML classique** suffit-il ? (ex. la classification de F3 en TF-IDF + régression logistique)
3. Un **petit modèle** spécialisé (encodeur, NER, embeddings) ?
4. Un **petit LLM** + un bon RAG ?
5. Seulement ensuite : un gros LLM, des agents, du fine-tuning.

**Mémoire nécessaire pour un LLM**
- **Poids** ≈ nombre de paramètres × octets par paramètre.

  | Précision | Octets / paramètre | Modèle 3B | Modèle 7-8B |
  |---|---|---|---|
  | FP32 | 4 | 12 Go | 28-32 Go |
  | FP16 / BF16 | 2 | 6 Go | 14-16 Go |
  | INT8 | 1 | 3 Go | 7-8 Go |
  | 4 bits (ex. Q4_K_M) | ~0,6 | ~2 Go | ~4,5-5 Go |

- **KV cache** (mémoire de l'attention pendant la génération) : il grandit **linéairement avec la longueur du contexte**. Un contexte de 32k tokens peut ajouter plusieurs Go. Ne pas allouer plus de contexte que nécessaire.
- **Règle pratique** : RAM à prévoir = poids + KV cache + 1 à 2 Go de marge, plus le reste de l'appli (embeddings, Qdrant, front).

**Vitesse sur CPU**
- Deux phases :
  - **prefill** : lecture du prompt, limitée par le **calcul**. C'est elle qui détermine le TTFT.
  - **decode** : génération token par token, limitée par la **bande passante mémoire**.
- **Ordre de grandeur du decode** : tokens/s ≈ bande passante mémoire ÷ taille du modèle en mémoire.
  - Exemple : RAM à ~40 Go/s et modèle de 4,5 Go → au mieux ~9 tokens/s.
  - Conséquence : **diviser la taille du modèle par 2 double à peu près la vitesse**. D'où l'intérêt de la quantification.
- **Latence totale** ≈ TTFT + (tokens générés ÷ débit). Un prompt long retarde le premier token ; une réponse longue allonge tout le reste.

**Dimensionner l'index vectoriel**
- Taille ≈ nombre de chunks × dimension × 4 octets (float32).
- Exemple : 100 000 chunks × 768 dimensions × 4 o ≈ **307 Mo**, plus l'index HNSW.
- Leviers : modèle d'embedding plus petit (384 dimensions), quantification des vecteurs (int8 : ÷4 ; binaire : ÷32, avec un reranking derrière).

**Estimer un coût API** (TransfIA tourne en local, mais le calcul vaut pour tout passage au cloud)
- Coût/jour = requêtes/jour × (tokens d'entrée × prix d'entrée + tokens de sortie × prix de sortie).
- Les tokens de sortie coûtent généralement **plus cher** que ceux d'entrée.
- En RAG, **le contexte injecté domine souvent le coût** : 10 chunks de 500 tokens = 5 000 tokens par question.

**Mesurer l'impact environnemental**
- Énergie (kWh) = puissance × temps ; carbone = énergie × intensité carbone de l'électricité (faible en France, très variable selon les pays).
- Outils : **CodeCarbon** (mesure du code qui tourne), **EcoLogits** (estimation pour des LLM appelés par API).
- Référentiel français : **AFNOR SPEC 2314** sur l'IA frugale (2024).
- Ne pas oublier : la **fabrication du matériel** (GPU, serveurs) pèse lourd dans l'empreinte totale, pas seulement l'électricité.

**Leviers de frugalité**
- Le plus petit modèle suffisant + quantification.
- **Routing / cascade** : petit modèle par défaut, gros modèle seulement si nécessaire.
- **Cache** :
  - exact : même question → même réponse ;
  - sémantique : question très proche → réponse réutilisée, au-delà d'un seuil de similarité.
- **Réduire le contexte** : un meilleur retrieval et un reranker permettent d'envoyer 3 chunks au lieu de 10.
- **Limiter la longueur de sortie** (`max_tokens`) et demander des réponses concises.
- **Un workflow plutôt qu'un agent** quand les étapes sont connues : moins d'appels au LLM.
- **Traitements batch** (indexation, extraction) plutôt qu'en temps réel ; indexation **incrémentale**.
- **Distillation** : faire apprendre à un petit modèle ce que fait un gros, sur une tâche précise.

### 11.2 Choisir un modèle

- **Les benchmarks publics** (MMLU, leaderboards…) sont un premier filtre, **pas une preuve** :
  - *contamination* : le modèle a pu voir les questions pendant son entraînement ;
  - *loi de Goodhart* : quand une mesure devient un objectif, elle cesse d'être une bonne mesure ;
  - ils sont souvent en anglais, sur des tâches éloignées de la tienne.
- **La seule évaluation qui compte : la tienne**, sur tes données et ta tâche (F11).
- Critères de choix : qualité sur ta tâche, langue (français), taille et vitesse, fenêtre de contexte, support du tool calling et des sorties structurées, **licence**.

### 11.3 Évaluation : compléments à F11

- **Eval-driven development** : on écrit l'évaluation **avant** d'optimiser, comme des tests en TDD.
- **Analyse d'erreurs** :
  - lire réellement 30 à 50 sorties ;
  - classer les erreurs par type (mauvais passage retrouvé, mauvaise lecture du passage, hallucination, format…) ;
  - s'attaquer d'abord à la catégorie la plus fréquente.
  - C'est souvent plus utile que n'importe quelle métrique.
- **Évaluer chaque étage séparément** : le retrieval (le bon passage est-il remonté ?), puis la génération (la réponse est-elle fidèle au passage ?). Sinon on ne sait pas où corriger.
- **Non-déterminisme** : un LLM ne répond pas toujours pareil, même à température 0. Il faut lancer l'éval plusieurs fois et regarder la variance.
- **Significativité** : sur 50 questions, l'incertitude sur un taux de réussite est d'environ ±14 points (intervalle à 95 %). Un écart de 3 points entre deux configurations **ne prouve rien**. Il faut plus de questions, ou un test statistique.
- **Évaluation multi-critère** : qualité + latence + coût + énergie (F14). Pas de qualité sans son coût.
- **Offline vs online** :
  - offline = jeu de test, avant mise en ligne ;
  - online = retours utilisateurs (pouce haut/bas), taux de reformulation, A/B tests, détection de **dérive** (les vraies questions changent avec le temps).
- **Évaluation de sécurité** : red teaming, questions pièges, tentatives d'injection, mesure du taux de refus injustifiés.

### 11.4 Cadre légal, éthique et licences

**AI Act (règlement UE 2024/1689)**
- **4 niveaux de risque** : inacceptable (interdit), haut risque, risque limité (obligations de transparence), risque minimal.
- **Calendrier** :
  - application progressive depuis 2024 ;
  - l'**omnibus numérique IA** (règlement UE 2026/1744), en vigueur depuis le 27 juillet 2026, a reporté les obligations des systèmes à haut risque de l'annexe III au **2 décembre 2027**, et celles de l'annexe I (produits réglementés) au **2 août 2028**.
- **Le report ne concerne que le haut risque.** Les **interdictions**, les obligations de **transparence** (art. 50) et de **maîtrise de l'IA** par le personnel (art. 4) restent applicables.
- **L'éducation fait partie des domaines à haut risque de l'annexe III** (ex. un système qui évalue ou oriente des apprenants).
  - TransfIA **génère des supports** pour les formateurs ; il n'évalue pas les élèves. Il relève donc a priori de la transparence, pas du haut risque.
  - Si on ajoutait une **notation automatique des apprenants**, on changerait de catégorie. C'est un bon exemple de qualification du risque.
- **Transparence** : indiquer clairement qu'un contenu est **généré par IA**.

**RGPD**
- Minimisation : ne traiter que les données personnelles nécessaires.
- Ne pas envoyer de données personnelles dans des prompts vers des API tierces sans base légale ni garanties.
- Analyse d'impact (AIPD) pour les traitements à risque. La CNIL publie des recommandations dédiées à l'IA.
- TransfIA : corpus scientifique public, sans données personnelles hormis les noms d'auteurs (déjà publics). Il faut quand même éviter de stocker les questions des utilisateurs sans le dire.

**Biais et équité**
- Les modèles reproduisent les biais de leurs données (langue, culture, représentation).
- Dans TransfIA : si certaines espèces ou filières sont surreprésentées dans le corpus, les réponses le seront aussi. Il faut le mesurer (F3) et l'afficher.

**Documentation responsable**
- **Model card** : à quoi sert le modèle, ses limites, comment il a été évalué.
- **Datasheet du dataset** : origine, licence, composition, biais connus.
- Les deux vont dans `docs/` du repo.

**Licences (3 niveaux à distinguer)**
- **Code** : MIT, Apache 2.0…
- **Données** : CC BY (attribution obligatoire), CC BY-NC (pas d'usage commercial)…
- **Modèles** : Apache 2.0 ou MIT (vraiment libres) **vs licences « open-weights » propriétaires**, avec des restrictions d'usage (ex. certaines licences communautaires).
- **Open source ≠ open weights** : un modèle « open weights » publie ses poids, mais pas forcément ses données ni son code d'entraînement, et sa licence peut restreindre l'usage.

### 11.5 Sécurité : compléments à F12

- **Moindre privilège pour les outils** : un agent n'a accès qu'aux outils et aux données strictement nécessaires (lecture seule si possible).
- **Code généré par un LLM** : toujours exécuté dans un **bac à sable** (conteneur isolé), jamais directement sur la machine.
- **Exfiltration** : une injection peut pousser un agent à envoyer des données via un outil (URL, e-mail). Il faut contrôler les destinations autorisées.
- **Secrets** : jamais dans le code ni dans les prompts. On utilise des variables d'environnement, un fichier `.env` ignoré par git, et un scan des secrets.
- **Chaîne d'approvisionnement** :
  - préférer le format **safetensors** aux fichiers pickle, qui peuvent exécuter du code au chargement ;
  - se méfier de `trust_remote_code=True` ;
  - figer les versions des dépendances.
- **Serveurs MCP tiers** : ce sont du code et des descriptions d'outils qui entrent dans ton contexte. Il faut les auditer comme une dépendance.

### 11.6 Fiabilité et ingénierie

- **Tout versionner** : code, prompts, skills, modèle (nom + quantification), modèle d'embedding, index, jeu de test. Un résultat d'éval sans ces versions est inexploitable.
- **Reproductibilité** : versions figées (fichier de lock), configuration centralisée, seeds, et la configuration enregistrée avec chaque résultat.
- **CI avec seuils d'éval** (*eval gate*) : une modification qui fait baisser la qualité sous un seuil bloque le merge.
- **Résilience** :
  - timeouts sur chaque appel ;
  - retries avec attente exponentielle ;
  - **fallback** : modèle de secours, ou réponse « dégradée » honnête ;
  - validation de toutes les sorties du LLM avant usage.
- **Logs structurés** et traces (F12) : pouvoir rejouer une requête problématique.
- **Séparer** l'expérimentation (notebooks) du code applicatif (modules testés).

### 11.7 L'humain dans la boucle

- **Supervision humaine** : l'IA propose, l'humain valide, surtout pour un contenu diffusé à des apprenants.
- **UX de l'incertitude** : afficher les sources, signaler les affirmations non vérifiées, oser afficher « je ne sais pas ».
- **Biais d'automatisation** : les utilisateurs font trop confiance à une machine qui paraît sûre d'elle. Le design doit inviter à vérifier.
- **Formation des utilisateurs** : c'est aussi l'obligation de maîtrise de l'IA de l'art. 4 de l'AI Act.
- **Mesurer la valeur métier**, pas seulement la qualité technique : temps gagné par le formateur, taux d'adoption, contenus réellement utilisés.

### 11.8 Checklist « IA sereine » avant de livrer

- [ ] Le besoin justifie l'IA (échelle de frugalité, 11.1)
- [ ] Le plus petit modèle suffisant est retenu, chiffres à l'appui (F14)
- [ ] Un jeu de test versionné existe, avec une partie validée par des humains (F11)
- [ ] Le retrieval et la génération sont évalués séparément
- [ ] Les sorties sont sourcées et vérifiées, et le système sait dire « je ne sais pas » (F5)
- [ ] Les garde-fous d'entrée et de sortie sont testés, y compris contre l'injection (F12)
- [ ] Traces, latence, tokens et énergie sont mesurés (F12, F14)
- [ ] Les licences (code, données, modèles) sont vérifiées et documentées
- [ ] RGPD et AI Act : catégorie de risque identifiée, mention « généré par IA » présente
- [ ] Model card et datasheet rédigées
- [ ] Versions figées, CI avec seuil d'éval, timeouts et fallback en place
- [ ] Un humain valide avant toute diffusion à des apprenants

---

## 12. TransfIA dans le cloud : architecture portable

> **Objectif** : montrer une compétence d'architecte. Le même système tourne **en local et open source** (souveraineté, frugalité) **ou** sur un cloud public, en changeant la configuration, pas le code.

### 12.1 Pourquoi c'est un bon exercice d'architecte

- **Le discours d'architecte** devient : « J'ai conçu en on-premise et open source pour des raisons de souveraineté. Je connais les équivalents cloud et je sais quand choisir l'un ou l'autre. »
- **Ça apprend les concepts propres au cloud** : identités et droits (IAM), régions, serverless, facturation à l'usage, quotas.
- **Ça apprend à comparer** local vs cloud avec des chiffres : qualité, latence, coût, énergie (F11, F14).

### 12.2 Tableau d'équivalences

Chaque brique de TransfIA a son équivalent cloud (les noms des services changent souvent, à vérifier) :

| Brique TransfIA | Azure | AWS | GCP |
|---|---|---|---|
| LLM local (Ollama) | Azure OpenAI / AI Foundry | Bedrock | Vertex AI |
| Base vectorielle (Qdrant) | Azure AI Search | OpenSearch | Vertex Vector Search |
| Parsing PDF (Docling) | Document Intelligence | Textract | Document AI |
| Agents (LangGraph) | Agent Service de Foundry | Bedrock Agents | Agent Builder |
| Conteneurs (Docker) | Container Apps | ECS / Fargate | Cloud Run |
| Traces (Phoenix) | Application Insights | CloudWatch | Cloud Trace |

**Compléments** (les briques « socle » qu'un architecte doit aussi savoir nommer) :

| Brique | Azure | AWS | GCP |
|---|---|---|---|
| Stockage de fichiers (PDF) | Blob Storage | S3 | Cloud Storage |
| Base NoSQL (avec recherche vectorielle) | Cosmos DB | DynamoDB | Firestore |
| Entrepôt de données (SQL analytique) | Fabric / Synapse | Redshift | BigQuery |
| Fonctions serverless | Functions | Lambda | Cloud Run functions |
| Secrets | Key Vault | Secrets Manager | Secret Manager |
| Identités et droits | Entra ID + RBAC | IAM | IAM |
| CI/CD et images | DevOps / GitHub Actions + Container Registry | CodeBuild + ECR | Cloud Build + Artifact Registry |
| Notebooks / ML managé | Azure Machine Learning | SageMaker | Vertex AI Workbench |

### 12.3 La couche d'abstraction (le cœur de la portabilité)

- **Principe** : le code métier ne dépend que d'**interfaces**. Chaque fournisseur (local, GCP, Azure) les implémente.
- **Pour les LLM : LiteLLM**. Une seule API pour Ollama, Gemini, Azure OpenAI, Bedrock… Changer de fournisseur = changer une chaîne de configuration.
- **Pour le reste** : des interfaces Python simples (`Protocol`).

```python
# backend/core/ports.py
from typing import Protocol

class VectorStore(Protocol):
    def upsert(self, chunks: list["Chunk"]) -> None: ...
    def search(self, vector: list[float], k: int, filters: dict | None = None) -> list["Hit"]: ...

class DocumentParser(Protocol):
    def parse(self, pdf_path: str) -> "ParsedDocument": ...

class FileStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
```

```yaml
# config/local.yaml          # config/gcp.yaml               # config/azure.yaml
llm: ollama/qwen-4b          # llm: gemini/gemini-flash-lite  # llm: azure/gpt-mini
vector_store: qdrant         # vector_store: bigquery        # vector_store: azure_ai_search
parser: docling              # parser: docling               # parser: azure_doc_intelligence
file_store: local            # file_store: gcs               # file_store: blob
```

- Nom du pattern : **architecture hexagonale** (ports et adaptateurs). Un motif d'architecture central.
- Bonus : le banc d'éval (F11) tourne sur les 3 configurations et produit un **comparatif chiffré local vs GCP vs Azure**.

### 12.4 Le marché : pourquoi regarder GCP

- **GCP est le cloud qui grandit le plus vite** : au trimestre clos fin juin 2026, ses revenus ont progressé de 82 % sur un an. Sa part de marché est passée de 13 % à 15 % en un an, pendant qu'AWS passait de 30 % à 28 % et qu'Azure restait à 20 % (source : The Motley Fool, données Synergy Research).
- **Un atout différenciant** : GCP est moins répandu que ses concurrents dans les compétences du marché, et Google est très fort en IA (Gemini, Vertex AI).
- **Nuance** : dans le conseil et les ESN en France, **Azure reste très présent** dans les offres (beaucoup de clients sont déjà équipés Microsoft). Le bon combo : **GCP comme atout différenciant + les bases d'Azure**. D'où les chapitres 13 et 14.

### 12.5 Ce qui est gratuit (état octobre 2026, à revérifier avant de commencer)

| | GCP | Azure | AWS |
|---|---|---|---|
| **Crédit de départ** | 300 $ pour les nouveaux clients | 200 $ à utiliser **en 30 jours** | Jusqu'à 200 $ (100 $ à l'inscription + 100 $ en utilisant certains services) |
| **Durée** | Pas de facturation tant que tu n'actives pas le compte payant | 30 jours, puis passage au paiement à l'usage pour continuer | Plan gratuit de 6 mois (ou moins si les crédits sont épuisés) |
| **Gratuit en permanence (extraits)** | Cloud Run (2 M de requêtes/mois), Cloud Storage (5 Go), Firestore (1 Go), BigQuery (1 To de requêtes/mois), Cloud Build (120 min/jour), Agent Engine (50 h de vCPU/mois) | AI Search (50 Mo, 3 index), Document Intelligence (500 pages/mois), Container Apps (180 000 vCPU-s, 2 M de requêtes), Functions (1 M de requêtes), Cosmos DB (25 Go) | Plus de 30 services gratuits en permanence |
| **LLM gratuit** | **Oui** : API Gemini (Google AI Studio), niveau gratuit limité en requêtes par jour | **Non** : Azure OpenAI seulement avec le crédit | **Non** : Bedrock avec les crédits |
| **Verdict pour toi** | ✅ Le plus généreux pour l'IA | ✅ Le plus généreux en services IA « socle » gratuits (recherche, parsing) | ⏸️ À garder pour plus tard |

**Règles de prudence (valables pour les 3)**
- Une **carte bancaire** est généralement demandée à l'inscription pour vérifier l'identité.
- Créer **dès le premier jour une alerte de budget** (ex. 5 €) et vérifier la facture chaque semaine.
- **Ne jamais exposer publiquement une API qui appelle un LLM payant sans authentification** : quelqu'un pourrait la saturer et faire exploser la facture.
- **Supprimer les ressources** après chaque essai (ou tout mettre dans un projet / groupe de ressources dédié, supprimable d'un coup).
- Les secrets (clés d'API) ne vont **jamais dans git** (11.5).

---

## 13. Implémentation sur Google Cloud (GCP)

### 13.1 Les briques retenues (en privilégiant le gratuit)

| Brique TransfIA | Service GCP choisi | Gratuit ? | Pourquoi ce choix |
|---|---|---|---|
| LLM | **API Gemini** (modèles Flash / Flash-Lite) via LiteLLM | ✅ Niveau gratuit (quotas journaliers) | Seul LLM cloud gratuit des 3 |
| LLM (alternative) | **Vertex AI** (Gemini) | 💳 Sur le crédit de 300 $ | Version « entreprise » : la plus demandée en mission |
| Modèle ouvert hébergé | **Gemma** (modèle open-weights de Google) via l'API | ✅ Listé dans le niveau gratuit | Fait le pont avec l'open source |
| Embeddings | **Gemini Embedding** via l'API | ✅ Niveau gratuit | Comparable au e5 local dans F11 |
| Stockage des PDF | **Cloud Storage** | ✅ 5 Go | Le corpus tient largement |
| Base vectorielle | **BigQuery** (fonction `VECTOR_SEARCH`) ou **Firestore** (recherche vectorielle) | ✅ 1 To de requêtes/mois (BigQuery), 1 Go (Firestore) | ⚠️ Éviter Vertex AI Vector Search : son point d'accès est facturé en continu, même sans requête |
| Parsing PDF | **Docling dans un job Cloud Run** (comparé à **Document AI** sur quelques pages) | ✅ Docling / 💳 Document AI sur crédit | Comparer open source vs service managé |
| API FastAPI | **Cloud Run** | ✅ 2 M de requêtes/mois | Serverless, s'éteint à zéro quand personne ne l'utilise |
| Front React | **Cloud Run** (ou Firebase Hosting) | ✅ | Même déploiement que l'API |
| Agents | **LangGraph sur Cloud Run**, puis essai de **Vertex AI Agent Engine** + **ADK** (framework d'agents open source de Google) | ✅ Agent Engine : 50 h de vCPU/mois | Comparer « mon agent » vs l'agent managé |
| Serveur MCP | **Cloud Run** | ✅ | Le même conteneur qu'en local |
| Observabilité | **Cloud Trace + Cloud Logging** via OpenTelemetry | ✅ Allocation mensuelle gratuite | Le même instrumentage OpenTelemetry que Phoenix |
| Secrets | **Secret Manager** | ✅ Petit volume gratuit (à vérifier) | Clé API Gemini hors du code |
| CI/CD | **Cloud Build + Artifact Registry** | ✅ 120 min/jour ; ⚠️ 0,5 Go d'images seulement | Une image avec Docling dépasse 0,5 Go : léger surcoût pris sur le crédit |

### 13.2 Deux points d'attention importants

- **Niveau gratuit de l'API Gemini = données réutilisées.** Google indique que les contenus envoyés au niveau gratuit servent à améliorer ses produits. Pour TransfIA, ce n'est pas un problème (corpus public CC BY), mais **jamais de données sensibles** sur ce niveau.
- **Utilisateurs européens.** Les conditions de l'API Gemini réservent le niveau payant aux applications **proposées à des utilisateurs de l'Espace économique européen** (dont la France). Donc :
  - pour **tes essais personnels de développement** : niveau gratuit ;
  - pour une **démo publique** ouverte à d'autres personnes : passer par Vertex AI (sur le crédit) ou rester sur la version locale Ollama.
  - C'est un excellent exemple concret de **contrainte juridique qui change l'architecture**.
- **Précision (conditions relues le 29/10/2026)** : pour un utilisateur situé **dans l'EEE, en Suisse ou au Royaume-Uni**, les conditions de traitement des données du niveau **payant** s'appliquent à tous les services, **même gratuits**. La règle « données réutilisées » du premier point vise donc surtout les utilisateurs hors de ces pays. La prudence reste la même : aucune donnée sensible ou personnelle sur le niveau gratuit. 
- **Sans carte bancaire (choix du projet)** : seule l'API Gemini est utilisable (le niveau gratuit permanent de Google Cloud exige lui aussi un compte de facturation). Le périmètre réalisé est donc : même code sur Gemini via `config/gcp.yaml`, banc d'essai comparatif sur un échantillon ; l'architecture Cloud Run / BigQuery (13.3) reste conçue, non déployée.
- **Quotas** : le niveau gratuit est limité à quelques dizaines (modèles Flash) ou centaines (modèles Flash-Lite) de requêtes par jour, d'après les retours de développeurs (Google affiche les limites exactes de chaque projet dans AI Studio). Il faut donc **mettre en cache** et lancer le banc d'éval sur un petit échantillon.

### 13.3 Architecture GCP de TransfIA

```
  Utilisateur
      │
      ▼
 ┌──────────────┐     ┌─────────────────────────┐
 │ Cloud Run    │────▶│ Cloud Run : API FastAPI │──────▶ API Gemini / Vertex AI (LLM, embeddings)
 │ Front React  │     │ (+ agents LangGraph)    │──────▶ BigQuery VECTOR_SEARCH (passages)
 └──────────────┘     └───────────┬─────────────┘──────▶ Secret Manager (clé API)
                                  │ OpenTelemetry
                                  ▼
                        Cloud Trace / Cloud Logging

 Batch d'ingestion :
 Cloud Storage (PDF) ──▶ Job Cloud Run (Docling + embeddings) ──▶ BigQuery (chunks + vecteurs)

 Essai agent managé : Vertex AI Agent Engine + ADK  ◀──▶  serveur MCP sur Cloud Run
```

### 13.4 Étapes d'implémentation

1. **Compte et garde-fous** : créer un projet dédié `transfia`, activer l'essai gratuit, poser une **alerte de budget**, choisir une région.
2. **LLM via LiteLLM** : clé API Gemini dans Secret Manager ; `config/gcp.yaml` ; lancer le banc d'éval F11 en local avec Gemini → **premier comparatif Ollama vs Gemini**.
3. **Stockage** : envoyer les PDF dans Cloud Storage, adapter `FileStore`.
4. **Base vectorielle** : table BigQuery (chunks, métadonnées, vecteur) + adaptateur `VectorStore` basé sur `VECTOR_SEARCH`.
5. **Déploiement** : Dockerfile → Cloud Build → Artifact Registry → Cloud Run (API, puis front). Accès protégé (authentification), pas d'URL publique ouverte.
6. **Observabilité** : exporter les traces OpenTelemetry vers Cloud Trace.
7. **Agent managé** : réécrire l'agent F8 avec ADK, le déployer sur Agent Engine, le brancher sur le serveur MCP → comparer avec la version LangGraph.
8. **Bilan chiffré** : qualité, latence, coût réel (facture), énergie estimée → une page « local vs GCP » dans `docs/`.

### 13.5 Ce que ça apprend

- **IAM et comptes de service** : chaque service a sa propre identité, avec le minimum de droits (moindre privilège).
- **Serverless et scale-to-zero** : on ne paie que quand ça tourne ; en contrepartie, le premier appel est plus lent (*cold start*).
- **Choix d'une base vectorielle selon le coût** : service dédié facturé en continu vs requêtes analytiques facturées à l'usage.
- **Contraintes juridiques** (conditions d'usage, données, utilisateurs européens) qui pèsent sur l'architecture.
- **FinOps** : budgets, alertes, coût par requête.

---

## 14. Implémentation sur Azure (alternative gratuite)

### 14.1 Pourquoi Azure en second

- **Le plus de services IA « socle » gratuits en permanence** : recherche (AI Search) et parsing (Document Intelligence).
- **Très présent chez les clients des ESN et cabinets de conseil en France.**
- **Limite** : pas de LLM gratuit. Azure OpenAI ne s'utilise qu'avec le crédit de 200 $, **valable 30 jours**. Il faut donc **concentrer les essais LLM Azure sur une fenêtre courte** et bien préparée.

### 14.2 Les briques retenues

| Brique TransfIA | Service Azure choisi | Gratuit ? | Remarque |
|---|---|---|---|
| LLM | **Azure OpenAI** (AI Foundry) via LiteLLM | 💳 Crédit de 200 $ / 30 jours | Préparer le banc d'éval avant d'ouvrir le compte |
| LLM (alternative prototypage) | **GitHub Models** (modèles hébergés sur Azure, accessibles avec un compte GitHub) | ✅ Usage de prototypage limité (à vérifier) | Pratique pour tester sans compte Azure |
| Base vectorielle | **Azure AI Search** (hybride BM25 + vecteurs intégré) | ✅ 50 Mo, 3 index | ~quelques centaines d'articles : suffisant pour un sous-corpus |
| Base vectorielle (alternative) | **Cosmos DB** (recherche vectorielle) | ✅ 25 Go | Plus de place, moins de fonctions de recherche |
| Parsing PDF | **Document Intelligence**, comparé à Docling | ✅ 500 pages/mois | ≈ 40 articles par mois : parfait pour une comparaison |
| Stockage des PDF | **Blob Storage** | ✅ Volume gratuit les 12 premiers mois (à vérifier) | |
| API et front | **Container Apps** | ✅ 180 000 vCPU-s et 2 M de requêtes/mois | Équivalent de Cloud Run |
| Agents | **LangGraph sur Container Apps**, puis essai de l'**Agent Service de Foundry** | 💳 Sur le crédit | |
| Observabilité | **Application Insights** via OpenTelemetry | ✅ Allocation gratuite (à vérifier) | Même instrumentage qu'en local |
| Secrets | **Key Vault** | 💳 Quelques centimes | |

### 14.3 Ce qui est intéressant à comparer

- **Azure AI Search** fait nativement la **recherche hybride + reranking sémantique** : à comparer avec ton implémentation from scratch (F4). Tu verras ce que le service managé fait « à ta place ».
- **Document Intelligence vs Docling** sur les mêmes 40 articles : qualité des tableaux, ordre de lecture, temps, coût.
- **Azure OpenAI vs Gemini vs Ollama** dans le banc d'éval : qualité, latence, coût par question.

### 14.4 Étapes d'implémentation (en fenêtre de 30 jours)

1. **Avant d'ouvrir le compte** : banc d'éval prêt, adaptateurs `VectorStore` et `DocumentParser` Azure écrits et testés avec les services gratuits (GitHub Models pour le LLM).
2. **Jour 1** : compte, groupe de ressources `rg-transfia`, alerte de budget.
3. **Semaine 1** : AI Search + Document Intelligence (gratuits) ; ingestion d'un sous-corpus.
4. **Semaine 2** : Azure OpenAI via LiteLLM ; lancement du banc d'éval complet.
5. **Semaine 3** : déploiement sur Container Apps, traces dans Application Insights.
6. **Fin** : bilan chiffré, puis **suppression du groupe de ressources** (tout disparaît d'un coup).

### 14.5 Et AWS ?

- Offre : jusqu'à 200 $ de crédits et un plan gratuit de 6 mois, mais **peu de services IA gratuits en permanence**.
- À garder pour plus tard. Grâce à la couche d'abstraction (12.3), l'ajouter ne demandera que de nouveaux adaptateurs (Bedrock, OpenSearch, Textract, S3).

### 14.6 Sources (chapitres 12 à 14)

- Offre gratuite GCP : https://cloud.google.com/free-trial/
- Évolutions du niveau gratuit de l'API Gemini (2026) : https://agentdeals.dev/gemini-api-pricing-changes
- Compte gratuit Azure : https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account
- Niveau gratuit AWS : https://aws.amazon.com/about-aws/whats-new/2025/07/aws-free-tier-credits-month-free-plan/
- Croissance de Google Cloud (2026) : https://www.fool.com/investing/2026/09/09/google-cloud-grew-quarter-why-amazon-microsoft/

> ⚠️ Les offres gratuites, quotas et noms de services changent souvent : on revérifie tout le jour où l'on ouvre un compte.

---

*Données : articles de la revue INRAE Productions Animales sous licence CC BY 4.0. Chaque contenu généré cite les auteurs et le DOI de l'article source.*
