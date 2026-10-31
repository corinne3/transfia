# TransfIA — Journal des tests (matière première du rapport final)

> Tenu au fil de l'eau. Chaque ligne : ce qu'on a lancé, ce qu'on a vu, la cause, la correction, la leçon.
> Machine : PC Windows, CPU seul, 8 Go de RAM. Corpus : 10 articles INRAE Productions Animales 2026 (CC BY).
> Modèle : **qwen2.5:1.5b** (Ollama), sauf mention contraire. Le rapport final (phase 3) sera rédigé à partir de ce journal
> et des bancs d'essai petit modèle / gros modèle.

## Brique 1-2 — Installation, modèles, ingestion

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| 1.1 | `transfia doctor` + `pytest` | 1 échec, erreur SSE « Token was created in a different Context » | le contexte de trace (contextvars) traverse un générateur de streaming | repli `set(parent)` si `reset` échoue | le streaming casse les hypothèses du code synchrone |
| 1.2 | `ingest` / `search` | le code contactait Hugging Face sur Internet | modèles chargés par leur nom | `transfia models download` copie les modèles dans `models/`, chargement local, `HF_HUB_OFFLINE=1` | « 100 % open source » ≠ « 100 % hors ligne » : il faut l'organiser |
| 1.3 | `collect` + `ingest` (vrai corpus) | 10 articles, 10 PDF, 796 passages | — | — | — |

## Brique 3-4 — Recherche et RAG

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| 3.1 | `search --mode bm25` | résultats identiques au mode dense | le reranker réordonne tout derrière | `--no-rerank` pour comparer | isoler une brique pour la mesurer |
| 4.1 | `ask` (1re fois) | 7,5 min par question | cascade vers le 3B + 1,5B + e5 + reranker en RAM → swap ; affichage d'usage faux | routage = **option** (`routing: false`) ; usage additionné | dimensionner la RAM avant d'empiler les modèles |
| 4.2 | `ask` | 0 % de fidélité, aucune citation | le 1,5B oublie les [n] | few-shot + attribution des citations après coup | un exemple vaut mieux qu'une consigne |
| 4.3 | `ask` | citation [3] injustifiée | — | `bad_citations` (précision des citations) | fidélité ≠ précision des citations |
| 4.4 | `ask` grippe aviaire | réponse inventée affichée | aucune phrase soutenue | **blocage** des réponses non soutenues | un garde-fou de sortie, pas seulement d'entrée |
| 4.5 | `ask` injection, tiramisu | refusés | garde-fous d'entrée | — | — |
| 4.6 | `ask` vrai corpus | reformulation qui perd « entérique », liste Markdown coupée à 700 tokens, rerank 51 s, 171 s au total | reformulation sans historique ; question reformulée envoyée au LLM ; `rerank_top` 20 | reformulation seulement avec historique ; question d'origine au LLM ; `max_tokens` 350 ; rerank 10 × 256 tokens | 171 s → 50 s ; rerank 51 s → 1,1 s |

## Brique 5 — Génération de contenus (F7)

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| 5.1 | `generate quiz` méthane | JSON parfait mais 2 lettres sur 3 fausses (explication de B, réponse C) | l'index est choisi AVANT l'explication | explication avant l'index dans le schéma + contrôle de cohérence explication ↔ choix | **valide ≠ juste** : le schéma garantit la forme, pas le fond |
| 5.2 | idem | Q2 et Q3 ambiguës, Q3 = mêmes choix que Q2 | limite du 1,5B | détection des doublons + **boucle de réparation** (critique → correction) | la réparation n'est gardée que si elle est meilleure |
| 5.3 | idem | réparation « pas mieux » | limite du modèle | questions non réparées **retirées** (1 question juste gardée) ; `generation_model` / `--model` | un QCM faux enseigne l'erreur ; routage par tâche |
| 5.4 | idem | résultat identique d'un essai à l'autre | cache SQLite (même prompt) | `--no-cache` | savoir quand on lit le cache |

## Brique 6 — Agents et MCP (F8)

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| 6.1 | `agent` méthane | réponse directe, 0 outil, fausse (« bactéries de l'estomac » au lieu d'archées du rumen) | le 1,5B répond de mémoire malgré la consigne | **garde-fou d'ancrage** : renvoyée une fois | une consigne n'est pas un contrôle |
| 6.2 | idem | « je vais vérifier »… sans outil | le modèle dit qu'il agit sans agir | **plan B** : le code lance la recherche | dégradation gracieuse |
| 6.3 | idem | passages trouvés mais réponse recopiée | **ancrage sur l'historique** | contexte NEUF + pipeline RAG complet | réinitialiser le contexte après une erreur |
| 6.4 | idem | ✔ archées méthanogènes du rumen, [1], fidélité 100 % | — | — | — |
| 6.5 | `--mode text` | mauvais outil, mauvais argument (`skill`), 8 fois le même appel, « réponse » = reste de Thought/Action | limites du 1,5B ; erreur Pydantic illisible | erreur lisible + exemple ; réparation d'arguments ; détection de boucle ; nettoyage | 3 limites d'agent : étapes max, boucle, budget |
| 6.6 | idem | `cas-pratique` au lieu de `cas_pratique` ; dérive de tâche | — | normalisation des enums ; plan B rattrape la dérive | la dérive ne se répare pas par le code |
| 6.7 | `--mcp` colostrum | 4 outils découverts ✔ ; réponse non citée, qui dérive | plan B côté client sans pipeline RAG | outil MCP **haut niveau** `ask_corpus` | granularité des outils |
| 6.8 | idem | refus `no_context` | « Que dit le corpus sur… » = bruit : reranker −5,7 (vs −2,6 pour « colostrum ») | nettoyage des tournures avant la recherche | mesurer avant de corriger |
| 6.9 | idem | −3,6 avec « le colostrum » (vs −2,6 sans « le ») | un score de cross-encoder n'est pas une probabilité | refus à **double critère** (score bas ET mots absents) | calibration (Platt) = la solution rigoureuse |
| 6.10 | idem + grippe aviaire | ✔ réponse citée 100 % ; ✔ refus maintenu | — | — | tester un cas qui passe ET un cas bloqué |
| 6.11 | mémoire long terme | préférence enregistrée mais perdue au plan B | le pipeline RAG ne recevait pas la mémoire | la mémoire suit le relais (RAG + MCP) | propager le contexte à chaque composant |
| 6.12 | idem | ✔ « préférences appliquées », mais pas un cas pratique | limite du modèle ; demande de contenu routée vers le RAG | *(option)* routage d'intention | — |
| 6.13 | `plan` poules pondeuses | livrable « [Rédiger ici] », Sources vides, « viande de poule », pas de progression | pas de contrôle de couverture ni de qualité | couverture avant de planifier, preuves communes, contrôle qualité, Sources par le code, progression | vérifier la faisabilité avant de dépenser |
| 6.14 | idem | poules : plan non lancé ✔ (vrai négatif) ; méthane : non lancé ✘ (faux négatif) | « Construis une séquence de… » = consigne, pas sujet | retrait des consignes de tâche | faux positifs ET faux négatifs |
| 6.15 | `plan` méthane | étapes « je n'ai pas trouvé » alors que les passages étaient fournis | agents d'étape avec outils = distraction | 1 appel LLM par étape, sans outils | le bon niveau d'autonomie |
| 6.16 | idem | contenu sur le sujet mais non cité, liste répétitive | **plafond du 1,5B** | contrôles au vert, échec visible | avec un petit modèle, workflow > agent |

## Brique 7 — Évaluation et sobriété (F11, F14)

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| 7.1 | `eval retrieval` | 0,0 partout | jeu écrit pour les articles d'exemple | contrôle de compatibilité du jeu avec l'index | valider l'instrument avant de mesurer |
| 7.2 | `eval synth --n 30` + retrieval | BM25 1,0 ; dense 0,6 ; hybride 0,9 ; rerank 0,83 (960 ms) | pseudo-questions = morceaux du texte → biais lexical | — | le jeu de test choisit le gagnant |
| 7.3 | `eval synth --n 10 --llm` | BM25 0,9 ; dense 0,8 ; hybride 0,9 ; rerank 0,9 (MRR 0,85) | questions reformulées | — | 10 questions : ±20 points, tendance seulement |
| 7.4 | `eval rag --limit 5` | fidélité 0,95 ; F1 0,67 ; source_hit 1,0 ; refus 1,0 ; 26 s p50 / 76 s p95 ; 2 220 tokens in / 92 out ; 0,29 Wh | refus = 1,0 sans question hors corpus | questions hors corpus ajoutées (maj30) | regarder le p95 ; coût = le contexte |
| 7.5 | `size 3B 4 bits 8k` | 1,8 Go + KV 0,3 Go + marge = 3,6 Go | — | — | quantification, KV cache, CPU limité par la mémoire |
| 7.7 | `eval retrieval` (index nettoyé, 30 pseudo-questions) | BM25 1,0 ; dense 0,67 ; hybride 0,97 ; rerank 0,93 (MRR 0,815, 893 ms) | même biais lexical qu'en 7.2 | — | après nettoyage, l'hybride passe de 0,9 à 0,97 |
| 7.8 | `eval rag` petit-ref (10 couvertes + 8 hors corpus) | fidélité 0,977 ; source_hit 0,9 ; **refus corrects 0,83** (3 erreurs sur 18) ; F1 0,35 ; p95 116 s | les moyennes mélangent réponses et refus quasi gratuits (tokens 1 338 contre 2 220) | matrice de confusion du refus + coût calculé sur les seules réponses ; `eval show` | une moyenne sur deux populations ment |
| 7.9 | `eval show` petit-ref | refus à tort : 1 = question « …présentée dans **ce passage** ? » (question invalide, le refus était JUSTE) ; réponses à tort : 2 = **grippe aviaire** (« traitement = ensilage de maïs + ray-grass ») et **autruche** | jeu : consigne ignorée par le 1,5B ; RAG : l'index nettoyé fait passer le seuil, et le vérificateur lexical est trompé (mots du passage + sujet de la question) | jeu : filtre des questions non autonomes ; RAG : **ancrage sur le sujet** (mots de la question absents de tout le corpus → refus si score < 0, phrases qui les contiennent = non soutenues) | une amélioration (nettoyage) peut en casser une autre (seuil) : rejouer TOUTE l'éval après chaque changement |
| 7.10 | `eval rag` petit-ref2 (après ancrage) | refus corrects **1,0** : 0 hallucination hors corpus, 0 refus à tort ; fidélité 1,0 ; MAIS latence 0,8 s (au lieu de 22 s) | les 9 réponses relues dans le **cache LLM** (mêmes prompts) | l'éval `rag` désactive le cache (`--cache` pour l'autoriser), et le rapport le signale | qualité valable, coûts faux : un cache fausse les mesures de performance |
| 7.6 | `pareto` | 1 seul point | 1 seul run | — | il faut ≥ 2 configurations |

## Brique 9 — Interface web (F13)

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| 9.1 | `transfia serve` + `npm run dev` | interface affichée sans erreur, branchée sur l'API (qwen2.5:1.5b · 694 passages · config local) | — | — | — |
| 9.3 | Questions : « colostrum important pour le veau ? » | réponse correcte **bloquée** (0/2 phrases soutenues) | **régression de maj34** : le chemin streaming n'avait pas l'attribution des citations après coup → phrases sans [n] → blocage | attribution ajoutée au streaming, texte cité renvoyé à l'interface | un garde-fou ajouté sans ses prérequis crée des faux positifs |
| 9.4 | Recherche : 3 modes | colonnes « Hybride + rerank » et « BM25 » identiques | l'API appliquait le reranker aux 3 modes | rerank seulement pour l'hybride | comparer des modes = les comparer bruts (déjà vu en CLI, brique 3) |
| 9.5 | Générateur : glossaire « colostrum », BTS | 15 termes tous cités [1], dont 3-NOP, méthane, engraissement… absents du passage ; définitions de mémoire | aucun contrôle de fond sur les éléments sourcés | chaque terme doit être DANS sa source, chaque définition y être soutenue → avertissements | même leçon que le QCM : valide ≠ juste |
| 9.6 | Assistant natif : 3 QCM sevrage | `generate_content` appelé, puis rappel, plan B, RAG → refus | observation de l'outil à examiner ; plan B = RAG pour une demande de CONTENU | *(à analyser)* routage d'intention | — |
| 9.7 | Assistant ReAct textuel | 4 fois `search_corpus`, « réponse » = plan Thought/Action brut | le 1,5B écrit tout son plan d'un coup ; sortie finale non nettoyée | nettoyage aussi sans appel d'outil | — |
| 9.8 | Graphe | étiquettes du haut coupées, nœuds plaqués aux bords | aucune marge ; répulsion trop forte | marges + gravité vers le centre | — |
| 9.9 | Évaluation | impossible de savoir quel tableau correspond à quelle manip ; en-têtes faux pour les rapports RAG | rapports « recherche » sans date ni jeu ; tableau RAG avec les colonnes de la recherche | date (heure du PC), type, jeu de test, nb de questions, config + empreinte, fichier ; `--note "…"` ; tableaux distincts | une mesure sans contexte est inexploitable |
| 9.10 | Questions : « facteurs de risque des mammites ? » (exemple de l'interface) | **hallucination validée 1/1** : « crustacés, viandes transformées, cuisson à haute température » [4] | passages retenus sur la viande ; le mot « mammite » existe dans le corpus (donc pas « inconnu ») mais dans AUCUN passage retenu ; la vérification lexicale ne regarde que le recouvrement | **ancrage sur les termes RARES** de la question : absents de tous les passages → refus ; phrase qui en parle en citant un passage qui n'en parle pas → non soutenue ; exemples de l'interface adaptés au vrai corpus | une vérification par recouvrement de mots ne vérifie pas le SUJET |
| 9.11 | Recherche (après maj35) | 3 colonnes différentes ✔ ; le reranker remonte le bon passage « porcelets au sevrage » en 1er | — | — | le reranker vaut ici son coût |
| 9.12 | Générateur : fiche master colostrum | 1 avertissement « élément peu soutenu » ✔, sources regroupées ✔ | — | — | — |
| 9.13 | Assistant natif : fiche caprins master | `generate_content` OK, puis l'agent **réécrit** la fiche en texte libre (vocabulaire vide, sources « Bluet et al. » ×3 sans titre ni DOI) | le LLM reformule la sortie de l'outil | **return_direct** : le livrable de l'outil est rendu tel quel | ne pas laisser un LLM réécrire un résultat déjà structuré et vérifié |
| 9.14 | Assistant ReAct textuel : fiche caprins | le modèle écrit lui-même « Observation : je trouve plusieurs articles… » (**observation inventée**) et la donne comme réponse | absence de « stop sequence » | tout ce qui suit « Observation : » dans la sortie du modèle est coupé | seul le code a le droit d'écrire une observation |
| 9.15 | `eval rag` petit-ref3 (non-régression après maj36) | ✔ 0 refus à tort, 0 réponse à tort, fidélité 1,0 ; **coûts réels (sans cache)** : réponse p50 8,3 s, p95 36,6 s, 2 282 tokens en entrée | — | — | référence fiable du petit modèle |
| 9.16 | Questions « mammites » (après maj36) | ✔ nouvelle hallucination (« réactions allergiques, contaminants chimiques… ») **bloquée** en sortie | — | — | — |
| 9.17 | Assistant textuel « fiche caprins » | ✔ plus d'observation inventée ; ✘ `niveau: "bac+2"` refusé ; ✘ « réponse » = plan Thought/Action | pas de synonymes métier ; nettoyage qui rendait le texte d'origine quand tout était retiré | synonymes (bac+2 → bts, fiche-synthese → fiche…) ; réponse vide → plan B | — |
| 9.18 | Assistant natif « fiche caprins » | ✘ fiche encore réécrite en texte libre ; l'observation est au format d'AVANT maj36 (dictionnaire JSON) | le serveur exécutait encore l'ancien `corpus_tools.py` (le paquet maj36 était correct) ; rien ne permettait de voir quelle version tournait | numéro de version (`BUILD`) affiché dans l'en-tête de l'interface et par `transfia doctor`, avec le dossier d'où le code est chargé | rendre visible la version déployée (comme un tag d'image en production) |
| 9.19 | Assistant natif « fiche caprins » (maj38 affiché) | ✔ fiche rendue TELLE QUELLE : objectifs, points clés cités [n], sources construites par le code avec DOI | — | — | `return_direct` : la sortie structurée et vérifiée n'est plus réécrite |
| 9.2 | relecture du code avant usage | le chemin **streaming** (onglet Questions) n'avait ni l'ancrage (maj32) ni le blocage des réponses non soutenues | logique dupliquée entre `answer()` et `stream()` | contrôles **partagés** (`precheck`) + blocage en sortie qui remplace le texte déjà affiché | deux chemins de code dupliqués finissent par diverger : factoriser les garde-fous |

## Bonus F9 — Atelier multi-agents (rédacteur → relecteur → vulgarisateur)

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| F9.1 | `editorial "le colostrum" --niveau grand_public` (rédacteur qwen2.5:1.5b, relecteur **llama3.2:1b**) | 3 tours de relecture avec « 0 problème » à chaque fois | l'arête de la machine à états suivait le drapeau `valide` du LLM (faux, sans problème listé) | routage sur les PROBLÈMES ouverts | ne pas se fier au drapeau d'un LLM : le recalculer |
| F9.2 | idem | relecteur qui ne trouve jamais rien | petit juge complaisant | relecteur AUTOMATIQUE en plus (vérification lexicale de chaque phrase) | juge LLM + règles |
| F9.4 | idem après maj39 | ✔ « 20 % » conservé, texte en prose ; ✔ relecteur automatique : « substances protectrices » (absente des passages) signalée ; ✔ contrôle final la retrouve | — | — | les règles trouvent ce que le petit juge rate |
| F9.5 | idem | ✘ relecteur llama3.2:1b : sortie illisible 3 fois sur 3 ; ✘ même phrase signalée 3 fois, jamais corrigée ; ✘ fausse alerte « chiffres perdus : 2012, 2021, 2024 » | petit modèle incapable du JSON ; rédacteur têtu ; années et numéros [n] comptés comme des quantités | repli sur le modèle du rédacteur si le relecteur est illisible ; **élagage** par le code des phrases toujours non soutenues ; années et [n] ignorés | un relecteur d'une autre famille ne sert que s'il sait répondre |
| F9.6 | idem après maj40 | ✔ repli du relecteur (llama illisible → qwen) ; ✔ élagage d'1 phrase non soutenue ; ✔ plus de fausse alerte (années) ; texte final sourcé, chiffres conservés (20 %, 34 %) ; ✘ reste : « une grande partie (20 %) » (jugement ajouté) ; le relecteur qwen juge « contredite » une phrase recopiée de la source (faux positif du juge) | limite des petits juges, dans les deux sens | — | un juge de 1,5 Md se trompe dans les DEUX sens (complaisant puis trop sévère) : à mesurer contre un humain (`eval agreement`) |
| F9.3 | idem | texte final : « Les vaches Charolaise ont des colostrums faibles » (source : 20 % des colostrums) ; liste numérotée ; citations perdues ; dérive (lésions podales) | la dernière étape (vulgarisation) n'était relue par personne | consignes « garder chiffres et nuances » + **contrôle final** : chiffres perdus, phrases sans appui | toute étape qui modifie le texte doit être contrôlée, surtout la dernière |

## Préparation phase 2 et F10 — licences et distillation

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| L.1 | vérification des licences avant F10 | qwen2.5:**3b** sous *Qwen Research License* (non commercial), alors que 1,5b est Apache 2.0 ; qwen3:**4b** Apache 2.0 | licence différente selon la TAILLE dans une même famille | 2 configurations : 3B (comparaison pure de taille) et 4B (version open source, professeur de F10) | vérifier la licence de CHAQUE modèle, pas de la famille |
| L.3 | `build_dataset.py --n 40 --teacher qwen3:4b` | 40 exemples en ~2 h (≈ 2,5 min / exemple, interrompu par un rendez-vous et repris ✔) ; **6 QCM ambigus rejetés** par le filtre | 4B ≈ 2,5× plus lent que 1,5B sur CPU | — | la reprise au fil de l'eau a servi en vrai |
| L.4 | relecture du notebook Colab avant usage | évaluation jamais déclenchée (10 pas au total, `eval_steps=20`) ; paramètres TRL renommés selon les versions ; essai final sans mesure ; nom du `.gguf` variable | notebook écrit pour un gros jeu et une version figée | évaluation à chaque époque, journal à chaque pas ; code tolérant aux versions de TRL ; **mesure base vs fine-tuné** (JSON valide, QCM conforme) sur la validation ; recherche du `.gguf` produit | adapter les hyperparamètres à la TAILLE du jeu |
| L.5 | Colab : chargement QLoRA | 18,5 M paramètres entraînés sur 1,56 Md (**1,18 %**), modèle de base gelé en 4 bits | — | — | QLoRA = fine-tuning sur GPU gratuit |
| L.6 | Colab : entraînement (3 époques, 15 pas, 1 min 05) | perte entraînement 1,77 → 1,31 ; validation 1,63 → 1,46 → **1,40** (gain qui ralentit : début d'écart) | 36 exemples seulement | — | surveiller l'écart train/validation (early stopping) ; 3 époques = le bon choix |
| L.7 | Colab : mesure base vs fine-tuné | plantage de la mesure : le modèle de base a mis `"bonne_reponse": "Spectroscopie MIR"` (texte au lieu d'un numéro) | fonction de mesure qui ne prévoyait pas un mauvais type | mauvais type = « non conforme » | une métrique doit résister aux sorties imprévues |
| L.8 | Colab : mesure (après correction) | **base : JSON valide 3/4, QCM conforme 0/4 ; fine-tuné : JSON valide 3/4, QCM conforme 3/4** ; MAIS l'exemple fine-tuné donne `bonne_reponse: 2` (« Génome sequencing ») avec une explication sur la spectroscopie MIR (index 1) | le fine-tuning a appris le FORMAT, pas la justesse ; même défaut que le QCM du 1,5B (brique 5) | le contrôle de cohérence de F7 l'attraperait | **le fine-tuning apprend un comportement, pas la vérité** : les contrôles restent nécessaires ; 4 exemples = effet net mais pas significatif |
| L.9 | Colab : export GGUF | fusion des poids LoRA en 16 bits, conversion GGUF f16 puis quantification **q4_k_m** : `Qwen2.5-1.5B-Instruct.Q4_K_M.gguf`, **0,99 Go** (llama.cpp précompilé, pas de compilation) | le nom du fichier ne suit pas celui demandé | la cellule cherche et affiche le fichier produit | prévoir que l'outil nomme les fichiers à sa façon |
| L.10 | `ollama create transfia-qcm` + `generate quiz ... --model transfia-qcm` | ✘ **pire que le modèle de base** : les 4 choix recopiés de l'EXEMPLE de style (« l'eau est propre et accessible »…), 3 questions identiques, explication = passage recopié ; ✔ garde-fous : « ambiguë » ×3, doublons ×2, « À REVOIR entièrement » | **décalage de distribution** : entraîné avec 1 consigne + 1 passage, appelé avec skill + exemple + 5 passages | **contrat de prompt** : un modèle fine-tuné est appelé EXACTEMENT comme à l'entraînement ; consigne système UNIQUE partagée par l'entraînement et l'utilisation | un fine-tuning crée un contrat d'interface ; le casser dégrade le modèle |
| L.11 | `generate quiz ... --model transfia-qcm` avec le contrat de prompt (maj44) | ✔ 1 question **sur le sujet** (méthane), réponse B **cohérente** avec l'explication ; plus de recopie de l'exemple ; ✘ énoncé qui n'est pas une question, distracteur faible, explication « méta », 1 question au lieu de 3 | 36 exemples apprennent la FORME (JSON conforme, cohérence), pas la pédagogie ; le professeur 4B avait lui-même des défauts | — | **F10 validé** : bilan = format appris (conforme 0/4 → 3/4), justesse portée par le contrat de prompt, pédagogie non apprise ; pour aller plus loin : plus d'exemples relus par un humain |
| L.2 | relecture de `build_dataset.py` avant 1 h de génération | filtre qualité limité aux sources inventées ; rien n'est sauvegardé avant la fin ; échecs silencieux | — | rejet des QCM ambigus / en doublon / à termes absents ; sauvegarde à chaque exemple et reprise ; erreurs comptées | en distillation, l'élève recopie aussi les défauts du professeur |

## Brique 8 — ML classique, clustering, graphe (F3, F6)

| # | Test | Constat | Cause | Correction | Leçon |
|---|---|---|---|---|---|
| 8.1 | `classify` | F1 0,91 ; mots « aubrac, 1994, vab » pour *économie* | **fuite par groupe** : passages d'un même article en train ET en test | CV groupée par article | lire ce que le modèle a appris |
| 8.2 | idem, à périmètre égal | TF-IDF 0,991 → 0,851 (fuite +0,14) ; embeddings 0,96 → 0,87 (+0,09) ; classement inversé | 1re comparaison faussée (4 thèmes vs 2) | comparaison à périmètre égal | une fuite peut faire choisir le mauvais modèle |
| 8.3 | idem | economie et environnement = 1 article chacun | corpus trop petit | retirés de la CV groupée | ≥ 2 articles par thème |
| 8.4 | `themes` | un cluster = bibliographie, un autre « br », « al » partout ; silhouette 0,10-0,14 | données sales | bibliographie écartée, `<br>` nettoyés, orphelins supprimés ; 796 → 694 passages | le non supervisé audite les données |
| 8.5 | `themes` (après) | 8 vrais sujets ; silhouette 0,136, k au bord | sujets qui se chevauchent | alertes silhouette | k au bord de la plage = pas de k naturel |
| 8.6 | `graph build` | 60 entités, 485 liens, 4 communautés cohérentes | — | — | — |
| 8.7 | `graph query méthane` | ~470 liens (tout le graphe) | arêtes des VOISINS | arêtes du nœud seulement | — |
| 8.8 | `graph global` | « pas d'éléments » (×2) | map sans texte ; puis échappatoire « RIEN » prise | extraits par communauté ; pas d'échappatoire ; **trace map** | diagnostiquer par la trace |
| 8.9 | idem | ✔ 4/4 gardées, synthèse sur tout le corpus ; réponse coupée | limite 400 tokens | 5 points max + coupure signalée | — |

## Banc d'essai (phase 1d) — référence « petit »

| # | Test | Résultat | Cause / lecture | Correction | Leçon |
|---|---|---|---|---|---|
| B.1 | `transfia bench --label petit` (qwen2.5:1.5b, cache coupé, 30,7 min, 12,8 Wh estimés) | RAG : fidélité 1,00, refus corrects 1,00, **0 hallucination hors corpus, 0 refus à tort** ; quiz : **3 questions gardées / 9** (2 doublons, 1 réparation ratée, 1 question retirée) ; glossaire : 1 terme absent de sa source ; agent : **0 tâche sur 3 menée seul** (plan B à chaque fois) ; multi-agents : 3 tours, élagage, 1 alerte finale | les garde-fous tiennent (RAG parfait) ; ce qui reste faible dépend de la capacité du modèle (outils, QCM) | — | référence pour la phase 2 (3B et 4B) |
| B.2 | latence RAG du banc vs `petit-ref3` | p50 **20,2 s** (vs 8,3 s), p95 40,8 s (vs 36,6 s) | `doctor` : **0,7 Go de RAM libre** sur 8,4 → mémoire saturée (navigateur, Teams…) | fermer les applications avant un banc | une latence se mesure dans des conditions fixées (RAM libre, secteur) |
| B.3 | `bench --label 3b` (qwen2.5:3b, 35,7 min, 14,9 Wh) | RAG : fidélité **0,94** (vs 1,00), refus 1,00, 0 hallucination, 0 refus à tort, latence p50 **44,7 s** (×2,2) ; quiz **7/9** (vs 3/9 ; restent 2 ambiguës, 1 lettre corrigée, 1 réparation ratée) ; glossaire : 1 terme absent ; **agent 3/3 mené seul** (vs 0/3) ; multi-agents : ✘ ReadTimeout | plus gros = meilleur sur les tâches de **capacité** (outils, QCM), pas sur la **fidélité** ; les erreurs de QCM persistent | — | les garde-fous restent nécessaires : ils servent moins souvent, mais ils servent |
| B.4 | `bench --label 4b` (qwen3:4b, 48,2 min) | ✘ **ReadTimeout** sur RAG, glossaire, multi-agents ; quiz 0/9 et agent « 0 / 0 / 3 » = appels en erreur affichés comme des zéros | délai max de 300 s par appel trop court pour un 4B sur CPU (prompt ≈ 2 300 tokens + 700 en sortie) ; RAM saturée (chargement e5 13 s au lieu de 0,2 s : le modèle 3B restait 10 min en mémoire) ; réflexion qwen3 peut-être non coupée (version d'Ollama à vérifier) | `timeout_s: 900` dans les configs 3B / 4B ; le banc décharge les autres modèles, chauffe le modèle, note la RAM libre, la version d'Ollama et la réflexion détectée ; erreurs comptées dans le tableau (plus de faux zéros) | un tableau ne doit jamais afficher une panne comme un résultat |
| B.5 | `bench compare … \| Tee-Object` | accents abîmés (« fid├®lit├® ») | PowerShell relit la sortie d'un programme dans l'encodage de la console | `bench compare … --out fichier.md` écrit en UTF-8 ; rapports de même étiquette **fusionnés** (scénario rejoué = remplace la panne) | — |
| B.6 | `bench --label 4b` après maj47 (délai 900 s) | conditions : **0,8 Go de RAM libre**, Ollama 0.40.6, aucun autre modèle chargé, chauffe 21,6 s, **réflexion non détectée** ; RAG ✘ (1 appel > 900 s), quiz 45 min, glossaire ✘, agent 45 min ; arrêté après 2 h 30 pendant le multi-agents | ni la réflexion ni un autre modèle : la **RAM** (4B 2,5 Go + embeddings > mémoire libre → disque) | banc arrêté ; mesures perdues car le fichier n'était écrit qu'à la fin → **sauvegarde après chaque scénario** (maj48) | **la contrainte matérielle décide du modèle** : sur 8 Go, le 3B est le plafond |
| B.7 | `bench compare` petit vs 3B | tableau UTF-8 correct (`--out`) ; le quiz du 1,5B montre « ⚠️ 1 en erreur » (invisible avant maj47) | 1 thème sur 3 en erreur côté 1,5B | — | le comptage des erreurs révèle aussi des pannes passées inaperçues |

## Phase 4 — GCP sans carte (API Gemini, niveau gratuit)

| # | Test | Résultat | Cause / lecture | Correction | Leçon |
|---|---|---|---|---|---|
| G.0 | clé d'API dans `.env` par remplacement de ligne | rien d'écrit : le `.env` existant n'avait pas de ligne `GEMINI_API_KEY` ; `ask` réessayait sans clé | — | ajout de la ligne ; toute clé exposée par erreur est **révoquée et recréée** | un secret exposé se remplace (rotation) |
| G.1 | `ask` colostrum + capitale de l'Australie (gemini-3.1-flash-lite) | ✔ réponse sur le sujet, citée (20 % de colostrums pauvres, Charolaise) ; fidélité 50 % (1 phrase reformulée jugée non soutenue) ; 1 722 → 85 tokens, 29,5 s (surtout le chargement local) ; ✔ hors corpus **refusé sans appeler Gemini** | Gemini reformule ; le vérificateur est lexical | — | changer de modèle change ce que mesurent les garde-fous |
| G.2 | `bench --label gemini --only rag --max-questions 10` | fidélité 0,96 ; refus 1,00 ; 0 hallucination, 0 refus à tort ; latence p50 **8,6 s** / p95 28,4 s ; 1,5 min ; 0,63 Wh (PC seul) ; RAM libre 2,2 Go | serveurs Google pour le LLM ; recherche locale | `DeprecationWarning` LiteLLM (température Gemini 3) masqué (maj49) | même code, mêmes garanties de refus, latence ÷ 2,4 ; énergie du cloud non mesurable côté client |
| G.3 | `bench --label petit10 --only rag --max-questions 10` (1,5B, mêmes 10 questions que G.2) | fidélité **1,00** ; refus 1,00 ; 0 / 0 ; p50 **23,8 s** / p95 47,3 s ; 2,6 min ; 1,08 Wh ; RAM libre 0,9 Go | à échantillon égal : Gemini ×2,8 plus rapide, fidélité 0,96 vs 1,00 | — | comparer sur le MÊME échantillon avant de conclure |

## À venir
- Phase 1 : éval refaite après ré-ingestion (avec questions hors corpus), brique 9 (web), F9, F10, banc d'essai de référence.
  - ✔ fait : éval, brique 9, F9, F10. Reste : **banc d'essai de référence** (`transfia bench --label petit`, créé en maj45).
- Phase 2 : même banc d'essai avec qwen2.5:3b, comparaison.
- Phase 3 : rapport final rédigé à partir de ce journal.
- Phase 4 : GCP.
