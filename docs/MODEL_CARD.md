# Model card — TransfIA (système)

TransfIA n'entraîne pas de grand modèle : il **assemble** des modèles open-weights. Cette fiche décrit le système et les modèles utilisés.

## Usage prévu
- **Pour** : aider des formateurs en élevage à trouver des résultats de recherche et à préparer des supports (QCM, fiches, cas pratiques), **toujours relus par un humain**.
- **Pas pour** : évaluer, noter ou orienter des apprenants (domaine à haut risque de l'AI Act, annexe III) ; donner un avis vétérinaire ; répondre hors du corpus.

## Modèles (configuration `local.yaml`)
| Rôle | Modèle | Taille | Licence (à vérifier sur la fiche du modèle) |
|---|---|---|---|
| LLM principal | qwen2.5:1.5b (Ollama, 4 bits) | ~1 Go | Apache 2.0 |
| Gros modèle (routage) | qwen2.5:3b | ~1,9 Go | Licence Qwen (restrictions) |
| 2e juge | llama3.2:1b | ~1,3 Go | Llama 3.2 Community License (restrictions) |
| Embeddings | intfloat/multilingual-e5-small | 118 M | MIT |
| Reranker | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 118 M | Apache 2.0 |
| VLM (optionnel) | moondream | ~1,7 Go | Apache 2.0 |

« Open weights » ne veut pas dire « open source » : certaines licences restreignent l'usage (voir dossier 11.4).

## Évaluation
- Banc d'éval : `transfia eval retrieval` et `transfia eval rag` (fidélité, F1, sources, refus, jury de LLM, latence, énergie).
- Les résultats dépendent du corpus et du matériel : les rapports datés sont dans `data/eval/` (non commités).
- Limite : le jeu de test d'exemple ne contient que 15 questions ; les intervalles de confiance sont larges.

## Limites connues
- Un petit modèle peut mal suivre la consigne de citation ou se tromper d'outil : la vérification et les garde-fous le signalent, sans l'empêcher totalement.
- La vérification lexicale ne détecte pas les négations ; la vérification NLI est plus robuste mais plus lourde.
- Biais du corpus : une revue francophone, des filières plus représentées que d'autres.
- Les descriptions de figures par VLM peuvent contenir des valeurs inventées : elles sont marquées comme générées.

## Transparence
Chaque contenu généré porte la mention : « Contenu généré par IA à partir des sources citées : à vérifier avant diffusion. »
