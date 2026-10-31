# Fine-tuning (F10)

Le seul moment du projet où l'on sort du CPU : l'entraînement se fait sur un GPU gratuit (Colab ou Kaggle), l'utilisation se fait ensuite en local.

1. **Données** (sur ton PC, avec Ollama) :
   `python finetuning/build_dataset.py --n 200 --teacher qwen2.5:3b`
2. **Entraînement** : ouvre `train_lora_colab.ipynb` dans Colab (GPU T4), téléverse `finetuning/data/train.jsonl` et `val.jsonl`, exécute les cellules.
3. **Retour en local** : télécharge le `.gguf`, puis `ollama create transfia-qcm -f finetuning/Modelfile`.
4. **Comparaison** (banc d'éval) : copie `config/local.yaml` en `config/finetuned.yaml`, mets `model: transfia-qcm`, puis compare les générations de QCM (taux de JSON valide du premier coup, notes du jury) avec le modèle de base.

À retenir : le fine-tuning apprend un **comportement** (format, style), pas de nouvelles **connaissances**. Les faits viennent toujours du RAG.

Pour comprendre LoRA sans GPU : `transfia foundations --only lora_quant`.


## Mise à jour du 29/10 (licences, filtre qualité, reprise)

- **Licence du professeur** : `qwen2.5:3b` est sous *Qwen Research License* (non commercial) ; `qwen3:4b` est sous **Apache 2.0**.
  Pour un projet 100 % open source, le professeur recommandé est donc :
  `python finetuning/build_dataset.py --n 40 --teacher qwen3:4b`
  (sa « réflexion » `<think>` est désactivée automatiquement par le client Ollama).
- **Filtre qualité renforcé** : en distillation, l'élève recopie aussi les DÉFAUTS du professeur. Sont rejetés les QCM
  avec source inventée, question ambiguë ou en doublon, questions retirées, termes absents de leur source.
- **Sauvegarde au fil de l'eau** dans `finetuning/data/all.jsonl` : si tu interromps (Ctrl+C), relance la même commande
  pour reprendre, ou `python finetuning/build_dataset.py --split-only` pour découper ce qui existe en `train`/`val`.
- **Durée** sur CPU 8 Go : compter 1 à 2 min par exemple avec un modèle de 4 Md → `--n 40` ≈ 1 h à 1 h 30.
