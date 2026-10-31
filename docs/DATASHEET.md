# Datasheet du corpus

## Origine
- Revue **INRAE Productions Animales** (INRAE), revue de synthèse francophone sur les productions animales.
- Métadonnées récupérées via l'API **OpenAlex** (source `S4210213780`).
- Filtre : articles publiés à partir de 2020, licence **CC BY** vérifiée article par article.

## Licence et redistribution
- CC BY 4.0 : réutilisation et transformation autorisées, **attribution obligatoire** (auteurs, titre, DOI).
- Les PDF ne sont **pas** redistribués dans le dépôt : ils sont téléchargés localement par `transfia collect`.

## Composition
- Articles de synthèse (« reviews ») : nutrition, santé, génétique, bien-être, environnement, économie, qualité des produits.
- Espèces : surtout bovins, porcins, volailles, ovins, caprins, poissons.
- Langue : français (quelques résumés en anglais).

## Biais connus
- Une seule revue, une seule institution principale : vision partielle de la recherche.
- Répartition inégale des espèces et des thèmes (à mesurer avec `transfia classify` et `transfia themes`).
- Articles de synthèse : moins de résultats bruts, plus de recul (plutôt un avantage pour la pédagogie).

## Données personnelles
- Seuls les noms des auteurs (déjà publics) sont traités.
- Les questions des utilisateurs ne sont enregistrées que dans les traces locales et le fichier de feedback (`data/`, non commité).

## Corpus d'exemple (`data/samples/`)
- 8 textes courts rédigés pour le projet (CC0), marqués « [Exemple] » : ils servent aux tests et à la démo hors ligne.
