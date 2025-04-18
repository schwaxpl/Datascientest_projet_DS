# Projet Data Science Utilisation des données TrustPilot par Camille HAMEL

Pour ce projet vous aurez besoin de spacy, de cuda ( voir ici https://stackoverflow.com/questions/75355264/how-to-enable-cuda-gpu-acceleration-for-spacy-on-windows )

Mais aussi du modèle FR de spacy

python -m spacy download fr_core_news_sm


Procédure :

## Récupération des données

### Web Crawling

(Non conseillé)
Si TrustPilot n'a pas changé sa mise en page web depuis le développement
Aller dans la partie Crawler
Saisir un secteur d'activité ( vous pouvez le trouver en navigant dans les catégories sur trustpilot.com, c'est la dernière partie de l'url https://fr.trustpilot.com/categories/distance_learning_center par exemple donnera distance_learning_center pour les établissements de formation à distance)

Cliquez sur "Obtenir les entreprises"

Choisissez ensuite quelle "tranche" vous allez importer. Attention, l'import peut être très long.
Si vous cochez la case "écraser les avis précédents", cela crééra un nouveau DF, sinon, cela mettra à jour le DF existant en ajoutant les nouvelles données.

Vous pouvez ensuite exporter ces données dans l'onglet Importer ou Exporter

### Import CSV
(Conseillé)
Aller dans Importer ou exporter
Importer le fichier TrustPilot_400

## preprocessing
Ici vous pouvez transformer les données dans un type plus adapté ( Note en Int, Date en Date)

Cliquez sur "Appliquer les modifications

Vous pouvez aussi appliquer un sur ou sous échantillonage en cliquant sur les boutons respectifs.

## Feature engineering
Laissez le programme faire ses calculs puis enregistrez les données.

## Modélisation

Vous pouvez maintenant explorer les différentes pages de modélisation.