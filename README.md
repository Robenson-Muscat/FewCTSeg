# FewCTSeg

## Contexte
Les CT-scans offrent des images 3D très précises du corps humain (jusqu’à 0,5 mm de résolution) et permettent ainsi de saisir l’anatomie humaine.
L’objectif de ce challenge est de segmenter automatiquement les structures anatomiques du corps humain, ainsi que les tumeurs, sur un CT-scan. Autrement dit, il s’agit d’identifier les formes visibles sur un CT-scan.
Sur l’image ci-dessous, d’un CT-scan abdominal, les différentes structures ont été segmentées :

![Exemple d'un CT scan abdominals](images/raidium_2024_1.png).

## But

Le but de ce challenge est de segmenter les structures en utilisant leur forme, mais sans annotations exhaustives.
Les données d’entraînement sont composées de deux types d’images:

Des images de CT-scanner partiellement annotées, avec des masques de segmentation anatomiques de structures individuelles.
Elles agissent comme la définition de vérité terrain de ce qu’est une structure anatomique.
Cependant, elles ne sont pas censées être représentatives de toutes les structures possibles et leur diversité, mais peuvent toujours être utilisées comme matériau d’entraînement.
Les masques ne contiennent pas la totalité des organes annotés sur l’ensemble du dataset. Par exemple, sur deux images abdominales,
le masque pour A contiendra le foie et la rate, alors que le masque pour B contiendra uniquement la rate (alors que le foie est visible sur l’image).
Des images de CT-scanner brutes, sans aucune structure segmentée
Elles peuvent être utilisées comme matériau d’entraînement supplémentaire, dans le cadre d’un entraînement non supervisé.
Le jeu de test est composé d’images nouvelles avec la totalité des structures segmentées correspondantes, et la métrique mesure la capacité à correctement segmenter et séparer les différentes structures sur une image.


