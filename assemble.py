import os
import settings
import pandas as pd
import numpy as np

HEADERS = {
	"akas": [
		"titleId",
		"language",
	],
	"basics": [
		"tconst",
		"titleType",
		"primaryTitle",
		"originalTitle",
		"isAdult",
		"startYear",
		"endYear",
		"runtimeMinutes",
		"genres"
	],
	"ratings": [
		"tconst",
		"averageRating",
		"numVotes"
	]
}

SELECT = {
    "basics": [
		"tconst",
		"titleType",
		"primaryTitle",
		"startYear",
		"genres",
	],
	"ratings": [
		"tconst",
		"averageRating"
	]
}

#AKAS
akas = pd.read_csv(os.path.join(settings.DATA_DIR, "akas.tsv"), sep="\t", header=0).rename(columns={'titleId':'tconst'})
akas = akas[(akas['language'] == 'en')]
#Basics
basics = pd.read_csv(os.path.join(settings.DATA_DIR, "basics.tsv"), sep="\t", header=0, names=HEADERS['basics'], index_col=False)
basics = basics[SELECT['basics']]
basics['startYear'] = pd.to_numeric(basics['startYear'], errors='coerce', downcast='integer').dropna()
##Save only movies in English made before the end of our test period
basics = basics[(basics['titleType'] == 'movie') & (basics['startYear'] <= 2018)].drop(['titleType'], axis=1)
basics = basics.merge(akas, on='tconst')

#Ratings
ratings = pd.read_csv(os.path.join(settings.DATA_DIR, "ratings.tsv"), sep="\t", header=0, names=HEADERS['ratings'], index_col=False)
ratings = ratings[SELECT['ratings']]
#Save only ratings of the selected movies and only movies with ratings
ratings = ratings[ratings['tconst'].isin(basics['tconst'])]
basics = basics[basics['tconst'].isin(ratings['tconst'])]

X = basics
for g in settings.GENRES:
	X[g] = X['genres'].str.contains(g)

X = X.merge(ratings, on='tconst')

X.to_csv(os.path.join(settings.PROCESSED_DIR, "train.csv"), columns = ['primaryTitle'] + settings.GENRES + ['averageRating'])