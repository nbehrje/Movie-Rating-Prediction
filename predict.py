import pandas as pd
import os
import settings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

train = pd.read_csv(os.path.join(settings.PROCESSED_DIR, "train.csv"))

X = train.filter(settings.GENRES)
y = train.filter(["averageRating"])
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)

clf = LinearRegression().fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print("Mean absolute error (training): {}".format(mean_absolute_error(y_pred_train, y_train)))
print("Mean absolute error (test): {}".format(mean_absolute_error(y_pred_test, y_test)))

min_error = 11
min_error_genre = ""
max_error = -1
max_error_genre = ""
for g in settings.GENRES:
	g_index = X_test[X_test[g]].index
	if g_index.empty:
		continue
	else:
		error = mean_absolute_error(y_pred_test[g_index], y_test.loc[g_index])
		if error < min_error:
			min_error = error
			min_error_genre = g
		elif error > max_error:
			max_error = error
			max_error_genre = g

print("Genre with lowest error: {} ({})".format(min_error_genre, min_error))
print("Genre with highest error: {} ({})".format(max_error_genre, max_error))