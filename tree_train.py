# importy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_breast_cancer
import graphviz
# zestaw danych
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print("Dokładność 1:{:.5f}".format((tree.score(X_train, y_train))))
print("Dokładność 2:{:.5f}".format(tree.score(X_test, y_test)))

# Wyświetl graficznie

export_graphviz(tree, out_file="drzewo.dot", class_names=["zlosliwy", "lagodny"], feature_names=cancer.feature_names, impurity=False, filled=True)
with open("drzewo.dot") as f:
    dot_graph = f.read()
print(graphviz.Source(dot_graph))