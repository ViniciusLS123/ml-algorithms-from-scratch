import random
from collections import Counter
 
# ==============================================
# Random Forest — from scratch
# Author: Vinícius De Lima Silva
# Course: Curso Técnico em IA — Colégio FECAP
# ==============================================
#
# Dataset: [study_hours, sleep_hours, exercise] -> passed(1) or not(0)
 
dataset = [
    [8, 7, 1, 1], [2, 5, 0, 0], [6, 8, 1, 1],
    [1, 4, 0, 0], [7, 6, 1, 1], [3, 6, 0, 0],
    [9, 7, 1, 1], [2, 8, 0, 0], [5, 5, 1, 1],
    [1, 3, 0, 0], [8, 8, 1, 1], [4, 6, 1, 1],
    [2, 4, 0, 0], [6, 7, 0, 1], [3, 5, 1, 0],
]
 
# --------------------------------------------------
# Gini impurity — measures how "mixed" a split is
# Lower is better (0 = perfect split)
# --------------------------------------------------
def gini(groups, classes):
    total = sum(len(g) for g in groups)
    score = 0.0
    for group in groups:
        if not group:
            continue
        size = len(group)
        for c in classes:
            p = [r[-1] for r in group].count(c) / size
            score += p * p
    return 1.0 - score / len(groups)
 
# --------------------------------------------------
# Split dataset by a feature index and threshold
# --------------------------------------------------
def split(index, value, dataset):
    left  = [r for r in dataset if r[index] <= value]
    right = [r for r in dataset if r[index] >  value]
    return left, right
 
# --------------------------------------------------
# Find the best split using a random subset of features
# (this is what makes it a *random* forest)
# --------------------------------------------------
def best_split(dataset, n_features):
    classes  = list(set(r[-1] for r in dataset))
    features = random.sample(range(len(dataset[0]) - 1), n_features)
    best = {'score': 1, 'index': None, 'value': None, 'groups': None}
    for idx in features:
        for row in dataset:
            groups = split(idx, row[idx], dataset)
            score  = gini(groups, classes)
            if score < best['score']:
                best = {'score': score, 'index': idx,
                        'value': row[idx], 'groups': groups}
    return best
 
# --------------------------------------------------
# Leaf node — returns the majority class
# --------------------------------------------------
def leaf(group):
    return Counter(r[-1] for r in group).most_common(1)[0][0]
 
# --------------------------------------------------
# Build one Decision Tree recursively
# --------------------------------------------------
def build_tree(dataset, depth=0, max_depth=3, min_size=2, n_features=2):
    node = best_split(dataset, n_features)
    left, right = node['groups']
    del node['groups']
 
    if not left or not right:
        node['left'] = node['right'] = leaf(left + right)
        return node
 
    if depth >= max_depth:
        node['left']  = leaf(left)
        node['right'] = leaf(right)
        return node
 
    node['left']  = leaf(left)  if len(left)  <= min_size else build_tree(left,  depth+1, max_depth, min_size, n_features)
    node['right'] = leaf(right) if len(right) <= min_size else build_tree(right, depth+1, max_depth, min_size, n_features)
    return node
 
# --------------------------------------------------
# Predict with a single tree
# --------------------------------------------------
def predict_tree(node, row):
    direction = 'left' if row[node['index']] <= node['value'] else 'right'
    branch = node[direction]
    return predict_tree(branch, row) if isinstance(branch, dict) else branch
 
# --------------------------------------------------
# Build the forest — n_trees bootstrapped trees
# --------------------------------------------------
def random_forest(train, n_trees=10, n_features=2):
    forest = []
    for _ in range(n_trees):
        sample = [random.choice(train) for _ in train]   # bootstrap sample
        forest.append(build_tree(sample, n_features=n_features))
    return forest
 
# --------------------------------------------------
# Predict with the full forest — majority vote
# --------------------------------------------------
def predict(forest, row):
    votes = [predict_tree(tree, row) for tree in forest]
    return Counter(votes).most_common(1)[0][0]
 
 
# ==============================================
# Run
# ==============================================
random.seed(42)
forest = random_forest(dataset, n_trees=10, n_features=2)
 
tests = [
    ([8, 7, 1], "Student A — 8h study, 7h sleep, exercises"),
    ([1, 4, 0], "Student B — 1h study, 4h sleep, no exercise"),
    ([5, 6, 1], "Student C — 5h study, 6h sleep, exercises"),
]
 
labels = {1: "PASSED", 0: "DID NOT PASS"}
 
print("=" * 52)
print("   Random Forest — Student Performance Classifier")
print("=" * 52)
for row, desc in tests:
    result = predict(forest, row)
    print(f"  {desc}")
    print(f"  -> {labels[result]}")
    print("-" * 52)
 
