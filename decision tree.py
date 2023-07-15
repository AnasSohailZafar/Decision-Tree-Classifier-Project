#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import random
import math
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self):
        self.tree = None

    def load_data(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            self.header = next(reader)
            self.data = list(reader)

    def split_data(self, data, feature_index):
        split_data = {}
        for row in data:
            value = row[feature_index]
            if value not in split_data:
                split_data[value] = []
            split_data[value].append(row)
        return split_data

    def calculate_entropy(self, data):
        class_counts = {}
        for row in data:
            label = row[-1]
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        entropy = 0.0
        for count in class_counts.values():
            probability = count / len(data)
            entropy -= probability * math.log2(probability)
        return entropy

    def calculate_information_gain(self, data, feature_index):
        feature_values = set(row[feature_index] for row in data)
        feature_entropy = 0.0
        for value in feature_values:
            subset = [row for row in data if row[feature_index] == value]
            probability = len(subset) / len(data)
            feature_entropy += probability * self.calculate_entropy(subset)
        return self.calculate_entropy(data) - feature_entropy

    def majority_vote(self, labels):
        label_counts = {}
        for label in labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        majority_label = max(label_counts, key=label_counts.get)
        return majority_label

    def build_tree(self, data, remaining_features):
        labels = [row[-1] for row in data]

        # Base cases
        if len(set(labels)) == 1:
            return labels[0]
        if len(remaining_features) == 0:
            return self.majority_vote(labels)

        best_feature_index = 0
        best_info_gain = 0.0

        for i, feature in enumerate(remaining_features):
            info_gain = self.calculate_information_gain(data, i)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = i

        best_feature = remaining_features[best_feature_index]
        tree = {best_feature: {}}

        remaining_features = remaining_features[:best_feature_index] + remaining_features[best_feature_index + 1:]

        feature_values = set(row[best_feature_index] for row in data)
        for value in feature_values:
            subset = [row for row in data if row[best_feature_index] == value]
            subtree = self.build_tree(subset, remaining_features)
            tree[best_feature][value] = subtree

        return tree

    def train(self, filename):
        self.load_data(filename)
        remaining_features = self.header[:-1]
        self.tree = self.build_tree(self.data, remaining_features)

    def predict_instance(self, instance, tree):
        if not isinstance(tree, dict):
            return tree

        feature = list(tree.keys())[0]
        value = instance[self.header.index(feature)]
        subtree = tree[feature].get(value)

        if subtree is None:
            return self.majority_vote([row[-1] for row in self.data])

        return self.predict_instance(instance, subtree)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            prediction = self.predict_instance(instance, self.tree)
            predictions.append(prediction)
        return predictions

    def evaluate(self, test_data):
        instances = [row[:-1] for row in test_data]
        labels = [row[-1] for row in test_data]
        predictions = self.predict(instances)
        accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels)
        return accuracy

    def print_tree(self, tree, indent=''):
        if not isinstance(tree, dict):
            print(tree)
        else:
            feature = list(tree.keys())[0]
            print(feature)
            for value, subtree in tree[feature].items():
                print(indent + '|', feature, '=', value, end=' ')
                self.print_tree(subtree, indent + '|    ')


    def create_learning_curve(self, train_data, test_data, train_sizes):
        accuracies = []
        for train_size in train_sizes:
            # Sample a subset of the training data
            subset_train_data = random.sample(train_data, train_size)

            # Train the model on the subset of training data
            remaining_features = self.header[:-1]
            self.tree = self.build_tree(subset_train_data, remaining_features)

            # Evaluate the model on the test data
            accuracy = self.evaluate(test_data)
            accuracies.append(accuracy)

        return accuracies

    def calculate_precision(self, true_positives, false_positives):
        if true_positives + false_positives == 0:
            return 0.0
        return true_positives / (true_positives + false_positives)

    def calculate_recall(self, true_positives, false_negatives):
        if true_positives + false_negatives == 0:
            return 0.0
        return true_positives / (true_positives + false_negatives)

    def calculate_f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * ((precision * recall) / (precision + recall))

    def calculate_evaluation_metrics(self, true_positives, false_positives, false_negatives):
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)
        f1_score = self.calculate_f1_score(precision, recall)
        return precision, recall, f1_score

    def calculate_metrics_by_class(self, true_labels, predicted_labels, classes):
        metrics_by_class = {}
        for class_name in classes:
            true_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == class_name and pred == class_name)
            false_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != class_name and pred == class_name)
            false_negatives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == class_name and pred != class_name)
            precision, recall, f1_score = self.calculate_evaluation_metrics(true_positives, false_positives, false_negatives)
            metrics_by_class[class_name] = {
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1_score
            }
        return metrics_by_class

    def calculate_macro_average(self, metrics_by_class):
        num_classes = len(metrics_by_class)
        macro_precision = sum(metrics['Precision'] for metrics in metrics_by_class.values()) / num_classes
        macro_recall = sum(metrics['Recall'] for metrics in metrics_by_class.values()) / num_classes
        macro_f1_score = sum(metrics['F1-score'] for metrics in metrics_by_class.values()) / num_classes
        return macro_precision, macro_recall, macro_f1_score

    def calculate_weighted_average(self, metrics_by_class, true_labels):
        total_instances = len(true_labels)
        weighted_precision = sum(metrics['Precision'] * (true_labels.count(class_name) / total_instances) for class_name, metrics in metrics_by_class.items())
        weighted_recall = sum(metrics['Recall'] * (true_labels.count(class_name) / total_instances) for class_name, metrics in metrics_by_class.items())
        weighted_f1_score = sum(metrics['F1-score'] * (true_labels.count(class_name) / total_instances) for class_name, metrics in metrics_by_class.items())
        return weighted_precision, weighted_recall, weighted_f1_score

    def evaluate_with_metrics(self, test_data):
        instances = [row[:-1] for row in test_data]
        true_labels = [row[-1] for row in test_data]
        predicted_labels = self.predict(instances)
        classes = set(true_labels)
        metrics_by_class = self.calculate_metrics_by_class(true_labels, predicted_labels, classes)
        macro_precision, macro_recall, macro_f1_score = self.calculate_macro_average(metrics_by_class)
        weighted_precision, weighted_recall, weighted_f1_score = self.calculate_weighted_average(metrics_by_class, true_labels)
        return metrics_by_class, macro_precision, macro_recall, macro_f1_score, weighted_precision, weighted_recall, weighted_f1_score
     
    def calculate_confusion_matrix(self, true_labels, predicted_labels, classes):
        confusion_matrix = {}
        for true_class in classes:
            confusion_matrix[true_class] = {}
            for predicted_class in classes:
                confusion_matrix[true_class][predicted_class] = 0

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            confusion_matrix[true_label][predicted_label] += 1

        return confusion_matrix

    def calculate_accuracy(self, true_labels, predicted_labels):
        correct_predictions = sum(1 for true_label, predicted_label in zip(true_labels, predicted_labels) if true_label == predicted_label)
        total_predictions = len(true_labels)
        accuracy = correct_predictions / total_predictions
        return accuracy

    def evaluate_with_metrics_and_confusion_matrix(self, test_data):
        instances = [row[:-1] for row in test_data]
        true_labels = [row[-1] for row in test_data]
        predicted_labels = self.predict(instances)
        classes = set(true_labels)
        metrics_by_class = self.calculate_metrics_by_class(true_labels, predicted_labels, classes)
        macro_precision, macro_recall, macro_f1_score = self.calculate_macro_average(metrics_by_class)
        weighted_precision, weighted_recall, weighted_f1_score = self.calculate_weighted_average(metrics_by_class, true_labels)
        confusion_matrix = self.calculate_confusion_matrix(true_labels, predicted_labels, classes)
        accuracy = self.calculate_accuracy(true_labels, predicted_labels)
        return metrics_by_class, macro_precision, macro_recall, macro_f1_score, weighted_precision, weighted_recall, weighted_f1_score, confusion_matrix, accuracy


# Usage example
random.seed(42)
dt = DecisionTree()
dt.load_data('votes.csv')

# Split the data into training and testing sets
train_size = int(0.8 * len(dt.data))
train_data = random.sample(dt.data, train_size)
test_data = [row for row in dt.data if row not in train_data]

# Specify the range of training set sizes for the learning curve
train_sizes = [int(train_size * s) for s in [0.1, 0.3, 0.5, 0.7, 0.9]]

# Create the learning curve
accuracies = dt.create_learning_curve(train_data, test_data, train_sizes)

# Evaluate the model with metrics, confusion matrix, and accuracy
metrics_by_class, macro_precision, macro_recall, macro_f1_score, weighted_precision, weighted_recall, weighted_f1_score, confusion_matrix, accuracy = dt.evaluate_with_metrics_and_confusion_matrix(test_data)

# Print the evaluation metrics
print("Metrics by Class:")
for class_name, metrics in metrics_by_class.items():
    print("Class:", class_name)
    print("Precision:", metrics['Precision'])
    print("Recall:", metrics['Recall'])
    print("F1-score:", metrics['F1-score'])
    print()

print("Macro Average:")
print("Precision:", macro_precision)
print("Recall:", macro_recall)
print("F1-score:", macro_f1_score)
print()

print("Weighted Average:")
print("Precision:", weighted_precision)
print("Recall:", weighted_recall)
print("F1-score:", weighted_f1_score)
print()

print("Confusion Matrix:")
for true_class, row in confusion_matrix.items():
    for predicted_class, count in row.items():
        print(true_class, "->", predicted_class, ":", count)

        print()
print("Accuracy:", accuracy)

# Plot the learning curve
plt.plot(train_sizes, accuracies)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.show()



# In[ ]:




