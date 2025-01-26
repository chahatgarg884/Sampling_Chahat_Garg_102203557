import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek

# Load dataset
data_url = "https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv?raw=true"
data = pd.read_csv(data_url)

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Balance the dataset using SMOTE-Tomek (Hybrid Sampling)
smote_tomek = SMOTETomek()
X_balanced, y_balanced = smote_tomek.fit_resample(X, y)

# Define samples using different random seeds
samples = [X_balanced.sample(frac=0.2, random_state=i) for i in range(5)]
sample_targets = [y_balanced.loc[sample.index] for sample in samples]

# Define models
models = [
    DecisionTreeClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier(),
    SVC(),
    GaussianNB()
]

# Define sampling techniques (for naming purposes)
techniques = ["SamplingA", "SamplingB", "SamplingC", "SamplingD", "SamplingE"]

# Dictionary to store results
results = {}

# Evaluate each model with each sample
for technique, X_sample, y_sample in zip(techniques, samples, sample_targets):
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    for model in models:
        model_name = type(model).__name__
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save accuracy in results dictionary
        results[f"{technique}_{model_name}"] = accuracy

# Determine the best sampling technique for each model
best_techniques = {}
for model in models:
    model_name = type(model).__name__
    model_results = {key: value for key, value in results.items() if model_name in key}
    best_technique = max(model_results, key=model_results.get)
    best_techniques[model_name] = best_technique

# Output the results
print("Best techniques for each model:")
for model_name, technique in best_techniques.items():
    print(f"{model_name}: {technique}")
