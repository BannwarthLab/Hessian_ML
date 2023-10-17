from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD 

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR 


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

# Define preprocessing steps
scaler = StandardScaler()
feature_selector = SelectKBest(k=10)  # Select the top 10 features
classifier = RandomForestClassifier(n_estimators=100)

# Create a pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', scaler),
    ('feature_selector', feature_selector),
])

# Combine preprocessing and modeling steps
full_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', classifier),
])
