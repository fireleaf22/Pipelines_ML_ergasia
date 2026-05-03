import streamlit as st #used for making the web app
import pandas as pd #used for data management
import numpy as np #used for numerical operations
 
from sklearn.model_selection import train_test_split #train/test split
from sklearn.preprocessing import StandardScaler #for scaling and encoding
from sklearn.compose import ColumnTransformer #ColumnTransformer for combining preprocessing
from sklearn.pipeline import Pipeline #Pipeline for preprocessing + model
from sklearn.metrics import ( #regression & classification metrics
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer #SimpleImputer for missing values
from sklearn.preprocessing import OneHotEncoder #for scaling and encoding categorical

from sklearn.linear_model import LinearRegression, LogisticRegression #ML algorithms
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier #ML algorithms
from sklearn.decomposition import PCA #PCA for dimensionality reduction (μείωση διαστάσεων)
from sklearn.cluster import KMeans #KMeans clustering (oμαδοποίηση K-μέσων)

import matplotlib.pyplot as plt #used for graphs
import seaborn as sns #used for graphs

# ---------------------------------------------------------
# 🖥️ Ρυθμίσεις Streamlit
# ---------------------------------------------------------

st.set_page_config(page_title="Student Performance ML App", layout="wide")
st.title("Student Performance – Data Science & ML Pipeline")

# Περιγραφή της εφαρμογής
st.markdown("""
Αυτή η εφαρμογή υλοποιεί μια end-to-end ροή:
- Φόρτωση & προεπεξεργασία δεδομένων
- Διερευνητική ανάλυση (EDA)
- ML διεργασίες (Regression & Classification) με σύγκριση αλγορίθμων
""")

# ---------------------------------------------------------
# 1️⃣ Στάδιο 1: Φόρτωση Δεδομένων / Upload Data
# ---------------------------------------------------------

st.header("1. Φόρτωση Δεδομένων & Προεπεξεργασία")

# Upload Csv by the user
uploaded_file = st.file_uploader("Φόρτωσε το CSV (student-por.csv)", type=["csv"])

# If no file was uploaded, then use the default csv
if uploaded_file is None:
    st.info("Αν δεν ανεβάσεις αρχείο, θα χρησιμοποιηθεί το τοπικό 'student-por.csv' (αν υπάρχει).")
    try:
        df = pd.read_csv("student-por.csv")
        st.success("Φορτώθηκε τοπικά το student-por.csv")
    except Exception:
        st.warning("Δεν βρέθηκε τοπικό αρχείο. Ανέβασε το CSV για να συνεχίσεις.")
        st.stop()
else:
    df = pd.read_csv(uploaded_file)
    st.success("Το dataset φορτώθηκε επιτυχώς!")

# Προεπισκόπηση δεδομένων / Preview Dataset
st.subheader("Προεπισκόπηση")
st.write("Διαστάσεις:", df.shape)
st.dataframe(df.head())

# Τύποι στηλών / Column types
st.subheader("Τύποι στηλών")
st.write(df.dtypes)

# ---------------------------------------------------------
# Επιλογή target & features
# Choose target & features
# ---------------------------------------------------------

all_columns = df.columns.tolist()

# G3 default target if it exists
default_target = "G3" if "G3" in all_columns else all_columns[-1]

# User selects target
target_col = st.selectbox("Επίλεξε target variable", all_columns, index=all_columns.index(default_target))

# User selects features
feature_cols = st.multiselect(
    "Επίλεξε features (αν τα αφήσεις κενά, θα χρησιμοποιηθούν όλες οι υπόλοιπες στήλες)",
    [c for c in all_columns if c != target_col]
)

# All exept target if user does not choose
if len(feature_cols) == 0:
    feature_cols = [c for c in all_columns if c != target_col]

st.write("**Target:**", target_col)
st.write("**Features:**", feature_cols)

# Create X, y = features, target columns as input
X = df[feature_cols].copy()
y = df[target_col].copy()

# ---------------------------------------------------------
# Χωρισμός σε αριθμητικά & κατηγορικά features
# Split numeric and categorical features
# ---------------------------------------------------------

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

st.write("Αριθμητικά features:", numeric_features)
st.write("Κατηγορικά features:", categorical_features)

# ---------------------------------------------------------
# Ρυθμίσεις Προεπεξεργασίας / Preprocessing settings
# ---------------------------------------------------------

st.subheader("Ρυθμίσεις Προεπεξεργασίας")

# Choose strategy for missing values
missing_strategy = st.selectbox(
    "Στρατηγική για ελλιπείς τιμές",
    ["mean", "median", "most_frequent", "constant"],
    index=0
)

# Choose scaling
scale_numeric = st.checkbox("Κανονικοποίηση/Τυποποίηση αριθμητικών features (StandardScaler)", value=True)

# Numeric transformer for missing values and scaling the data
numeric_transformer_steps = [("imputer", SimpleImputer(strategy=missing_strategy))]
if scale_numeric:
    numeric_transformer_steps.append(("scaler", StandardScaler()))

numeric_transformer = Pipeline(steps=numeric_transformer_steps)

# Categorical transformer for missing values and encoding categoric to numeric
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ColumnTransformer = combine numeric + categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

st.success("Ορίστηκε preprocessor (impute + encoding + scaling).")

# ---------------------------------------------------------
# 2️⃣ Στάδιο 2: Διερευνητική Ανάλυση / EDA
# ---------------------------------------------------------

st.header("2. Διερευνητική Ανάλυση (EDA)")

# Tabs for EDA
eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Κατανομές", "Συσχετίσεις", "PCA 2D"])

# ---------------------------------------------------------
# TAB 1: Κατανομές / Distributions
# ---------------------------------------------------------

with eda_tab1:
    st.subheader("Κατανομές Features")
    col_to_plot = st.selectbox("Επίλεξε στήλη για histogram/boxplot", all_columns)

    if pd.api.types.is_numeric_dtype(df[col_to_plot]):            # Histogram + Boxplot (spread, skewness, outliers) if numerical
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax[0])
        ax[0].set_title(f"Histogram of {col_to_plot}")
        sns.boxplot(x=df[col_to_plot], ax=ax[1])
        ax[1].set_title(f"Boxplot of {col_to_plot}")
        st.pyplot(fig)
    else:                                                          # Barplot if categorical
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col_to_plot].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Counts of {col_to_plot}")
        st.pyplot(fig)

# ---------------------------------------------------------
# TAB 2: Correlation Heatmap / Θερμικός Χάρτης Συσχέτισης
# ---------------------------------------------------------

with eda_tab2:
    st.subheader("Correlation Heatmap (μόνο αριθμητικά)")
    num_df = df.select_dtypes(include=["int64", "float64"])

    if num_df.shape[1] > 1:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("Δεν υπάρχουν αρκετές αριθμητικές στήλες για heatmap.")

# ---------------------------------------------------------
# TAB 3: PCA 2D (Principal Component Analysis) / Ανάλυση Κύριων Συνιστωσών 
# ---------------------------------------------------------

with eda_tab3:
    st.subheader("PCA 2D (σε αριθμητικά features)")
    num_df = df[numeric_features].dropna()

    if num_df.shape[1] >= 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(num_df)

        pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA 2D Projection")
        st.pyplot(fig)

        st.write("Explained variance ratio:", pca.explained_variance_ratio_)
    else:
        st.write("Χρειάζονται τουλάχιστον 2 αριθμητικά features για PCA.")

# ---------------------------------------------------------
# 3️⃣ Στάδιο 3: ML Pipeline
# ---------------------------------------------------------

st.header("3. ML Pipeline")

ml_tab1, ml_tab2, ml_tab3 = st.tabs(["Regression (G3)", "Classification (Pass/Fail)", "Clustering (KMeans)"])

# ---------------------------------------------------------
# TAB 1: Regression / Οπισθοδρόμηση
# ---------------------------------------------------------

with ml_tab1:
    st.subheader("Regression για πρόβλεψη του G3")

    # Warning if target != G3
    if target_col != "G3":
        st.info("Για πιο λογική regression, συνήθως target = G3. Παρ' όλα αυτά, θα τρέξουμε με το επιλεγμένο target.")

    # Choose train/test split
    test_size = st.slider("Test size (ποσοστό test set)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=999, value=42, step=1)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    st.write("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Choose regression Model
    reg_model_name = st.selectbox("Επίλεξε regression αλγόριθμο", ["LinearRegression", "RandomForestRegressor"])

    if reg_model_name == "LinearRegression":
        reg_model = LinearRegression()
    else:
        n_estimators = st.slider("n_estimators (RandomForest)", 50, 300, 100, 50)
        max_depth = st.slider("max_depth (RandomForest)", 2, 20, 8, 1)
        reg_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    # Pipeline = preprocessing + model
    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", reg_model)
        ]
    )

    # Run Regression
    if st.button("Τρέξε Regression"):
        reg_pipeline.fit(X_train, y_train)
        y_pred = reg_pipeline.predict(X_test)

        # Μετρικές / Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.write("**RMSE:**", rmse)
        st.write("**R²:**", r2)

        # Plot: True vs Predicted
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("True vs Predicted")
        st.pyplot(fig)

# ---------------------------------------------------------
# TAB 2: Classification / Ταξινόμηση
# ---------------------------------------------------------

with ml_tab2:
    st.subheader("Classification – Pass/Fail από G3")

    #Check if G3 exists
    if "G3" not in df.columns:
        st.warning("Δεν υπάρχει στήλη G3 για να ορίσουμε pass/fail.")
    else:
        # Create binary target for pass/fail
        y_class = (df["G3"] >= 10).astype(int)
        st.write("0 = fail, 1 = pass")

        X_class = df[feature_cols].copy()

        # Train/test split
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42
        )

        # Choose Classification Model
        clf_model_name = st.selectbox("Επίλεξε classification αλγόριθμο", ["LogisticRegression", "RandomForestClassifier"])

        if clf_model_name == "LogisticRegression":
            C = st.slider("C (inverse regularization strength)", 0.01, 10.0, 1.0, 0.01)
            clf_model = LogisticRegression(max_iter=1000, C=C)
        else:
            n_estimators_c = st.slider("n_estimators (RandomForest)", 50, 300, 100, 50)
            max_depth_c = st.slider("max_depth (RandomForest)", 2, 20, 8, 1)
            clf_model = RandomForestClassifier(
                n_estimators=n_estimators_c,
                max_depth=max_depth_c,
                random_state=42
            )

        # Pipeline
        clf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clf_model)
            ]
        )

        # Run Classification
        if st.button("Τρέξε Classification"):
            clf_pipeline.fit(X_train_c, y_train_c)
            y_pred_c = clf_pipeline.predict(X_test_c)

            # Accuracy
            acc = accuracy_score(y_test_c, y_pred_c)
            st.write("**Accuracy:**", acc)

            # Classification report
            st.text("Classification report:")
            st.text(classification_report(y_test_c, y_pred_c))

            # Confusion matrix
            cm = confusion_matrix(y_test_c, y_pred_c)
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3: Clustering
# ---------------------------------------------------------

with ml_tab3:
    st.subheader("Unsupervised Clustering – KMeans")

    # Use only numerical features
    num_X = df[numeric_features].dropna()

    if num_X.shape[1] < 2:
        st.write("Χρειάζονται τουλάχιστον 2 αριθμητικά features για clustering.")
    else:
        n_clusters = st.slider("Αριθμός clusters (k)", 2, 10, 3, 1)  #choose how many clusters

        # Scaling for clustering
        scaler_clust = StandardScaler()
        X_scaled = scaler_clust.fit_transform(num_X)

        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_scaled)

        st.write("Cluster counts:", pd.Series(labels).value_counts())

        # PCA for visualization
        pca_clust = PCA(n_components=2)
        X_pca_clust = pca_clust.fit_transform(X_scaled)

        pca_df = pd.DataFrame(X_pca_clust, columns=["PC1", "PC2"])
        pca_df["cluster"] = labels

        # Scatter plot for visualization of the groups/clusters
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="cluster",
            palette="tab10",
            ax=ax
        )
        ax.set_title("KMeans Clusters (PCA 2D)")
        st.pyplot(fig)
