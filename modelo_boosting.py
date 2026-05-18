import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# cargar archivos
train = pd.read_csv("competencia_icd_seguros_train.csv")
test = pd.read_csv("competencia_icd_seguros_test.csv")

print(test.columns)


# pasar variables categoricas a numeros
def preparar_datos(df):

    df = df.copy()

    df["Gender"] = df["Gender"].replace({
        "Female": 0,
        "Male": 1
    })

    df["Vehicle_Age"] = df["Vehicle_Age"].replace({
        "< 1 Year": 0,
        "1-2 Year": 1,
        "> 2 Years": 2
    })

    df["Vehicle_Damage"] = df["Vehicle_Damage"].replace({
        "No": 0,
        "Yes": 1
    })

    return df


train = preparar_datos(train)
test = preparar_datos(test)


# separar variable objetivo
y = train["Response"]
X = train.drop("Response", axis=1)


# usar numero de fila como id
ids = range(len(test))


# dividir para validar
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# modelo boosting
modelo = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

modelo.fit(X_train, y_train)


# evaluar
pred_valid = modelo.predict_proba(X_valid)[:, 1]

score = roc_auc_score(
    y_valid,
    pred_valid
)

print("ROC AUC:", score)


# predecir test
pred_test = modelo.predict_proba(test)[:, 1]


# generar archivo para kaggle
submission = pd.DataFrame({
    "test_client_id": ids,
    "Response": pred_test
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("submission.csv generado")