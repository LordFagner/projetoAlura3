import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split,
                                     cross_validate,
                                      StratifiedKFold,)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.feature_selection import mutual_info_classif
from imblearn.under_sampling import NearMiss
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



url = '/home/fagner/ProjetoAlura3/data/TelecomX_Data_Normalizado_limpo.csv'
df = pd.read_csv(url)

#print(df.info())
#print(df['remainder__churn'].value_counts())
#rint(df.corr)
df_clean = df.dropna()
print(df.shape)
print(df_clean.shape)


y = df_clean['remainder__churn']
x = df_clean.drop(axis=1, columns= 'remainder__churn')

mic = mutual_info_classif(x,y,discrete_features='auto') 

mic_df = pd.DataFrame({'Features': x.columns, 'MI': mic}).reset_index()
mic_df.sort_values('MI', ascending=False, inplace=True)
#print(mic_df)


limite = 0.01 
features_Relevantes = mic_df[mic_df['MI'] >= limite ]['Features'].tolist()
x_relevante = x[features_Relevantes]



#organizando dados 

''' 
dados relevantes em relação a x e y : 
 index                           Feature        MI
5       5         onehotencoder__contract_2  0.085471
24     24                 remainder__tenure  0.073468
6       6         onehotencoder__contract_3  0.055854
8       8    onehotencoder__paymentmethod_2  0.047189
3       3  onehotencoder__internetservice_2  0.040675
19     19          remainder__charges_total  0.039975
18     18        remainder__charges_monthly  0.035922
1       1  onehotencoder__internetservice_0  0.021691
17     17       remainder__paperlessbilling  0.019522
4       4         onehotencoder__contract_1  0.017663
14     14            remainder__techsupport  0.015361
11     11         remainder__onlinesecurity  0.014797
10     10    onehotencoder__paymentmethod_4  0.011757

paymentMethod{'Mailed check': 1, 'Electronic check': 2, 'Credit card (automatic)': 3, 'Bank transfer (automatic)': 4}
InternetService{'DSL': 1, 'Fiber optic': 2, 'no': 0}
Contract{one-year : 1, month-to-month:2 ,two-year : 3}

'''
# Removendo imports não utilizados
# Removidos: confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def resultado(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'Intervalo de confiança: [{media - 2*desvio_padrao}, {min(media + 2*desvio_padrao, 1)}]')


# Separação treino/teste
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
x_t, x_teste, y_t, y_teste = train_test_split(x, y, test_size=0.15, stratify=y, random_state=5)

# === MODELO 1: Decision Tree ===
undersample = NearMiss(version=3)
x_balance, y_balance = undersample.fit_resample(x_t, y_t)
modelo = DecisionTreeClassifier(max_depth=10)
pipeline = imbpipeline([
    ('undersampling', NearMiss(version=3)),
    ('arvore', modelo)
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
cv_resultados = cross_validate(pipeline, x_t, y_t, cv=skf, scoring='recall')
resultado(cv_resultados)

undersample = NearMiss(version=3)
x_balance, y_balance = undersample.fit_resample(x_t, y_t)
modelo.fit(x_balance, y_balance)
y_prev = modelo.predict(x_teste)

print("Decision Tree - Relatório de Classificação:")
print(classification_report(y_teste, y_prev))
ConfusionMatrixDisplay.from_predictions(y_teste, y_prev).plot()
plt.title("Decision Tree")
plt.show()

# === MODELO 2: Random Forest ===
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=5)
pipeline_rf = imbpipeline([
    ('undersampling', NearMiss(version=3)),
    ('rf', modelo_rf)
])
cv_rf = cross_validate(pipeline_rf, x_t, y_t, cv=skf, scoring='recall')
resultado(cv_rf)

x_balance_rf, y_balance_rf = undersample.fit_resample(x_t, y_t)
modelo_rf.fit(x_balance_rf, y_balance_rf)
y_prev_rf = modelo_rf.predict(x_teste)

print("Random Forest - Relatório de Classificação:")
print(classification_report(y_teste, y_prev_rf))
ConfusionMatrixDisplay.from_predictions(y_teste, y_prev_rf).plot()
plt.title("Random Forest")
plt.show()

# === MODELO 3: Regressão Logística ===
modelo_lr = LogisticRegression(max_iter=1000, solver='liblinear')
pipeline_lr = imbpipeline([
    ('undersampling', NearMiss(version=3)),
    ('lr', modelo_lr)
])
cv_lr = cross_validate(pipeline_lr, x_t, y_t, cv=skf, scoring='recall')
resultado(cv_lr)

x_balance_lr, y_balance_lr = undersample.fit_resample(x_t, y_t)
modelo_lr.fit(x_balance_lr, y_balance_lr)
y_prev_lr = modelo_lr.predict(x_teste)

print("Regressão Logística - Relatório de Classificação:")
print(classification_report(y_teste, y_prev_lr))
ConfusionMatrixDisplay.from_predictions(y_teste, y_prev_lr).plot()
plt.title("Regressão Logística")
plt.show()
