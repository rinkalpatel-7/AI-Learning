import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


X = train_data.drop('income_>50K',axis=1)
y = train_data['income_>50K']

X_train, X_test, y_train,  y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# print(X_train)
# print(y_train)

num_features = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']
num_transformer = Pipeline(steps=
                           [
                               ('imputer',SimpleImputer(strategy='median')),
                               ('scaler',StandardScaler())
                           ]
                           )

cat_features = ['workclass','education','marital-status','occupation','relationship','race','gender','native-country']
cat_transformer = Pipeline(steps=
                           [
                               ('imputer',SimpleImputer(strategy='most_frequent' )),
                               ('onehot',OneHotEncoder(handle_unknown='ignore'))
                           ])

pre_processor = ColumnTransformer(
                        transformers =  [
                                            ('num',num_transformer,num_features),
                                            ('cat',cat_transformer,cat_features)
                                        ]
                                )


model = Pipeline(
    steps=
    [
        ('preprocessor',pre_processor),
        ('classifier',RandomForestClassifier(n_estimators=100,random_state=42))
    ]
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))

new_pred = model.predict(test_data)

test_data['income_>50K'] = new_pred

test_data.to_csv('new_pred.csv',index=False)