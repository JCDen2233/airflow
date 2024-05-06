import pandas as pd
import dill

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def apply_boundaries(df):
    import copy
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries
    
    def short_model(x):
        import pandas as pd
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x
    
    b_df = copy.copy(df)
    boundaries = calculate_outliers(b_df['bollinger_high'])

    b_df.loc[b_df['bollinger_high'] < boundaries[0], 'bollinger_high'] = round(boundaries[0])
    b_df.loc[b_df['bollinger_high'] > boundaries[1], 'bollinger_high'] = round(boundaries[1])
    
    boundaries = calculate_outliers(b_df['bollinger_low'])
    
    b_df.loc[b_df['bollinger_low'] < boundaries[0], 'bollinger_low'] = round(boundaries[0])
    b_df.loc[b_df['bollinger_low'] > boundaries[1], 'bollinger_low'] = round(boundaries[1])
    
    boundaries = calculate_outliers(b_df['fast_MA'])
    
    b_df.loc[b_df['fast_MA'] < boundaries[0], 'fast_MA'] = round(boundaries[0])
    b_df.loc[b_df['fast_MA'] > boundaries[1], 'fast_MA'] = round(boundaries[1])
    
    boundaries = calculate_outliers(b_df['slow_MA'])
    
    b_df.loc[b_df['slow_MA'] < boundaries[0], 'slow_MA'] = round(boundaries[0])
    b_df.loc[b_df['slow_MA'] > boundaries[1], 'slow_MA'] = round(boundaries[1])

    boundaries = calculate_outliers(b_df['close'])
    
    b_df.loc[b_df['close'] < boundaries[0], 'close'] = round(boundaries[0])
    b_df.loc[b_df['close'] > boundaries[1], 'close'] = round(boundaries[1])
    
    boundaries = calculate_outliers(b_df['open'])
    
    b_df.loc[b_df['open'] < boundaries[0], 'open'] = round(boundaries[0])
    b_df.loc[b_df['open'] > boundaries[1], 'open'] = round(boundaries[1])
    
    boundaries = calculate_outliers(b_df['high'])
    
    b_df.loc[b_df['high'] < boundaries[0], 'high'] = round(boundaries[0])
    b_df.loc[b_df['high'] > boundaries[1], 'high'] = round(boundaries[1])
    
    boundaries = calculate_outliers(b_df['low'])
    
    b_df.loc[b_df['low'] < boundaries[0], 'low'] = round(boundaries[0])
    b_df.loc[b_df['low'] > boundaries[1], 'low'] = round(boundaries[1])
    
    return b_df

def main():
    print("Starting car model price category prediction Pipeline\n")
    df = pd.read_csv('train_data.csv')
    
    print(df['out'].value_counts())
    
    df = df.groupby('out').head(df.groupby('out').size().min())
    
    print(df['out'].value_counts())

    X = df.drop(['out'], axis=1)
    Y = df['out']    
    
    types = {
    'int64': 'int',
    'float64': 'float'
    }
    
    s_types = ""
    
    for k, v in X.dtypes.iteritems():
        s_types += f'{k}: {types.get(str(v), "str")}\n'

    print(s_types)
    
    prep_df = Pipeline(steps=[
        ('boundaries_df', FunctionTransformer(apply_boundaries)),
    ])
        
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=['object']))
    ])

    models = [
        LogisticRegression(solver='liblinear', C=1500),
        RandomForestClassifier(n_estimators=500, min_samples_split=12, min_samples_leaf=50, max_features='sqrt', max_depth=100, bootstrap=False),
        SVC()
    ]

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        #if type(model).__name__ == "RandomForestClassifier":
        #    print('start search')
        #    from sklearn.model_selection import RandomizedSearchCV
        #    import numpy as np
        #    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
        #    max_features = ['log2', 'sqrt']
        #    max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
        #    min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
        #    min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
        #    bootstrap = [True, False]
        #   param_dist = {'classifier__n_estimators': n_estimators,
        #                   'classifier__max_features': max_features,
        #                   'classifier__max_depth': max_depth,
        #                   'classifier__min_samples_split': min_samples_split,
        #                   'classifier__min_samples_leaf': min_samples_leaf,
        #                   'classifier__bootstrap': bootstrap}
        #    rs = RandomizedSearchCV(pipe, 
        #                            param_dist, 
        #                            n_iter = 100, 
        #                            cv = 3, 
        #                            verbose = 1, 
        #                            n_jobs=-1, 
        #                            random_state=0)
        #    rs.fit(X, Y)
        #    print(rs.best_params_)
        
        score = cross_val_score(pipe, X, Y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(X)
            
    print("\n" + f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    
    best_pipe.fit(X,Y)
    m_y = best_pipe.predict(X)
    

    
    print(confusion_matrix(Y, m_y))
    
    file_name = 'car_cat_pipe.pickle'
    dump_object = {
        'model' : best_pipe,
        'pydantic' : s_types,
        'meta' : {
            'name' : 'car model price category prediction Pipeline',
            'author' : 'Sergey',
            'version' : '1.0',
            'predictor' : type(best_pipe.named_steps["classifier"]).__name__,
            'accuracy' : best_score,
        }
    }
    
    with open(file_name, 'wb') as file:
        dill.dump(dump_object, file)
    
    print(f'\nModel saved in file: {file_name}')
    print("\nWe're done here. Next!")


if __name__ == '__main__':
    main()

