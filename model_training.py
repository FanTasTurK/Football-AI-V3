from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import joblib
import os

# Models klasörü yolu
MODELS_DIR = 'models'

def evaluate_model(model, X, y, model_type, name, cv=5):
    """
    Model performansını time series cross validation ile değerlendirir.
    """
    # TimeSeriesSplit kullanarak zaman bazlı cross-validation
    tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.2))
    
    if model_type == 'regression':
        # Regresyon modelleri için RMSE kullan
        scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
        scores = cross_val_score(model, X, y, cv=tscv, scoring=scorer)
        print(f"\n{name} Cross-Validation RMSE Skorları:")
        print(f"Ortalama: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    else:
        # Sınıflandırma modelleri için accuracy kullan
        scores = cross_val_score(model, X, y, cv=tscv, scoring=make_scorer(accuracy_score))
        print(f"\n{name} Cross-Validation Accuracy Skorları:")
        print(f"Ortalama: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores.mean()

def train_models(X, y, y_score=None, y_htft=None, y_btts=None):
    """
    Birden fazla model eğitir ve en iyi modelleri seçer.
    Regularizasyon ve zaman bazlı cross-validation kullanır.
    """
    # Eksik değerleri kontrol et
    if X.isnull().any().any():
        print("Uyarı: Veride eksik değerler var. Sıfır ile doldurulacak.")
        X = X.fillna(0)
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Son 2 sezonu test seti olarak ayır (yaklaşık %20)
    test_size = int(len(X) * 0.2)
    X_train = X_scaled[:-test_size]
    X_test = X_scaled[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]
    
    models = {}
    
    # LazyPredict ile model karşılaştırması
    print("\nLazyPredict ile model karşılaştırması yapılıyor...")
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    lazy_pred = clf.fit(X_train, X_test, y_train, y_test)
    print("\nTüm modellerin performans karşılaştırması:")
    print(lazy_pred)
    
    # Maç sonucu modelleri (regularizasyon eklenmiş)
    print("\nMaç sonucu modelleri eğitiliyor...")
    models['match_result'] = {
        'RandomForest': RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'SVC': SVC(
            probability=True,
            C=0.8,
            kernel='rbf',
            random_state=42
        )
    }
    
    for name, model in models['match_result'].items():
        print(f"\n{name} için cross validation yapılıyor...")
        evaluate_model(model, X_train, y_train, 'classification', name)
        model.fit(X_train, y_train)
    
    # Skor tahmin modelleri
    if y_score is not None:
        print("\nSkor tahmin modelleri eğitiliyor...")
        
        # LazyRegressor ile model karşılaştırması
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        lazy_reg_pred = reg.fit(X_train, X_test, y_train, y_test)
        print("\nRegresyon modellerinin performans karşılaştırması:")
        print(lazy_reg_pred)
        
        models['score'] = {
            'RandomForest': RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'SVR': SVR(
                C=0.8,
                kernel='rbf',
                epsilon=0.2
            )
        }
        for name, model in models['score'].items():
            print(f"\n{name} için cross validation yapılıyor...")
            evaluate_model(model, X_train, y_score[:-test_size], 'regression', name)
            model.fit(X_train, y_score[:-test_size])
    
    # İY/MS tahmin modelleri
    if y_htft is not None:
        print("\nİY/MS tahmin modelleri eğitiliyor...")
        models['htft'] = {
            'RandomForest': RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'SVC': SVC(
                probability=True,
                C=0.8,
                kernel='rbf',
                random_state=42
            )
        }
        for name, model in models['htft'].items():
            print(f"\n{name} için cross validation yapılıyor...")
            evaluate_model(model, X_train, y_htft[:-test_size], 'classification', name)
            model.fit(X_train, y_htft[:-test_size])
    
    # KG tahmin modelleri (daha sıkı regularizasyon)
    if y_btts is not None:
        print("\nKG tahmin modelleri eğitiliyor...")
        models['btts'] = {
            'RandomForest': RandomForestClassifier(
                n_estimators=2000,
                max_depth=3,
                min_samples_split=30,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=2000,
                learning_rate=0.001,
                max_depth=2,
                min_samples_split=30,
                min_samples_leaf=10,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            ),
            'SVC': SVC(
                probability=True,
                C=0.3,
                kernel='rbf',
                class_weight='balanced',
                random_state=42
            )
        }
        
        # Feature selection için SelectFromModel kullan
        selector = SelectFromModel(
            RandomForestClassifier(
                n_estimators=2000,
                max_depth=3,
                min_samples_split=30,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42
            ),
            max_features=8,  # En önemli 8 özelliği seç
            threshold='median'
        )
        X_selected = selector.fit_transform(X_train, y_btts[:-test_size])
        X_test_selected = selector.transform(X_test)
        
        # Selector'ı modeller sözlüğüne ekle
        models['btts_selector'] = selector
        
        # Seçilen özellikleri kullanarak modelleri eğit
        for name, model in models['btts'].items():
            print(f"\n{name} için cross validation yapılıyor...")
            evaluate_model(model, X_selected, y_btts[:-test_size], 'classification', name)
            model.fit(X_selected, y_btts[:-test_size])
            
        # Feature importance bilgisini kaydet ve göster
        feature_importance = pd.DataFrame({
            'feature': X.columns[selector.get_support()],
            'importance': selector.estimator_.feature_importances_[selector.get_support()]
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print("\nSeçilen 15 Önemli Özellik:")
        print(feature_importance)
    
    return models, scaler

def save_models(models, scaler, prefix=''):
    """
    Eğitilmiş modelleri ve scaler'ı models klasörüne kaydeder.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_type, model_dict in models.items():
        if model_type == 'btts_selector':
            # Selector'ı ayrı kaydet
            model_path = os.path.join(MODELS_DIR, f'{prefix}{model_type}.joblib')
            joblib.dump(model_dict, model_path)
        else:
            # Normal modelleri kaydet
            for name, model in model_dict.items():
                model_path = os.path.join(MODELS_DIR, f'{prefix}{model_type}_{name}_model.joblib')
                joblib.dump(model, model_path)
    
    scaler_path = os.path.join(MODELS_DIR, f'{prefix}scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModeller {MODELS_DIR} klasörüne kaydedildi.")

def load_models(prefix=''):
    """
    Kaydedilmiş modelleri ve scaler'ı models klasöründen yükler.
    """
    try:
        models = {
            'match_result': {},
            'score': {},
            'htft': {},
            'btts': {}
        }
        
        model_types = ['match_result', 'score', 'htft', 'btts']
        model_names = ['RandomForest', 'GradientBoosting', 'SVC']
        
        for model_type in model_types:
            for name in model_names:
                model_path = os.path.join(MODELS_DIR, f'{prefix}{model_type}_{name}_model.joblib')
                if os.path.exists(model_path):
                    models[model_type][name] = joblib.load(model_path)
        
        # Selector'ı yükle
        selector_path = os.path.join(MODELS_DIR, f'{prefix}btts_selector.joblib')
        if os.path.exists(selector_path):
            models['btts_selector'] = joblib.load(selector_path)
        
        scaler_path = os.path.join(MODELS_DIR, f'{prefix}scaler.joblib')
        scaler = joblib.load(scaler_path)
        
        return models, scaler
    except Exception as e:
        print(f"Modeller yüklenirken hata oluştu: {str(e)}")
        print("Modeller yeniden eğitilecek.")
        return None, None 