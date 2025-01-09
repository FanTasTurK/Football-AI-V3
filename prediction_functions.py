import numpy as np
from data_preprocessing import get_team_stats, get_head_to_head_stats, prepare_features

def predict_match_result(home_team, away_team, models, scaler):
    """
    İki takım arasındaki maç sonucunu tahmin eder.
    """
    try:
        # Takımların son maçlarındaki performanslarını al
        home_data = get_team_stats(home_team)
        away_data = get_team_stats(away_team)
        
        # Son 5 maçın ortalamasını al
        home_recent = home_data.head(5)
        away_recent = away_data.head(5)
        
        # Geçmiş karşılaşmaları al
        h2h_home, h2h_away = get_head_to_head_stats(home_team, away_team)
        
        # Özellikleri hazırla
        X_home, _, _, _, _ = prepare_features(home_recent)
        X_away, _, _, _, _ = prepare_features(away_recent)
        
        # Ortalama değerleri al
        X_home_mean = X_home.mean().values.reshape(1, -1)
        X_away_mean = X_away.mean().values.reshape(1, -1)
        
        # Verileri ölçeklendir
        X_home_scaled = scaler.transform(X_home_mean)
        X_away_scaled = scaler.transform(X_away_mean)
        
        # Her model için tahminleri al
        predictions = {}
        for name, model in models['match_result'].items():
            home_pred = model.predict_proba(X_home_scaled)[0]
            away_pred = model.predict_proba(X_away_scaled)[0]
            
            # Ev sahibi avantajını hesaba kat
            home_advantage = 0.1
            predictions[name] = {
                'home_win': (home_pred[2] + home_advantage) * 0.6 + away_pred[0] * 0.4,  # Galip (2)
                'draw': home_pred[1] * 0.5 + away_pred[1] * 0.5,                         # Berabere (1)
                'away_win': home_pred[0] * 0.4 + (away_pred[2] + home_advantage) * 0.6   # Mağlup (0)
            }
        
        # Tüm modellerin tahminlerini ortala
        final_pred = {
            'home_win': np.mean([p['home_win'] for p in predictions.values()]),
            'draw': np.mean([p['draw'] for p in predictions.values()]),
            'away_win': np.mean([p['away_win'] for p in predictions.values()])
        }
        
        # Olasılıkları normalize et
        total = sum(final_pred.values())
        if total > 0:
            for key in final_pred:
                final_pred[key] /= total
        
        return final_pred
    except Exception as e:
        print(f"Tahmin sırasında bir hata oluştu: {str(e)}")
        return {'home_win': 0.33, 'draw': 0.34, 'away_win': 0.33}

def predict_score(home_team, away_team, models, scaler):
    """
    Maç skorunu ve olasılığını tahmin eder.
    """
    try:
        # Takımların son maçlarındaki performanslarını al
        home_data = get_team_stats(home_team)
        away_data = get_team_stats(away_team)
        
        # Son 5 maçın ortalamasını al
        home_recent = home_data.head(5)
        away_recent = away_data.head(5)
        
        # Özellikleri hazırla
        X_home, _, _, _, _ = prepare_features(home_recent)
        X_away, _, _, _, _ = prepare_features(away_recent)
        
        # Ortalama değerleri al
        X_home_mean = X_home.mean().values.reshape(1, -1)
        X_away_mean = X_away.mean().values.reshape(1, -1)
        
        # Verileri ölçeklendir
        X_home_scaled = scaler.transform(X_home_mean)
        X_away_scaled = scaler.transform(X_away_mean)
        
        # Her model için tahminleri al
        home_goals_pred = []
        away_goals_pred = []
        
        for name, model in models['score'].items():
            home_goals_pred.append(model.predict(X_home_scaled)[0])
            away_goals_pred.append(model.predict(X_away_scaled)[0])
        
        # Tahminlerin ortalamasını al ve yuvarla
        home_goals = round(np.mean(home_goals_pred))
        away_goals = round(np.mean(away_goals_pred))
        
        # Olasılık hesapla
        # Tahminlerin standart sapması ve ortalaması arasındaki farka göre olasılık hesapla
        home_std = np.std(home_goals_pred)
        away_std = np.std(away_goals_pred)
        home_mean = np.mean(home_goals_pred)
        away_mean = np.mean(away_goals_pred)
        
        # Tahmin edilen skor ile ortalama arasındaki farkı hesapla
        home_diff = abs(home_goals - home_mean)
        away_diff = abs(away_goals - away_mean)
        
        # Olasılığı hesapla (ne kadar sapma varsa o kadar düşük olasılık)
        score_prob = np.exp(-(home_diff + away_diff) / 2) * 0.7  # Max %70 olasılık
        
        return max(0, home_goals), max(0, away_goals), score_prob
    except Exception as e:
        print(f"Skor tahmini sırasında bir hata oluştu: {str(e)}")
        return 1, 1, 0.33

def predict_ht_ft(home_team, away_team, models, scaler):
    """
    İlk Yarı / Maç Sonucu tahminini ve olasılığını yapar.
    """
    try:
        # Takımların son maçlarındaki performanslarını al
        home_data = get_team_stats(home_team)
        away_data = get_team_stats(away_team)
        
        # Son 5 maçın ortalamasını al
        home_recent = home_data.head(5)
        away_recent = away_data.head(5)
        
        # Özellikleri hazırla
        X_home, _, _, _, _ = prepare_features(home_recent)
        X_away, _, _, _, _ = prepare_features(away_recent)
        
        # Ortalama değerleri al
        X_home_mean = X_home.mean().values.reshape(1, -1)
        X_away_mean = X_away.mean().values.reshape(1, -1)
        
        # Verileri ölçeklendir
        X_home_scaled = scaler.transform(X_home_mean)
        X_away_scaled = scaler.transform(X_away_mean)
        
        # Her model için tahminleri al
        predictions = {}
        probabilities = []
        
        for name, model in models['htft'].items():
            home_pred = model.predict(X_home_scaled)[0]
            away_pred = model.predict(X_away_scaled)[0]
            home_prob = np.max(model.predict_proba(X_home_scaled)[0])
            away_prob = np.max(model.predict_proba(X_away_scaled)[0])
            
            predictions[name] = home_pred if home_prob > away_prob else away_pred
            probabilities.append(max(home_prob, away_prob))
        
        # En çok tahmin edilen sonucu bul
        from collections import Counter
        most_common = Counter(predictions.values()).most_common(1)[0][0]
        
        # Olasılık hesapla
        # Tahminlerin standart sapmasına göre olasılığı azalt
        prob_std = np.std(probabilities)
        avg_prob = np.mean(probabilities)
        
        # Olasılığı sınırla (maksimum %65)
        final_prob = min(0.65, avg_prob * np.exp(-prob_std))
        
        return most_common, final_prob
    except Exception as e:
        print(f"İY/MS tahmini sırasında bir hata oluştu: {str(e)}")
        return "X-X", 0.33

def predict_btts(home_team, away_team, models, scaler):
    """
    Karşılıklı Gol tahminini ve olasılığını yapar.
    """
    try:
        # Takımların son maçlarındaki performanslarını al
        home_data = get_team_stats(home_team)
        away_data = get_team_stats(away_team)
        
        # Son 5 maçın ortalamasını al
        home_recent = home_data.head(5)
        away_recent = away_data.head(5)
        
        # Özellikleri hazırla
        X_home, _, _, _, _ = prepare_features(home_recent)
        X_away, _, _, _, _ = prepare_features(away_recent)
        
        # Ortalama değerleri al
        X_home_mean = X_home.mean().values.reshape(1, -1)
        X_away_mean = X_away.mean().values.reshape(1, -1)
        
        # Verileri ölçeklendir
        X_home_scaled = scaler.transform(X_home_mean)
        X_away_scaled = scaler.transform(X_away_mean)
        
        # Feature selection uygula
        selector = models.get('btts_selector')
        if selector:
            X_home_scaled = selector.transform(X_home_scaled)
            X_away_scaled = selector.transform(X_away_scaled)
        
        # Her model için tahminleri al
        predictions = {}
        probabilities = []
        
        for name, model in models['btts'].items():
            home_pred = model.predict(X_home_scaled)[0]
            away_pred = model.predict(X_away_scaled)[0]
            home_prob = np.max(model.predict_proba(X_home_scaled)[0])
            away_prob = np.max(model.predict_proba(X_away_scaled)[0])
            
            predictions[name] = bool(home_pred or away_pred)
            probabilities.append(max(home_prob, away_prob))
        
        # En çok tahmin edilen sonucu bul
        from collections import Counter
        most_common = Counter(predictions.values()).most_common(1)[0][0]
        
        # Olasılık hesapla
        # Tahminlerin tutarlılığına göre olasılığı ayarla
        prob_std = np.std(probabilities)
        avg_prob = np.mean(probabilities)
        
        # Olasılığı sınırla (maksimum %65)
        final_prob = min(0.65, avg_prob * np.exp(-prob_std))
        
        return "VAR" if most_common else "YOK", final_prob
    except Exception as e:
        print(f"KG tahmini sırasında bir hata oluştu: {str(e)}")
        return "YOK", 0.33

def get_highest_probability_prediction(final_pred, score_prob, htft_prob, btts_prob):
    """
    En yüksek olasılıklı tahmini belirler.
    """
    predictions = {
        'Ev Sahibi Kazanır': final_pred['home_win'],
        'Beraberlik': final_pred['draw'],
        'Deplasman Kazanır': final_pred['away_win'],
        'Tahmini Skor': score_prob,
        'İY/MS': htft_prob,
        'Karşılıklı Gol': btts_prob
    }
    
    # En yüksek olasılıklı tahmini bul
    max_pred = max(predictions.items(), key=lambda x: x[1])
    return max_pred[0], max_pred[1] 