from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from data_preprocessing import get_team_stats, prepare_features
from model_training import train_models, save_models, load_models, MODELS_DIR
from prediction_functions import predict_match_result, predict_score, predict_ht_ft, predict_btts
from prediction import get_team_performance_stats

app = Flask(__name__)

def init_models():
    """Modelleri yükler veya yeniden eğitir."""
    try:
        # Önce kaydedilmiş modelleri yüklemeyi dene
        models, scaler = load_models()
        if models is not None and scaler is not None:
            print("Kaydedilmiş modeller başarıyla yüklendi.")
            return models, scaler
        
        print("Kaydedilmiş modeller bulunamadı. Yeniden eğitiliyor...")
        
        # Tüm takımların verilerini birleştir
        all_data = []
        for file in os.listdir('stats'):
            if file.endswith('.csv'):
                team_name = file[:-4]
                team_data = get_team_stats(team_name)
                all_data.append(team_data)
        
        # Özellikleri hazırla
        features, y_match, y_score, y_htft, y_btts = prepare_features(pd.concat(all_data))
        
        # Modelleri eğit
        models, scaler = train_models(features, y_match, y_score, y_htft, y_btts)
        
        # Modelleri kaydet
        save_models(models, scaler)
        print("Modeller eğitildi ve kaydedildi.")
        return models, scaler
    except Exception as e:
        print(f"Model eğitimi sırasında hata oluştu: {str(e)}")
        return None, None

def get_available_teams():
    """Mevcut takımların listesini döndürür."""
    teams = []
    for file in os.listdir('stats'):
        if file.endswith('.csv'):
            teams.append(file[:-4])
    return sorted(teams)

def analyze_team_comparison(home_team, away_team, home_stats, away_stats):
    """Takımların karşılaştırmalı analizini yapar."""
    home_better = []
    away_better = []
    
    # Puan durumu
    if home_stats['Puan'] > away_stats['Puan']:
        home_better.append(f"{home_team} son 5 maçta {home_stats['Puan']} puan topladı (daha iyi form)")
    else:
        away_better.append(f"{away_team} son 5 maçta {away_stats['Puan']} puan topladı (daha iyi form)")
    
    # Gol performansı
    if home_stats['MS Gol Ort'] > away_stats['MS Gol Ort']:
        home_better.append(f"{home_team} daha fazla gol atıyor (maç başı {home_stats['MS Gol Ort']:.1f} gol)")
    else:
        away_better.append(f"{away_team} daha fazla gol atıyor (maç başı {away_stats['MS Gol Ort']:.1f} gol)")
    
    # İlk yarı gol performansı
    if home_stats['İY Gol Ort'] > away_stats['İY Gol Ort']:
        home_better.append(f"{home_team} ilk yarıda daha etkili (maç başı {home_stats['İY Gol Ort']:.1f} gol)")
    else:
        away_better.append(f"{away_team} ilk yarıda daha etkili (maç başı {away_stats['İY Gol Ort']:.1f} gol)")
    
    # Top kontrolü
    if home_stats['Topla Oynama'] > away_stats['Topla Oynama']:
        home_better.append(f"{home_team} topla daha fazla oynuyor (%{home_stats['Topla Oynama']:.1f})")
    else:
        away_better.append(f"{away_team} topla daha fazla oynuyor (%{away_stats['Topla Oynama']:.1f})")
    
    # Pas organizasyonu
    if home_stats['Pas İsabeti'] > away_stats['Pas İsabeti']:
        home_better.append(f"{home_team}'nin pas isabeti daha yüksek (%{home_stats['Pas İsabeti']:.1f})")
    else:
        away_better.append(f"{away_team}'nin pas isabeti daha yüksek (%{away_stats['Pas İsabeti']:.1f})")
    
    # Şut etkinliği
    home_shot_accuracy = (home_stats['İsabetli Şut'] / home_stats['Toplam Şut'] * 100) if home_stats['Toplam Şut'] > 0 else 0
    away_shot_accuracy = (away_stats['İsabetli Şut'] / away_stats['Toplam Şut'] * 100) if away_stats['Toplam Şut'] > 0 else 0
    
    if home_shot_accuracy > away_shot_accuracy:
        home_better.append(f"{home_team}'nin şut isabeti daha yüksek (%{home_shot_accuracy:.1f})")
    else:
        away_better.append(f"{away_team}'nin şut isabeti daha yüksek (%{away_shot_accuracy:.1f})")
    
    # Gol beklentisi
    if home_stats['Gol Beklentisi'] > away_stats['Gol Beklentisi']:
        home_better.append(f"{home_team}'nin gol beklentisi daha yüksek ({home_stats['Gol Beklentisi']:.2f})")
    else:
        away_better.append(f"{away_team}'nin gol beklentisi daha yüksek ({away_stats['Gol Beklentisi']:.2f})")
    
    # Ceza sahası etkinliği
    if home_stats['Ceza Sahası Etkinliği'] > away_stats['Ceza Sahası Etkinliği']:
        home_better.append(f"{home_team} ceza sahasında daha etkili ({home_stats['Ceza Sahası Etkinliği']:.1f} kez)")
    else:
        away_better.append(f"{away_team} ceza sahasında daha etkili ({away_stats['Ceza Sahası Etkinliği']:.1f} kez)")
    
    # İkili mücadele performansı
    if home_stats['İkili Mücadele'] > away_stats['İkili Mücadele']:
        home_better.append(f"{home_team} ikili mücadelelerde daha başarılı ({home_stats['İkili Mücadele']:.1f})")
    else:
        away_better.append(f"{away_team} ikili mücadelelerde daha başarılı ({away_stats['İkili Mücadele']:.1f})")
    
    # Hava topu hakimiyeti
    if home_stats['Hava Topu'] > away_stats['Hava Topu']:
        home_better.append(f"{home_team} hava toplarında daha etkili ({home_stats['Hava Topu']:.1f})")
    else:
        away_better.append(f"{away_team} hava toplarında daha etkili ({away_stats['Hava Topu']:.1f})")
    
    # Pas arası yapma
    if home_stats['Pas Arası'] > away_stats['Pas Arası']:
        home_better.append(f"{home_team} daha fazla pas arası yapıyor ({home_stats['Pas Arası']:.1f})")
    else:
        away_better.append(f"{away_team} daha fazla pas arası yapıyor ({away_stats['Pas Arası']:.1f})")
    
    # Galibiyet serisi
    if home_stats['Galibiyet'] > away_stats['Galibiyet']:
        home_better.append(f"{home_team} son 5 maçta daha fazla galibiyet aldı ({home_stats['Galibiyet']} galibiyet)")
    elif away_stats['Galibiyet'] > home_stats['Galibiyet']:
        away_better.append(f"{away_team} son 5 maçta daha fazla galibiyet aldı ({away_stats['Galibiyet']} galibiyet)")
    
    # Mağlubiyet durumu
    if home_stats['Mağlubiyet'] < away_stats['Mağlubiyet']:
        home_better.append(f"{home_team} son 5 maçta daha az mağlup oldu ({home_stats['Mağlubiyet']} mağlubiyet)")
    elif away_stats['Mağlubiyet'] < home_stats['Mağlubiyet']:
        away_better.append(f"{away_team} son 5 maçta daha az mağlup oldu ({away_stats['Mağlubiyet']} mağlubiyet)")
    
    # Genel değerlendirme
    if len(home_better) > len(away_better):
        conclusion = (
            f"{home_team} {len(home_better)} alanda daha iyi performans gösteriyor. "
            f"Ev sahibi avantajı ile birlikte {home_team} maçın favorisi."
        )
    elif len(away_better) > len(home_better):
        conclusion = (
            f"{away_team} {len(away_better)} alanda daha iyi performans gösteriyor. "
            f"Ev sahibi avantajına rağmen {away_team} daha avantajlı."
        )
    else:
        conclusion = (
            "İki takım da benzer performans gösteriyor. "
            "Ev sahibi avantajı belirleyici olabilir."
        )
    
    return {
        'home_advantages': home_better,
        'away_advantages': away_better,
        'conclusion': conclusion
    }

def get_last_matches(team_data, count=5):
    """Takımın son maçlarını döndürür."""
    last_matches = []
    for _, match in team_data.head(count).iterrows():
        result = {
            'date': match['Tarih'],
            'opponent': match['Rakip'],
            'is_home': match['Ev Sahibi/Deplasman'] == 'Ev Sahibi',
            'score': f"{match['MS Gol']}-{match['MS Yenilen Gol']}",
            'result': 'Galibiyet' if match['MS Gol'] > match['MS Yenilen Gol'] else 
                     'Beraberlik' if match['MS Gol'] == match['MS Yenilen Gol'] else 'Mağlubiyet'
        }
        last_matches.append(result)
    return last_matches

def get_head_to_head_matches(home_team, away_team, home_data, away_data):
    """İki takım arasındaki son karşılaşmaları döndürür."""
    h2h_matches = []
    
    # Ev sahibi takımın maçlarından bul
    for _, match in home_data.iterrows():
        if match['Rakip'] == away_team:
            h2h_matches.append({
                'date': match['Tarih'],
                'home_team': home_team,
                'away_team': away_team,
                'score': f"{match['MS Gol']}-{match['MS Yenilen Gol']}",
                'result': 'Ev Sahibi Galip' if match['MS Gol'] > match['MS Yenilen Gol'] else 
                         'Beraberlik' if match['MS Gol'] == match['MS Yenilen Gol'] else 'Deplasman Galip'
            })
    
    # Deplasman takımın maçlarından bul
    for _, match in away_data.iterrows():
        if match['Rakip'] == home_team:
            h2h_matches.append({
                'date': match['Tarih'],
                'home_team': away_team,
                'away_team': home_team,
                'score': f"{match['MS Yenilen Gol']}-{match['MS Gol']}",  # Skorları ters çevir
                'result': 'Ev Sahibi Galip' if match['MS Gol'] > match['MS Yenilen Gol'] else 
                         'Beraberlik' if match['MS Gol'] == match['MS Yenilen Gol'] else 'Deplasman Galip'
            })
    
    # Tarihe göre sırala
    h2h_matches.sort(key=lambda x: x['date'], reverse=True)
    return h2h_matches[:5]  # Son 5 maç

# Modelleri yükle
print("\nModeller yükleniyor...")
models, scaler = init_models()

if models is None or scaler is None:
    print("HATA: Modeller yüklenemedi!")
    exit(1)

# Takım listesini al
teams = get_available_teams()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')
        
        if not home_team or not away_team:
            return render_template('index.html', teams=teams, error="Lütfen her iki takımı da seçin.")
        
        if home_team == away_team:
            return render_template('index.html', teams=teams, error="Aynı takımı iki kez seçemezsiniz.")
        
        # Takım istatistiklerini al
        home_data = get_team_stats(home_team)
        away_data = get_team_stats(away_team)
        
        # Son 5 maç istatistiklerini al
        home_stats = get_team_performance_stats(home_data)
        away_stats = get_team_performance_stats(away_data)
        
        # Son maçları al
        home_last_matches = get_last_matches(home_data)
        away_last_matches = get_last_matches(away_data)
        h2h_matches = get_head_to_head_matches(home_team, away_team, home_data, away_data)
        
        # Tahminleri yap
        match_result = predict_match_result(home_team, away_team, models, scaler)
        home_goals, away_goals, score_prob = predict_score(home_team, away_team, models, scaler)
        ht_ft_result, ht_ft_prob = predict_ht_ft(home_team, away_team, models, scaler)
        btts_result, btts_prob = predict_btts(home_team, away_team, models, scaler)
        
        # Takım karşılaştırma analizini yap
        analysis = analyze_team_comparison(home_team, away_team, home_stats, away_stats)
        
        predictions = {
            'match_result': {
                'home_win': f"%{match_result['home_win']*100:.1f}",
                'draw': f"%{match_result['draw']*100:.1f}",
                'away_win': f"%{match_result['away_win']*100:.1f}"
            },
            'score': f"{home_goals}-{away_goals} (%{score_prob*100:.1f})",
            'iy_ms': f"{ht_ft_result} (%{ht_ft_prob*100:.1f})",
            'kg': f"{btts_result} (%{btts_prob*100:.1f})",
            'stats': {
                'home': home_stats,
                'away': away_stats
            },
            'analysis': analysis,
            'matches': {
                'home_last': home_last_matches,
                'away_last': away_last_matches,
                'h2h': h2h_matches
            }
        }
        
        return render_template('index.html', teams=teams, predictions=predictions, 
                             home_team=home_team, away_team=away_team)
    
    return render_template('index.html', teams=teams)

if __name__ == '__main__':
    app.run(debug=True) 