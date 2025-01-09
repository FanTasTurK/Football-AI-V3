import numpy as np
from data_preprocessing import get_team_stats
from model_training import load_models
from prediction_functions import predict_match_result, predict_score, predict_ht_ft, predict_btts, get_highest_probability_prediction
from datetime import datetime
from io import StringIO

def format_date(date):
    """
    datetime nesnesini istenen formatta string'e dönüştürür.
    """
    if isinstance(date, datetime):
        return date.strftime('%d.%m.%Y')
    return str(date)

def get_team_performance_stats(team_data, last_n=5):
    """
    Takımın son n maçtaki performans istatistiklerini hesaplar.
    """
    # Veri zaten tarih sırasına göre sıralı olmalı
    recent_data = team_data.head(last_n)
    
    stats = {
        'MS Gol Ort': recent_data['MS Gol'].mean(),
        'İY Gol Ort': recent_data['İY Gol'].mean(),
        'Topla Oynama': recent_data['Topla Oynama'].mean(),
        'Toplam Pas': recent_data['Toplam Pas'].mean(),
        'Pas İsabeti': recent_data['Pas İsabeti %'].mean(),
        'Toplam Şut': recent_data['Toplam Şut'].mean(),
        'İsabetli Şut': recent_data['İsabetli Şut'].mean(),
        'Gol Beklentisi': recent_data['Gol Beklentisi (xG)'].sum(),
        'Ceza Sahası Etkinliği': recent_data['Rakip Ceza Sahasında Topla Buluşma'].mean(),
        'İkili Mücadele': recent_data['İkili Mücadele Kazanma'].mean(),
        'Hava Topu': recent_data['Hava Topu Kazanma'].mean(),
        'Pas Arası': recent_data['Pas Arası'].mean(),
    }
    
    # Son maçların sonuçları
    results = recent_data['Sonuç'].value_counts()
    stats['Galibiyet'] = results.get('Galip', 0)
    stats['Beraberlik'] = results.get('Berabere', 0)
    stats['Mağlubiyet'] = results.get('Mağlup', 0)
    stats['Puan'] = (stats['Galibiyet'] * 3) + stats['Beraberlik']
    
    # Son maçların detayları
    stats['Son Maçlar'] = []
    for _, match in recent_data.iterrows():
        match_info = {
            'Tarih': format_date(match['Tarih']),
            'Rakip': match['Rakip'],
            'Gol': match['MS Gol'],
            'Sonuç': match['Sonuç']
        }
        stats['Son Maçlar'].append(match_info)
    
    return stats

def display_predictions(home_team, away_team):
    """
    Tüm tahminleri gösterir ve result.txt dosyasına kaydeder.
    """
    try:
        # Çıktıyı kaydetmek için StringIO kullan
        output = StringIO()
        
        # Modelleri yükle
        models, scaler = load_models()
        
        # Takım verilerini al
        home_data = get_team_stats(home_team)
        away_data = get_team_stats(away_team)
        
        # Son 5 maç istatistiklerini al
        home_stats = get_team_performance_stats(home_data)
        away_stats = get_team_performance_stats(away_data)
        
        # Maç sonucu tahminleri
        final_pred = predict_match_result(home_team, away_team, models, scaler)
        
        # Skor tahmini
        home_goals, away_goals, score_prob = predict_score(home_team, away_team, models, scaler)
        
        # İY/MS tahmini
        ht_ft, htft_prob = predict_ht_ft(home_team, away_team, models, scaler)
        
        # KG tahmini
        btts, btts_prob = predict_btts(home_team, away_team, models, scaler)
        
        # En yüksek olasılıklı tahmini bul
        best_pred, best_prob = get_highest_probability_prediction(final_pred, score_prob, htft_prob, btts_prob)
        
        # Tahmin zamanını al
        prediction_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        
        # Başlık ve zaman bilgisi
        header = f"\n{home_team} vs {away_team} Maç Tahminleri\nTahmin Tarihi: {prediction_time}"
        print(header)
        output.write(header + "\n")
        
        separator = "-" * 50
        print(separator)
        output.write(separator + "\n")
        
        # Kazanma olasılıkları
        odds = (
            f"Kazanma Olasılıkları:\n"
            f"{home_team}: %{final_pred['home_win']*100:.1f}\n"
            f"Beraberlik: %{final_pred['draw']*100:.1f}\n"
            f"{away_team}: %{final_pred['away_win']*100:.1f}\n"
            f"\nTahmini Skor: {home_goals}-{away_goals} (%{score_prob*100:.1f})\n"
            f"İY/MS: {ht_ft} (%{htft_prob*100:.1f})\n"
            f"Karşılıklı Gol: {btts} (%{btts_prob*100:.1f})\n"
            f"\nEn Yüksek Olasılıklı Tahmin: {best_pred} (%{best_prob*100:.1f})"
        )
        print(odds)
        output.write(odds + "\n")
        
        #İstatistikler başlığı
        stats_header = f"\n{separator}\nTemel İstatistikler:"
        print(stats_header)
        output.write(stats_header + "\n")
        
        # Temel istatistikler
        basic_stats = (
            f"Maç Sonu Gol Ort.: {home_team}: {home_stats['MS Gol Ort']:.1f} - {away_team}: {away_stats['MS Gol Ort']:.1f}\n"
            f"İlk Yarı Gol Ort.: {home_team}: {home_stats['İY Gol Ort']:.1f} - {away_team}: {away_stats['İY Gol Ort']:.1f}"
        )
        print(basic_stats)
        output.write(basic_stats + "\n")
        
        # Top kontrolü ve pas
        possession_stats = (
            f"\nTop Kontrolü ve Pas:\n"
            f"Topla Oynama: {home_team}: %{home_stats['Topla Oynama']:.1f} - {away_team}: %{away_stats['Topla Oynama']:.1f}\n"
            f"Toplam Pas: {home_team}: {home_stats['Toplam Pas']:.1f} - {away_team}: {away_stats['Toplam Pas']:.1f}\n"
            f"Pas İsabeti: {home_team}: %{home_stats['Pas İsabeti']:.1f} - {away_team}: %{away_stats['Pas İsabeti']:.1f}"
        )
        print(possession_stats)
        output.write(possession_stats + "\n")
        
        # Hücum etkinliği
        attack_stats = (
            f"\nHücum Etkinliği:\n"
            f"Toplam Şut: {home_team}: {home_stats['Toplam Şut']:.1f} - {away_team}: {away_stats['Toplam Şut']:.1f}\n"
            f"İsabetli Şut: {home_team}: {home_stats['İsabetli Şut']:.1f} - {away_team}: {away_stats['İsabetli Şut']:.1f}\n"
            f"Gol Beklentisi: {home_team}: {home_stats['Gol Beklentisi']:.2f} - {away_team}: {away_stats['Gol Beklentisi']:.2f}\n"
            f"Ceza Sahası Etkinliği: {home_team}: {home_stats['Ceza Sahası Etkinliği']:.1f} - {away_team}: {away_stats['Ceza Sahası Etkinliği']:.1f}"
        )
        print(attack_stats)
        output.write(attack_stats + "\n")
        
        # Savunma ve mücadele
        defense_stats = (
            f"\nSavunma ve Mücadele:\n"
            f"İkili Mücadele: {home_team}: {home_stats['İkili Mücadele']:.1f} - {away_team}: {away_stats['İkili Mücadele']:.1f}\n"
            f"Hava Topu: {home_team}: {home_stats['Hava Topu']:.1f} - {away_team}: {away_stats['Hava Topu']:.1f}\n"
            f"Pas Arası: {home_team}: {home_stats['Pas Arası']:.1f} - {away_team}: {away_stats['Pas Arası']:.1f}"
        )
        print(defense_stats)
        output.write(defense_stats + "\n")
        
        # Son 5 maç performansı
        performance = (
            f"\nSon 5 Maç Performansı:\n"
            f"{home_team}: {home_stats['Galibiyet']} Galibiyet, {home_stats['Beraberlik']} Beraberlik, {home_stats['Mağlubiyet']} Mağlubiyet\n"
            f"{away_team}: {away_stats['Galibiyet']} Galibiyet, {away_stats['Beraberlik']} Beraberlik, {away_stats['Mağlubiyet']} Mağlubiyet"
        )
        print(performance)
        output.write(performance + "\n")
        
        # Gerekçeler başlığı
        reasons_header = f"{separator}\nGerekçeler:\n"
        print(reasons_header)
        output.write(reasons_header + "\n")
        
        # Son 5 maç detayları
        for team, stats in [(away_team, away_stats), (home_team, home_stats)]:
            team_details = (
                f"{team}:\n"
                f"- Galibiyet: {stats['Galibiyet']} maç\n"
                f"- Beraberlik: {stats['Beraberlik']} maç\n"
                f"- Mağlubiyet: {stats['Mağlubiyet']} maç\n"
                f"- Puan: {stats['Puan']} ({stats['Galibiyet']*3} + {stats['Beraberlik']})\n"
                f"\nSon Maçlar:\n"
            )
            print(team_details)
            output.write(team_details)
            
            for match in stats['Son Maçlar']:
                result_emoji = "✅" if match['Sonuç'] == 'Galip' else "🟡" if match['Sonuç'] == 'Berabere' else "❌"
                match_detail = f"[{match['Tarih']}] {team} vs {match['Rakip']}: {match['Gol']} gol ({match['Sonuç']}) {result_emoji}\n"
                print(match_detail)
                output.write(match_detail)
            print()
            output.write("\n")
        
        # Form değerlendirmesi
        print("Form Değerlendirmesi:")
        output.write("Form Değerlendirmesi:\n")
        
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
        
        print("\nEv Sahibi Avantajları:")
        output.write("\nEv Sahibi Avantajları:\n")
        if home_better:
            for advantage in home_better:
                print(f"- {advantage}")
                output.write(f"- {advantage}\n")
        else:
            print("- Belirgin bir avantaj bulunamadı")
            output.write("- Belirgin bir avantaj bulunamadı\n")
        
        print("\nDeplasman Avantajları:")
        output.write("\nDeplasman Avantajları:\n")
        if away_better:
            for advantage in away_better:
                print(f"- {advantage}")
                output.write(f"- {advantage}\n")
        else:
            print("- Belirgin bir avantaj bulunamadı")
            output.write("- Belirgin bir avantaj bulunamadı\n")
        
        # Genel değerlendirme
        print("\nGenel Değerlendirme:")
        output.write("\nGenel Değerlendirme:\n")
        if len(home_better) > len(away_better):
            conclusion = (
                f"- {home_team} {len(home_better)} alanda daha iyi performans gösteriyor\n"
                f"- Ev sahibi avantajı ile birlikte {home_team} maçın favorisi"
            )
        elif len(away_better) > len(home_better):
            conclusion = (
                f"- {away_team} {len(away_better)} alanda daha iyi performans gösteriyor\n"
                f"- Ev sahibi avantajına rağmen {away_team} daha avantajlı"
            )
        else:
            conclusion = (
                "- İki takım da benzer performans gösteriyor\n"
                "- Ev sahibi avantajı belirleyici olabilir"
            )
        print(conclusion)
        output.write(conclusion + "\n")
        
        # Dosyaya kaydet
        with open('result.txt', 'w', encoding='utf-8') as f:
            f.write(output.getvalue())
        
        # StringIO'yu kapat
        output.close()
        
    except Exception as e:
        error_msg = f"Tahminleri gösterirken bir hata oluştu: {str(e)}\nLütfen tekrar deneyin."
        print(error_msg)
        with open('result.txt', 'w', encoding='utf-8') as f:
            f.write(error_msg) 