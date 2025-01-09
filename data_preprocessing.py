import pandas as pd
import numpy as np
from datetime import datetime

def clean_percentage(value):
    """
    Yüzdelik değerleri temizler ve float'a dönüştürür.
    """
    if isinstance(value, str):
        # Yüzde işaretini ve boşlukları kaldır
        value = value.strip('%').strip()
        # Virgülü noktaya çevir
        value = value.replace(',', '.')
        return float(value)
    return value

def clean_numeric(value):
    """
    Sayısal değerleri temizler ve float'a dönüştürür.
    """
    if isinstance(value, str):
        # Virgülü noktaya çevir
        value = value.replace(',', '.')
        try:
            return float(value)
        except ValueError:
            return 0.0
    return value if pd.notnull(value) else 0.0

def parse_date(date_str):
    """
    Tarih string'ini datetime nesnesine dönüştürür.
    """
    try:
        # Tarih formatı: DD.MM.YYYY
        return datetime.strptime(date_str, '%d.%m.%Y')
    except ValueError as e:
        print(f"Tarih dönüştürme hatası: {str(e)}")
        return None

def preprocess_team_data(file_path):
    """
    Takım verilerini okur ve ön işleme yapar.
    """
    try:
        # CSV'yi okurken hataları yönet
        df = pd.read_csv(file_path, 
                        encoding='utf-8',
                        on_bad_lines='skip',  # Hatalı satırları atla
                        sep=',',              # Ayırıcı olarak virgül kullan
                        engine='python')       # Python engine'i kullan
        
        # Tarihi datetime'a dönüştür
        df['Tarih'] = df['Tarih'].apply(parse_date)
        
        # Tarihe göre sırala (en yeni maç en üstte)
        df = df.sort_values('Tarih', ascending=False).reset_index(drop=True)
        
        # Tüm sayısal sütunları temizle
        numeric_columns = [
            'MS Gol', 'İY Gol', 'MS Yenilen Gol', 'İY Yenilen Gol',
            'Topla Oynama', 'İkili Mücadele Kazanma',
            'Hava Topu Kazanma', 'Pas Arası', 'Toplam Pas', 'İsabetli Pas',
            'Pas İsabeti %', 'Toplam Orta', 'İsabetli Orta', 'Toplam Şut',
            'İsabetli Şut', 'İsabetsiz Şut', 'Engellenen Şut', 'Gol Beklentisi (xG)',
            'Rakip Ceza Sahasında Topla Buluşma', 'Uzaklaştırma', 'Faul', 'Ofsayt'
        ]
        
        # Yüzdelik sütunlar
        percentage_columns = ['Topla Oynama', 'Pas İsabeti %']
        
        # Diğer sayısal sütunlar
        other_numeric = [col for col in numeric_columns if col not in percentage_columns]
        
        # Yüzdelik değerleri temizle
        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_percentage)
        
        # Diğer sayısal değerleri temizle
        for col in other_numeric:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
        
        # Eksik sütunları kontrol et ve gerekirse ekle
        required_columns = numeric_columns + ['Tarih', 'Rakip', 'Sonuç']
        for col in required_columns:
            if col not in df.columns:
                print(f"Uyarı: {col} sütunu eksik. Sıfır ile doldurulacak.")
                df[col] = 0
        
        return df
        
    except Exception as e:
        print(f"Veri okuma hatası: {str(e)}")
        print(f"Dosya: {file_path}")
        # Sütun sayısını ve başlıkları kontrol et
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            print(f"Sütun sayısı: {len(header.split(','))}")
            print(f"Sütunlar: {header}")
        raise

def get_team_stats(team_name):
    """
    Bir takımın istatistiklerini getirir.
    """
    file_path = f'stats/{team_name}.csv'
    return preprocess_team_data(file_path)

def get_head_to_head_stats(team1_name, team2_name):
    """
    İki takım arasındaki geçmiş maç istatistiklerini getirir.
    """
    team1_data = get_team_stats(team1_name)
    team2_data = get_team_stats(team2_name)
    
    # Takım 1'in takım 2 ile olan maçları
    team1_vs_team2 = team1_data[team1_data['Rakip'] == team2_name]
    
    # Takım 2'nin takım 1 ile olan maçları
    team2_vs_team1 = team2_data[team2_data['Rakip'] == team1_name]
    
    return team1_vs_team2, team2_vs_team1

def prepare_features(team_data):
    """
    Model için özellikleri hazırlar.
    """
    # Türetilmiş özellikleri hesapla
    team_data['Şut İsabet Oranı'] = team_data['İsabetli Şut'] / team_data['Toplam Şut'].replace(0, 1)
    team_data['Orta İsabet Oranı'] = team_data['İsabetli Orta'] / team_data['Toplam Orta'].replace(0, 1)
    team_data['Gol Dönüşüm Oranı'] = team_data['MS Gol'] / team_data['İsabetli Şut'].replace(0, 1)
    
    positive_features = [
        # Temel gol istatistikleri
        'MS Gol', 'İY Gol',
        'Gol Beklentisi (xG)',
        'Gol Dönüşüm Oranı',
        
        # Top kontrolü
        'Topla Oynama',
        'İkili Mücadele Kazanma',
        'Hava Topu Kazanma',
        
        # Pas istatistikleri
        'Pas Arası',
        'Toplam Pas',
        'İsabetli Pas',
        'Pas İsabeti %',
        
        # Orta istatistikleri
        'Toplam Orta',
        'İsabetli Orta',
        'Orta İsabet Oranı',
        
        # Şut istatistikleri
        'Toplam Şut',
        'İsabetli Şut',
        'Şut İsabet Oranı',
        
        # Diğer pozitif istatistikler
        'Rakip Ceza Sahasında Topla Buluşma',
        'Uzaklaştırma'
    ]
    
    negative_features = [
        # Yenilen goller
        'MS Yenilen Gol',
        'İY Yenilen Gol',
        
        # Kartlar
        'Sarı Kart',
        'İkinci Sarıdan Kırmızı Kart',
        'Kırmızı Kart',
        
        # Diğer negatif istatistikler
        'Ofsayt',
        'İsabetsiz Şut',
        'Engellenen Şut',
        'Faul'
    ]
    
    # Eksik değerleri 0 ile doldur
    features = team_data[positive_features + negative_features].fillna(0)
    
    # Maç sonucu için hedef değişken (0: Mağlup, 1: Berabere, 2: Galip)
    result_mapping = {'Galip': 2, 'Berabere': 1, 'Mağlup': 0}
    y_match = team_data['Sonuç'].map(result_mapping)
    
    # Skor tahmini için hedef değişkenler
    y_score = team_data['MS Gol']
    
    # İY/MS için hedef değişken
    def get_htft_label(row):
        ht_goals = row['İY Gol']
        ft_goals = row['MS Gol']
        ht_result = '1' if ht_goals > row['İY Yenilen Gol'] else 'X' if ht_goals == row['İY Yenilen Gol'] else '2'
        ft_result = '1' if ft_goals > row['MS Yenilen Gol'] else 'X' if ft_goals == row['MS Yenilen Gol'] else '2'
        return f"{ht_result}-{ft_result}"
    
    y_htft = team_data.apply(get_htft_label, axis=1)
    
    # KG tahmini için hedef değişken
    y_btts = (team_data['MS Gol'] > 0) & (team_data['MS Yenilen Gol'] > 0)
    
    return features, y_match, y_score, y_htft, y_btts 