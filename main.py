from utils import get_team_selection
from data_preprocessing import get_team_stats, prepare_features
from model_training import train_models, save_models, load_models, MODELS_DIR
from prediction import display_predictions
import os
import pandas as pd

def main():
    """
    Ana program akışı
    """
    print("Futbol Maç Tahmin Sistemi")
    print("=" * 50)
    
    # Model dosyalarını kontrol et
    model_files_exist = all(
        os.path.exists(os.path.join(MODELS_DIR, f'{name}_model.joblib'))
        for name in ['RandomForest', 'GradientBoosting', 'SVC']
    ) and os.path.exists(os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    # Modeller yoksa veya yüklenemezse yeniden eğit
    if not model_files_exist:
        print("\nModeller eğitiliyor...")
        
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
    else:
        # Modelleri yüklemeyi dene
        models, scaler = load_models()
        if models is None:  # Yükleme başarısız olduysa yeniden eğit
            print("\nModeller eğitiliyor...")
            
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
        else:
            print("\nKaydedilmiş modeller yüklendi.")
    
    while True:
        # Takım seçimi
        home_team, away_team = get_team_selection()
        
        # Tahminleri göster
        display_predictions(home_team, away_team)
        
        # Devam etmek istiyor mu?
        print("\nBaşka bir maç tahmini yapmak ister misiniz? (E/H)")
        if input().upper() != 'E':
            break
    
    print("\nProgram sonlandırıldı.")

if __name__ == "__main__":
    main() 