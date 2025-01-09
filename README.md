# Football Match Prediction System / Futbol Maç Tahmin Sistemi

[English](#english) | [Türkçe](#turkish)

<a name="english"></a>
## English

### Description
This project is an artificial intelligence model that analyzes football match results and statistics to calculate various probabilities for a match between two teams. The model uses machine learning algorithms to make predictions based on historical match data.

### Features
- Match result prediction (Home win, Draw, Away win)
- Score prediction
- First Half / Full Time (HT/FT) prediction
- Both Teams to Score (BTTS) prediction
- Analysis of historical match statistics
- Home advantage consideration
- Multiple model usage and model optimization
- Detailed form analysis
- Chronological match history
- Performance comparison in various aspects
- Saving predictions to file with timestamp

### Technical Details
The project uses three different machine learning models:
1. Random Forest
2. Gradient Boosting
3. Support Vector Machine

Each model is optimized using hyperparameter tuning and cross-validation.

### Requirements
- Python 3.8+
- Required packages:
  ```
  pandas==2.1.4
  numpy==1.26.2
  scikit-learn==1.3.2
  lazypredict==0.2.12
  joblib==1.3.2
  ```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/username/football-prediction.git
cd football-prediction
```

2. Create and activate virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage
1. Run the program:
```bash
python main.py
```

2. Select teams:
   - The program will show a numbered list of available teams
   - Enter the number for the home team
   - Enter the number for the away team

3. View predictions:
   - The program will display detailed predictions and statistics
   - Results will also be saved to `result.txt`

### Project Structure
- `main.py`: Main program flow
- `data_preprocessing.py`: Data preprocessing operations
- `model_training.py`: Model training and evaluation
- `prediction.py`: Prediction operations
- `prediction_functions.py`: Core prediction functions
- `utils.py`: Helper functions
- `models/`: Directory containing trained models
- `stats/`: Directory containing team statistics CSV files

### Data Structure
Each CSV file in the `stats` directory contains a team's match statistics:
- Match Date
- Opponent
- Home/Away
- Full Time Goals
- First Half Goals
- Ball Possession
- Duels Won
- Aerial Duels Won
- Interceptions
- Total Passes
- Accurate Passes
- Pass Accuracy
- Total Crosses
- Accurate Crosses
- Total Shots
- Shots on Target
- Expected Goals (xG)
- Touches in Opposition Box
- Clearances
- Fouls
- Offsides

### Sample Output
```
Fenerbahçe vs Galatasaray Match Predictions
Prediction Time: 06.01.2024 15:30:45
--------------------------------------------------
Win Probabilities:
Fenerbahçe: %33.6
Draw: %15.3
Galatasaray: %51.2

Predicted Score: 1-4
HT/FT: 2-2
Both Teams to Score: YES

--------------------------------------------------
Basic Statistics:
Full Time Goals Avg.: Fenerbahçe: 1.8 - Galatasaray: 3.2
First Half Goals Avg.: Fenerbahçe: 1.0 - Galatasaray: 1.6

Ball Control and Passing:
Possession: Fenerbahçe: %57.8 - Galatasaray: %57.0
Total Passes: Fenerbahçe: 482.8 - Galatasaray: 398.8
Pass Accuracy: Fenerbahçe: %80.7 - Galatasaray: %79.5

[... continues with detailed statistics and analysis ...]
```

---

<a name="turkish"></a>
## Türkçe

### Açıklama
Bu proje, futbol maç sonuçlarını ve istatistiklerini analiz ederek, iki takım arasında oynanacak bir maç için çeşitli olasılıkları hesaplayan bir yapay zeka modelidir. Model, geçmiş maç verilerini kullanarak makine öğrenimi algoritmaları ile tahminler yapar.

### Özellikler
- Maç sonucu tahmini (Ev sahibi kazanır, Beraberlik, Deplasman kazanır)
- Skor tahmini
- İlk Yarı / Maç Sonu (İY/MS) tahmini
- Karşılıklı Gol (KG) tahmini
- Geçmiş maç istatistiklerinin analizi
- Ev sahibi avantajının hesaba katılması
- Çoklu model kullanımı ve model optimizasyonu
- Detaylı form analizi
- Kronolojik maç geçmişi
- Çeşitli alanlarda performans karşılaştırması
- Tahminlerin zaman damgalı olarak dosyaya kaydedilmesi

### Teknik Detaylar
Proje üç farklı makine öğrenimi modeli kullanır:
1. Random Forest (Rastgele Orman)
2. Gradient Boosting (Gradyan Artırma)
3. Support Vector Machine (Destek Vektör Makinesi)

Her model için hiperparametre optimizasyonu ve çapraz doğrulama kullanılır.

### Gereksinimler
- Python 3.8+
- Gerekli paketler:
  ```
  pandas==2.1.4
  numpy==1.26.2
  scikit-learn==1.3.2
  lazypredict==0.2.12
  joblib==1.3.2
  ```

### Kurulum
1. Projeyi klonlayın:
```bash
git clone https://github.com/kullanici_adi/football-prediction.git
cd football-prediction
```

2. Sanal ortam oluşturun ve aktifleştirin (önerilen):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

### Kullanım
1. Programı çalıştırın:
```bash
python main.py
```

2. Takımları seçin:
   - Program mevcut takımların numaralandırılmış listesini gösterecek
   - Ev sahibi takımın numarasını girin
   - Deplasman takımının numarasını girin

3. Tahminleri görüntüleyin:
   - Program detaylı tahminleri ve istatistikleri gösterecek
   - Sonuçlar ayrıca `result.txt` dosyasına kaydedilecek

### Proje Yapısı
- `main.py`: Ana program akışı
- `data_preprocessing.py`: Veri ön işleme işlemleri
- `model_training.py`: Model eğitimi ve değerlendirme
- `prediction.py`: Tahmin işlemleri
- `prediction_functions.py`: Temel tahmin fonksiyonları
- `utils.py`: Yardımcı fonksiyonlar
- `models/`: Eğitilmiş modellerin bulunduğu dizin
- `stats/`: Takım istatistiklerinin bulunduğu CSV dosyaları

### Veri Yapısı
`stats` dizinindeki her CSV dosyası bir takımın maç istatistiklerini içerir:
- Maç Tarihi
- Rakip
- Ev Sahibi/Deplasman
- Maç Sonu Gol
- İlk Yarı Gol
- Topla Oynama
- İkili Mücadele Kazanma
- Hava Topu Kazanma
- Pas Arası
- Toplam Pas
- İsabetli Pas
- Pas İsabeti
- Toplam Orta
- İsabetli Orta
- Toplam Şut
- İsabetli Şut
- Gol Beklentisi (xG)
- Rakip Ceza Sahasında Topla Buluşma
- Uzaklaştırma
- Faul
- Ofsayt

### Örnek Çıktı
```
Fenerbahçe vs Galatasaray Maç Tahminleri
Tahmin Tarihi: 06.01.2024 15:30:45
--------------------------------------------------
Kazanma Olasılıkları:
Fenerbahçe: %33.6
Beraberlik: %15.3
Galatasaray: %51.2

Tahmini Skor: 1-4
İY/MS: 2-2
Karşılıklı Gol: VAR

--------------------------------------------------
Temel İstatistikler:
Maç Sonu Gol Ort.: Fenerbahçe: 1.8 - Galatasaray: 3.2
İlk Yarı Gol Ort.: Fenerbahçe: 1.0 - Galatasaray: 1.6

Top Kontrolü ve Pas:
Topla Oynama: Fenerbahçe: %57.8 - Galatasaray: %57.0
Toplam Pas: Fenerbahçe: 482.8 - Galatasaray: 398.8
Pas İsabeti: Fenerbahçe: %80.7 - Galatasaray: %79.5

[... detaylı istatistikler ve analizler devam eder ...]
```

### Katkıda Bulunma
1. Bu projeyi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

### Lisans
Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın. 