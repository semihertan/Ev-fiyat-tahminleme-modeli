import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Uyarıları kapat
warnings.filterwarnings('ignore')
# Grafik stili
sns.set_style('whitegrid')
plt.rc('figure', figsize=(12, 6))

print("--- 2. ADIM: MODEL EĞİTİMİ (4 ALGORİTMA) ---")

# 1. Hazırlanmış Veriyi Yükleme
print("Veriler yükleniyor...")
try:
    X = pd.read_csv('train_processed_X.csv')
    y_df = pd.read_csv('train_processed_y.csv')
    y = y_df.iloc[:, 0]
    print("Veriler başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: Temizlenmiş veri dosyaları bulunamadı. Lütfen önce 'Veri Ön İşleme' kodunu çalıştırın.")
    exit()

# 2. Eğitim ve Doğrulama Ayrımı
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. Modellerin Tanımlanması ve Eğitimi
print("\nModeller eğitiliyor (Bu işlem biraz zaman alabilir)...")

# Model 2: Lasso (Özellik Seçici)
lasso_model = Lasso(alpha=0.0005, random_state=42)
lasso_model.fit(X_train, y_train)
print("2. Lasso Modeli Tamamlandı.")

# Model 3: Random Forest (Bagging)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("3. Random Forest Modeli Tamamlandı.")

# Model 4: Gradient Boosting (Boosting - YENİ!)
# Hatalardan ders çıkararak ilerleyen güçlü bir model
gb_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
print("4. Gradient Boosting Modeli Tamamlandı.")


# 4. Performans Değerlendirme
def evaluate_model(model):
    # Tahmin (Logaritmik)
    preds_log = model.predict(X_val)
    # Gerçek Değer (Dolar)
    preds_orig = np.expm1(preds_log)
    y_true_orig = np.expm1(y_val)

    rmse = np.sqrt(mean_squared_error(y_true_orig, preds_orig))
    r2 = r2_score(y_val, preds_log)
    return rmse, r2


rmse_ridge, r2_ridge = evaluate_model(ridge_model)
rmse_lasso, r2_lasso = evaluate_model(lasso_model)
rmse_rf, r2_rf = evaluate_model(rf_model)
rmse_gb, r2_gb = evaluate_model(gb_model)

# 5. Sonuç Tablosu
results = pd.DataFrame({
    'Model': ['Lasso', 'Random Forest', 'Gradient Boosting'],
    'RMSE ($)': [rmse_lasso, rmse_rf, rmse_gb],
    'R² Score': [r2_lasso, r2_rf, r2_gb]
})

print("\n--- 4 MODELİN KARŞILAŞTIRMA SONUÇLARI ---")
print(results.sort_values('RMSE ($)'))

# 6. Karşılaştırma Grafiği
plt.figure(figsize=(10, 6))
sns.barplot(x='RMSE ($)', y='Model', data=results.sort_values('RMSE ($)'), palette='viridis')
plt.title('Model Hata Oranları Karşılaştırması (Düşük Olan Kazanır)')
plt.xlabel('Ortalama Hata Payı ($)')
# Çubukların ucuna değerleri yazalım
for index, row in enumerate(results.sort_values('RMSE ($)').itertuples()):
    plt.text(row._2 + 500, index, f"${row._2:,.0f}", color='black', va='center')
plt.tight_layout()
plt.savefig('sonuc_4_model_karsilastirma.png')
print("\n'sonuc_4_model_karsilastirma.png' grafiği kaydedildi.")

# 7. En İyi Modelin (Muhtemelen GB veya Lasso) Tahmin Grafiği
best_model = gb_model if rmse_gb < rmse_lasso else lasso_model
best_model_name = "Gradient Boosting" if rmse_gb < rmse_lasso else "Lasso"

preds_best = np.expm1(best_model.predict(X_val))
y_true = np.expm1(y_val)

plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_true, y=preds_best, alpha=0.6, color='purple')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.title(f'En İyi Model ({best_model_name}): Gerçek vs Tahmin')
plt.xlabel('Gerçek Fiyatlar')
plt.ylabel('Tahmin Edilen Fiyatlar')
plt.tight_layout()
plt.savefig('sonuc_en_iyi_tahmin.png')
print("'sonuc_en_iyi_tahmin.png' kaydedildi.")