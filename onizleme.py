import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Görselleştirme ayarları (daha okunaklı grafikler için)
sns.set_style('whitegrid')
plt.rc('figure', figsize=(12, 8))
plt.rc('font', size=12)
plt.rc('axes', titlesize=16, labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

print("Kütüphaneler yüklendi.")

# --- 1. VERİYİ YÜKLEME VE İLK BAKIŞ ---

# train.csv dosyasını yükle
try:
    df = pd.read_csv('train.csv')
    print("train.csv başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: train.csv dosyası bulunamadı. Lütfen dosyanın doğru klasörde olduğundan emin olun.")
    exit()

# data_description.txt'ye göre MSSubClass sayısal değil, kategoriktir.
# Bunu en başta dönüştürmek, analizlerin doğruluğu için kritiktir.
if 'MSSubClass' in df.columns:
    df['MSSubClass'] = df['MSSubClass'].astype(str)

print("\n--- Veri Seti İlk 5 Satır ---")
print(df.head())

print("\n--- Veri Seti Bilgisi (Sütun Tipleri ve Eksik Veriler) ---")
# .info() çıktısını bir değişkene alıp yazdırmak, 'None' çıktısını engeller
import io
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
print(s)

# --- 2. HEDEF DEĞİŞKEN (SalePrice) ANALİZİ ---

print("\nGrafikler oluşturuluyor...")

# Fiyat Dağılımı (Histogram)
plt.figure(figsize=(12, 6))
sns.histplot(df['SalePrice'], kde=True, bins=50)
plt.title('Ev Fiyatı (SalePrice) Dağılımı (Çarpık)')
plt.xlabel('Satış Fiyatı ($)')
plt.ylabel('Adet (Frekans)')
plt.tight_layout()
plt.savefig('01_saleprice_dagilimi.png')
plt.clf() # Figürü temizle

# Fiyat dağılımı sağa çarpık (right-skewed).
# Modeller bu tür çarpık verileri sevmez. Logaritmik dönüşüm uygulayarak düzeltebiliriz.
# Bu, "Öznitelik Mühendisliği" adımında yapılacak ama EDA'da göstermek çok etkilidir.
df['SalePrice_log'] = np.log1p(df['SalePrice'])

plt.figure(figsize=(12, 6))
sns.histplot(df['SalePrice_log'], kde=True, bins=50, color='green')
plt.title('Log-Dönüşümlü Ev Fiyatı Dağılımı (Normal Dağılıma Yakın)')
plt.xlabel('Log(Satış Fiyatı)')
plt.ylabel('Adet (Frekans)')
plt.tight_layout()
plt.savefig('02_saleprice_log_dagilimi.png')
plt.clf()

# --- 3. İLİŞKİ ANALİZİ: SAYISAL ÖZELLİKLER vs FİYAT (Scatter Plot) ---

# Yaşam Alanı (GrLivArea) vs Fiyat
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'])
plt.title('Yaşam Alanı (GrLivArea) vs. Satış Fiyatı')
plt.xlabel('Yaşam Alanı (Metrekare/SqFt)')
plt.ylabel('Satış Fiyatı ($)')
plt.tight_layout()
plt.savefig('03_yasamalani_vs_fiyat.png')
plt.clf()

# Toplam Bodrum Alanı (TotalBsmtSF) vs Fiyat
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['TotalBsmtSF'], y=df['SalePrice'])
plt.title('Toplam Bodrum Alanı (TotalBsmtSF) vs. Satış Fiyatı')
plt.xlabel('Toplam Bodrum Alanı (Metrekare/SqFt)')
plt.ylabel('Satış Fiyatı ($)')
plt.tight_layout()
plt.savefig('04_bodrumalani_vs_fiyat.png')
plt.clf()

# --- 4. İLİŞKİ ANALİZİ: KATEGORİK ÖZELLİKLER vs FİYAT (Box Plot) ---

# Genel Ev Kalitesi (OverallQual) vs Fiyat
# Bu, en önemli özelliklerden biridir.
plt.figure(figsize=(12, 7))
sns.boxplot(x=df['OverallQual'], y=df['SalePrice'])
plt.title('Genel Ev Kalitesi (OverallQual) vs. Satış Fiyatı')
plt.xlabel('Genel Kalite (1=Çok Kötü, 10=Çok İyi)')
plt.ylabel('Satış Fiyatı ($)')
plt.tight_layout()
plt.savefig('05_genelkalite_vs_fiyat.png')
plt.clf()

# Mutfak Kalitesi (KitchenQual) vs Fiyat
# Bu sıralı (ordinal) bir kategoridir. Doğru sıralamayı verelim.
qual_order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['KitchenQual'], y=df['SalePrice'], order=qual_order)
plt.title('Mutfak Kalitesi (KitchenQual) vs. Satış Fiyatı')
plt.xlabel('Mutfak Kalitesi (Po=Kötü, Ex=Mükemmel)')
plt.ylabel('Satış Fiyatı ($)')
plt.tight_layout()
plt.savefig('06_mutfakkalitesi_vs_fiyat.png')
plt.clf()

# Semt (Neighborhood) vs Fiyat
# Bu grafiğin okunaklı olması için:
# 1. Figürü genişlet
# 2. X ekseni etiketlerini döndür
# 3. Semtleri medyan fiyata göre sırala
plt.figure(figsize=(20, 10))
neighborhood_order = df.groupby('Neighborhood')['SalePrice'].median().sort_values().index
sns.boxplot(x=df['Neighborhood'], y=df['SalePrice'], order=neighborhood_order)
plt.title('Semtlere (Neighborhood) Göre Satış Fiyatları (Medyan Fiyata Göre Sıralı)')
plt.xlabel('Semt')
plt.ylabel('Satış Fiyatı ($)')
plt.xticks(rotation=90) # Etiketleri 90 derece döndür
plt.tight_layout()
plt.savefig('07_semtler_vs_fiyat.png')
plt.clf()

# --- 5. KORELASYON ANALİZİ (Heatmap ve Bar Plot) ---

# Sadece sayısal sütunları seç
df_numeric = df.select_dtypes(include=[np.number])

# Id ve log dönüşümlü sütunları analizden çıkar
if 'Id' in df_numeric.columns:
    df_numeric = df_numeric.drop(columns=['Id', 'SalePrice_log'])

# Korelasyon matrisini hesapla
corr_matrix = df_numeric.corr()

# Büyük Isı Haritası (Tüm özellikler)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Tüm Sayısal Özelliklerin Korelasyon Isı Haritası', fontsize=20)
plt.tight_layout()
plt.savefig('08_korelasyon_haritasi.png')
plt.clf()

# Sunum için daha da iyisi: Sadece SalePrice ile olan korelasyonları göstermek
plt.figure(figsize=(10, 12))
sale_price_corr = corr_matrix['SalePrice'].sort_values(ascending=False)
sns.barplot(x=sale_price_corr.values, y=sale_price_corr.index, palette='viridis')
plt.title('Hangi Özellikler Satış Fiyatı (SalePrice) ile En İlişkili?', fontsize=18)
plt.xlabel('Korelasyon Katsayısı')
plt.ylabel('Özellikler (Features)')
plt.tight_layout()
plt.savefig('09_fiyat_korelasyonlari_barplot.png')
plt.clf()

print("\nTüm grafikler başarıyla oluşturuldu ve kaydedildi.")
print("Oluşturulan dosyalar:")
print("- 01_saleprice_dagilimi.png")
print("- 02_saleprice_log_dagilimi.png")
print("- 03_yasamalani_vs_fiyat.png")
print("- 04_bodrumalani_vs_fiyat.png")
print("- 05_genelkalite_vs_fiyat.png")
print("- 06_mutfakkalitesi_vs_fiyat.png")
print("- 07_semtler_vs_fiyat.png")
print("- 08_korelasyon_haritasi.png")
print("- 09_fiyat_korelasyonlari_barplot.png")