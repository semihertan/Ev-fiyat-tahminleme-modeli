import pandas as pd
import numpy as np
import warnings

# Uyarıları kapat
warnings.filterwarnings('ignore')

print("--- 1. Adım: Veri Yükleme (train ve test) ---")

# Veri setlerini yükle
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("train.csv ve test.csv başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: train.csv veya test.csv dosyası bulunamadı. Lütfen iki dosyanın da klasörde olduğundan emin olun.")
    exit() # Kodun devam etmesini engelle

# Orijinal ID'leri ve satır sayılarını sakla
train_ID = df_train['Id']
test_ID = df_test['Id']
ntrain = df_train.shape[0]
ntest = df_test.shape[0]

print(f"Eğitim verisi satır sayısı: {ntrain}")
print(f"Test verisi satır sayısı: {ntest}")

# --- 2. Adım: Aykırı Değerleri (Outliers) Temizleme (SADECE EĞİTİM VERİSİNDEN) ---
print("\n--- 2. Adım: Aykırı Değerler (Outliers) Temizleniyor ---")
# Aykırı değerler, modelin öğrenmesini bozduğu için SADECE eğitim verisinden çıkarılır.
# Test verisi, modelin gerçek dünyadaki performansını ölçer, bu yüzden ona dokunulmaz.
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
print(f"Aykırı değerler temizlendi, eğitim verisinden {ntrain - df_train.shape[0]} satır çıkarıldı.")
# Yeni eğitim verisi satır sayısını güncelle
ntrain = df_train.shape[0]

# --- 3. Adım: Hedef Değişken (SalePrice) İşleme ---
print("\n--- 3. Adım: Hedef Değişken (SalePrice) İşleniyor ---")
# SalePrice'ı log(1+x) dönüşümü ile normalleştir
# ve y_train olarak ayır
y_train = np.log1p(df_train["SalePrice"])
# y_train'i bir pandas Series olarak tanımlayalım ve isimlendirelim. Bu, ileride oluşabilecek
# boyut uyuşmazlığı ve boş DataFrame hatalarını önlemek için kritik bir adımdır.
y_train.name = 'SalePrice_log'

# SalePrice'ı birleştirmeden önce ana eğitim setinden çıkar
df_train.drop('SalePrice', axis=1, inplace=True)
print("SalePrice logaritmik dönüşümü (log1p) uygulandı ve 'y_train' olarak ayrıldı.")

# ID sütunlarını modelleme için gereksiz olduğundan çıkar
df_train.drop("Id", axis = 1, inplace=True)
df_test.drop("Id", axis = 1, inplace=True)

# --- 4. Adım: Veri Setlerini Birleştirme ve Eksik Verileri Doldurma ---
print("\n--- 4. Adım: Veriler Birleştiriliyor ve Eksik Veriler Dolduruluyor ---")
# Tüm ön işleme adımlarının tutarlı olması için train ve test verisini birleştir
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
print(f"Birleştirilmiş 'all_data' boyutu: {all_data.shape}")

# Grup 1: 'Yok' Anlamına Gelen Kategorik NaN'ler
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType'):
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

# Grup 2: 'Yok' Anlamına Gelen Sayısal NaN'ler
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# Grup 3: Gerçekten Eksik Olanlar (Strateji ile)
# LotFrontage: Aynı semtteki (Neighborhood) medyan değer ile doldur
if 'LotFrontage' in all_data.columns:
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

# Diğerleri (Mod - en sık görülen değer ile doldur)
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities', 'Functional'):
    if col in all_data.columns:
        # test setinde NaN olabilecekler için mod()[0] kullanmak güvenlidir
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

print("Eksik veri doldurma tamamlandı.")

# --- 5. Adım: Öznitelik Mühendisliği (Feature Engineering) ---
print("\n--- 5. Adım: Öznitelik Mühendisliği ---")

# MSSubClass sayısal değil, kategoriktir
if 'MSSubClass' in all_data.columns:
    all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

# Sıralı (Ordinal) Kategorileri Sayısal Değerlere Dönüştürme (Label Encoding)
print("Sıralı (Ordinal) özellikler sayısallaştırılıyor...")
# Kalite haritası
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
for col in ('ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
            'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'):
    if col in all_data.columns:
        all_data[col] = all_data[col].map(quality_map).fillna(0)
        all_data[col] = all_data[col].astype(int)

# Diğer sıralı haritalar
if 'BsmtExposure' in all_data.columns:
    all_data['BsmtExposure'] = all_data['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}).astype(int)
if 'LotShape' in all_data.columns:
    all_data['LotShape'] = all_data['LotShape'].map({'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}).astype(int)
if 'LandSlope' in all_data.columns:
    all_data['LandSlope'] = all_data['LandSlope'].map({'Sev': 0, 'Mod': 1, 'Gtl': 2}).astype(int)
if 'GarageFinish' in all_data.columns:
    all_data['GarageFinish'] = all_data['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}).astype(int)
if 'PavedDrive' in all_data.columns:
    all_data['PavedDrive'] = all_data['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2}).astype(int)
if 'BsmtFinType1' in all_data.columns:
    all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}).astype(int)
if 'BsmtFinType2' in all_data.columns:
    all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}).astype(int)

# Sırasız (Nominal) Kategorileri One-Hot Encoding'e Çevirme
print("Sırasız (Nominal) özellikler One-Hot Encoding'e çevriliyor...")
all_data = pd.get_dummies(all_data)
print(f"One-Hot Encoding sonrası yeni sütun sayısı: {all_data.shape[1]}")

# --- 6. Adım: Çarpık Sayısal Özelliklerin Düzeltilmesi ---
print("\n--- 6. Adım: Çarpık Sayısal Özellikler Düzeltiliyor ---")
# Çarpıklığı (skewness) yüksek olan sayısal özellikleri bul
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
skewness = skewed_feats[abs(skewed_feats) > 0.75]
print(f"Çarpıklık (skewness) değeri 0.75'ten büyük {len(skewness)} adet özellik bulundu.")

# Bu özelliklere log(1+x) dönüşümü uygula
for feat in skewness.index:
    all_data[feat] = np.log1p(all_data[feat])

print("Çarpıklık düzeltme tamamlandı.")

# --- Sonuç: Modellemeye Hazır Veri Setlerini Ayırma ---
print("\n--- ÖN İŞLEME TAMAMLANDI ---")

# Birleştirilmiş veriyi tekrar train ve test olarak ayır
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
y = y_train

print(f"Model için hazır 'X_train' (özellikler) boyutu: {X_train.shape}")
print(f"Model için hazır 'X_test' (özellikler) boyutu: {X_test.shape}")
print(f"Model için hazır 'y' (hedef) boyutu: {y.shape}")

# Temizlenmiş verileri kaydet
X_train.to_csv('train_processed_X.csv', index=False)
# y verisini kaydederken .to_csv() metodunu doğrudan Series üzerinde kullanmak en güvenlisidir.
# Bu, boyut ve indeks hatalarını önler.
y.to_csv('train_processed_y.csv', index=False, header=True)
X_test.to_csv('test_processed_X.csv', index=False)

print("\nTemizlenmiş veriler 'train_processed_X.csv', 'train_processed_y.csv' ve 'test_processed_X.csv' olarak kaydedildi.")
print("\nArtık modelleme aşamasına geçmeye hazırsınız.")