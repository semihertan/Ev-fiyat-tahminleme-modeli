import pandas as pd
import io

# 1. Veriyi Yükleme
try:
    df = pd.read_csv('train.csv')
    print("Veri başarıyla yüklendi.\n")
except FileNotFoundError:
    print("Hata: train.csv dosyası bulunamadı.")
    exit()

# --- A. İLK BAKIŞ (df.head) ---
# Verinin ilk 5 satırını gösterir. Sütunların neye benzediğini anlamak için kullanılır.
print("### 1. VERİNİN İLK 5 SATIRI (df.head()) ###")
# Pandas'ın tüm sütunları göstermesini sağlamak için ayar (Sunumda kesik görünmemesi için)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())
print("\n" + "="*50 + "\n")


# --- B. VERİ YAPISI (df.info) ---
# Sütun isimleri, veri tipleri (int, float, object) ve dolu veri sayısını gösterir.
print("### 2. VERİ YAPISI VE TİPLERİ (df.info()) ###")
# df.info() normalde doğrudan konsola yazdırır.
# Bunu bir değişkene alıp temiz yazdırmak için 'buffer' kullanıyoruz.
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
print(s)
print("\n" + "="*50 + "\n")


# --- C. İSTATİSTİKSEL ÖZET (df.describe) ---
# Sayısal sütunların ortalaması, standart sapması, min/max değerlerini verir.
# Sunumda "Verimizdeki fiyatlar ortalama ne kadar?" sorusuna cevap verir.
print("### 3. İSTATİSTİKSEL ÖZET (df.describe()) ###")
print(df.describe().T) # .T işlemi tabloyu yan çevirir (Transpose), slaytta okuması daha kolaydır.