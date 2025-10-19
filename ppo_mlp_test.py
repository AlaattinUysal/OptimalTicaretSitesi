import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# İlgili proje dosyalarını import ediyoruz
from veri_hazirlama import veri_cek_ve_hazirla
from ticaret_ortami import TicaretOrtami

# --- TEST PARAMETRELERİ ---
MODEL_DOSYASI = "ppo_mlp_model.zip" # Yeni model dosyamızın adı
HİSSE_KODU = "THYAO.IS"
BASLANGIC_BAKIYE = 100000

# Test için daha önce hiç görmediği bir tarih aralığı
TEST_BASLANGIC_TARIHI = "2025-01-01"
TEST_BITIS_TARIHI = "2025-09-06"  # Bugünün tarihi

# --- 1. Veri ve Ortamı Hazırla ---
print("Test verisi çekiliyor...")
test_verisi = veri_cek_ve_hazirla(
    hisse_kodu=HİSSE_KODU,
    baslangic_tarihi=TEST_BASLANGIC_TARIHI,
    bitis_tarihi=TEST_BITIS_TARIHI
)

if test_verisi is None or len(test_verisi) < 2:
    print("Test için yeterli veri bulunamadı.")
    exit()

# Basitleştirilmiş Ticaret Ortamı'nı kullanıyoruz (LSTM olmadığı için sequence_length yok)
env = TicaretOrtami(df=test_verisi, baslangic_bakiye=BASLANGIC_BAKIYE)

# --- 2. Eğitilmiş PPO Modelini Yükle ---
model = PPO.load(MODEL_DOSYASI)
print(f"'{MODEL_DOSYASI}' başarıyla yüklendi.")

# --- 3. Test Simülasyonunu Çalıştır ---
print("Test simülasyonu başlatılıyor...")
obs, info = env.reset()
done = False
aksiyon_gecmisi = []

while not done:
    # deterministic=True: Ajanın her zaman en iyi bildiği hamleyi yapmasını sağlar (test modu)
    action, _states = model.predict(obs, deterministic=True)
    aksiyon_gecmisi.append(action)
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print("Test simülasyonu tamamlandı.")

# --- 4. Sonuçları Hesapla ve Göster ---
ppo_sonuc = info['toplam_portfoy_degeri']
ilk_fiyat = test_verisi['Close'].iloc[0]
son_fiyat = test_verisi['Close'].iloc[-1]
al_ve_tut_sonuc = (BASLANGIC_BAKIYE / ilk_fiyat) * son_fiyat

print("\n--- PPO (MLP) TEST SONUÇLARI ---")
print(f"Başlangıç Bakiyesi: {BASLANGIC_BAKIYE:,.2f} TL")
print("-" * 30)
print(f"PPO Ajanının Portföy Sonucu: {ppo_sonuc:,.2f} TL")
print(f"'Al ve Tut' Stratejisi Sonucu: {al_ve_tut_sonuc:,.2f} TL")
print("-" * 30)

ppo_getiri = ((ppo_sonuc / BASLANGIC_BAKIYE) - 1) * 100
al_ve_tut_getiri = ((al_ve_tut_sonuc / BASLANGIC_BAKIYE) - 1) * 100

print(f"PPO Ajanı Getirisi: %{ppo_getiri:.2f}")
print(f"'Al ve Tut' Getirisi: %{al_ve_tut_getiri:.2f}")

if ppo_getiri > al_ve_tut_getiri:
    print("\nSonuç: PPO ajanı, basit 'Al ve Tut' stratejisinden daha iyi bir performans gösterdi. ✅")
else:
    print("\nSonuç: PPO ajanı, basit 'Al ve Tut' stratejisini geçemedi. ❌")


# --- 5. İşlemleri Görselleştir ---
plt.figure(figsize=(16, 8))
plt.plot(test_verisi.index, test_verisi['Close'], label='THYAO Fiyat', color='blue', alpha=0.6)

alim_noktalari = [i for i, a in enumerate(aksiyon_gecmisi) if a == 0]
plt.plot(test_verisi.index[alim_noktalari], test_verisi['Close'].iloc[alim_noktalari], '^', markersize=10, color='green', label='Alım Yapıldı')

satim_noktalari = [i for i, a in enumerate(aksiyon_gecmisi) if a == 1]
plt.plot(test_verisi.index[satim_noktalari], test_verisi['Close'].iloc[satim_noktalari], 'v', markersize=10, color='red', label='Satım Yapıldı')

plt.title('PPO Ajanının (Basit) Test Verisi Üzerindeki Alım-Satım Noktaları')
plt.xlabel('Tarih')
plt.ylabel('Hisse Fiyatı (TL)')
plt.legend()
plt.grid(True)
plt.show()