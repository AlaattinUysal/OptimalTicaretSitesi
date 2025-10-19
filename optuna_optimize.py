import optuna
import pandas as pd
from stable_baselines3 import PPO
import torch as th
import sqlite3 # Optuna'nın veritabanı için

# Proje dosyalarımızı import ediyoruz
from veri_hazirlama import veri_cek_ve_hazirla
from ticaret_ortami import TicaretOrtami

# --- VERİYİ BİR KEZ BAŞTA YÜKLEYELİM ---
print("Optimizasyon için eğitim ve test verileri hazırlanıyor...")
EĞİTİM_VERİSİ = veri_cek_ve_hazirla(
    hisse_kodu="THYAO.IS",
    baslangic_tarihi="2020-01-01",
    bitis_tarihi="2024-12-31"
)
TEST_VERİSİ = veri_cek_ve_hazirla(
    hisse_kodu="THYAO.IS",
    baslangic_tarihi="2025-01-01",
    bitis_tarihi="2025-09-12" # Bugünün tarihi
)
# ------------------------------------

def objective(trial):
    # Bu fonksiyonun içeriği aynı, hiçbir değişiklik yok
    try:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        n_neurons = trial.suggest_int("n_neurons", 64, 256)
        
        policy_kwargs = dict(
            net_arch=dict(pi=[n_neurons, n_neurons], vf=[n_neurons, n_neurons]),
            activation_fn=th.nn.ReLU
        )

        eğitim_ortamı = TicaretOrtami(df=EĞİTİM_VERİSİ, baslangic_bakiye=100000)
        model = PPO("MlpPolicy", eğitim_ortamı, policy_kwargs=policy_kwargs, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, verbose=0)
        model.learn(total_timesteps=30000)

        test_ortamı = TicaretOrtami(df=TEST_VERİSİ, baslangic_bakiye=100000)
        obs, info = test_ortamı.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_ortamı.step(action)
            done = terminated or truncated
        
        final_portfolio_value = info['toplam_portfoy_degeri']
        return final_portfolio_value
    except Exception as e:
        print(f"Deneme başarısız oldu: {e}")
        return 0


if __name__ == "__main__":
    # --- YENİ EKLENEN KISIM: KALICI KAYIT AYARI ---
    # Sonuçları saklamak için bir veritabanı dosyası tanımlıyoruz.
    storage_name = "sqlite:///optuna_study.db"
    study_name = "ppo_optimization_v1" # Çalışmamıza bir isim veriyoruz

    # Optuna'ya bu veritabanını kullanmasını ve eğer varsa eski çalışmayı yüklemesini söylüyoruz.
    study = optuna.create_study(
        storage=storage_name,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True # Bu satır, kaldığı yerden devam etmeyi sağlar!
    )
    # ---------------------------------------------
    
    # n_trials: Toplamda kaç deneme yapılacağını belirtir.
    # Optimizasyon, bu sayıya ulaşana kadar devam eder.
    study.optimize(objective, n_trials=50)

    # --- En İyi Sonuçları Göster ---
    print("\n--- OPTİMİZASYON TAMAMLANDI ---")
    print(f"Toplam deneme sayısı: {len(study.trials)}")
    print("En iyi deneme:")
    trial = study.best_trial
    
    print(f"  Değer (Portföy Sonucu): {trial.value:,.2f} TL")
    print("  En İyi Hiperparametreler:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")