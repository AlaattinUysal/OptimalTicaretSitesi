import pandas as pd
from stable_baselines3 import PPO
import torch as th

from veri_hazirlama import veri_cek_ve_hazirla
from ticaret_ortami import TicaretOrtami

if __name__ == "__main__":
    hisse_verisi = veri_cek_ve_hazirla(
        hisse_kodu="THYAO.IS",
        baslangic_tarihi="2020-01-01",
        bitis_tarihi="2024-12-31"
    )

    if hisse_verisi is not None:
        env = TicaretOrtami(df=hisse_verisi, baslangic_bakiye=100000)

        # --- OPTUNA'DAN GELEN "ŞAMPİYON" AYARLARI KULLANIYORUZ ---
        policy_kwargs = dict(
            net_arch=dict(pi=[155, 155], vf=[155, 155]),
            activation_fn=th.nn.ReLU
        )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=4.431302727643054e-05,
            n_steps=2048,
            gamma=0.9770458406880964,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./ppo_tensorboard_logs/"
        )
        # --------------------------------------------------------

        print(">>> 'Şampiyon Ajan' (Optimize Edilmiş PPO Modeli) eğitiliyor...")
        # Şimdi bu en iyi ayarlarla tam süreli bir eğitim yapıyoruz
        model.learn(total_timesteps=150000) # Adım sayısını artırarak daha derin öğrenmesini sağlıyoruz

        # Eğitilmiş en iyi modeli kaydediyoruz
        model.save("ppo_champion_model")
        print("Şampiyon PPO modeli başarıyla eğitildi ve 'ppo_champion_model.zip' olarak kaydedildi.")
        
    else:
        print("Veri çekilemediği için program sonlandırılıyor.")