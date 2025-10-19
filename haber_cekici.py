import requests
from bs4 import BeautifulSoup
import datetime

def haberleri_getir(sirket_adi, gun):
    """
    Belirtilen şirket için Google News'te o güne ait haber başlıklarını arar.
    
    Args:
        sirket_adi (str): Aranacak şirket adı (örn: "Türk Hava Yolları").
        gun (datetime.date): Haberlerin aranacağı tarih.
    
    Returns:
        list: Bulunan haber başlıklarının bir listesi.
    """
    try:
        # Google News URL formatı: q=aranacak_kelime&after=YYYY-MM-DD&before=YYYY-MM-DD
        baslangic_tarihi = gun.strftime('%Y-%m-%d')
        bitis_tarihi = (gun + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Googlebot gibi görünmek için User-Agent bilgisi ekliyoruz
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        url = f"https://news.google.com/search?q={sirket_adi}%20after%3A{baslangic_tarihi}%20before%3A{bitis_tarihi}&hl=tr&gl=TR&ceid=TR%3Atr"
        
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Hata varsa (404, 500 vb.) exception fırlatır

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google News'teki haber başlıkları genellikle 'h3' veya 'h4' etiketleri içinde bulunur
        basliklar = soup.find_all('a', class_='JtKRv') # Bu class adı değişebilir
        
        if not basliklar:
             # Eğer ilk class bulunamazsa alternatif bir class deneyebiliriz
             basliklar = soup.find_all('h3')

        bulunan_basliklar = [baslik.text for baslik in basliklar]
        
        if not bulunan_basliklar:
            return ["O tarihte ilgili haber bulunamadı."]

        return bulunan_basliklar

    except requests.exceptions.RequestException as e:
        return [f"Haberler çekilirken bir ağ hatası oluştu: {e}"]
    except Exception as e:
        return [f"Beklenmedik bir hata oluştu: {e}"]

if __name__ == "__main__":
    # Dünü test edelim
    sirket = "THY"
    tarih = datetime.date.today() - datetime.timedelta(days=1)
    
    print(f"'{sirket}' için {tarih.strftime('%Y-%m-%d')} tarihli haberler aranıyor...")
    
    haber_listesi = haberleri_getir(sirket, tarih)
    
    print("\n--- BULUNAN HABER BAŞLIKLARI ---")
    for i, baslik in enumerate(haber_listesi):
        print(f"{i+1}. {baslik}")
        