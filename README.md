# Görüntü İşleme Tabanlı Sürücü Yorgunluk ve Uyku Tespit Sistemi

Bu proje, bilgisayarlı görü ve makine öğrenmesi teknikleri kullanılarak sürücülerin uykuya dalma ve yorgunluk durumlarını gerçek zamanlı olarak tespit etmeyi amaçlayan bir güvenlik asistanıdır.

## 🚀 Projenin Güncel Özellikleri (Final Sürümü)

Vize sonrasında projeye eklenen yeni donanımsal ve algoritmik çözümlerle sistemin kararlılığı maksimuma çıkarılmıştır:

* **EAR (Eye Aspect Ratio) ile Uyku Tespiti:** Sürücünün göz kapaklarının dikey ve yatay mesafeleri (Öklid uzaklığı) hesaplanarak uykuya dalma anı (3 saniye kapalılık) tespit edilir.
* **Kafa Eğimi (Head Tilt) Tespiti:** Sürücünün yorgunluğa bağlı olarak kafasının düşmesi (Roll açısının 20 dereceyi aşması) matematiksel `arctan2` fonksiyonu ile tespit edilip anında uyarı verilir.
* **Dinamik Zoom ve Gimbal Etkisi:** Araç içi sarsıntılardan etkilenmemek için yüzün geometrik merkezi sürekli takip edilir. LERP (Linear Interpolation) mantığıyla kamera görüntüsü dinamik olarak kırpılarak (zoom) sürücünün yüzü daima merkezde tutulur.
* **Hassas Görünürlük Kontrolü (Canny Edge):** Işık parlaması veya gözlük yansıması gibi durumlarda kameranın körleştiğini tespit etmek için Canny kenar bulma algoritması çalışır. Sistem gözleri göremediğinde asılsız alarm vermek yerine kendini korumaya alıp kullanıcıdan 'E' tuşu ile onay bekler.
* **Asenkron Alarm (Threading):** Alarm çalarken ana döngünün (FPS) yavaşlamaması için sesli ikaz sistemi arka planda paralel olarak çalıştırılır.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python
* **Yüz ve Landmark Tespiti:** Google MediaPipe (FaceLandmarker)
* **Görüntü İşleme:** OpenCV, Numpy
* **Yapay Zeka (Test Aşaması):** Scikit-Learn (Random Forest)

## 💻 Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri yükleyin:
   `pip install opencv-python mediapipe numpy`
2. `face_landmarker.task` dosyasının projenin ana dizininde olduğundan emin olun.
3. Kameranızı bağlayın ve ana scripti çalıştırın:
   `python main.py`
4. Programdan çıkmak için 'q', hata durumlarında onay vermek için 'e' tuşunu kullanın.
