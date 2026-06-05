import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_eye_images_from_folder(folder_path, label, max_images=5000):
    """
    Fotoğrafları okur, siyah-beyaza çevirir, 32x32 boyutuna küçültür
    ve makinenin anlayacağı düz bir piksel dizisine (array) çevirir.
    (Hız için her klasörden şimdilik maksimum 5000 fotoğraf alıyoruz)
    """
    features = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"HATA: {folder_path} klasörü bulunamadı!")
        return features, labels

    print(f"'{folder_path}' klasörü okunuyor...")
    count = 0
    
    for filename in os.listdir(folder_path):
        if count >= max_images:
            break 
            
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                
                img_resized = cv2.resize(img, (32, 32))
                
                
                img_flattened = img_resized.flatten()
                
                features.append(img_flattened)
                labels.append(label)
                count += 1
                
                
                if count % 1000 == 0:
                    print(f"  -> {count} fotoğraf işlendi...")
                    
    return features, labels

print("VERİLER HAZIRLANIYOR...\n")
open_X, open_y = load_eye_images_from_folder("data/train/open eyes", 0, max_images=5000)
close_X, close_y = load_eye_images_from_folder("data/train/close eyes", 1, max_images=5000)

X = np.array(open_X + close_X)
y = np.array(open_y + close_y)

if len(X) == 0:
    print("\nKRİTİK HATA: Klasörlerden hiçbir veri okunamadı.")
    exit()

# Veriyi %80 Eğitim, %20 Test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nVeri seti başarıyla bölündü: {len(X_train)} Eğitim verisi, {len(X_test)} Test verisi.")


print("\nYapay zeka modeli (Random Forest) eğitiliyor... (Bu işlem bilgisayar hızına göre 30-60 saniye sürebilir)")

model = RandomForestClassifier(n_estimators=30, max_depth=5, min_samples_split=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*45)
print(f"YAPAY ZEKA MODEL BAŞARI ORANI (ACCURACY): %{accuracy * 100:.2f}")
print("="*45)
print("\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Açık Göz (0)", "Kapalı Göz (1)"]))