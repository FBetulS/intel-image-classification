# Intel Image Classification

## 📖 Proje Açıklaması
Bu proje, Intel'in görüntü sınıflandırma veri setini kullanarak derin öğrenme tabanlı bir model geliştirmeyi amaçlamaktadır. Model, çeşitli görüntüleri sınıflandırmak için konvolüsyonel sinir ağı (CNN) mimarisi kullanır. Proje, görüntü artırma teknikleri ile modelin genel performansını artırmayı hedefler.

## 🔗 Veri Kümesi
Veri kümesi, [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data) adresinden alınmıştır. Bu veri seti, farklı sınıflara ait görüntüleri içerir ve modelin eğitim ve test aşamalarında kullanılmak üzere düzenlenmiştir.

## 🔗 Hugging Face Uygulaması
Ayrıca, projenin etkileşimli bir versiyonu [Intel Image Classification - Hugging Face Space](https://huggingface.co/spaces/btulftma/intel-image-classification) adresinde bulunmaktadır.

## 🛠️ Kullanılan Kütüphaneler
- `tensorflow`: Derin öğrenme modeli geliştirmek için.
- `matplotlib`: Görselleştirme için.
- `numpy`: Sayısal işlemler için.

## 📁 Veri Yapısı
- `seg_train`: Eğitim verileri.
- `seg_test`: Test verileri.
- `seg_pred`: Tahmin verileri.

## 📈 Model Mimarisi
Model aşağıdaki katmanları içerir:
1. **Conv2D**: Görüntüden özellik çıkarmak için.
2. **MaxPooling2D**: Boyutları küçültmek için.
3. **Flatten**: Çok boyutlu veriyi tek boyutlu hale getirmek için.
4. **Dense**: Sınıflandırma görevini gerçekleştirmek için.
5. **Dropout**: Aşırı öğrenmeyi önlemek için.

## 📊 Eğitim Süreci
Model, aşağıdaki gibi eğitim verileri ile eğitilir:
```python
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)
