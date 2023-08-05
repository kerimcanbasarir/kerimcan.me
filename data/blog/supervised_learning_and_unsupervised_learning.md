---
title: 'Supervised Learning and Unsupervised Learning'
date: 2023-08-05
tags: ['Supervised Learning', 'Machine Learning', 'Unsupervised Learning']
summary: 'Recursion in JavaScript and React Components'
---

## Table of Contents

<TOCInline toc={props.toc} exclude="Table of Contents" />

![giris](/static/giris.png)

# Supervised Learning and Unsupervised Learning

## Supervised Learning

Supervised learning, makine öğrenmesinde kullanılan önemli bir yöntemdir. Bu yöntem, bir modelin belirli bir veri seti üzerinde eğitilmesini ve modelin giriş verilerinden çıktıları tahmin etme yeteneğini geliştirmeyi amaçlar. Eğitim sürecinde, modelin doğru çıktıları tahmin etmesi için giriş verileriyle birlikte hedef çıktılar sağlanır. Model, bu eğitim veri setindeki desenleri ve ilişkileri analiz ederek, giriş verileriyle ilişkilendirilen doğru çıktıları öğrenir. Bu öğrenme süreci sonunda, model giriş verilerine dayalı olarak çıktıları tahmin edebilir hale gelir. Modelin performansını değerlendirmek için genellikle ayrı bir test veri seti kullanılır.

Supervised learning, **sınıflandırma ve regresyon** gibi farklı problemlerin çözümünde kullanılır. Gerçek dünyada geniş bir uygulama alanına sahiptir ve birçok sektörde başarıyla kullanılmaktadır.

### Regression

Lineer regresyon, istatistiksel bir yöntemdir ve bir bağımlı değişkenin (hedef değişken) bir veya daha fazla bağımsız değişkenle (girdi değişkenleri) ilişkisini modellemek için kullanılır. Bu yöntem, bağımlı değişkenin sürekli bir değer alması durumunda tercih edilir. Lineer regresyon, veri noktalarını en iyi şekilde bir doğru üzerine yerleştirerek, bağımsız değişkenlerin değerlerine dayalı olarak bağımlı değişkenin tahmin edilmesini amaçlar. Temel olarak, lineer regresyon veri analizi, tahminler ve ilişkisel modellerin oluşturulması gibi birçok alanda yaygın olarak kullanılan bir öngörü yöntemidir.

Gözetimli öğrenme, en yaygın kullanılan algoritmaları içerir. Bu öğrenme türünde, bir makine öğrenme modeli, veri setindeki örnekleri kullanarak x değişkeninin y sonucunu tahmin etmeyi veya giriş ile çıkış arasındaki ilişkiyi öğrenmeyi amaçlar. Gözetimli öğrenme, bu tür tahmin ve öğrenme problemlerini çözmek için kullanılan öğrenme algoritmalarını içerir.

![inputt](/static/inputt.png)

→ Gözetimsiz öğrenmenin ayırt edici özelliği bu algoritmaları öğrenmesi için örnek verilir. Bu örnekler doğru cevapları verilen x girişleri için elde ettiğimiz y sonuçlarını verir.

| **input (x)**    | output (y)   | Application         |
| ---------------- | ------------ | ------------------- |
| email →          | spam mı ?    | Spam Filtering      |
| english →        | Spanish      | Machine Translation |
| Ads, User Info → | click (0/1)? | Online Advertising  |

Bütün bu uygulamalarda yapılan şeyler aynıdır. Modeli x girdileriyle (örnekleriyle) ve y çıkışlarıyla (doğru cevaplarla) eğitileceklerdir. Model bu ve ona karşılık gelen y outputları üretebilir.

**Örnek**
Regression: Housing Price Prediction
Diyelim ki evin boyutuna göre ev fiyatlarını tahmin etmek istiyoruz. Bir miktar veri topladık ve dağılım grafiğini oluşturduk.

![house_price](/static/house_price.png)

750 feet²'lik bir evin fiyatını tahmin etmek istediğimizi düşünelim.

Basit bir düz çizgi çizerek (x₁), evin yaklaşık 150k değerinde olduğunu tahmin edebiliriz. Ancak, verilere uygun bir düz çizgi oluşturmak, kullanabileceğimiz tek algoritma değildir. Daha iyi sonuçlar elde etmemizi sağlayan daha gelişmiş algoritmalar da bulunmaktadır. Örneğin, basit bir düz çizgi yerine daha karmaşık ve kompleks bir eğrili fonksiyon çizgisi (x₂) kullanarak evin 200k değerinde olduğunu tahmin edebiliriz. Bu durumda, en iyi (en yüksek) fiyat tahminini belirlemek yerine, veri setine dayalı olarak en uygun modeli seçmek daha doğru sonuçlar sağlayacaktır. Bu algoritmanın amacı, ev fiyatlarını en doğru şekilde tahmin etmektir. İşte bu nedenle bu yönteme gözetimli öğrenme adı verilir.

![housee](/static/housee.png)

### Classification

Sınıflandırma, bir bağımlı değişkenin (hedef değişken) kategorik bir değer alması durumunda kullanılır. Bu yöntem, belirli bir örneğin bağımsız değişkenlerin (girdi değişkenleri) değerlerine dayalı olarak hangi sınıfa ait olduğunu tahmin etmeyi amaçlar. Sınıflandırma algoritmaları, önceden etiketlenmiş örneklerin bir eğitim seti üzerinde eğitilerek, yeni veri noktalarını sınıflandırmak için kullanılabilir. Sınıflandırma, örneklerin etiketlenmesi, spam filtreleme, hastalık teşhisi, görüntü tanıma gibi birçok uygulama alanında yaygın olarak kullanılan bir yöntemdir.

Örnek: Göğüs Kanseri Teşhisi (Breast Cancer Detection)

Bu örnekte doktorlar, göğüs kanseri teşhisine yardımcı olacak bir makine öğrenmesi modeli geliştiriyor. Makine öğrenmesi modeli, hastanın tıbbi kayıtlarını kullanarak tümörün kötü huylu veya iyi huylu olup olmadığını tahmin etmeye çalışır.

![classification](/static/classification.png)

Bu grafikte yatay eksen tümör boyutunu, dikey eksen ise tümörün iyi huylu (0) ve kötü huylu (1) olarak etiketlenmiş sonuçlarını temsil eder. Sınıflandırmanın regresyondan farklı olmasının temel nedeni, sınıflandırmada yalnızca belirli kategoriler arasında tahminde bulunulmasıdır. Bu durumda çıktı değerimiz sadece 0 (iyi huylu) veya 1 (kötü huylu) olabilir. Regresyon ise çıktı değerinin sonsuz sayılar arasından herhangi bir sayı olabileceği bir tahminleme yöntemidir. Sınıflandırmada ise çıkış değerimizin sadece iki farklı değer (0 ve 1) olabileceğini gözlemleyebiliriz.

Ayrıca, bu veri setini bir çizgi üzerinde görselleştirebiliriz.

![line_classification](/static/line_classification.png)

Bu sınıflandırma sadece 1,0 ve 2… gibi sonlu sayıları tahmin eder. Anca 0.5, 1.7, -1.5 gibi sayıları tahmin edemez.

#### Classification’da İki veya daha fazla girdi

Önceki örnekte sadece tek bir giriş değerine odaklanmıştık; tümör boyutu. Ancak tahmin yapabilmek için birden fazla giriş değerini kullanabiliriz. Örneğin, tümör boyutunun yanı sıra hastanın yaşı da önemli bir giriş değeri olabilir.

![two_classification_feature](/static/two_classification_feature.png)

Bu senaryoda, algoritmanın yapması gereken iyi huylu ve kötü huylu tümörler arasında bir ayrım sınırı belirlemektir. Yani öğrenme algoritması, eldeki verilere dayanarak bu sınır çizgisini nasıl oluşturacağını anlamalıdır. Algoritmanın bulduğu bu sınır çizgisi, doktorlara teşhis koymada yardımcı olabilir.

### Regresyon ve sınıflandırmanın temel farkları

**Amaçları:** Regresyon algoritmaları, bağımlı değişkeni tahmin etmek amacıyla kullanılırken, sınıflandırma algoritmaları, bağımlı değişkenin kategorik değerini tahmin etmek amacıyla kullanılır.

**Çıktılar:** Regresyon algoritmalarının çıktıları genellikle sürekli sayısal değerlerdir. Örneğin fiyat veya sıcaklık gibi. Sınıflandırma algoritmalarının çıktıları ise kategorik değerlerdir. Örneğin bir nesnenin türü veya bir kişinin cinsiyeti gibi.

**Veri Tipleri:** Regresyon algoritmaları genellikle sayısal verileri kullanırken, sınıflandırma algoritmaları kategorik verilerle çalışır.

**Değişkenler:** Regresyon algoritmaları, bağımlı ve bağımsız değişkenler arasındaki ilişkiyi analiz etmek için kullanılırken, sınıflandırma algoritmaları özellikle bağımsız değişkenleri (predictor) kullanır.

## Unsupervised Learning

Gözetimsiz öğrenme, makine öğrenimi yöntemlerinden biridir ve veri setindeki yapıları ve desenleri keşfetmek amacıyla kullanılır. Bu yöntemde, bir bağımlı değişken (hedef değişken) olmadan çalışılır, yani veri seti önceden etiketlenmemiştir. Gözetimsiz öğrenme, veri setindeki gizli ilişkileri ve kalıpları ortaya çıkarmak için istatistiksel yöntemler ve algoritmalar kullanır. Bu yaklaşımda, veri noktaları kendi aralarında benzerliklere veya farklılıklara göre gruplandırılabilir (kümeleme) veya veri setinin boyutunu azaltmak için özellik çıkarımı yapılabilir. Gözetimsiz öğrenme, veri keşfi, anomaliden kaçınma, veri sıkıştırma gibi birçok alanda kullanılır. Bu yöntem, etiketleme maliyetinden kaçınarak veri setlerindeki değerli bilgilerin keşfedilmesine yardımcı olur.

### Clustering

Kümeleme, bir makine öğrenimi yöntemi olarak kullanılan bir süreçtir ve benzer özelliklere sahip veri noktalarını gruplara (kümeler) ayırmayı amaçlar. Kümeleme, bir bağımlı değişkenin olmadığı durumlarda uygulanır ve bağımsız değişkenlerin (girdi değişkenleri) değerlerine dayanarak benzer veri noktalarını bir araya getirir. Bu yöntem, veri noktaları arasındaki benzerlik ölçütlerine göre kümeleme yaparak, homojen ve birbirinden farklı gruplar oluşturmayı hedefler. Kümeleme algoritmaları, veri setindeki yapıyı anlamak, segmentasyon, pazar analizi, müşteri segmentasyonu gibi birçok uygulama alanında kullanılır. Kümeleme, veri keşfi sürecinde bilinmeyen ilişkilerin keşfedilmesine yardımcı olan önemli bir araçtır.

Bir önceki sınıflandırma örneğinde veriler iyi huylu (0) ve kötü huylu (1) olarak etiketlenmişti (label). Ancak gözetimsiz öğrenmede hiçbir örnek veri etiketlenmez. Yani sadece x (bağımsız) değişkenleri bulunur.

![unsupervised_Learning](/static/unsupervised_Learning.png)

Diyelim hastanın yaşı ve tümörün boyutu gibi veriler elimizde bulunuyor, yani x değişkenleri mevcut. Ancak bu verilerin iyi huylu (0) ve kötü huylu (1) gibi etiketleri yok. Yani y değişkeni elimizde bulunmuyor. Bu örneği sağ üstteki grafik ile düşünelim. Etiketlerin olmadığı durumda, tümörün iyi huylu veya kötü huylu olup olmadığını belirlemek mümkün değil. Bunun yerine yapmamız gereken, verileri bir gruba, yakınlığa, benzerliğe, ilişkiye veya hiyerarşiye göre analiz etmektir. İşte buna gözetimsiz öğrenme (Unsupervised Learning) denir. Gözetimsiz öğrenme terimi kullanılır çünkü algoritma üzerinde kontrol veya gözetim sağlanamaz. Giriş (x) değerlerinin çıkış değerlerini (y) tahmin etmek yerine, sadece giriş değerlerini alıp bu veriler arasında ilginç yapılar veya benzerlikler aranmasını sağlarız. Örneğin yukarıdaki veri setine bakarsak, gözetimsiz öğrenme algoritması bu verileri iki farklı kümeye ayırmaya karar verebilir. Bu gözetimsiz öğrenmenin "kümeleme algoritması" olarak adlandırılan bir türüdür.

**Örnek: Google News ve Kümeleme**

Google News'da bir şey arattığımızda, çıkan sonuçlardaki başlıklar genellikle benzer içeriklerden oluşur. Örneğin, çıkan başlık: "Japonya'nın en eski hayvanat bahçesinde dev bir panda ikiz doğurdu."

![screen_s](/static/screen_s.png)

Burada geçen "Panda", "ikizler" ve "hayvanat bahçesi" kelimeleri altı farklı yazıda da bulunuyor. Yani kümeleme algoritması, o günün yüz binlerce haberi arasından benzer kelimeleri içeren yazıları gruplandırır. Bu algoritma kümelemeyi tamamen kendi başına yapar, insan müdahalesine ihtiyaç duyulmaz. Bu nedenle kümeleme algoritması, gözetimsiz öğrenmenin bir türü olarak kabul edilir.

**Örnek: Müşteri Gruplandırma (Clustering)**

Birçok şirket, müşterilere ait devasa veritabanlarına sahiptir. Bu verileri kullanarak müşterileri daha iyi anlamak ve onlara daha iyi hizmet sunmak için farklı pazar segmentlerine ayırmak mümkün müdür? Deeplearning.ai ekibi, topluluklarını daha iyi anlamak amacıyla böyle bir çalışma (anket) gerçekleştirdi.

Neden farklı insanlar bu dersi aldı?
Neden haftalık bülten aboneliği veya etkinliklere katılım gerçekleştirdiler?

![clustering](/static/clustering.png)

Pazar segmentasyon ekibi, deeplearning.ai'deki ana öğrenci tiplerini yukarıdaki grafikteki üç gruba ayırdı. Bu sayede daha uygun etkinlikler, fırsatlar ve çalışmalar sunarak pazarda daha etkili bir şekilde faaliyet gösterebilme amacı güdüyorlar.

### Unsupervised Learning Diğer türleri

- Clustering (Kümeleme) - Etiketlenmemiş verilerin anlamlılıklarına göre gruplandırılma yöntemi.
- Anomaly Detection (Anomali Tespiti) - Olağan dışı konuların tespiti için kullanılır.
- Dimensionality Reduction (Boyut Azaltma) - Büyük verilerin boyutlarının azaltılması işlemi.
