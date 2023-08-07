---
title: 'Linear Regression'
date: 2023-08-07
tags: ['Supervised Learning', 'Machine Learning', 'Linear Regression']
summary: 'machine learning notes'
---

## Table of Contents

<TOCInline toc={props.toc} exclude="Table of Contents" />

# Regression

Regresyon, bir makine öğrenimi yöntemi olarak kullanılan bir süreçtir ve bir bağımlı değişkenin (hedef değişken) bir veya daha fazla bağımsız değişkenle (girdi değişkenleri) ilişkisini modellemeyi amaçlar. Regresyon analizi, bağımlı değişkenin sürekli bir değer alması durumunda uygulanır. Bu yöntem, bağımsız değişkenlerin değerlerine dayanarak bağımlı değişkenin tahmin edilmesini sağlamayı hedefler. Regresyon modelleri, veri noktalarının bir fonksiyon veya eğri üzerinde en iyi şekilde uymasını sağlar. Regresyon analizi, tahmin yapma, trendleri belirleme, ilişkileri anlama gibi birçok uygulama alanında kullanılan önemli bir yöntemdir.

## Tek Değişkenli Linear Regresyon

Tek değişkenli regresyon, sadece bir bağımsız değişkenin (girdi değişkeni) bağımlı değişkenle (hedef değişken) ilişkisini modellemek için kullanılan bir regresyon yöntemidir. Bu yöntemde, bağımlı değişkenin sürekli bir değer alması durumunda uygulanır. Tek değişkenli regresyon analizi, bağımsız değişkenin değerine dayanarak bağımlı değişkenin tahmin edilmesini sağlamayı amaçlar. Bu analizde, veri noktalarının bir doğru üzerinde en iyi şekilde uyması hedeflenir. Tek değişkenli regresyon, ilişkiyi anlamak, tahmin yapmak veya trendleri belirlemek gibi birçok uygulama alanında kullanılan temel bir regresyon yöntemidir.

### Linear Regresyon Modeli

Diyelim ev boyuna göre fiyatını tahmin etmek istiyoruz. Burada X ekseni evin boyutu (size feet²), y ekseni de evin fiyatını 1000$ cinsinden temsil eden bir grafik var.

![regg](/static/regg.png)

Burada görünen her bir nokta, satılan evlerin boyutunu ve fiyatını temsil eden verilerdir. Bu grafik sayesinde bir evin yaklaşık değerini hesaplayabiliriz. 1250 fit²'lik bir evin fiyatını öğrenmek için veri setine dayalı olarak doğrusal regresyon modeli kullanmak, bizi yaklaşık bir değere ulaştıracaktır. Model, verilerimize göre düz bir çizgi oluşturur.

![regg](/static/reg_line.png)

Bu doğrusal çizgiye göre, 1250 fit²lik bir evin fiyatı yaklaşık 220 bin dolar olacaktır. Buna gözetimli öğrenme denir, çünkü modeli oluşturmak için cevapları verilmiş (ev fiyatı) örneklerden oluşan bir veri seti kullandık. Bu veri seti, hem evin boyutunu hem de evin fiyatını (cevaplarını) içerir. Böylece modelimiz, evin boyutuna göre ev fiyatını yaklaşık bir değerle tahmin etmeyi öğrenir.

Regresyon modeli, 200 bin, 1.5, -33.2 gibi sürekli sayıları tahmin eden gözetimli öğrenme problemlerini çözer. Burada önemli olan regresyonun kategorik veriler yerine sürekli verilerle ilgilendiğidir.

**Notasyon**

| **index ($i$)** | **Size in feet² ($𝐱$)** | **Price in 1000 ($𝐲$)** | **$(𝑥^i ,𝑦^i)$**          |
| --------------- | ----------------------- | ----------------------- | ------------------------- |
| 1               | 2104 $(𝐱¹)$             | 400 $(𝐲¹)$              | $(𝐱¹, 𝐲¹)$ = (2104, 400)  |
| 2               | 1416 $(𝐱²)$             | 232 $(𝐲²)$              | $(𝐱², 𝐲²)$ = 1416, 232)   |
| 3               | 1534 $(𝐱³)$             | 315 $(𝐲³)$              | $(𝐱³, 𝐲³)$ = (1534, 315)  |
| 4               | 852 $(𝐱⁴)$              | 178 $(𝐲⁴)$              | $(𝐱⁴,𝐲⁴)$ = (852, 178)    |
| …..             | …..                     | …..                     | …..                       |
| 47 ($m$)        | 3210 $(𝐱⁴⁷)$            | 870 $(𝐲⁴⁷)$             | $(𝐱⁴⁷, 𝐲⁴⁷)$ = (3210,870) |

| **Notasyon** | **Açıklama**                                                    | **Örnek**                | **Python variable** |
| ------------ | --------------------------------------------------------------- | ------------------------ | ------------------- |
| $𝐱$          | Training Example feature variable or input variable             | (𝐱³) = 1534              | x_train             |
| $𝐲$          | Training Example targets variable or output variable            | (𝐲³) = 315               | y_train             |
| $m$          | Number of training examples                                     | 47                       | m                   |
| $(𝑥^𝑖 ,𝑦^𝑖)$ | 𝑖𝑡ℎ Training Example                                            | (𝐱³ , 𝐲³) = (1534, 315 ) | x_i, y_i            |
| $𝑤$          | parameter: weight                                               |                          | w                   |
| $𝑏$          | parameter: bias                                                 |                          | b                   |
| $ŷ$          | estimation of the target variable (y-hat)                       | 𝐲 = 400 ŷ = 390.9        | y_hat               |
| $𝑓𝑤,𝑏(𝑥^𝑖)$  | The result of the model evaluation at 𝑥(𝑖)parameterized by 𝑤,𝑏: |                          | f_wb                |
| $𝑎$          | scalar, non bold                                                |                          |                     |
| $𝐚$          | vector, bold                                                    |                          |                     |

![learning_Alg](/static/learning_Algorithm.png)

1. Eğitim Seti, yani eğitim seti, iki bileşenden oluşmaktadır: Özellikler veya girdiler (x - evin boyutu) ve Hedefler veya çıktılar (y - evin fiyatı).
2. Modeli eğitmek için eğitim seti kullanılarak hem girdi özellikleri (özellikler) hem de çıktı hedefleri (hedefler), öğrenme algoritmasına beslenir.

3. Daha sonra gözetimli öğrenme algoritması, belirli bir işlevi üretir. Bu işlev genellikle $f$ ile gösterilir. $f$ fonksiyonunun amacı, yeni bir x değeri almak ve tahmini bir çıktı üretmektir. Bu tahmini çıktıya $ŷ$ (y-hat) adı verilir. $ŷ$, aslında gerçek $y$ değeri için bir tahmin veya yaklaşık bir değerdir.

Örneğin, Ev Genişliği özelliğini modele vererek tahmini ev fiyatı çıktısını elde ediyoruz.

**$f$ fonksiyonunu nasıl ifade etmeliyiz?**

$$𝑓𝑤,𝑏(𝑥) = wx + b $$

w ve b sayıları için seçilen değerler, x girdilerini kullanarak ŷ çıktısını tahmin etmek için kullanılır. (w ve b parametreleri aşağıda açıklanmıştır)

![f_fonk](/static/f_fonk.png)

Algoritma bu veriden öğrenir ve en iyi uyan çizgiyi oluşturmaya çalışır. Lineer fonksiyon $f_w, b(x) = wx + b$ ile ifade edilir. Bu fonksiyonun amacı, buradaki $f(x)$ fonksiyonu çizgisini kullanarak y değerini tahmin etmektir ($y → ŷ$). İşte lineer regresyonun temeli budur. Bundan sonraki adım, maliyet (cost) fonksiyonunu oluşturmaktır.

### Linear Regresyon - Cost Function

Lineer Regresyon uygulamak için yapılması gereken ilk adım, maliyet (cost) fonksiyonunu tanımlamaktır. Bu Cost fonksiyonu, modelin ne kadar başarılı olduğunu bize söyler ve daha iyi sonuçlar elde etmek için bu fonksiyonu minimize etmeye çalışırız.

**Linear Fonksiyon**
Eğitim setimiz (training set), girdi özellikleri (feature) x ve çıktı hedefleri (target) y içermektedir.

![training_set](/static/training_set.png)

Bu eğitim verisini (training data) beslemeyi amaçladığımız (fit) model, aşağıdaki gibi gösterilen bir lineer fonksiyondur:

Model : $$ f w,b(x) = wx + b $$

Makine öğrenmesinde bir modelin parametreleri, modeli geliştirmek için eğitim sürecinde ayarlayabileceğimiz değişkenlerdir. Bu parametreler $w$ ve $b$ olarak adlandırılır.

$b$ = katsayılar (coefficients)

$w$ = Ağırlıklar veya sabit (weights or constant) (eğim)

Şimdi $b$ ve $w$'nin ne işe yaradığına bakalım.

**$b$ ve $w$ Nedir?**

$w$ ve $b$ için seçtiğimiz değerlere bağlı olarak farklı bir $f(x)$ fonksiyonu elde edebiliriz. Bu da fonksiyonun grafikte farklı çizgiler çizmesine neden olur.

**örnek 1**

![f_fonk_1](/static/f_fonk_1.png)

Burada $w=0$ ve $b=1.5$ olduğunda, $f$ fonksiyonu yatay bir çizgi olur. Çünkü $w$ değeri 0 olduğu için $x$ ile çarpımı da 0 olacaktır. Yani $f$, her zaman sabit bir değeri temsil eder. Her zaman $ŷ$ (tahmini) değeri olarak 1.5 değerini alır.

**örnek 2**

![f_fonk_2](/static/f_fonk_2.png)

$w=0.5$ ve $b=0$ olduğunda:

$x = 0$ ise:
$f(0)$ = $f(x)= 0,5 x 0 + 0$ = 0 ---> 1.nokta oluşur.

$x = 1$ ise:
$f(1)$ = $f(x)= 0,5 x 1 + 0$ = 0.5 ---> 2.nokta oluşur.

$x=2$ ise:
$f(2)$ = $f(x)= 0,5 x 2 + 0$ = 1 ---> 3.nokta oluşur.

$x=3$ ise:
$f(3)$ = $f(x)= 0,5 x 3 + 0$ = 1.5 ---> 4.nokta oluşur.

**Örnek 3**

![f_fonk_3](/static/f_fonk_3.png)

$w=0.5$ ve $b=1$ olduğunda:

$x = 0$ ise:
$f(0)$ = $f(x)= 0,5 x 0 + 1$ = 1 ---> 1.nokta oluşur.

$x = 1$ ise:
$f(1)$ = $f(x)= 0,5 x 1 + 1$ = 1,5 ---> 2.nokta oluşur.

$x=2$ ise:
$f(2)$ = $f(x)= 0,5 x 2 + 1$ = 2 ---> 3.nokta oluşur.

$x=3$ ise:
$f(3)$ = $f(x)= 0,5 x 3 + 1$ = 2,5 ---> 4.nokta oluşur.

![y_hat](/static/y_hat.png)

$f(x)$ fonksiyonundan elde edilen düz çizgi, veriye uygun olmalıdır. Notasyonları hatırlatmak adına, bir eğitim örneği $(x^i, y^i)$ olarak tanımlanır. $x^i$ olarak belirtilen bu giriş, $f$ fonksiyonu tarafından $y$'nin tahmini bir değeri üretir. Bu üretilen değer $ŷ$ olarak gösterilir.

Şimdi sorulması gereken soru şudur: $ŷ$ değerini bulmak için $w$ ve $b$ değerlerini nasıl ayarlamalıyız, böylece bütün $x^i$ ve $y^i$ değerleri için en yakın sonucu tahmin edebiliriz? Bu sorunun cevabını bulmak için önce bir çizginin eğitim verisine ne kadar uyduğunu hesaplamayı bilmeliyiz. Bu amaçla, maliyet (cost) fonksiyonunu oluşturacağız.

**Cost (Maliyet) Fonksiyonu**

$$ MSE=j(w,b) = {1\over2m} \sum\_{i=1}^{m}{(ŷ^i - y^i)^2} $$

Bir makine öğrenmesi modelini eğitirken, modelin performansını ölçmek için bir maliyet fonksiyonu kullanılır. Bu fonksiyon, modelin tahminleri ile gerçek değerler arasındaki farkları hesaplayarak, modelin ne kadar doğru tahmin yaptığını belirler.

Maliyet fonksiyonu, tahmini değerden gerçek değeri çıkararak bir hata (error) değeri elde eder. Bu hata genellikle karesi alınarak hesaplanır. Her bir örnek için hata karesini hesaplamak ve toplam hatayı hesaplamak için örneklerin sırasına göre bir döngü oluşturulur. Bu nedenle maliyet fonksiyonunda hata karelerini hesaplamak için MSE (Mean Squared Error) metriği kullanılır. MSE yerine mse, rmse, mae, mape gibi metrikler de kullanılabilir. Bu bağlamda, burada MSE metriğini kullanmaktayız.

Eğitim seti büyüdükçe, maliyet fonksiyonu daha büyük değerler hesaplayabilir. Bu nedenle, hata karelerinin ortalamasını alarak hesaplanan bir maliyet fonksiyonu kullanmak daha uygundur. Böylece, eğitim setindeki örnek sayısı arttıkça, maliyet fonksiyonu otomatik olarak artmaz ve daha istikrarlı bir performans ölçümü gerçekleştirilir.

#### Cost Function Sezileri güçlendirme

Cost fonksiyonunun bir model için en iyi parametreleri nasıl bulunduğunu bir örnek üzerinden inceleyecek olursak:

Model: $f_{w,b}(x) = wx + b$

Parametreler: $w, b$

Cost fonksiyonu: $MSE = J(w,b) = \frac{1}{2m} \sum_{i=1}^{m}{(\hat{y}^i - y^i)^2}$

Hedef: $J(w,b)$'yi minimize etmek

Doğrusal modelimiz olan $f_{w,b}(x) = wx + b$, eğitim veri setimize tam olarak uyan bir düz çizgi elde etmek için $w$ ve $b$ parametreleri için en uygun değerleri seçmemiz gerekmektedir. Bu şekilde farklı $w$ ve $b$ değerlerini kullanarak farklı çizgiler elde edebiliriz. Daha sonra seçilen $w$ ve $b$ değerleri için $J$ cost fonksiyonunu ($J(w,b)$) oluşturarak modelin tahmin ettiği değer ile gerçek değer arasındaki farkı ölçeriz. Son olarak, $J(w,b)$ fonksiyonunu minimize ederek seçilen $w$ ve $b$ değerlerinin eğitim verisine ne kadar iyi uyduğunu ölçeriz. Bu sayede, doğrusal modelimiz en iyi performansını gösterir ve en uygun $w$ ve $b$ değerleri ile eğitim setine en uygun çizgiyi çizebilir.

**Basitleştirilmiş Örnek**

![basit](/static/basit.png)

Bu basitleştirilmiş örnekte yukarıdaki örneğin aynısını aldık, ancak basitleştirmek amacıyla $b$ parametresini denklemden çıkartarak (0'a eşitleyerek) $f_w(x) = wx$ modelini oluşturduk. Yani çizgimiz artık orijinden geçiyor, çünkü $x=0$ olduğunda $f(x)$ de 0 olacaktır. Bu basitleştirilmiş modelle amacımız, $w$ için $J(w)$ fonksiyonunu küçültecek bir değeri bulmaktır.

model : $fw(x) = wx$

cost function : $j(w) = {1\over2m}  \sum_{i=1}^{m}{(ŷ^i - y^i)^2}$

hedef : $minimize j(w)$

$w (eğim)= 1$

$j(1) = ?$

![orn](/static/orn.png)

Tabloya göre üç eğitim örneği sırasıyla (1,1), (2,2), (3,3) olarak gösterilmiştir. Bu veri seti özelinde $j$ cost fonksiyonunu $w = 1$ için hesaplayacağız, yani $j(1)$. Hatırlarsanız, $j$ cost fonksiyonu gerçek değerden tahmin edileni çıkarıp karesini alır ve her birini toplar, sonrasında $\frac{1}{2m}$'ye böler. Buna "hata kareler fonksiyonu" denir. Örneğimizde de bu formülü her bir eğitim verisi için uygulayacağız, yani $i=1$'den $m$'e kadar. Formül uygulandıktan sonra, bu veri seti özelinde $w=1$ iken $j(1) = 0$ çıktı. Soldaki grafiğimize çıkan sonucu işaretledik.

Ya $w$ değeri 1 yerine 0.5 olursa, o zaman bu grafik nasıl gözükürdü?

$w$ (eğim) = 0.5

$j(0.5) = ?$

![orn2](/static/orn2.png)

Aynı eğitim veri örnekleri için bu sefer $w = 0.5$ için $j$ cost fonksiyonunu hesaplayacağız, yani $j(0.5)$ değerini bulacağız. İlk örnekte $x=1$ olduğunda $f(x)=0.5$ değerini alır. Bu nedenle ilk örneğin hata karesi $(0.5-1)^2$ olarak hesaplanır. İkinci örnekte $x=2$ olduğunda $f(x)=1$, bu örneğin hata karesi ise $(1-2)^2$ olarak hesaplanır. Üçüncü örnekte $x=3$ olduğunda $f(x)=1.5$, bu örneğin hatası ise $(1.5-3)^2$ olarak hesaplanır. Tüm eğitim örnekleri hesaplandıktan sonra, bu değerler toplanıp $1/2m$ ile çarpılır. Böylece $j(0.5) = 0.583$ olarak hesaplanır. Çıkan sonucumuzu soldaki grafikteki yerine yazıyoruz.

$w=0$ desek nasıl olur? $w=0$ için $f$ ve $J$ fonksiyonlarının grafiği nasıl gözükürdü?

$w=0$

$f(0) = ?$

![orn3](/static/orn3.png)

Aynı eğitim veri örnekleri için bu sefer $j$ cost fonksiyonunu $w = 0$ için hesaplayacağız, yani $j(0)$. Burada $f(x)$ tam olarak yatay çizgi olacaktır. Bu sebeple bütün tahmin edilen değerler 0'a eşit olacaktır, yani $f(x) = 0$. $j$ fonksiyonu ${1\over2m} \times (1^2 + 2^2 + 3^2)$ olacaktır. Bütün işlemlerden sonra çıkan sonuç ise $2.33$ yapar. Yani $j(0) = 2.33$.

Farklı $w$ değerleri için cost fonksiyonunu hesaplayabilir ve grafiğini çizebiliriz. Bazı değerleri hesaplayarak, $j$ fonksiyonunun nasıl gözükeceğini görebiliriz.

![orn4](/static/orn4.png)

Yukarıdaki örneklerde seçtiğimiz farklı $w$ değerleri gibi daha farklı $w$ değerleri seçerek yukarıdaki grafiği oluşturabiliriz. Her bir nokta, farklı bir $f(x)$ çizgisini temsil etmektedir. Sonuç olarak çıkan bu "u" şeklindeki çizgimizin en düşük değeri $j(1)$ noktasıdır. Amacımız $min j(w)$ değerini bulmaktı ve böylelikle en uygun $w$ değerini elde ettik.

Bu örnekte daha kolay anlayabilmek için $b$ parametresi çıkartılmış ve örnek basitleştirilmiştir.

#### Cost Function Görselleştirme

Yukarıdaki basitleştirilmiş örneğimizde sadece $w$ parametresi alınarak $b = 0$'a eşitlenmiş ve 2 boyutlu görsel oluşturulmuştu. Bu görselleştirmede $w$ ve $b$ parametrelerini kullanarak nasıl bir görsel oluşturulduğunu ve asıl görselin nasıl okunduğunu açıklayacağız. Ancak bu tür görselleştirmeler büyük eğitim setlerinde etkili olmayabilir ve bu tür durumlarda Gradyan İnişi gibi diğer yöntemler kullanılmaktadır. Gradyan İnişi, en iyi $w$ ve $b$ değerlerini otomatik olarak bularak $J$ cost fonksiyonunu minimize etmeye yardımcı olur.

![img1](/static/img1.png)

Bu metindeki 3 boyutlu düzlem grafiği, $w$ ve $b$ olarak ayarlanmış eksenlerle gösterilir. $w$ ve $b$ değerleri değiştikçe, modelin iki parametresi olan $J(w,b)$ fonksiyonu için farklı değerler elde edilir. Her nokta, belirli bir $w$ ve $b$ değeri için $J$ noktasını temsil eder. Örneğin, $w=-10$ ve $b=-15$ ise, bu noktanın yüzeyden uzaklığı $J(-10, -15)$ noktasına karşılık gelen yükseklikle ölçülür.

$J$ fonksiyonunun daha kullanışlı ve anlaşılır bir görselleştirmesi de mevcuttur.

![img3](/static/img3.png)

Sağ üstte gördüğünüz şey, aynı $J$ fonksiyonunun kontür grafiğidir. Dikey eksen $b$, yatay eksen ise $w$ değerlerini temsil eder. Kontür grafiği, 3 boyutlu yüzeyin farklı yüksekliklerini gösterir. Her bir elips, 3 boyutlu yüzeyin merkez noktasının aynı yüksekliğine denk gelir. Başka bir deyişle, bu elipslerin her biri aynı $J$ değerine karşılık gelir.

Kontür grafiği oluşturmak için, 3 boyutlu grafiği yatay olarak kesiyoruz. Bu şekilde aynı yüksekliğe sahip noktaları elde ederiz. Bu kesiklerden her biri, elipslerden birini temsil eder.

Mavi, turuncu ve yeşil oklarla işaretlenen noktaların hepsi, farklı $w$ ve $b$ değerlerine sahip olmalarına rağmen aynı $J$ değerine sahiptir. Ancak bu üç nokta, ev fiyatı tahminleri için kötü sonuçlar verir. Kasenin dibi, $J$ fonksiyonunun en küçük değere sahip olduğu noktadır ve bu nokta eş merkezli elipslerin tam ortasında yer alır.

Sağ üstteki görseldeki $J$ fonksiyonunun en küçük değeri, halkaların tam ortasındaki noktaya karşılık gelir.

**Görselleştirme Örnekleri**

**Örnek 1**

$f(x) = -0.15x + 800$

![12d](/static/12d.png)

Grafikte belirli bir nokta gösterilmekte ve bu nokta $w$ ve $b$ parametrelerinin değerlerini temsil etmektedir. $w$ yaklaşık olarak $-0.15$ ve $b$ yaklaşık olarak $800$ olarak seçilmiştir. Bu parametrelerle oluşturulan $f(x)$ fonksiyonu sol tarafta çizilmiştir. Ancak bu fonksiyonun veri setine tam olarak uymadığı görülmektedir. Çünkü veri noktalarının çoğu bu fonksiyonun uzakında kalmaktadır. Bu nedenle $J$ grafiğindeki cost değeri oldukça yüksek ve minimum değere yakın değildir. Farklı $w$ ve $b$ değerleri seçilseydi, cost değeri daha düşük olabilirdi.

**Örnek 2**

$f(x) = 0x + 360$

![13d](/static/13d.png)

Bu örnek, veriye tam olarak uymasa da Örnek 1'den daha iyi bir seçim olabilir. "Kötünün iyisi" olarak adlandırılabilir. Bu nokta, $w$ ve $b$ parametrelerinin değerlerini temsil etmektedir. $w=0$ ve $b=360$ olarak seçilmiştir. Bu parametrelerle oluşturulan $f(x)$ fonksiyonu sol tarafta çizilmiştir. Bu fonksiyon yatay bir çizgidir çünkü $f(x) = 0 \cdot x + 360$ formülüne sahiptir, yani $w=0$ olduğu için.

**Örnek 3**

$f(x) = -0.15x + 500$

![14d](/static/14d.png)

Bu örnekte farklı $w$ ve $b$ değerleri kullanılarak oluşturulan başka bir $f(x)$ çizgisi görülmektedir. Ancak bu çizginin veriye uygun olmadığı açıktır. Önceki örneğe göre daha kötü bir seçimdir. Cost değeri oldukça yüksektir ve minimuma ulaşmaktan uzaktır. Minimum nokta, en küçük elipsin ortasında yer almalıdır.

**Örnek 3 (en uygun)**

$f(x) = 0.13x + 71$

![15d](/static/15d.png)

Son örnek olarak, sol taraftaki $f(x)$ fonksiyonunun veriye çok daha uygun olduğu görülmektedir. Sağdaki grafikte, cost (hata) fonksiyonunu temsil eden noktanın en küçük elipsin merkezine yakın olduğu görülmektedir. Tam olarak minimum değeri ifade etmese de oldukça yakındır. Bu $w$ ve $b$ değerleri ile sol taraftaki $f(x)$ çizgisi oluşturulmuştur. Her veri noktası için tahmin edilen değer ile gerçek değer arasındaki dikey mesafeyi ölçerek hatayı hesaplayabiliriz. Bütün veri noktaları için hata karelerinin toplamını hesapladığımızda, olası bütün düz çizgiler arasında en düşük hataya sahip olanı seçmiş oluruz.

Lineer regresyonda en iyi $w$ ve $b$ değerlerini bulmak için kontür grafiğini okumak yeterli değildir ve daha karmaşık makine öğrenmesi modellerinde işe yaramayabilir. Bu nedenle, en iyi $w$ ve $b$ değerlerini otomatik olarak bulan etkili bir algoritma olan "gradient descent (dereceli azalma)" algoritmasının kullanılması daha uygundur. Bu algoritma, J cost fonksiyonunu minimize etmeye yardımcı olur. Gradient descent ve farklı varyasyonları, yalnızca lineer regresyon eğitiminde değil, dünyanın en büyük ve karmaşık yapay zeka modellerinde bile kullanılır.

### Gradient Descent

Gradient Descent (GD), çok çeşitli problemlerde en uygun çözümleri bulma yeteneğine sahip genel bir optimizasyon algoritmasıdır. GD'nin temel fikri, bir maliyet fonksiyonunu en aza indirmek için parametreleri tekrarlayan şekilde güncellemektir.

Hayal edin ki yoğun bir sisin içinde dağlarda kayboldunuz. Sadece ayaklarınızın altındaki zeminin eğimini hissedebilirsiniz. Hızla vadinin dibine inmek isterseniz, en dik yokuşun yönünde aşağı inmek mantıklı olur. İşte bu durum, Gradient Descent'in temel prensibini açıklar: $w$ parametre vektörüne göre hata fonksiyonunun yerel gradyanını ölçer ve azalan gradyan yönünde adım atar. Eğimin sıfır olduğu noktada, minimuma ulaşılmış olur. Yani $J_{\text{min}}(w,b)$ elde edilir.

Gradient Descent formülü şu şekildedir:

$$w:=w-a {d\over dw}j(w,b)$$

$$b:=b-a {d\over db}j(w,b)$$

Burada "$:=$" işareti, sol taraftaki "$w$" parametresinin, sağ tarafındaki "$w - a \frac{d}{dw}j(w,b)$" ifadesine eşitlendiğini gösterir. Bu formül, $w$ parametresini güncellemek için kullanılır.

"$a$" parametresi, Yunan alfabesinin alfa harfini temsil eder. Bu denklemde, alfa genellikle 0 ile 1 arasında bir pozitif sayıdır, örneğin $0.01$ gibi. Alfa, ne kadar büyük adımlar atacağınızı belirler. Büyük bir alfa, daha hızlı bir yokuş aşağı inmeye neden olurken, küçük bir alfa daha dikkatli ve yavaş adımlar atmaya yol açar.

"$\frac{d}{dw}j(w,b)$", $j(w,b)$ fonksiyonunun $w$ parametresine göre türevidir. Bu türev, hangi yöne doğru adım atılması gerektiğini belirler.

"$j(w,b)$" ise maliyet fonksiyonunu temsil eder. Bu fonksiyon, modelin veri kümesindeki hataları ölçer ve modelin doğruluğunu artırmak için kullanılır. Bu fonksiyon, modelin $w$ ve $b$ parametrelerine bağlı olarak ifade edilir.

Gradient Descent algoritması, modelinizi daha iyi bir uyum sağlamak üzere otomatik olarak güncellemeye yardımcı olur. Bu sayede en iyi $w$ ve $b$ parametrelerini bulabilir ve maliyeti minimuma indirebilirsiniz. Bu algoritma, sadece lineer regresyon gibi temel modellerde değil, daha karmaşık yapay zeka modellerinde de yaygın olarak kullanılır.

#### Kısmi türevini alma

Kısmi türev, bir fonksiyonun birden fazla değişkeni olduğunda, sadece belirli bir değişkenin türevidir. Gradient Descent yöntemi, bu kısmi türevlerin kullanımını içerir ve bir fonksiyonun minimumunu veya maksimumunu bulmada yardımcı olur.

Bu örnekte sağ ve sol yönlerdeki gradient descent işlemlerini ele alalım:

**Sağa Doğru Gradient Descent:**

![right_gradient](/static/right_gradient.png)

Daha önce $b$ parametresini geçici olarak 0 olarak kabul ettiğimizi hatırlayalım. Bu durumda sadece $w$ parametresi bulunuyor. Eksenlerde yatayda $w$ parametresi, dikeyde ise $J(w)$ maliyet fonksiyonu yer alır. Dereceli azalmayı rastgele bir $w$ değeriyle başlatalım ve bu noktadan $J$ fonksiyonuna doğru bir çizgi çizelim. Dereceli azalma, $w$ değerini eski $w$ değerinden, alfa değeri çarpı türev ifadesinden çıkararak günceller. Bu türev ifadesi, $J(w)$ fonksiyonunun o noktadaki eğiminin negatifini temsil eder. Yani, bu çizgi, eğrinin o noktadaki teğetini temsil eder. Eğer teğet çizgisi sağa yukarıya doğru ilerliyorsa, eğim pozitif demektir (türev pozitif bir sayıdır). Bu durumda, güncellenen $w$ değeri, $w - \alpha$ pozitif bir sayı olur. Eğim pozitif olduğundan, sola doğru hareket ediyor ve $w$ değerini azaltıyoruz. Amacımız $J$ değerini azaltmak olduğundan, bu adım doğru bir yönde atılıyor gibi görünüyor.

**Sola Doğru Gradient Descent:**

![left_gradient](/static/left_gradient.png)

Aynı mantığı bir adım geriye giderek düşünelim. Bu kez $w$ parametresini sol tarafta bir başlangıç değeri seçelim. Bu, $J$ fonksiyonunun o noktasına karşılık gelir. Türev ifadesi $dJ(w)/dw$ olup, bu noktadaki teğet çizgisine bakarsak, eğim negatif olacaktır. Bu durumda, $w$'yi güncellediğimizde, $w - \alpha$ negatif bir sayı olur. Bu da $w$ değerini artırır, çünkü negatif bir sayıdan çıkarmak, pozitif bir sayı eklemekle aynıdır. Dereceli azalmanın bu adımı, $J$ fonksiyonunun azaldığı sağa doğru bir yönde hareketi temsil eder.

Sonuç olarak, dereceli azalma yöntemi, fonksiyonun minimumunu veya maksimumunu bulma sürecinde mantıklı adımlar atmayı sağlar. Sağa veya sola doğru gradient descent, fonksiyonun eğiminin pozitif veya negatif olduğu noktalarda adım atarak hedefe yaklaşmayı amaçlar. Bu yöntem, modelin parametrelerini güncellemek ve maliyet fonksiyonunu minimize etmek için yaygın olarak kullanılır.

#### Öğrenme Oranı (Learning Rate)

Bir modeli eğitirken, Gradient Descent yöntemini kullanarak parametreleri güncellemek ve maliyet fonksiyonunu minimuma yaklaştırmak amacıyla adımlar atılır. Bu adımların büyüklüğünü belirleyen önemli bir parametre, öğrenme oranıdır.

![gradian_inisi](/static/gradian_inisi.png)

Öğrenme oranı, algoritmanın her iterasyonda ne kadar büyük bir adım atacağını belirleyen hiperparametredir. Öğrenme oranı çok küçükse, algoritma yakınsamak için çok fazla iterasyon gerektirecektir, bu da eğitim süresini uzatabilir.

![gradient_small](/static/gradient_small.png)

Öte yandan, öğrenme oranı çok yüksekse, algoritma vadinin diğer tarafına atlayabilir ve hatta daha önce bulunandan daha yükseğe çıkabilir. Bu, algoritmanın yakınsamayı atlamasına ve istenmeyen sonuçlara yol açabilir.

![gradient_large](/static/gradient_large.png)

Ayrıca, tüm maliyet fonksiyonları düz bir "kase" şeklinde olmayabilir. Çukurlar, sırtlar, platolar gibi çeşitli topografyalara sahip olabilirler. Şekil 4-6, Gradient Descent'in iki temel zorluğunu gösterir: Rastgele başlatılan bir algoritma soldan başlarsa, yerel minimuma doğru yakınsar. Sağdan başlarsa, düz bir plato üzerinde çok uzun bir süre geçirebilir ve asla global minimuma ulaşamayabilir.

![gradient__](/static/gradient__.png)

Neyse ki, Doğrusal Regresyon modeli için MSE maliyet işlevi dışbükey bir yapıya sahiptir; bu, eğri üzerinde herhangi iki noktayı seçtiğinizde, bu noktaları birleştiren doğru parçasının eğriyi asla geçmeyeceği anlamına gelir. Bu, yerel minimumların olmadığı, yalnızca bir küresel minimum olduğu anlamına gelir. Aynı zamanda, eğimi aniden değişmeyen sürekli bir fonksiyondur. Bu iki gerçeğin büyük bir sonucu vardır: Gradient Descent, global minimuma ulaşmada rastgele yaklaşmayı garanti eder (yeterince uzun süre beklerseniz ve öğrenme oranı çok yüksek değilse).

Aslında, maliyet fonksiyonu bir "kase" şeklindedir, ancak özellikler farklı ölçeklere sahipse, bu kase uzun bir vadisi şeklini alabilir. Şekil 4-7, özellik 1 ve özellik 2'nin aynı ölçeğe sahip olduğu bir eğitim setinde (solda) ve özellik 1'in özellik 2'den çok daha küçük değerlere sahip olduğu bir eğitim setinde (sağda) Gradient Descent'i göstermektedir.

![[gradient_y|500]]

![gradient_y](/static/gradient_y.png)

Gördüğünüz gibi, soldaki Gradient Descent algoritması doğrudan minimuma gidiyor ve bu nedenle hızla ulaşıyor. Sağda ise önce global minimuma neredeyse dik bir yönde ilerliyor ve daha sonra uzun bir yolculukla sona eriyor. Neredeyse düz bir vadiden aşağı iniyor. Sonunda minimum seviyeye ulaşacak, ancak bu uzun zaman alacaktır.

Bu şema ayrıca, bir modelin eğitiminin, bir maliyet fonksiyonunu (eğitim seti üzerinden) en aza indiren model parametrelerinin bir kombinasyonunu aramak anlamına geldiğini gösterir. Bu, model parametre uzayında bir arama yapma işlemidir: Bir model ne kadar fazla parametreye sahipse, parametre uzayı o kadar fazla boyuta sahip olur ve arama o kadar zorlaşır. Örneğin, 300 boyutlu bir uzayda iğne aramak, üç boyutlu bir uzayda iğne aramaktan çok daha zordur. Neyse ki, Lineer Regresyon durumunda maliyet fonksiyonu dışbükey olduğu için, iğne kase şeklinin en altındadır.

#### Doğru Güncelleme

![update](/static/update.png)

**doğru güncelleme:**

İşte eş zamanlı olarak güncelleyen doğru bir dereceli azalma algoritması. `tmp_w` ve `tmp_b` değişkenlerine sağ taraftaki ifadeleri eşitleyip güncelleriz. Daha sonra, `tmp_w` değerini w'ye, `tmp_b` değerini de b'ye atadık. Ancak burada dikkat edilmesi gereken nokta, w değerinin güncellenmemiş halinin kullanılmasıdır. Bu durumda, güncellenmemiş w değeri, türev ifadesine de uygulanacaktır. Yani kısacası `tmp_b`, güncellenmemiş w ile güncellenecektir.

**yanlış güncelleme:**

Burada `tmp_w` değişkenini sağ taraftaki ifadeyle eşitleyip güncelledik. Daha sonra güncellenen değeri w değişkenine atadık. `tmp_b` değişkeni ise sağ taraftaki ifadeyle güncellendi, fakat yeni değer alan w ile güncellendiği için yanlış sonuç üretecektir.

Buradaki önemli nokta, her iki değişkenin eş zamanlı olarak güncellenip eş zamanlı olarak yeni değerlere atanmasıdır. Bu şekilde daha doğru sonuçlara ulaşmış oluruz.

Lineer Regresyon için Gradient Descent:

$$w = w-a{1\over m}  \sum_{i=1}^{m}{(ŷ^i - y^i)}x^i$$

$$b = b-a{1\over m}  \sum_{i=1}^{m}{(ŷ^i - y^i)}$$

### Örnek Kod

```py
import numpy as np

def cost_function(X, y, theta):
	# Maliyet fonksiyonunu hesaplayan fonksiyon
    n = len(X)

    y_hat = X.dot(theta)

    squared_errors = (y_hat - y) ** 2

    cost = 1 / (2 * n) * np.sum(squared_errors)

    return cost
```

```py

def gradient_descent(X, y, theta, alpha, num_iterations):
	# Gradyan iniş algoritması ile theta değerlerini güncelleyen fonksiyon
    m = len(X)

    costs = []

    for _ in range(num_iterations):

        predictions = X.dot(theta)

        errors = predictions - y

        gradient = (1 / m) * X.T.dot(errors)

        theta -= alpha * gradient

        cost = cost_function(X, y, theta)

        costs.append(cost)

    return theta, costs
   
```

```py
def linear_regression(X, y, alpha, num_iterations):
	# Doğrusal regresyon modelini oluşturan fonksiyon
    X = np.column_stack((np.ones(len(X)), X))  # X'in başına birler sütunu ekle

    theta = np.zeros(X.shape[1])  # Başlangıçta tüm theta değerlerini sıfır ata

    theta, costs = gradient_descent(X, y, theta, alpha, num_iterations)

    return theta, costs


X = np.array([1, 2, 3, 4, 5]) # Bağımsız değişken

y = np.array([2, 4, 6, 8, 10]) # Bağımlı değişken


alpha = 0.01  # Öğrenme oranı

num_iterations = 1000  # İterasyon sayısı


theta, costs = linear_regression(X, y, alpha, num_iterations)


print("Theta:", theta)

print("Maliyetler:", costs)


```

```py

import matplotlib.pyplot as plt


# Gerçek verilerin görselleştirilmesi

plt.scatter(X, y, color='blue', label='Gerçek Veriler')


# Tahminlerin görselleştirilmesi

X_pred = np.array([0, 6])

y_pred = theta[0] + theta[1] * X_pred

plt.plot(X_pred, y_pred, color='red', label='Doğrusal Regresyon Modeli')


plt.xlabel('X')

plt.ylabel('y')

plt.title('Doğrusal Regresyon')

plt.legend()

plt.show()
```

![örnk_kod](/static/örnk_kod.png)

### sklearn ile örnek kod

```py

# Gerekli kütüphaneleri içe aktarın

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Örnek veri setini oluşturun

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Bağımsız değişkenler

y = np.array([2, 4, 6, 8, 10])  # Bağımlı değişkenler

```

```py

# Lineer regresyon modelini oluşturun ve eğitin

model = LinearRegression()

model.fit(X, y)

```

```py

# Eğitilmiş modelin katsayılarını ve kesme noktasını alın

katsayilar = model.coef_

kesme_noktasi = model.intercept_

# Görselleştirme için tahminler yapın

y_pred = model.predict(X)

```

```py

# Veri noktalarını ve tahmin edilen doğruyu görselleştirin

plt.scatter(X, y, color='blue', label='Veri Noktaları')

plt.plot(X, y_pred, color='red', linewidth=2, label='Tahmin Edilen Doğru')

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Basit Doğrusal Regresyon')

plt.legend()

plt.show()

# Sonuçları yazdırın

print("Katsayılar:", katsayilar)

print("Kesme Noktası:", kesme_noktasi)
```

![örn_kod_2](/static/örn_kod_2.png)
