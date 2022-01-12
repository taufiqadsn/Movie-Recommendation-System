# Laporan Proyek Machine Learning - Taufiq Adjie Sanjaya
## Project Overview
Filmku adalah suatu aplikasi pemutar film yang sudah ada sejak tahun 2010 hingga sekarang. Sejak awal perilisannya, aplikasi tersebut mampu meraih 1 juta penonton tiap tahunnya dan terus bertambah seiring waktu. Namun, dua tahun belakangan ini jumlah penonton tersebut mengalami penurunan yang cukup signifikan. Tentu hal ini akan berimbas buruk untuk kedepannya jika tidak cepat diatasi. Owner Filmku menyadari bahwa hal ini disebabkan karena pada saat pengguna menonton sebuah film, pengguna tersebut tidak dapat menemukan film lain yang kemungkinan ia sukai berdasarkan genre film yang ia tonton saat ini.

Lantas Owner Filmku pun berencana akan membuat suatu sistem rekomendasi berdasarkan kategori tertentu agar para pengguna dapat tertarik kembali untuk menggunakan aplikasi tersebut. Sistem rekomendasi ini diharapkan dapat memberikan manfaat untuk meningkatkan jumlah penonton, click rate, serta lamanya pengguna dalam menggunakan aplikasi.

## Business Understanding
Pada tahap ini, Owner Filmku akan menguraikan beberapa masalah yang akan dihadapi serta jawaban dari masalah tersebut.
### Problem Statements
Berdasarkan pernyataan diatas, Owner Filmku akan membuat suatu sistem rekomendasi untuk menjawab permasalahan tersebut.
* Bedasarkan data pengguna yang dimiliki dari tahun 2010 hingga saat ini, bagaimana cara membuat sistem rekomendasi yang dipersonalisasi dengan teknik _content based filtering_ ?
* Lalu dengan data rating yang dimiliki, bagaimana Owner dapat merekomendasikan film lain yang mungkin ia sukai dan belum pernah ditonton ?

### Goals
Untuk menjawab permasalahan pada pernyataan diatas, kita akan membuat sistem rekomendasi dengan tujuan sebagai berikut :
* Menghasilkan sejumlah rekomendasi film untuk pengguna berdasarkan genre film yang pernah ia tonton dengan teknik _content based filtering_
* Menghasilkan sejumlah rekomendasi film yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya dengan teknik _collaborative filtering_

### Solution Statements
Berikut cara untuk meraih goals diatas :
* Melalui pendekatan machine learning dengan teknik TF-IDF Vectorizer, kita akan menemukan representasi fitur penting pada setiap genre film dan diteruskan dengan melakukan teknik Cosine Similarity untuk menghitung derajat kesamaan antar film sehingga kita dapat menghasilkan sejumlah rekomendasi film berdasarkan genre film tersebut.
* Pada _collaborative filtering_ kita menggunakan teknik embedding untuk menghitung skor kecocokan antara pengguna dan film dilanjut dengan membuat class _RecommenderNet_ yang terinspirasi dari tutorial situs [Keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) sehingga kita dapat menghasilkan sejumlah rekomendasi film yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya.

## Data Understanding
Data yang akan kita gunakan bersumber dari situs [Kaggle](https://www.kaggle.com/). Berikut data yang akan kita gunakan : [Movie Recommender System Dataset](https://www.kaggle.com/gargmanas/movierecommenderdataset). Data tersebut memiliki dua file terpisah yakni movies dan ratings. Data tersebut berisi sejumlah informasi tentang sejumlah film dengan genrenya serta rating dari para pengguna. Pada data movies berisi jumlah total 9742 baris serta 3 kolom sedangkan pada data ratings berisi jumlah total 100836 baris serta 4 kolom. Data tersebut akan dijelaskan melalui variabel dibawah ini.

**Variabel pada datasets Movie Recommender System Dataset sebagai beikut :**
* userId : yaitu ID pengguna
* movieId : yaitu ID film
* title : yaitu judul film
* genres : yaitu kategori film
* rating : yaitu penilaian pengguna terhadap film tersebut
* timestamp : yaitu waktu dimana penilaian tersebut diberikan

Untuk lebih memahami tentang data yang akan kita gunakan, kita akan melakukan beberapa tahapan diantaranya :
* Univariate Exploratory Analysis Data

Tahapan eksplorasi penting untuk memahami variabel-variabel pada data serta korelasi antar variabel. Pada kasus ini, kita akan mengeksplorasi variabel movies dan ratings.

**Movies**

Berikut gambaran dari data movies :

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/1.png?raw=true)

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/2.png?raw=true)

Berdasarkan pada gambar diatas, kita mengetahui bahwa pada data movies terdapat dua data tipe kategorik dan satu data tipe numerik serta berisi jumlah total 9742 baris serta 3 kolom. Gambar selanjutnya menerangkan bahwa pada data tersebut tidak terdapat nilai yang hilang (missing value). Selanjutnya kita cek genre film yang unik.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/10.png?raw=true)

Jika kita amati, pada gambar terdapat genre film dengan kategori '(no genres listed)'. Kemungkinan data pada indeks tersebut terjadi kesalahan saat input. Untuk mengatasi masalah ini kita perlu mengeksplorasi datanya lebih lanjut dan melakukan analisis. Mari kita analisa kategori '(no genres listed)' terdapat pada film apa saja. Berikut hasil setelah kita eksplorasi dengan perintah kode.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/11.png?raw=true)

Ternyata cukup banyak jumlah film dengan kategori genre '(no genres listed)'. Cara mengatasi permasalahan ini akan kita bahas pada tahapan persiapan data.

**Ratings**

Berikut gambaran dari data ratings :

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/3.png?raw=true)

Pada data ratings seluruh tipe datanya merupakan tipe numerik yang memiliki jumlah total 100836 baris serta 4 kolom.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/4.png?raw=true)

Dapat kita ketahui bahwa tidak terdapat missing value pada data ratings berdasarkan pada gambar diatas.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/5.png?raw=true)
<br>
Keterangan : 
<br>
       - count : yaitu jumlah data keseluruhan
       <br>
       - mean : yaitu nilai rata-rata pada kolom tertentu
       <br>
       - std : yaitu standar deviasi pada kolom tertentu
       <br>
       - min : yaitu nilai terkecil pada kolom tertentu
       <br>
       - 25% : yaitu kuartil bawah data pada kolom tertentu
       <br>
       - 50% : yaitu median atau nilai tengah data pada kolom tertentu
       <br>
       - 75% : yaitu kuartil bawah data pada kolom tertentu
       <br>
       - max : yaitu nilai terbesar pada kolom tertentu

Gambar tersebut merupakan nilai tendensi sentral pada data ratings. Di setiap kolomnya tidak terdapat nilai 0 pada baris _min_ yang berarti memperkuat data bahwa tidak terdapat missing value. Namun, kita juga perlu memastikan apakah ada nilai rating 0.5 yang diberikan oleh pengguna sebab jika tidak dipastikan kemungkinan data tersebut bisa saja merupakan anomali.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/38.png?raw=true)


## Data Preparation
Pada tahap ini kita akan melakukan beberapa persiapan data sebagai berikut :
<br>
### Data Cleaning

* Menggabungkan Data Movies dan Ratings

Berikut hasil setelah kedua data tersebut digabungkan menjadi satu.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/6.png?raw=true)

Data tersebut kini telah menjadi satu dan mengalami perubahan struktur data yang kini berisi jumlah total 100836 baris serta 5 kolom. Pada proses penggabungan ini memungkinkan adanya missing value pada setiap kolom. Lalu, apakah data tersebut terdapat missing value atau tidak kita akan mengetahuinya pada gambar berikut.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/7.png?raw=true)

Data tersebut tidak terdapat missing value setelah mengalami penggabungan data. Data ini berarti bersih dan siap dilanjutkan ke tahap berikutnya.
<br>
* Menghilangkan Noise 

Untuk menghilangkan noise kita akan mengubah semua huruf menjadi huruf kecil pada kolom _title_ dan _genres_. Berikut hasil output setelah dijalankan dengan perintah kode.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/8.png?raw=true)

* Menyamakan Genre Film

Pada tahap ini kita akan mengawali dengan mengurutkan film berdasarkan kolom _movieId_. Berikut hasil output setelah kita jalankan dengan perintah kode.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/9.png?raw=true)

Judul film pada data terlihat lebih rapi dan terurut dibanding sebelumnya. Pada tahapan pemahaman data kita mendapati bahwa ada kategori (no genres listed) pada kolom genre film. Kita akan analisa satu persatu film untuk mengetahui apakah ada kategori genre lain pada setiap film yang berkategori (no genres listed).

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/12.png?raw=true)

Gambar diatas merupakan salah satu contoh dari beberapa film yang berkategori '(no genres listed)'. Pada sample tersebut ternyata tidak ada kategori genre lain yang berarti ada kemungkinan kesalahan saat input. Ada berbagai cara untuk mengatasi permasalahan ini, namun pada kasus ini kita akan mengganti genre tersebut dengan unknown. Berikut hasil yang telah diterapkan dengan perintah kode.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/13.png?raw=true)

* Menghapus Data Duplikat

Pada tahapan ini kita akan menghapus data duplikat dan hanya menggunakan data unik saja untuk dimasukkan ke dalam proses pemodelan. Kita membuang data duplikat pada kolom _movieId_ dan hasilnya sebagai berikut.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/14.png?raw=true)

Dapat dilihat bahwa tidak ada film yang sama pada tabel.
<br>
### Data Transform

* Mengkonversi Data Series Menjadi List

Pada tahap ini kita akan mengkonversi data series menjadi list. Berikut hasil output dari perintah kode yang kita jalankan.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/39.png?raw=true)

Tahap selanjutnya kita akan membuat dictionary untuk menentukan pasangan key value pada data yang telah kita siapkan sebelumnya. Berikut hasilnya.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/16.png?raw=true)

Data sudah siap untuk dimasukkan ke dalam proses pemodelan dengan teknik _content based filtering_. Karena kita juga menerapkan teknik _collaborative filtering_ maka ada beberapa tahapan lagi pada data preparation sebelum dimasukkan ke dalam proses pemodelan dengan teknik _collaborative filtering_. Berikut tahapan selanjutnya.

* Encode fitur

Pada tahap ini, kita melakukan persiapan data untuk menyandikan (encode) fitur _'userId'_ dan _'movieId'_ ke dalam indeks integer. Berikut output dari perintah kode yang kita jalankan.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/17.png?raw=true)

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/18.png?raw=true)

* Normalisasi

Selanjutnya, kita perlu memetakan (mapping) data user dan film menjadi satu value terlebih dahulu. Lalu, buatlah rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training. Berikut hasil setelah kita jalankan dengan perintah kode.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/40.png?raw=true)

* Membagi Data Menjadi Training dan Validasi

Sebelum memasuki tahap pembagian dataset, kita akan mengacak datanya terlebih dahulu agar distribusinya menjadi random.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/20.png?raw=true)

Selanjutnya kita akan melakukan pembagian data kita menjadi data training dan data validasi. Berikut hasil setelah kita jalankan dengan perintah kode.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/48.png?raw=true)

Kini data sudah siap untuk dimasukkan ke dalam proses pemodelan.

## Modeling
Pada tahap ini kita akan membuat sebuah model sistem rekomendasi dengan teknik _content based filtering_ dan _collaborative filtering_ dengan data yang telah kita siapkan sebelumnya.

* Content Based Filtering

Pada tahap ini, kita akan membangun sistem rekomendasi sederhana berdasarkan genre pada sebuah film. Ada dua teknik yang akan kita gunakan pada kasus ini, yaitu TF-IDF Vectorizer dan Cosine Similarity. TF-IDF Vectorizer akan kita gunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap genre film, sedangkan Cosine Similarity digunakan untuk menghitung derajat kesamaan antar film.

**TF-IDF Vectorizer**

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/41.png?raw=true)

Hasil output dari kode yang telah kita jalankan pada gambar diatas dihasilkan seluruh fitur penting dari genre film. Berikutnya, kita akan melakukan fit dan transformasi ke dalam bentuk matriks.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/42.png?raw=true)

Hasil output tersebut menunjukkan matriks yang kita miliki berukuran (9724, 22). Nilai 9724 merupakan ukuran data dan 22 merupakan matriks genre film. Seharusnya jumlah metriks genre film adalah 20, karena pada genre film _'sci-fi'_ dan _'film-noir'_ terdapat tanda (-) sehingga dianggap terdapat dua nilai namun ini tidak jadi masalah karena sistem tetap menganggapnya sebagai satu kesatuan. Untuk menghasilkan vektor tf-idf dalam bentuk matriks, kita akan menggunakan fungsi todense().

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/43.png?raw=true)

Selanjutnya, kita lihat matriks tf-idf untuk beberapa film dan genre film tersebut.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/24.png?raw=true)

Pada hasil output matriks diatas menunjukkan bahwa film _shadow world (2016)_ merupakan genre film documentary. Hal ini didapat dari nilai matriks 1.0 pada genre documentary. Lalu, pada film _eyes of laura mars (1978)_ memiliki dua nilai matriks pada kolom genre thriller dan mystery. Hal ini disebabkan karena pada film _eyes of laura mars (1978)_ memiliki lebih dari satu kategori genre film. Selanjutnya, kita akan menghitung derajat kesamaan antara satu film dengan film lainnya untuk menghasilkan kandidat film yang akan direkomendasikan.

**Cosine Similarity**

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/44.png?raw=true)

Kode diatas menghasilkan keluaran berupa matriks kesamaan dalam bentuk array. Selanjutnya, mari kita lihat matriks kesamaan setiap film dengan menampilkan judul film dalam 5 sampel kolom dan 10 sampel baris.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/26.png?raw=true)

Berdasarkan tabel di atas terdapat beberapa nilai yang mengindikasikan kesamaan antara film yang berada dalam kolom dengan film yang berada pada baris. Semakin nilai tersebut mendekati angka 1.0 maka film tersebut terindikasi sama. Sebagai contoh, pada film _requiem for a heavyweight (1962)_ teridentifikasi sama dengan film _white nights (1985)_ dan _elephant (2003)_ karena memiliki nilai 1.0 antara keduanya.

**Mendapatkan Rekomendasi**

Tahap ini merupakan tahap terakhir dimana kita akan membuat suatu fungsi untuk dalam membuat sistem rekomendasi. Kita akan ambil satu sample film dan menemukan rekomendasi film berdasarkan kesamaan genrenya.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/46.png?raw=true)

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/47.png?raw=true)

Pada gambar diatas kita ambil sample film _toy story (1995)_ yang memiliki lima kategori genre yaitu adventure, animation, children, comedy, dan fantasy. Lalu, sistem kita memberikan rekomendasi 10 film dengan kategori genre yang sama. Hal ini menandakan bahwa sistem rekomendasi yang kita buat sudah berhasil.

* Collaborative Filtering

Pada tahap ini, model menghitung skor kecocokan antara pengguna dan film dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan film. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan film. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan film. Skor kecocokan ditetapkan dalam skala (0,1) dengan fungsi aktivasi sigmoid. Selanjutnya, lakukan proses compile terhadap model dan mulailah proses training. Berikut sebagian hasil dari training.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/45.png?raw=true)

Untuk mendapatkan visualisasi tentang hasil training, kita akan membuat plot matriks evaluasi.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/31.png?raw=true)

Perhatikanlah, proses training model cukup smooth dan model konvergen pada epochs sekitar 100. Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.19 dan error pada data validasi sebesar 0.20. Nilai tersebut cukup bagus untuk sistem rekomendasi. Selanjutnya, kita akan menuliskan kode untuk membuat sistem dimana sistem tersebut akan merekomendasikan film yang belum pernah ia tonton yang mungkin cocok untuk pengguna berdasarkan rating pada beberapa film yang sudah ditonton pengguna sebelumnya. Berikut hasil dari sistem tersebut.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/33.png?raw=true)

Dari output tersebut, kita dapat membandingkan antara movie with high rating from user dan top 10 recommendation. Beberapa film rekomendasi memiliki sejumlah genre yang sesuai dengan rating user. Dengan ini model kita memiliki prediksi yang cukup sesuai.

## Evaluation
Tahap akhir ini kita akan menjelaskan hasil proyek berdasarkan matriks evaluasi pada kedua teknik tersebut.

* Content Based Filtering

Pada teknik ini, kita akan menghitung manual nilai presisi sebagai evaluasi. Presisi disini merupakan jumlah film rekomendasi yang relevan dengan kategori film yang dipilih. Berikut formula dari evaluasi presisi.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/34.png?raw=true)

Hasil rekomendasi sebagai berikut:

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/46.png?raw=true)

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/47.png?raw=true)

Berdasarkan hasil rekomendasi di atas, dapat diketahui bahwa film _toy story (1995)_ merupakan film dengan genre adventure, animation, children, comedy, dan fantasy. Lalu sistem memberikan 10 rekomendasi film. Dari 10 film yang direkomendasikan, semuanya memiliki kesamaan genre secara keseluruhan dengan film _toy story (1995)_. Artinya, nilai presisi sistem kita sebesar 10/10 atau 100%.

* Collaborative Filtering

Pada teknik ini kita menggunakan Root Mean Squared Error (RMSE) sebagai metriks evaluasi. Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai observasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati. Formula RMSE sebagai berikut:

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/35.png?raw=true)

Berikut hasil perhitungan RMSE pada model yang kita buat.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/36.png?raw=true)

Pada data training diperoleh nilai error akhir sebesar 0.19 dan pada data validasi sebesar 0.20. Artinya, model kita sudah cukup baik karena pada RMSE semakin kecil nilai yang diperoleh maka semakin dekat nilai hasil prediksi. 
