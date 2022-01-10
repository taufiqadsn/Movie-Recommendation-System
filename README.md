# Laporan Proyek Machine Learning - Taufiq Adjie Sanjaya
## Project Overview
Filmku adalah suatu aplikasi pemutar film yang sudah ada sejak tahun 2010 hingga sekarang. Sejak awal perilisannya, aplikasi tersebut mampu meraih 1 juta penonton tiap tahunnya dan terus bertambah seiring waktu. Namun, dua tahun belakangan ini jumlah penonton tersebut mengalami penurunan yang cukup signifikan. Tentu hal ini akan berimbas buruk untuk kedepannya jika tidak ada cepat diatasi. Owner Filmku menyadari bahwa hal ini disebabkan karena pada saat pengguna menonton sebuah film, pengguna tersebut tidak dapat menemukan film lain yang kemungkinan ia sukai berdasarkan genre film yang ia tonton saat ini.

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

Berdasarkan pada gambar diatas, kita mengetahui bahwa pada data movies terdapat dua data tipe kategorik dan satu data tipe numerik serta berisi jumlah total 9742 baris serta 3 kolom. Gambar selanjutnya menerangkan bahwa pada data tersebut tidak terdapat nilai yang hilang (missing value).

**Ratings**

Berikut gambaran dari data ratings :

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/3.png?raw=true)

Pada data ratings seluruh tipe datanya merupakan tipe numerik yang memiliki jumlah total 100836 baris serta 4 kolom.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/4.png?raw=true)

Dapat kita ketahui bahwa tidak terdapat missing value pada data ratings berdasarkan pada gambar diatas.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/5.png?raw=true)

Gambar tersebut merupakan nilai tendensi sentral pada data ratings. Di setiap kolomnya tidak terdapat nilai 0 pada baris _min_ yang berarti memperkuat data bahwa tidak terdapat missing value.
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
## Data Preparation
Pada tahap ini kita akan melakukan beberapa persiapan data sebagai berikut :
<br>
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

Judul film pada data terlihat lebih rapi dan terurut dibanding sebelumnya. Selanjutnya kita cek genre film yang unik.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/10.png?raw=true)

Jika kita amati, pada gambar terdapat genre film dengan kategori '(no genres listed)'. Kemungkinan data pada indeks tersebut terjadi kesalahan saat input. Untuk mengatasi masalah ini kita perlu mengeksplorasi datanya lebih lanjut dan melakukan analisis. Mari kita analisa kategori '(no genres listed)' terdapat pada film apa saja. Berikut hasil setelah kita eksplorasi dengan perintah kode.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/11.png?raw=true)

Ternyata cukup banyak jumlah film dengan kategori genre '(no genres listed)'. Selanjutnya kita akan analisa satu persatu film untuk mengetahui apakah ada kategori genre lain pada setiap film.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/12.png?raw=true)

Gambar diatas merupakan salah satu contoh dari beberapa film yang berkategori '(no genres listed)'. Pada sample tersebut ternyata tidak ada kategori genre lain yang berarti ada kemungkinan kesalahan saat input. Ada berbagai cara untuk mengatasi permasalahan ini, namun pada kasus ini kita akan mengganti genre tersebut dengan unknown. Berikut hasil yang telah diterapkan dengan perintah kode.

![](https://github.com/cumapemula/Movie-Recommendation-System/blob/main/Images/13.png?raw=true)

