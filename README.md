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

