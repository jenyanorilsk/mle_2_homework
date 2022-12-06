
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix

# для вывода номера строки в отладке
from inspect import currentframe, getframeinfo


# входной файл
INPUT = "./data/generated.csv"
# число фичей TF
FEATURES_COUNT = 10000

# параметры spark
MASTER = "local"
APP_NAME = "second_lab"
NUM_PROCESSORS = "3"
NUM_EXECUTORS = "1"
# для тестового запуска на маленьком датасете 
# большое число партиций только тормозит
#NUM_PARTITIONS = 40
NUM_PARTITIONS = None

# вывод отладочной информации
DEBUG_PRINT = True

def debug_print(*args):
    if DEBUG_PRINT:
        print('\n>>>>>>>> [', currentframe().f_back.f_lineno, '] >>>', *args, '\n')

# общий метод для подсчёта ранга фильмов для рекомендации
def recomend(all_watched_matrix, ordered_similarity, user_watched):

    # преобразуем типы, чтобы использовать умножение, идея в следующе:
    # будем использовать похожесть пользователя как вес для просмотренных им фильмов
    # таким образом мы сможем посчитать взвешенную по похожести зрителей сумму фильма (ранг)

    users_sim_matrix = IndexedRowMatrix(ordered_similarity)
    # а вот и ранг каждого фильма, взвешенный на похожесть пользователей
    multpl = users_sim_matrix.toBlockMatrix().transpose().multiply(all_watched_matrix.toBlockMatrix())
    
    ranked_movies = multpl.transpose().toIndexedRowMatrix().rows.sortBy(lambda row: row.vector.values[0], ascending=False)

    for row in ranked_movies.collect():
        debug_print('RECOMENDATION: MOVIE #', row.index, f'({row.vector}) {"WATCHED" if row.index in user_watched else ""}')

    pass


conf = SparkConf()

conf.set("spark.app.name", APP_NAME)
conf.set("spark.master", MASTER)
conf.set("spark.executor.cores", NUM_PROCESSORS)
conf.set("spark.executor.instances", NUM_EXECUTORS)
conf.set("spark.executor.memory", "8g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer.max", "2000")
conf.set("spark.executor.heartbeatInterval", "6000s")
conf.set("spark.network.timeout", "10000000s")
conf.set("spark.shuffle.spill", "true")
conf.set("spark.driver.memory", "8g")
conf.set("spark.driver.maxResultSize", "8g")

# создание контекста, вывод конфига

sc = SparkContext(conf=conf)
spark = SparkSession(sc)

print('\t>>>>', 'spark config:')
for conf in sc.getConf().getAll():
    print('\t>>>>', conf[0].upper(), ' = ', conf[1])
print()

raw = sc.textFile(INPUT, NUM_PARTITIONS)

# записи, сгруппированные по user_id

grouped = raw.map(lambda x: map(int, x.split())).groupByKey() \
    .map(lambda x : (x[0], list(x[1])))

debug_print('grouped.first():', grouped.first())

count_users = grouped.count()
count_movies = grouped.flatMap(lambda x: x[1]).max() + 1

debug_print('USERS COUNT:', count_users)
debug_print('MOVIES COUNT:', count_movies)

# чтобы сделать всё правильно, используем DataFrame из-за 
# его встроенного метода columnSimilarities, позволяющего
# считать косинусное сходство между колонкам

df = grouped.toDF(schema=["user_id", "movie_ids"])

debug_print('df.printSchema():', df.printSchema())
debug_print('df.first():', df.first())

# считаем TF - частоту токенов (фильмов), должна быть 1,
#  т.к. пользователь либо посмотрел, либо не посмотрел фильм

hashingTF = HashingTF(inputCol="movie_ids", outputCol="rawFeatures", numFeatures=FEATURES_COUNT)
tf_features = hashingTF.transform(df)

debug_print('tf_features.first():', tf_features.first())

# считаем IDF - здесь уже будут дробные значения, т.к. учёт по пользователям, это и будут фичи

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf_features)
idf_features = idfModel.transform(tf_features)

# вектор IDF - фичей

debug_print('idf_features.first():', idf_features.first())
debug_print('idf_features.first()["features"]', idf_features.first()["features"])
debug_print('type(idf_features.first()["features"])', type(idf_features.first()["features"]))


# возьмём индексы просмотренных фильмов в нужном нам типе, чтобы можно было
# умножить их на похожесть пользователя

all_watched_matrix = CoordinateMatrix(grouped.flatMapValues(lambda x: x).map(lambda x: MatrixEntry(x[0], x[1], 1.0)))


##########################################################################
# Пример 1: рекомендации для существующего пользователя
##########################################################################

# получим матрицу (пользователи, фичи)

temp_matrix = IndexedRowMatrix(idf_features.rdd.map(
    lambda row: IndexedRow(row["user_id"], Vectors.dense(row["features"]))
))
temp_block = temp_matrix.toBlockMatrix()

debug_print('type(temp_block)', type(temp_block))
debug_print('temp_block.numRows()', temp_block.numRows(), 'temp_block.numCols()', temp_block.numCols())

# транспонируем матрицу, чтобы получить (фичи, пользователи)
# это нужно для метода columnSimilarities, который считает косинусное сходство

similarities = temp_block.transpose().toIndexedRowMatrix().columnSimilarities()

debug_print('type(similarities)', type(similarities))
debug_print('similarities.numRows()', similarities.numRows(), 'similarities.numCols()', similarities.numCols())
debug_print('similarities.entries.first()', similarities.entries.first())

# использовалось для вывода матриц, чтобы проверять в Excel

if count_movies < 25 and count_users < 25:

    local = similarities.toBlockMatrix().toLocalMatrix()
    debug_print('local.toArray()', local.toArray())

    simetryc = similarities.toBlockMatrix().add(similarities.toBlockMatrix().transpose()).toLocalMatrix()
    debug_print('simetryc.toArray()', simetryc.toArray())

# берём одного случайного пользователя

random_user = df.rdd.takeSample(False, 1, seed=0)[0]
debug_print('random_user:', random_user)

target_userid = random_user["user_id"]
debug_print('target_userid:', target_userid)

# мы получили верхнюю треугольную матрицу похожести пользователей, поэтому, чтобы найти
# для конкретного пользователя ближайшего похожего, мы берём id этого пользователя
# и отбираем значения похожестей

filtered = similarities.entries.filter(lambda x: x.i == target_userid or x.j == target_userid)
debug_print('filtered.collect()', filtered.collect())

# отсортировав по убыванию мы получим самых похожих пользователей среди первых элементов

ordered_similarity = filtered.sortBy(lambda x: x.value, ascending=False) \
    .map(lambda x: IndexedRow(x.j if x.i == target_userid else x.i, Vectors.dense(x.value)))
debug_print('ordered_similarity.collect()', ordered_similarity.collect())

# рекомендации для одного сэмплированного пользователя

recomend(all_watched_matrix, ordered_similarity, random_user["movie_ids"])

##########################################################################
# Пример 2: рекомендации для нового пользователя
##########################################################################

# если мы хотим взять нового пользователя, которого ещё нет в датасете и рекомендовать ему
# фильмы, то сэмплируем значения просмотренных им фильмов (здесь заданы жёстко для проверки)
# и оборачиваем в dataframe, т.к. модели у нас уже настроены на работу с ним

watched_movies = [0, 6, 8, 9]

newdf = sc.parallelize([[-1, watched_movies]]).toDF(schema=["user_id", "movie_ids"])
debug_print('newdf.printSchema():', newdf.printSchema())
debug_print('newdf.first():', newdf.first())

new_tf_features = hashingTF.transform(newdf)
debug_print('new_tf_features.first():', new_tf_features.first())

new_idf_features = idfModel.transform(new_tf_features)

debug_print('new_idf_features.first():', new_idf_features.first())
debug_print('new_idf_features.first()["features"]:', new_idf_features.first()["features"])

new_idf_features = new_idf_features.first()["features"]

# рассчитаем похожесть нового пользователя с теми, кто уже есть в датасете
# здесь считаем косинусное расстояние по формуле (оно здесь не нормируется, но нам и не нужно)
similarities = idf_features.rdd.map(
    lambda row: IndexedRow(
        row["user_id"],
        Vectors.dense(new_idf_features.dot(row["features"] / (new_idf_features.norm(2) * row["features"].norm(2))))
    )
)

debug_print('similarities.first():', similarities.first())
debug_print('similarities.collect():', similarities.collect())

ordered_similarity = similarities.sortBy(lambda x: x.vector.values[0], ascending=False)
recomend(all_watched_matrix, ordered_similarity, random_user["movie_ids"])
exit(0)
