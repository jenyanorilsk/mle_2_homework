import os
import shutil
import traceback
import configparser

from pyspark.ml.feature import HashingTF, IDF, IDFModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix

# для генерации случайного вектора просмотренных фильмов новым пользователем
import numpy as np

from adapter import SparkAdapter

from logger import Logger

SHOW_LOG = True

class Processor():

    def __init__(self):
        """
        default initialization
        """

        self.config = configparser.ConfigParser()
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)

        try:
            self.adapter = SparkAdapter()
            self.sc = self.adapter.get_context()
            self.spark = self.adapter.get_session()
        except:
            self.log.error(traceback.format_exc())

        if not self._load_models():
            raise Exception('Can\'t load models')
        
        self.log.info("Processor is ready")
        pass

    def _load_watched(self) -> bool:
        path = self.config.get("MODEL", "WATCHED_PATH")
        if path is None or not os.path.exists(path):
            self.log.error('Matrix of watched movies doesn\'t exists')
            return False
        self.log.info(f'Reading {path}')
        try:
            self.watched = CoordinateMatrix(self.spark.read.parquet(path) \
                .rdd.map(lambda row: MatrixEntry(*row)))
        except:
            self.log.error(traceback.format_exc())
            return False
        return True
    
    def _load_tf(self) -> bool:
        path = self.config.get("MODEL", "TF_PATH")
        if path is None or not os.path.exists(path):
            self.log.error('TF model doesn\'t exists')
            return False
        self.log.info(f'Reading {path}')
        try:
            self.hashingTF = HashingTF.load(path)
        except:
            self.log.error(traceback.format_exc())
            return False
        return True
    
    def _load_idf(self) -> bool:
        path = self.config.get("MODEL", "IDF_PATH")
        if path is None or not os.path.exists(path):
            self.log.error('IDF model doesn\'t exists')
            return False
        self.log.info(f'Reading {path}')
        try:
            self.idf = IDFModel.load(path)
        except:
            self.log.error(traceback.format_exc())
            return False
        return True
    
    def _load_idf_features(self) -> bool:
        path = self.config.get("MODEL", "IDF_FEATURES_PATH")
        if path is None or not os.path.exists(path):
            self.log.error('IDF features doesn\'t exists')
            return False
        self.log.info(f'Reading {path}')
        try:
            self.idf_features = self.spark.read.load(path)
        except:
            self.log.error(traceback.format_exc())
            return False
        return True

    def _load_models(self) -> bool:
        
        self.log.info('Loading Matrix of watched movies')
        if not self._load_watched():
            return False
        
        self.log.info('Loading TF model')
        if not self._load_tf():
            return False

        self.log.info('Loading IDF model')
        if not self._load_idf():
            return False

        self.log.info('Loading IDF features')
        if not self._load_idf_features():
            return False

        return True
    
    def _get_recomendation(self, ordered_similarity, max_count=5):
        
        # преобразуем типы, чтобы использовать умножение, идея в следующе:
        # будем использовать похожесть пользователя как вес для просмотренных им фильмов
        # таким образом мы сможем посчитать взвешенную по похожести зрителей сумму фильма (ранг)
        users_sim_matrix = IndexedRowMatrix(ordered_similarity)
        
        # а вот и ранг каждого фильма, взвешенный на похожесть пользователей
        multpl = users_sim_matrix.toBlockMatrix().transpose().multiply(self.watched.toBlockMatrix())
        
        ranked_movies = multpl.transpose().toIndexedRowMatrix().rows.sortBy(lambda row: row.vector.values[0], ascending=False)

        result = []
        for i, row in enumerate(ranked_movies.collect()):
            if i >= max_count:
                break
            result.append((row.index, row.vector[0]))
        return result

    def sample(self):
        # получим матрицу (пользователи, фичи)
        temp_matrix = IndexedRowMatrix(self.idf_features.rdd.map(
            lambda row: IndexedRow(row["user_id"], Vectors.dense(row["features"]))
        ))
        temp_block = temp_matrix.toBlockMatrix()

        # транспонируем матрицу, чтобы получить (фичи, пользователи)
        # это нужно для метода columnSimilarities, который считает косинусное сходство
        similarities = temp_block.transpose().toIndexedRowMatrix().columnSimilarities()

        # берём одного случайного пользователя
        #random_user = df.rdd.takeSample(False, 1, seed=0)[0]
        #target_userid = random_user["user_id"]
        user_id = np.random.randint(low=0, high=self.watched.numCols())
        self.log.info(f'Random user ID: {user_id}')

        # мы получили верхнюю треугольную матрицу похожести пользователей, поэтому, чтобы найти
        # для конкретного пользователя ближайшего похожего, мы берём id этого пользователя
        # и отбираем значения похожестей
        filtered = similarities.entries.filter(lambda x: x.i == user_id or x.j == user_id)

        # отсортировав по убыванию мы получим самых похожих пользователей среди первых элементов
        ordered_similarity = filtered.sortBy(lambda x: x.value, ascending=False) \
            .map(lambda x: IndexedRow(x.j if x.i == user_id else x.i, Vectors.dense(x.value)))

        recomendations = self._get_recomendation(ordered_similarity)
        self.log.info('TOP recomendations for existing user:')
        for movie_id, rank in recomendations:
            self.log.info(f'- movie # {movie_id} (rank: {rank})')

        pass

    def random(self):
        
        watched_movies = np.random.randint(low=0, high=self.watched.numCols(), size=int(self.watched.numCols()/4)).tolist()
        newdf = self.sc.parallelize([[-1, watched_movies]]).toDF(schema=["user_id", "movie_ids"])
        new_tf_features = self.hashingTF.transform(newdf)
        new_idf_features = self.idf.transform(new_tf_features)
        new_idf_features = new_idf_features.first()["features"]

        # рассчитаем похожесть нового пользователя с теми, кто уже есть в датасете
        # здесь считаем косинусное расстояние по формуле (оно здесь не нормируется, но нам и не нужно)
        similarities = self.idf_features.rdd.map(
            lambda row: IndexedRow(
                row["user_id"],
                Vectors.dense(new_idf_features.dot(row["features"] / (new_idf_features.norm(2) * row["features"].norm(2))))
            )
        )
        ordered_similarity = similarities.sortBy(lambda x: x.vector.values[0], ascending=False)
        recomendations = self._get_recomendation(ordered_similarity)
        self.log.info('TOP recomendations for random user:')
        for movie_id, rank in recomendations:
            self.log.info(f'- movie # {movie_id} (rank: {rank})')
        pass

if __name__ == "__main__":
    processor = Processor()
    processor.sample()
    #processor.random()