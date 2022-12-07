import os
import shutil
import traceback
import configparser

from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix

from adapter import SparkAdapter

from logger import Logger

SHOW_LOG = True

class Trainer():

    def __init__(self) -> None:
        """
        default initialization
        """

        self.config = configparser.ConfigParser()
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)
        
        self.log.info("Trainer is ready")
        pass

    def _remove_stored(self, path) -> bool:
        """
        Удаление сохраненной ранее модели
        """
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        if os.path.exists(path):
            self.log.error(f'Can\'t remove {path}')
            return False
        return True

    def _calc_watched_matrix(self, grouped, path='./models/WATCHED') -> bool:
        """
        Подготовка матрицы просмотренных фильмов
        """

        if not self._remove_stored(path):
            return False

        # возьмём индексы просмотренных фильмов в нужном нам типе, чтобы можно было
        # умножить их на похожесть пользователя
        matrix = CoordinateMatrix(grouped.flatMapValues(lambda x: x).map(lambda x: MatrixEntry(x[0], x[1], 1.0)))
        
        try:
            matrix.entries.toDF().write.parquet(path)
            self.config["MODEL"]["WATCHED_PATH"] = path
            self.log.info(f"Matrix of watched movies is stored at {path}")
        except:
            self.log.error(traceback.format_exc())
            return False
        return os.path.exists(path)
    
    def _train_tf(self, grouped, path='./models/TF_MODEL'):
        """
        Обучение TF модели
        """

        if not self._remove_stored(path):
            return None

        # чтобы сделать всё правильно, используем DataFrame из-за его встроенного метода columnSimilarities,
        # позволяющего считать косинусное сходство между колонкам
        df = grouped.toDF(schema=["user_id", "movie_ids"])

        FEATURES_COUNT = self.config.getint("MODEL", "FEATURES_COUNT", fallback=10000)
        self.log.info(f'TF-IDF features count = {FEATURES_COUNT}')

        # считаем TF - частоту токенов (фильмов), должна быть 1, т.к. пользователь либо посмотрел, либо не посмотрел фильм
        hashingTF = HashingTF(inputCol="movie_ids", outputCol="rawFeatures", numFeatures=FEATURES_COUNT)
        tf_features = hashingTF.transform(df)

        try:
            hashingTF.write().overwrite().save(path)
            self.config["MODEL"]["TF_PATH"] = path
            self.log.info(f"TF model stored at {path}")
        except:
            self.log.error(traceback.format_exc())
            return None
        
        return tf_features
    
    def _save_idf_features(self, idf_features, path='./models/IDF_FEATURES') -> bool:
        
        if not self._remove_stored(path):
            return False

        try:
            idf_features.write.format("parquet").save(path, mode='overwrite')
            self.config["MODEL"]["IDF_FEATURES_PATH"] = path
            self.log.info(f"IDF features stored at {path}")
        except:
            self.log.error(traceback.format_exc())
            return False
        return True

    
    def _train_idf(self, tf_features, path='./models/IDF_MODEL') -> bool:
        """
        Обучение IDF модели
        """

        if not self._remove_stored(path):
            return False

        # считаем IDF - здесь уже будут дробные значения, т.к. учёт по пользователям, это и будут фичи
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf = idf.fit(tf_features)

        self.log.info(f"IDF model type: {type(idf)}")

        try:
            idf.write().overwrite().save(path)
            self.config["MODEL"]["IDF_PATH"] = path
            self.log.info(f"IDF model stored at {path}")
        except:
            self.log.error(traceback.format_exc())
            return False

        # считаем значения фичей для существующих пользователей и сохраняем их
        idf_features = idf.transform(tf_features)
        if not self._save_idf_features(idf_features):
            return False
        
        return True


    def train_models(self, input_filename=None) -> bool:
        
        try:
            adapter = SparkAdapter()
            sc = adapter.get_context()
            spark = adapter.get_session()
        except:
            self.log.error(traceback.format_exc())
            return False
        
        if input_filename is None:
            INPUT_FILENAME = self.config.get("DATA", "INPUT_FILE", fallback="./data/generated.csv")
        else:
            INPUT_FILENAME = input_filename
        self.log.info(f'train data filename = {INPUT_FILENAME}')

        # чтение файла, группировка записей по user_id
        grouped = sc.textFile(INPUT_FILENAME, adapter.num_parts) \
            .map(lambda x: map(int, x.split())).groupByKey() \
            .map(lambda x : (x[0], list(x[1])))
        
        # расчёт матрицы просмотренных фильмов
        self.log.info('Calculating matrix of watched movies')
        if not self._calc_watched_matrix(grouped):
            return False
        
        # обучение TF модели, получение фичей
        self.log.info('Train TF model')
        tf_features = self._train_tf(grouped)
        if tf_features is None:
            return False
        
        # обучение IDF модели
        self.log.info('Train IDF model')
        if not self._train_idf(tf_features):
            return False

        # сохраняем пути в конфиге
        os.remove(self.config_path)
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)
        
        return True

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_models()