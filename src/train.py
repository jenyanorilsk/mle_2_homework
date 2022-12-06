import os
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

    def train_models(self) -> bool:
        try:
            adapter = SparkAdapter()
            sc = adapter.get_context()
            spark = adapter.get_session()
        except:
            self.log.error(traceback.format_exc())
        
        INPUT_FILENAME = self.config.get("DATA", "INPUT_FILE", fallback="./data/generated.csv")
        self.log.info(f'train data filename = {INPUT_FILENAME}')

        FEATURES_COUNT = self.config.getint("MODEL", "FEATURES_COUNT", fallback=10000)
        self.log.info(f'TF-IDF features count = {FEATURES_COUNT}')

        # чтение файла как есть
        raw = sc.textFile(INPUT_FILENAME, adapter.num_partitions)
        
        # записи, сгруппированные по user_id
        grouped = raw.map(lambda x: map(int, x.split())).groupByKey() \
            .map(lambda x : (x[0], list(x[1])))

        # возьмём индексы просмотренных фильмов в нужном нам типе, чтобы можно было
        # умножить их на похожесть пользователя
        all_watched_matrix = CoordinateMatrix(grouped.flatMapValues(lambda x: x).map(lambda x: MatrixEntry(x[0], x[1], 1.0)))
        
        # чтобы сделать всё правильно, используем DataFrame из-за 
        # его встроенного метода columnSimilarities, позволяющего
        # считать косинусное сходство между колонкам
        df = grouped.toDF(schema=["user_id", "movie_ids"])

        # считаем TF - частоту токенов (фильмов), должна быть 1,
        #  т.к. пользователь либо посмотрел, либо не посмотрел фильм
        hashingTF = HashingTF(inputCol="movie_ids", outputCol="rawFeatures", numFeatures=FEATURES_COUNT)
        tf_features = hashingTF.transform(df)

        model_path = "./models/TF_MODEL"
        try:
            hashingTF.save(model_path)
            self.config["MODEL"]["TF_PATH"] = model_path
            self.log.info(f"TF model stored at {model_path}")
        except:
            self.log.error(traceback.format_exc())

        # считаем IDF - здесь уже будут дробные значения, т.к. учёт по пользователям, это и будут фичи
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idfModel = idf.fit(tf_features)

        model_path = "./models/IDF_MODEL"
        try:
            idfModel.save(model_path)
            self.config["MODEL"]["IDF_PATH"] = model_path
            self.log.info(f"IDF model stored at {model_path}")
        except:
            self.log.error(traceback.format_exc())


        idf_features = idfModel.transform(tf_features)

        # сохраняем изменения
        os.remove(self.config_path)
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_models()