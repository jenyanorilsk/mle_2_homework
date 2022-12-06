import os
import traceback
import configparser

from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

from logger import Logger

SHOW_LOG = True

class SparkAdapter():

    def __init__(self) -> None:
        """
        default initialization
        """

        self.config = configparser.ConfigParser()
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)
        
        self.spark_config = SparkConf()
        self.spark_config.set("spark.app.name", "homework")
        self.spark_config.set("spark.master", "local")
        self.spark_config.set("spark.executor.cores", \
            self.config.get("SPARK", "NUM_PROCESSORS", fallback="3"))
        self.spark_config.set("spark.executor.instances", \
            self.config.get("SPARK", "NUM_EXECUTORS", fallback="1"))
        self.spark_config.set("spark.executor.memory", "8g")
        self.spark_config.set("spark.locality.wait", "0")
        self.spark_config.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        self.spark_config.set("spark.kryoserializer.buffer.max", "2000")
        self.spark_config.set("spark.executor.heartbeatInterval", "6000s")
        self.spark_config.set("spark.network.timeout", "10000000s")
        self.spark_config.set("spark.shuffle.spill", "true")
        self.spark_config.set("spark.driver.memory", "8g")
        self.spark_config.set("spark.driver.maxResultSize", "8g")

        self.num_partitions = self.config.get("SPARK", "NUM_PARTITIONS", fallback=None)

        self.log.info("Spark config:")
        for conf in self.spark_config.getAll():
            self.log.info(f'{conf[0].upper()} = {conf[1]}')
        
        self.sc = None
        self.spark = None

        self.log.info("Spark adapter is ready")
        pass

    def get_context(self) -> SparkContext:
        """
        Возвращает Spark-контекст:
        Если не инициализирован, то инициализируется и сохраняется в атрибут sc
        """
        if self.sc is None:
            try:
                self.sc = SparkContext(conf=self.spark_config)
                self.log.info("SparkContext is initialized")
            except:
                self.log.error(traceback.format_exc())
        return self.sc
    
    def get_session(self) -> SparkSession:
        """
        Возвращает Spark-сессию:
        Если не инициализирована, то инициализируется и сохраняется в атрибут spark
        """
        if self.spark is None:
            try:
                self.spark = SparkSession(self.get_context())
                self.log.info("SparkSession is initialized")
            except:
                self.log.error(traceback.format_exc())
        return self.spark