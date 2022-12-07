
import os
import shutil
import traceback
import configparser

import requests
import re

from logger import Logger

SHOW_LOG = True

class Preprocessor():
    
    def __init__(self):

        self.config = configparser.ConfigParser()
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)

        self.log.info(f'Current working directory: {os.getcwd()}')

        pass

    def _download(self) -> bool:
        url = self.config.get("DATA", "SOURCE_URL")
        if url is None or url == '':
            self.log.error('File source url is not specified')
            raise Exception('File source url is not specified')
        
        self.log.info(f'Downloading from {url}')
        response = requests.get(url)
        self.log.info(f'Response status: {response.status_code}')
        
        if response.status_code != 200:
            return False
        
        if "Content-Disposition" in response.headers.keys():
            filename = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
            if path.endswith(filename):
                open(path, 'wb').write(response.content)
                self.log.info(f'Saved to {path}')
            else:
                save_path = os.path.join(os.getcwd(), 'data', filename)
                self.log.info(f'Save to {saved_path}')
        else:
            open(path, 'wb').write(response.content)
            self.log.info(f'Saved to {path}')
        pass

    def get_data(self) -> bool:

        path = self.config.get("DATA", "INPUT_FILE")
        if path is None or path == '':
            self.log.error('Input file is not specified')
            raise Exception('Input file is not specified')
        self.log.info(f'Data file is {path}')

        if not os.path.exists(path):
            self.log.info(f'Data file doesn\'t exist')
            return self._download()
        
        self.log.info('Everything is ok')
        return True

        pass

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.get_data()
