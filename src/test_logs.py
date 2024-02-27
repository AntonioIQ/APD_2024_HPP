import unittest
import os
import logging
from logs import configure_logger

class TestLogs(unittest.TestCase):
    def test_configure_logger(self):
        # Configurar el logger
        logger = configure_logger('test_script')

        # Verificar que el logger es una instancia de la clase Logger
        self.assertIsInstance(logger, logging.Logger)

        # Verificar que el nivel del logger es DEBUG
        self.assertEqual(logger.level, logging.DEBUG)

        # Verificar que se ha creado el archivo de log
        log_file = logger.handlers[0].baseFilename
        self.assertTrue(os.path.isfile(log_file))

        # Verificar que el archivo de log tiene el nombre correcto
        self.assertTrue('test_script.log' in log_file)

if __name__ == '__main__':
    unittest.main()
