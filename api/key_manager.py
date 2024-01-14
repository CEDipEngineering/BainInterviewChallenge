from __future__ import annotations
import uuid # Will be used to generate the API Keys
import logging
from pathlib import Path
from datetime import date
from datetime import datetime
import pandas as pd

# Create a log path, ensure it's available to write on
log_path = Path(f"logs/apikey_log_{str(date.today())}.txt").resolve()
log_path.parent.mkdir(mode=0o754, parents=True, exist_ok=True)
log_path.touch(mode=0o754, exist_ok=True)

# New log file every day, very basic
# WON'T generate new file for next day, unless the server restarts
# Eventually logging should be done in a more cloud-friendly way, sending the log message to a trusted separate server for safe-keeping.
logging.basicConfig(
    format="%(levelname)s:\t[%(asctime)s] %(message)s",
    filename=log_path, 
    level=logging.DEBUG
)

# Create a file path to save valid api_keys 
key_file_storage_path = Path(__file__).parent.resolve() / "secrets" / "key_file.csv"
key_file_storage_path.parent.mkdir(mode=0o754, parents=True, exist_ok=True)
key_file_storage_path.touch(mode=0o754, exist_ok=True)

class KeyManager():
    def __init__(self) -> None:
        self.keys = None
        pass
    
    def validateKey(self, apiKey: str) -> bool:
        """
        Returns whether or not a key is valid.
        To validate the key, two checks must be made.
        1. The key must exist in our key storage vault (.csv file currently)
        2. The key must be within its lifespan (every key is created with a lifespan)
        """
        logging.info(f"Attempting validation of key {apiKey}")
        if self.keys is None:
            try:
                self.keys = pd.read_csv(key_file_storage_path, index_col=0)
            except Exception as e:
                logging.error(f"Error loading api key file: {e}")
                return False
        known_key = apiKey in self.keys["API_Key"].to_list()
        if not known_key: 
            logging.info(f"Validation for key {apiKey} failed, key is not known")
            return False
        keyInfo = self.keys[self.keys["API_Key"] == apiKey]
        current_timestamp = datetime.now()
        key_timestamp = datetime.strptime(keyInfo["creationTimestamp"].to_list()[0], "%Y-%m-%d %H:%M:%S.%f")
        elapsed_time = current_timestamp - key_timestamp
        elapsed_hours = divmod(elapsed_time.total_seconds(), 86400)[0]
        if elapsed_hours < keyInfo["keyLifespan"].to_list()[0]: 
            logging.info(f"Validation successful for key {apiKey}, remaining lifespan {keyInfo['keyLifespan'].to_list()[0] - elapsed_hours}h")
            return True
        logging.info(f"Validation failed for key {apiKey}, key lifespan has expired, please generate a new key or contact system administrator")
        return False

    def generateNewKey(self, keyLifespan: float | None) -> str:
        """
        Generates a new key, potentially with a specified lifespan in hours (defaults to 24h)
        To generate the key, we simply get a new random uuid.
        We then save the key to our storage medium of choice (.csv) and keep track of when it was created and its lifespan
        """
        if keyLifespan is None: keyLifespan = 24 # Handles direct None inputs
        new_key = uuid.uuid4() # New random uuid, will be the new key
        curr_timestamp = datetime.now() # get current timestamp
        new_line = {
            "API_Key": new_key,
            "creationTimestamp": curr_timestamp,
            "keyLifespan": keyLifespan,
        }
        if self.keys is None:
            self.keys = pd.DataFrame(new_line, index=[0])
        else:
            self.keys.append(new_line, ignore_index=True)
        self.keys.to_csv(key_file_storage_path)
        logging.info(f"Generated new key {new_key} with lifespan of {keyLifespan}h")
        return new_key
    
if __name__ == "__main__":
    km = KeyManager()
    # key = km.generateNewKey(keyLifespan=24)
    # print(key)
    print("Validating wrong key: ", km.validateKey("Totally a valid key"))
    print("Validating good key: ", km.validateKey("b4beedad-7aa5-4605-8ca1-e3a96807a978"))