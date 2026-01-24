from src.Dataprep.dataprep import Dataprep
from config.config import CONFIG



def main():
    dataprep = Dataprep(CONFIG)
    dataprep.prepare_data()