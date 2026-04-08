import sys

from src.app import LigandMinerApplication, progress


def main(uniprot_id):
    LigandMinerApplication().main(uniprot_id, progress)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        uniprot = sys.argv[1]
        main(uniprot)
    else:
        print("Usage: python main.py uniprot_id")

