import sys
from swe_project import ui, classification

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'util':
        classification.main()
    else:
        ui.main()

