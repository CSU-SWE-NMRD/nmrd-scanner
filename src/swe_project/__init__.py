import sys


if len(sys.argv) >= 2:
    match sys.argv[1]:
        case "noah":
            from swe_project import noah
            noah.main()
        case "mason":
            from swe_project import mason
            mason.main()
        case "rj":
            from swe_project import rj
            rj.main()
        case "darius":
            from swe_project import darius
            darius.main()

def main():
    pass
