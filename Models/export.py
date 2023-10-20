import argparse

import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="all",
                        choices=["pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"],
                        help="which model to verify pensieve, decima, lindex, cardinality, bloom_filter, aurora")

    args = parser.parse_args()
    if args.model == "all":
        os.system("cd Pensieve")
        os.system("python export.py")
        os.system("cd ..")

        os.system("cd Aurora")
        os.system("python export.py")
        os.system("cd ..")

        os.system("cd Decima")
        os.system("python export.py")
        os.system("cd ..")

        os.system("cd Learned_index")
        os.system("python export.py")
        os.system("cd ..")

        os.system("cd Cardinality")
        os.system("python export.py")
        os.system("cd ..")

        os.system("cd Bloom_filter")
        os.system("python export.py")
        os.system("cd ..")

    if args.model == "pensieve":
        os.system("cd Pensieve && python export.py && cd ..")

    if args.model == "decima":
        os.system("cd Decima")
        os.system("python export.py")
        os.system("cd ..")
    if args.model == "lindex":
        os.system("cd Learned_index")
        os.system("python export.py")
        os.system("cd ..")
    if args.model == "cardinality":
        os.system("cd Cardinality")
        os.system("python export.py")
        os.system("cd ..")
    if args.model == "bloom_filter":
        os.system("cd Bloom_filter")
        os.system("python export.py")
        os.system("cd ..")
    if args.model == "aurora":
        os.system("cd Aurora")
        os.system("python export.py")
        os.system("cd ..")


if __name__ == "__main__":
    main()
