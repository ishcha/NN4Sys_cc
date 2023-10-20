import argparse
import Aurora_abcrown_run
import Pensieve_abcrown_run
import Bloom_filter_abcrown_run
import Lindex_abcrown_run
import Cardinality_abcrown_run
import Decima_abcrown_run


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="all",
                        choices=["pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"],
                        help="which model to verify pensieve, decima, lindex, cardinality, bloom_filter, aurora")

    args = parser.parse_args()
    if args.model == "all":
        Pensieve_abcrown_run.main()
        Decima_abcrown_run.main()
        Lindex_abcrown_run.main()
        Cardinality_abcrown_run.main()
        Bloom_filter_abcrown_run.main()
        Aurora_abcrown_run.main()
    if args.model == "pensieve":
        Pensieve_abcrown_run.main()
    if args.model == "decima":
        Decima_abcrown_run.main()
    if args.model == "lindex":
        Lindex_abcrown_run.main()
    if args.model == "cardinality":
        Cardinality_abcrown_run.main()
    if args.model == "bloom_filter":
        Bloom_filter_abcrown_run.main()
    if args.model == "aurora":
        Aurora_abcrown_run.main()


if __name__ == "__main__":
    main()
