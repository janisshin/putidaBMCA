import argparse
import os

import runBMCA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runName", metavar="FILENAME", type=str,
                        help="output as a pickle file")
    parser.add_argument("--iter", metavar="N", type=int, default=8,
                        help="how many iterations to run")
    args = parser.parse_args()

    if args.iter:
        runBMCA.runBMCA(args.runName, args.iter)
    else:
        runBMCA.runBMCA(args.runName)
    