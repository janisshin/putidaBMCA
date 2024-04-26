import argparse
import os

# should we use cloudpickle instead?
# import pickle5 as pickle 

import src.putidabmca.runBMCA as runBMCA
#  import analysis

OUTPUT_FOLDER = 'output'
ANALYSIS_FOLDER = OUTPUT_FOLDER + 'analysis/'

try:
    os.makedirs(OUTPUT_FOLDER)
    print('folder was made')
except FileExistsError:
    print('file exists')
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runName", metavar="FILENAME", type=str,
                        help="output as a pickle file")
    parser.add_argument("--iter", metavar="N", type=int, default=8,
                        help="how many iterations to run")
    args = parser.parse_args()
    
    if not args.runName:
        args.runName='test'
    if args.iter:
        runBMCA.runBMCA(args.runName, args.iter)
    else:
        runBMCA.runBMCA(args.runName)
