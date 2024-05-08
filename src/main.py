import argparse
import os

import runBMCA
from datetime import datetime
startTime = datetime.now()

OUTPUT_FOLDER = 'output/'
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
    parser.add_argument("--iter", metavar="N", type=int, default=50000,
                        help="how many iterations to run")
    parser.add_argument("--chunk", metavar="ch", type=int, default=1,
                        help="how many iterations to run")
    args = parser.parse_args()
    
    if not args.runName:
        args.runName='test'
    if args.chunk != 1:
        firstPickle = runBMCA.runBMCA(args.runName, args.iter/args.chunk)
        resumePickle = firstPickle
        for i in range(args.chunk-1):
            resumePickle = runBMCA.resumeBMCA(resumePickle, i)
    else:
        runBMCA.runBMCA(args.runName)

with open('elapsed_time.txt', 'w') as f:
    f.write(str(datetime.now() - startTime))