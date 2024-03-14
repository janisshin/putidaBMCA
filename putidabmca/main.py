import argparse
import os
import pickle5 as pickle

import runBMCA
import analysis

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
    
    with open('output/' + args.runName + '.pgz', "rb") as f:
        pickleJar = pickle.load(f)

    # plot the ELBO convergence
    analysis.plot_ELBO_convergence(pickleJar, args.runName, args.iter)        
    # save csv of sampled elasticity values
    analysis.save_sampled_elasticities(pickleJar, args.runName)
    # calculate the median FCC values
    

    