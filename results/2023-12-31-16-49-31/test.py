import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# # print all files in cwd
# print(os.listdir("."))

# objectivefunc = ["F5"]
# optimizer = ["CMAES", "SSA", "PSO", "GWO"]
# Iterations = 120
# j = 0
# fileResultsData = pd.read_csv("results//2023-12-31-16-49-31/experiment_details.csv")


# objective_name = objectivefunc[j]

# startIteration = 0
# if "SSA" in optimizer:
#     startIteration = 1
#     allGenerations = [x + 1 for x in range(startIteration, Iterations)]
# for i in range(len(optimizer)):
#     optimizer_name = optimizer[i]

#     row = fileResultsData[
#         (fileResultsData["Optimizer"] == optimizer_name)
#         & (fileResultsData["objfname"] == objective_name)
#     ]
#     row = row.iloc[:, 3 + startIteration :]
#     # calculate the mean and 95% confidence interval of row
#     quantiles=np.array(row.quantile([0.1,0.5,0.9]))
#     plt.figure()
#     plt.fill_between(np.arange(1,row.shape[1]+1),quantiles[0],quantiles[2],alpha=0.2)
#     plt.yscale("log")
#     plt.xlabel("Iterations")
#     plt.ylabel("Fitness")
#     plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
#     plt.title("Convergence plots of optimizers for " + objective_name)
#     plt.grid()
#     fig_name = results_directory + "/convergence-" + objective_name + ".png"
#     plt.savefig(fig_name, bbox_inches="tight")
#     plt.show()


def run(results_directory, optimizer, objectivefunc, Iterations):
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + "/experiment.csv")
    # initialize the list of default matplotlib colours
    colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    for j in range(0, len(objectivefunc)):
        plt.figure()
        objective_name = objectivefunc[j]

        startIteration = 0
        if "SSA" in optimizer:
            startIteration = 1
        allGenerations = [x + 1 for x in range(startIteration, Iterations)]
        for i in range(len(optimizer)):
            optimizer_name = optimizer[i]

            row = fileResultsData[
                (fileResultsData["Optimizer"] == optimizer_name)
                & (fileResultsData["objfname"] == objective_name)
            ]
            row = row.iloc[:, 3 + startIteration :]
                    
            quantiles=np.array(row.quantile([0.1,0.5,0.9]))
            plt.fill_between(np.arange(1,row.shape[1]+1),quantiles[0],quantiles[2],alpha=0.2,color=colours[i%len(colours)])
            plt.plot(np.arange(1,row.shape[1]+1), quantiles[1], label=optimizer_name,color=colours[i%len(colours)])

        # make the y axis log scale
        plt.yscale("log")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
        plt.title("Convergence plots of optimizers for " + objective_name)
        plt.grid()
        fig_name = results_directory + "/convergence-" + objective_name + ".png"
        # plt.savefig(fig_name, bbox_inches="tight")
        plt.clf()
        plt.close()
        plt.show()
