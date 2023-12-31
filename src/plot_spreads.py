import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def run(results_directory, optimizer, objectivefunc, Iterations):
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + "/experiment_details.csv")
    # initialize the list of default matplotlib colours
    colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    for j in range(0, len(objectivefunc)):
        objective_name = objectivefunc[j]

        startIteration = 1
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
            plt.fill_between(allGenerations,quantiles[0,1:],quantiles[2,1:],alpha=0.2,color=colours[i%len(colours)])
            plt.plot(np.arange(2,row.shape[1]+1), quantiles[1,1:], label=optimizer_name,color=colours[i%len(colours)])

        # make the y axis log scale
        plt.yscale("log")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend(loc="upper right")
        plt.title("Convergence spread plots of optimizers for " + objective_name)
        plt.grid()
        fig_name = results_directory + "/spreads-" + objective_name + ".png"
        plt.savefig(fig_name, bbox_inches="tight")
        plt.clf()
        # plt.show()


# run("results/2023-12-31-19-02-06", ["CMAES", "SSA", "PSO", "GWO"], ["F5", "F12", "F13"], 12)