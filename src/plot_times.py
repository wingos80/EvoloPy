import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run(results_directory, optimizer, objectivefunc, Iterations):
    plt.ioff()

    fileResultsDetailsData = pd.read_csv(results_directory + "/experiment_details.csv")
    for j in range(0, len(objectivefunc)):
        plt.figure()

        # Box Plot
        data = []

        for i in range(len(optimizer)):
            objective_name = objectivefunc[j]
            optimizer_name = optimizer[i]

            detailedData = fileResultsDetailsData[
                (fileResultsDetailsData["Optimizer"] == optimizer_name)
                & (fileResultsDetailsData["objfname"] == objective_name)
            ]
            detailedData = detailedData["ExecutionTime"]
            detailedData = np.array(detailedData).T.tolist()
            data.append(detailedData)

        # , notch=True
        box = plt.boxplot(data, patch_artist=True, labels=optimizer)

        colors = [
            "#5c9eb7",
            "#f77199",
            "#cf81d2",
            "#4a5e6a",
            "#f45b18",
            "#ffbd35",
            "#6ba5a1",
            "#fcd1a1",
            "#c3ffc1",
            "#68549d",
            "#1c8c44",
            "#a44c40",
            "#404636",
        ]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        plt.legend(
            handles=box["boxes"],
            labels=optimizer,
            loc="upper right",
            bbox_to_anchor=(1.2, 1.02),
        )
        # make the y axis log scale
        plt.yscale("log")
        plt.grid()
        # Make a y axis label
        plt.ylabel("Run Time [s]")
        # make a title 
        plt.title("Run time of optimizers for " + objective_name)
        fig_name = results_directory + "/timeplot-" + objective_name + ".png"
        plt.savefig(fig_name, bbox_inches="tight")
        plt.clf()
    # plt.show()

# run("2023-12-31-15-36-22/", ["CMAES", "SSA", "PSO", "GWO"], ["F5", "F12","F14"], 200)