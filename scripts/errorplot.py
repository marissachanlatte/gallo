import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

problem_name = "KAIST-MOX"
experiment_name = "kaist"

# Load Files
nda = np.loadtxt("../experiments/" + problem_name + "/nda_"
                 + experiment_name + ".out")
tgnda = np.loadtxt("../experiments/" + problem_name + "/tgnda_"
                 + experiment_name + ".out")
num_groups = len(nda)

# Calculate Difference
rel = []
for g in range(num_groups):
    rel.append(np.abs(nda[g] - tgnda[g])/np.max(nda[g]))

# Make Data Frame
dfs = []
for g in range(7):
    tmp = pd.DataFrame(rel[g], columns=["rel_err"])
    tmp["group"]=g+1
    dfs.append(tmp)
df = pd.concat(dfs)

# Plot
g = sns.boxplot(data=df, x="group", y="rel_err")
plt.ylabel("Relative Error")
plt.xlabel("Group Number")
plt.ticklabel_format(style='sci', axis='y', useMathText=True, scilimits=(0,1))
plt.title("Relative Difference in Flux NDA vs. TG-NDA")
g.set_yscale('log')

# Save Figure
fig = g.get_figure()
fig.savefig("relative_error_" + experiment_name + ".png")
