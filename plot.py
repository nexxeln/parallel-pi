import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results.csv")

grouped = (
    df.groupby(["mode", "threads"])
    .agg(
        pi_estimate_mean=("pi_estimate", "mean"),
        pi_estimate_std=("pi_estimate", "std"),
        error_percent_mean=("error_percent", "mean"),
        error_percent_std=("error_percent", "std"),
        time_seconds_mean=("time_seconds", "mean"),
        time_seconds_std=("time_seconds", "std"),
        trials=("trial", "count"),
    )
    .reset_index()
)

serial_time = grouped[(grouped["mode"] == "serial")]["time_seconds_mean"].values[0]
grouped["speedup"] = serial_time / grouped["time_seconds_mean"]

parallel = grouped[grouped["mode"] == "parallel"]
serial = grouped[grouped["mode"] == "serial"]

plt.figure(figsize=(8, 5))
plt.plot(parallel["threads"], parallel["speedup"], marker="o", label="Parallel")
plt.axhline(1, color="gray", linestyle="--", label="Serial")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs. Number of Threads")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.errorbar(
    parallel["threads"],
    parallel["time_seconds_mean"],
    yerr=parallel["time_seconds_std"],
    marker="o",
    label="Parallel",
)
plt.axhline(
    serial["time_seconds_mean"].values[0], color="gray", linestyle="--", label="Serial"
)
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs. Number of Threads")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.errorbar(
    parallel["threads"],
    parallel["pi_estimate_mean"],
    yerr=parallel["pi_estimate_std"],
    marker="o",
    label="Parallel",
)
plt.axhline(np.pi, color="red", linestyle="--", label="True π")
plt.xlabel("Number of Threads")
plt.ylabel("π Estimate")
plt.title("π Estimate vs. Number of Threads")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.errorbar(
    parallel["threads"],
    parallel["error_percent_mean"],
    yerr=parallel["error_percent_std"],
    marker="o",
    label="Parallel",
)
plt.xlabel("Number of Threads")
plt.ylabel("Error (%)")
plt.title("Error (%) vs. Number of Threads")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
