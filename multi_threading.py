

##############################################################################
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates KitNET's ability to incrementally learn, and detect anomalies.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 2.7.14   #######################



def process_data(start, end, X, RMSEs, K):
    for i in range(start, end):
        RMSEs[i] = K.process(X[i,])

if __name__ == "__main__":
    import KitNET as kit
    import numpy as np
    import pandas as pd
    import time
    import multiprocessing
    import threading
    import concurrent.futures

    print("Reading Sample dataset...")
    # X = pd.read_csv("mirai3.csv",header=None).as_matrix() #an m-by-n dataset with m observations
    X = pd.read_csv("mirai3.csv", header=None).values  # an m-by-n dataset with m observations

    # KitNET params:
    maxAE = 10  # maximum size for any autoencoder in the ensemble layer
    FMgrace = 5000  # the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = 50000 # the number of instances used to train the anomaly detector (ensemble itself)

    # Build KitNET
    K = kit.KitNET(X.shape[1], maxAE, FMgrace, ADgrace)
    RMSEs = np.zeros(X.shape[0])  # a place to save the scores

    print("Running KitNET:")
    start_time = time.time()
    # Here we process (train/execute) each individual observation.
    # In this way, X is essentially a stream, and each observation is discarded after performing process() method.
    for i in range(FMgrace+ADgrace):
        if i % 1000 == 0:
            print(i)
        RMSEs[i] = K.process(X[i,]) #will train during the grace periods, then execute on all the rest.

    # num_threads = multiprocessing.cpu_count()  # Get the number of available CPU cores
    num_threads = 10  # Get the number of available CPU cores
    chunk_size = (X.shape[0] - (FMgrace+ADgrace+1)) // num_threads  # Calculate chunk size for each process
    print(RMSEs)

    processes = []
    for i in range(num_threads):
        start = (FMgrace+ADgrace+1) + (i * chunk_size)
        end = start + chunk_size if i < num_threads - 1 else X.shape[0] # Ensure the last process gets the remaining data
        # end = start + chunk_size
        print("Chunk Size ", start, end)
        p = threading.Thread(target=process_data, args=(start, end, X, RMSEs, K))
        processes.append(p)
        p.start()

    # for p in processes:
    #     p.join()

    print(RMSEs)

    stop_time = time.time()
    print("Complete. Time elapsed: "+ str(stop_time - start_time))
    from matplotlib import pyplot as plt
    from scipy.stats import norm
    plt.figure(figsize=(10, 5))
    # Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
    # from scipy.stats import norm
    # benignSample = np.log(RMSEs[FMgrace+ADgrace+1:71000])
    # logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

    benignSample = np.log(RMSEs[FMgrace + ADgrace + 2:71000])
    logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

    # plot the RMSE anomaly scores
    print("Plotting results")

    from matplotlib import cm

    # timestamps = pd.read_csv("mirai3_ts.csv",header=None).values
    # fig = plt.scatter(timestamps[FMgrace+ADgrace+1:],RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
    fig = plt.scatter(range(FMgrace+ADgrace+1,len(RMSEs)),RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
    plt.yscale("log")
    plt.title("Anomaly Scores from KitNET's Execution Phase")
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("Time elapsed [min]")
    # plt.annotate('Mirai C&C channel opened [Telnet]', xy=(timestamps[71662],RMSEs[71662]), xytext=(timestamps[58000],1),arrowprops=dict(facecolor='black', shrink=0.05),)
    # plt.annotate('Mirai Bot Activated\nMirai scans network for vulnerable devices', xy=(timestamps[72662],1), xytext=(timestamps[55000],5),arrowprops=dict(facecolor='black', shrink=0.05),)
    figbar=plt.colorbar()
    figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
    plt.show()
