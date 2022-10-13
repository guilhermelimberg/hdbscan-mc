# hdbscan-mc
Application of HDBSCAN combined with Monte Carlo resamples

An almost identical version of this code was originally described in [Limberg+2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...907...10L/abstract) paper in ApJ. 

The algorithm strategy goes as follows. 
1) Use HDBSCAN to produce what I refer to "nominal" clusters (from the nominal values of the quantities in your parameter space).
2) Generating Monte Carlo realizations from uncertainties.
3) Throw these disturbed data back into the nominal parameter space.
4) Count the number of instances each data point falls into a given cluster. We take this value divided by the total amount of Monte Carlo realizations as a "membership probability" of each given stars belonging to a group. 

This same code was utilized in several subsequent papers, including [Gudin+2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...908...79G/abstract) and [Shank+2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...926...26S/abstract). This strategy was copied by [Santos-Silva+2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1033S/abstract) for a different purpose. Check out their paper (:

If you need help to implement this code for your specific problem, do not hesitate to reach out at: guilherme.limberg@usp.br

Finally, you will notice that the provided code is calling generic names for the input data sets. You should replace these with your own sample(s) in order to run the code. I do not provide any data originally used for publication as I still utilize them for other purposes.

PS: recently, some authors have built on our original work to develop a more robust way of applying HDBSCAN combined with Monte Carlo to account for uncertainties and membership probabilities. Check out their efforts: [Ou+2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220801056O/abstract).
