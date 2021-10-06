from TD3 import TD3

td3res = TD3(2, lr=0.001, memorySize=2000, batchSize=64, decayRate=0.1, numOfEpisodes=50000, stepsPerEpisode=12
              , tau=0.005, policy_freq=2, updateAfter=2)
td3res.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1,
                          benchmarkFlag=True)
