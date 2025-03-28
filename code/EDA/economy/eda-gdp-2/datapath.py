import os

root = os.path.join(os.getcwd(), '..','datasets')
# print(root)
rgdpPath = os.path.join(root, 'Regional gross domestic product(all ITL).xlsx')
trafficPath = os.path.join(root, 'dft_traffic_counts_raw_counts.csv')
trafficPathCleaned = os.path.join(root, 'traffic_data_cleaned.csv')
ladPath = os.path.join(root, 'Local_Authority_District_(April_2023)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_(January_2021)Lookup.csv')
# print(ladPath)

def getRgdp():
    return rgdpPath

def getTraffic():
    return trafficPath

def getTrafficCleaned():
    return trafficPathCleaned

def getLad():
    return ladPath
