

import numpy as np


def Comm(input,metric='(Ct)Gbps'):
    time_eISLlength = input


    # compute
    fGHz = 12  # Ghz
    log_2_10 = 3.321928094887362
    lg12 = 1.0791812
    EIRP = 31  # dBW
    GT = 20  # dBi/K
    Gt = 40  # dB
    Gr = 40  # dB
    k = -228.6  # dBW/K
    L = 30  # dB
    B=1#GHz
    Bn = 10*np.log10(1000000000)  # GHz
    Tn = np.log10(324.81) # K~ 50oC
            # d = df['Range (km)'].max() - df['Range (km)']
    d = time_eISLlength

    Lf = 92.45 + 20 * np.log10(d/1000) + 20 * np.log10(fGHz)


    # Cno = EIRP + Gt  - Lf - k -Tn
    Cno = EIRP + GT - Lf - k


    # CN = EIRP + Gt  - Lf - k -Tn - Bn
    CN = Cno - Bn

    # Ct = B * log_2_10 * (EIRP + Gt - Lf - k - Tn - Bn) / 10
    Ct =  B * log_2_10 * CN/10


    if metric == 'Cno(dBHz)':
        return Cno
    if metric =='CN(dB)':
        return CN
    if metric == 'CT(dBW/K)':
        CT = EIRP + GT - Lf

    if metric == 'Pr(dBW)':
        Pr = EIRP + Gr - Lf + 100
    if metric == '(Ct)Gbps':
        return Ct

def cdf(arr,bins=10,NUM=False):
    bins,ys = np.histogram(arr,bins=bins)[1][:-1], np.histogram(arr,bins=bins)[0]
    ys2=[]
    for i in range(len(ys)):
        ys2.append( ys[:i+1].sum())
    ys2 = np.array(ys2)
    if NUM ==False:
        ys2=100*ys2/ys2[-1]

    return bins,ys2
def pdf(arr,bins=10,NUM=False):
    arr = np.array(arr)
    min_value, max_value = arr.min(),arr.max()
    intervals = np.linspace(start=0, stop=max_value+20, num=bins)
    count=[]
    for pre,post in zip(intervals[:-1],intervals[1:]):
        count.append(sum((arr<post)*(pre<arr)))
    count = np.array(count)
    count = np.expand_dims(count,axis=1)
    return intervals[:-1],count/sum(count)*100
