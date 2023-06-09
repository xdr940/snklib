import numpy as np
import math
import datetime
import pymap3d as pm
import json

def getCoord(czml_dict):
    frame = {}
    for k,v in czml_dict.items():
        positions = v['position']['cartesian']
        if len(positions)%4 !=0:
            print('error in sat {}'.format(k))
            continue
        coor_len = int(len(positions)/4)


        positions= np.array(positions).reshape([coor_len,4])
        frame[k] = positions
    return frame

def eci2ecef_np(time_postions):
    start = datetime.datetime(2000,1,1)

    ecef_coord = []
    datetime_arr=[]
    for delta in time_postions[:,0]:
        datetime_arr.append(start+datetime.timedelta(seconds=delta))
    datetime_arr = np.array(datetime_arr)
    # for line in time_postions:
    x,y,z =  pm.eci2ecef(time_postions[:,1],time_postions[:,2],time_postions[:,3],datetime_arr)
    time_postions[:, 1]=x
    time_postions[:, 2]=y
    time_postions[:, 3]=z


    # print(ecef_coord)
    return time_postions

def eci2geo_np(time,positions):
    pass

def tle2description(tle):
    temp = "<!--HTML--> <ul> <li>{}</li> <li>{}</li> </ul>".format(tle[0][1:],tle[1][1:])
    return temp
def cartesian3(longitude,latitude,h=0):
    longitude = math.radians(longitude)
    latitude = math.radians(latitude)


    R = 6378137.0 +h  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)
    return  [X,Y,Z]

def lalong_str2num(lalong_str):
    la_str, long_str = lalong_str.split(',')
    longitude=None
    latitude=None
    if long_str[-1] == 'E':
        longitude = float(long_str[:-1])
    elif long_str[-1] == 'W':
        longitude = -float(long_str[:-1])

    if la_str[-1] == 'N':
        latitude = float(la_str[:-1])
    elif la_str[-1] == 'S':
        latitude = -float(la_str[:-1])

    return latitude,longitude

def mid_longlat(oneLon, oneLat, twoLon, twoLat):
    # oneLon：第一个点的经度；oneLat：第一个点的纬度；twoLon：第二个点的经度；twoLat：第二个点的纬度；
    bLon = float(oneLon) - float(twoLon)
    bLat = float(oneLat) - float(twoLat)
    # //Math.abs()绝对值
    if bLon > 0:
        aLon = float(oneLon) - abs(bLon) / 2
    else:
        aLon = float(twoLon) - abs(bLon) / 2

    if bLat > 0:
        aLat = float(oneLat) - abs(bLat) / 2
    else:
        aLat = float(twoLat) - abs(bLat) / 2

    return aLon, aLat

def association_stamps2json(entity_type,access_stamp):
    ret_ass={}
    for k,v in access_stamp.items():
        # access_stamp[k] =
        newv = []
        for i in range(len(v)):
            newv.append( [v[i][0],v[i][1]])
        ret_ass['{}-{}-{}'.format(entity_type,Id2numId(k[0]),Id2numId(k[1]))] =  newv
    return ret_ass



def time_format(time_str):
    if len(time_str) == 20:
        return datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
    elif len(time_str )> 20:
        return datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%fZ')



#========


def homogen2D(arrs,pad_value=np.nan):
    length = max(map(len, arrs))
    y = np.array([xi + [pad_value] * (length - len(xi)) for xi in arrs])
    return y

def homogen3D(arrs,pad_value=np.nan):
    shapes=[]
    for arr in arrs:

        shapes.append(np.shape(arr))

    shapes = np.array(shapes)
    nums,peers = np.max(shapes, axis=0)
    arrs_pad = []
    for arr in arrs:
        arr_pad = np.pad(arr, ((0, nums-arr.shape[0]), (0, peers-arr.shape[1])), mode='constant', constant_values=((0, pad_value), (0, pad_value)))
        arrs_pad.append(np.expand_dims(arr_pad,axis=0))
    arrs_pad = np.concatenate(arrs_pad,axis=0)
    return arrs_pad


def Id2numId(entity_id):
    assert type(entity_id) == str
    return entity_id.split("-")[1]

def numId2Id(entity_type,num_id):
    assert entity_type in ['SAT','GS','MS','ISL']
    assert type(num_id) == str
    return entity_type+'-'+num_id


def nan2zero(arr):
    nan_mask = (1 - (arr > 0)).astype(np.bool8)

    arr[nan_mask] = 0
    return arr

def zero2nan(arr):
    zero_mask = (arr == 0).astype(np.bool8)

    arr[zero_mask] = np.nan

    return arr
def zero2Inf(arr):

    zero_mask = (arr == 0).astype(np.bool8)

    arr[zero_mask] = np.Inf

    return arr


# ============== FWDS =========




def sats2fwds(sats_paths,adj_tab=None,tem_adj_tab=None):
    '''
    using:
        sats2Fwds(sats)
        satsFwds(sats,adj_tab,tem_adj_tab)
    :param sats:
    :param adj_tab:
    :param tem_adj_tab:
    :return:
    '''
    add_fwds =set()

    if adj_tab== None:
        for sats in sats_paths:
            i = 0
            j = 1
            while j < len(sats):
                add_fwds.add("FWD-{}-{}".format(Id2numId(sats[i]), Id2numId(sats[j])))
                j += 1
                i += 1
        return add_fwds
    else:
        for sats in sats_paths:
            i = 0
            j = 1
            while j < len(sats):
                if sats[i] in adj_tab.keys() and sats[j] in adj_tab[sats[i]]:
                    add_fwds.add("FWD-{}-{}".format(Id2numId(sats[i]), Id2numId(sats[j])))

                if tem_adj_tab and sats[i] in tem_adj_tab.keys() and sats[j] in tem_adj_tab[sats[i]]:
                    add_fwds.add("tFWD-{}-{}".format(Id2numId(sats[i]), Id2numId(sats[j])))

                j += 1
                i += 1
        return add_fwds


