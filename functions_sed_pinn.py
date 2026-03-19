# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:52:18 2025

@author: haddadchia
"""


#%%
def ded (c,di, c_alpha):
    # function to calculate deposition rate [kg/m^2/s]
    # Input: 
    # sediment concentration - c [kg/m3] as array or dataframe
    # sediment size - di [m] as ingle value
    # average to base concentration - c_alpha [-] (put as 1 if no data)
    # output:
    # deposition rate - ded [kg/m^2/s]
    
    
    # calculate fall velocity [m/s]
    Rrho=1.65       # R=((rohs/rho)-1),rhos=2650, rho=1000 - no unit
    g=9.806         # acceleration gravity [m/s^2]
    nu=0.0000010533 # kinematic viscosity [m^2/s]
    distar=(((Rrho*g)/(nu**2))**(1/3))*(di)
    falveli= (nu/di)*(((25+(1.2*(distar**2)))**(0.5))-5)**(1.5)     # fall velocity in [m/s]
    
    # calculate deposition rate 
    ded_out=falveli*c_alpha*c
    return ded_out

#%%

def read_h_q (firstdate, lastdate,main_dir, flow_dir_name, di):
    
    import pandas as pd
    import numpy as np
    import os
    import scipy
    import pylab
    import matplotlib.pyplot as plt
    from datetime import datetime
    #%% reading hydroDF file which has all nzsegv2, HYDSEQ, FROM_NODE, TO_NODE
    
    routing_file='Manawatu_Routing_calib-4points.csv'
    hydroDF=pd.read_csv(os.path.join(main_dir,routing_file)
                        ).sort_values(by = ["HYDSEQ"]) # read routing file  
    US_reach=hydroDF["nzsegv2"].values 
    US_reach=hydroDF["nzsegv2"].values 
    hydroDF = hydroDF.set_index("nzsegv2") # US_reach is now index in dataframe
    fromArray = hydroDF["FROM_NODE"].values      
    toArray = hydroDF["TO_NODE"].values
    nodeArray = np.append(toArray, fromArray) #join arrays to make single array with all nodes
    uniqueNodeArray = np.unique(nodeArray) #remove duplicates since nodes are found in both arrays
    uniqueNodeList =uniqueNodeArray.tolist() #make list of keys 
    entryArray = np.zeros(len(uniqueNodeList)).astype(float) #make array of initial entries, i.e, zero
    nodeDict = dict(zip(uniqueNodeList, entryArray.T)) #zip the keys and entries into a dictionary
    fromArrayList=fromArray.tolist()    # list for fromArray (which is FROM_NODE)
    uniquetoArray=np.unique(toArray)
    toArrayList=uniquetoArray.tolist()
    
    #%% import median riverbed diameter and critical stream power coefficients as dataframe
    # read median sediment, critical stream power, and dispersion factor
    
    median_sed_file='median-diam_critical_dispers_depos-depth_calib-4points_No4.csv'    #^^^^^^changed for each flood No^^^^^^
    DmDF=pd.read_csv(os.path.join(main_dir,median_sed_file), header=0, sep=',').sort_values(by = ["HYDSEQ"]) 
    # Dictionary for critical coefficients and max deposition depth
    # Dictionary xpoints
    xpoints_dict={} # 
    for n in fromArrayList:
        reach_interest=hydroDF.index[hydroDF['FROM_NODE']==n]
        xpoints_dict[n]=DmDF.loc[DmDF['nzsegv2']==reach_interest[0], 'xpoints'].iloc[0]
    
    #%% import reach characteristics data (slope and distance from do)
    slope_dict={}
    reach_distance_dict={}
    reachID_dict={}
    Dm_dict={}
    Dispersioncoeff_dict={}
    
    reach_charac_file='reach_characteristics-calibrate-4points.csv'
    reach_charac=pd.read_csv(os.path.join(main_dir,reach_charac_file), header=0, sep=',')         # reach characteristics
    for n in uniqueNodeList:
        res_temp=fromArrayList.count(n)
        if res_temp>0:
            reach_interest=hydroDF.index[hydroDF['FROM_NODE']==n]
            slope=reach_charac.loc[reach_charac['nzsegv2']==reach_interest[0],
                                   'rch_slope_grad'].iloc[0]
            slope_dict[n]=slope # River bed slope [m/m]
            reach_distance=reach_charac.loc[reach_charac['nzsegv2']==reach_interest[0],
                                            'rch_length_m'].iloc[0]
            Dm_dict[n]=DmDF.loc[DmDF['nzsegv2']==reach_interest[0], 'median_diameter'].iloc[0]  # dictionary for the median size of sediment [m] for each reach
            Dispersioncoeff_dict[n]=DmDF.loc[DmDF['nzsegv2']==reach_interest[0], 'Dispersion'].iloc[0]  # dictionary for the dispersion coefficient for each reach - Dispersion has a multiplier of 250
            reach_distance_dict[n]=reach_distance
            reachID_dict[n]=reach_interest[0]
            del slope, reach_distance        
        else: 
            reach_interest=hydroDF.index[hydroDF['TO_NODE']==n]
            print ('This is the sink reach', reach_interest[0])
    #%% read flow data from TopNet
    
    
    firststpart_file='Manawatu_TeacherCollege-STEC-Hrly_output-nzsegment_'  # first part of the flow csv files (common on all files)
    lastpart_file='_20170101-20191231_P1..csv.gz'   # last part of the flow csv files (common on all files)
    # Insert dates for the start and end of flood
    # Hours in-between firstdate and lastdate hours will be included 
    #(i.e. firstdate='2018-08-17 12:00:00' will start '2018-08-17 13:00:00'
    # i.e. lastdate='2018-08-19 5:00:00' will finish '2018-08-19 4:00:00' )
    # Flood No 6
    # firstdate='2018-08-17 12:00:00'     # first date to include the data, format= 'yyyy-mm-dd HH:MM:SS'
    # lastdate='2018-08-19 5:00:00'      # last date to include the data, format= 'yyyy-mm-dd HH:MM:SS'
    
    
    difference_dates=datetime.strptime(lastdate, '%Y-%m-%d %H:%M:%S')-datetime.strptime(firstdate, '%Y-%m-%d %H:%M:%S')
    hours=int(difference_dates.total_seconds() / 3600)
    
    #%% read flow data and make timeseries
    datearray=pd.DataFrame()
    unitflowarray=np.zeros((len(hydroDF),(hours-1)))
    Qarray=np.zeros((len(hydroDF),(hours-1)))
    deptharray=np.zeros((len(hydroDF),(hours-1)))
    # a loop to read flow data and depth data from list of csv files
    for m in range(len(hydroDF)):    
        print (US_reach[m])       
        interestreach_file=firststpart_file+str(US_reach[m])+lastpart_file        
        #rows_to_ignore=list(range (1,firstrow_to_include))+list(range ((lastrow_to_include+1),totalrow_flowdata))
        #print (rows_to_ignore)
        # flowdata=pd.read_csv(os.path.join(flow_dir_name,interestreach_file), 
        #                      compression='gzip',skiprows=lambda x: x in rows_to_ignore, 
        #                      header=0, sep=',', quotechar='"')         # nrows=100, , error_bad_lines=False
        
        #++++++++++++++++++++++++++++++
        # Alternative flow extraction using dates
        flowdata1=pd.read_csv(os.path.join(flow_dir_name,interestreach_file), 
                             compression='gzip', header=0, sep=',',
                             parse_dates=["datetime"], quotechar='"')         # nrows=100, , error_bad_lines=False
        #flowdata=pd.read_csv(os.path.join(flow_dir_name,interestreach_file), compression='gzip',skiprows=lambda x: x in rows_to_ignore, header=0, sep=',',parse_dates=["datetime"], quotechar='"')# flow selection based on row number (not first and last date)
        mask=(flowdata1['datetime']>firstdate)& (flowdata1['datetime']<lastdate)
        flowdata=flowdata1.loc[mask]
        #+++++++++++++++++++++++++++++++
        
        datetime=flowdata['datetime'].values
        temp_datetime=pd.DataFrame({'datetime': datetime})
        unitflow=flowdata['unit_flow'].values
        depth=flowdata['water_level'].values
        Q=flowdata['mod_streamq'].values
        datearray=pd.concat([datearray,temp_datetime], axis=1)
        unitflowarray[m,:]=unitflow
        deptharray[m,:]=depth
        Qarray[m,:]=Q
    datetimearray=np.array(datearray, dtype='datetime64[s]')
    #%% make unitflow, flow and depth dictionary including fromArray (FROM_NODE) as index and arrays of input flow for each reach
    listunitflow=unitflowarray.tolist()
    unitflowdict=dict(zip(fromArrayList,listunitflow)) # dictionary of fromArray(index) and flow data
    listdepth=deptharray.tolist()
    depthdict=dict(zip(fromArrayList,listdepth)) # dictionary of fromArray(index) and depth data
    listQ=Qarray.tolist()
    Qdict=dict(zip(fromArrayList,listQ)) # dictionary of fromArray(index) and depth data
    #%--------------------Calculations-----------------------
    #%% Interpolation of unitflow (qinterp_dict), flow (Qinterp_dict), depth (depthinterp_dict)
    #xpoints=10       # points in-between upstream and downstream data, all rows would be: xpoints+2
    qinterp_dict={}     # empty disctionary for interpolated unitflow
    depthinterp_dict={} # empty dictionary for interpolated depth
    Qinterp_dict={} # empty dictionary for interpolated flow
    segment=0
    for n in fromArrayList:
        if n!=fromArrayList[-1]:
            print(n)
            #print(segment)
            q_US=np.array(unitflowdict[n])
            q_DS=np.array(unitflowdict[toArray[segment]])
            depth_US=np.array(depthdict[n])
            depth_DS=np.array(depthdict[toArray[segment]])
            Q_US=np.array(Qdict[n])
            Q_DS=np.array(Qdict[toArray[segment]])
            qinterp=np.zeros((len(q_US),xpoints_dict[n]+2))
            qinterp_struct=scipy.interpolate.interp1d([1,xpoints_dict[n]+2], np.vstack([q_US,q_DS]),kind='linear',axis=0)
            depthinterp=np.zeros((len(depth_US),xpoints_dict[n]+2))
            depthinterp_struct=scipy.interpolate.interp1d([1,xpoints_dict[n]+2], np.vstack([depth_US,depth_DS]),kind='linear',axis=0)
            Qinterp=np.zeros((len(Q_US),xpoints_dict[n]+2))
            Qinterp_struct=scipy.interpolate.interp1d([1,xpoints_dict[n]+2], np.vstack([Q_US,Q_DS]),kind='linear',axis=0)
            for i in range(xpoints_dict[n]+2):
                #depthinterp[:,i]=depthinterp_struct(i+1)
                qinterp[:,i]=qinterp_struct(i+1)
                depthinterp[:,i]=depthinterp_struct(i+1)
                Qinterp[:,i]=Qinterp_struct(i+1)
            qinterpt=np.transpose(qinterp)
            qinterp_dict[n]=qinterpt  # filled dictionary for interpolated unitflow
            depthinterpt=np.transpose(depthinterp)
            depthinterp_dict[n]=depthinterpt # filled dictionary for interpolated depth
            Qinterpt=np.transpose(Qinterp)
            Qinterp_dict[n]=Qinterpt # filled dictionary for interpolated flow
            segment=segment+1
        else:
            print ('Sink!',n)
    interp_out=pd.DataFrame({'depth_interp':depthinterp_dict,'q_interp':qinterp_dict}) # output of interpolations as data frame
    #%%
    SF=0.1  # sand fraction in proportion (0 - 1)
    rho=1000        # Density of water [kg/m^3]
    g=9.806         # acceleration gravity [m/s^2]
    Rrho=1.65       # R=((rohs/rho)-1),rhos=2650, rho=1000 - no uni
    mcoef=1             # drag/roughness coefficients (1-10) [-]
    kv=0.4          # von-Karman coefficient
    #%% Calculate critical discharge and critical streampower (regression)
    
    interceptregressi_dict={} # intercept of regression 
    sloperegressi_dict={}
    qcr_dict={} # unit critical flow for each reach (n=1,2...), and each fraction (i=1,2,3,4)
    Qcr_dict={} # critical flow for each reach (n=1,2...), each fraction (i=1,2,3,4) and each interpolated flow ([xinterp+2])
    Strmpowcri_dict={} # critical stream power for each reach (n=1,2...), each fraction (i=1,2,3,4) and each interpolated flow ([xinterp+2])
    Strmpow_dict={} # streampower for each reach (n=1,2...), and each interpolated flow ([xinterp+2])
    widthcr_dict={} # width for critical flow for each reach (n=1,2...), each fraction (i=1,2,3,4) and each interpolated flow ([xinterp+2])
    
    for n in fromArrayList:
        if n!=fromArrayList[-1]:
        # calculate tetarm, taurm, bfunc, tetari for each fraction
            tetarm=0.021+(0.015*np.exp(-20*SF))
            taurm=tetarm*rho*g*Rrho*Dm_dict[n]
            bfunci=0.67/(1+(np.exp(1.5-(di/Dm_dict[n]))))
            tetari=taurm*(((di/Dm_dict[n])**bfunci)/(rho*Rrho*g*di))        
            Logi=np.log10((30*tetari*Rrho*di)/(2.718*mcoef*slope_dict[n]*Dm_dict[n]))
            Wcri=(2.3/kv)*rho*((tetari*Rrho*g*di)**1.5)*Logi #Unit critical streampower [w/m2]
            qcri=Wcri/(rho*g*slope_dict[n])      # critical unit discharge [m^2/s]
                
            Qcr=np.zeros((len(Qinterp_dict[n]),1))
            widthcr=np.zeros((len(Qinterp_dict[n]),1))
            Strmpowcri=np.zeros((len(Qinterp_dict[n]),1))
            sloperegressi=np.zeros(len(Qinterp_dict[n]))    # slope for rows of interpolated flow [xinterp+2]
            interceptregressi=np.zeros(len(Qinterp_dict[n])) # intercept for rows of interpolatedflow [xinterp+2]
            
            #calculate stream power
            Qtemp=np.array(Qinterp_dict[n])
            Strmpow=rho*g*Qtemp*slope_dict[n]
            Strmpow_dict[n]=Strmpow        
            
            for l in range (len(Qinterp_dict[n])):
                # calculate regression 
                sloperegressi[l],interceptregressi[l], r_value, p_value, std_err=scipy.stats.linregress (qinterp_dict[n][l],(Qinterp_dict[n][l]/qinterp_dict[n][l]))
                
                widthcr[l]=sloperegressi[l]*qcri + interceptregressi[l]
                Qcr[l]=widthcr[l]*qcri
                Strmpowcri[l]=Wcri*widthcr[l]
            
            interceptregressi_dict[n]=interceptregressi        
            sloperegressi_dict[n]=sloperegressi
            Strmpowcri_dict[n]=Strmpowcri
            Qcr_dict[n]=Qcr
            qcr_dict[n]=qcri
            widthcr_dict[n]=widthcr
        else:
            print ('Sink!',n)
    
    # read data
    
    h1=depthinterp_dict[1]
    h2=depthinterp_dict[2]
    h3=depthinterp_dict[3]
    q1=qinterp_dict[1]
    q2=qinterp_dict[2]
    q3=qinterp_dict[3]
    Q1=Qinterp_dict[1]
    Q2=Qinterp_dict[2]
    Q3=Qinterp_dict[3]
    Strmpowcri1=Strmpowcri_dict[1]
    Strmpowcri2=Strmpowcri_dict[2]
    Strmpowcri3=Strmpowcri_dict[3]
    slope1=slope_dict[1]
    slope2=slope_dict[2]
    slope3=slope_dict[3]
    return (h1,h2,h3,q1,q2,q3,Q1,Q2,Q3,Strmpowcri1,Strmpowcri2,Strmpowcri3,slope1,slope2,slope3)
#%%
def read_h_q_xls (new_directory, xls_file):
    
    import pandas as pd
    import numpy as np
    import os
# read directly from xls


    qinterp_dict={}     # empty disctionary for interpolated unitflow
    depthinterp_dict={} # empty dictionary for interpolated depth
    Qinterp_dict={} # empty dictionary for interpolated flow
    Strmpowcri_dict={} # critical stream power for each reach (n=1,2...), each fraction (i=1,2,3,4) and each interpolated flow ([xinterp+2])
    
    depthinterp_dict[1]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='h1',header=None)
    depthinterp_dict[2]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='h2',header=None)        
    depthinterp_dict[3]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='h3',header=None) 
    qinterp_dict[1]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='q1',header=None) 
    qinterp_dict[2]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='q2',header=None) 
    qinterp_dict[3]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='q3',header=None) 
    Qinterp_dict[1]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='Flow1',header=None)
    Qinterp_dict[2]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='Flow2',header=None)
    Qinterp_dict[3]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='Flow3',header=None)
    Strmpowcri_dict[1]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='Strmpowcri1',header=None)
    Strmpowcri_dict[2]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='Strmpowcri2',header=None)
    Strmpowcri_dict[3]=pd.read_excel(os.path.join(new_directory,xls_file),
                       sheet_name='Strmpowcri3',header=None)
    
    
    h1=depthinterp_dict[1].to_numpy()
    h2=depthinterp_dict[2].to_numpy()
    h3=depthinterp_dict[3].to_numpy()
    q1=qinterp_dict[1].to_numpy()
    q2=qinterp_dict[2].to_numpy()
    q3=qinterp_dict[3].to_numpy()
    Q1=Qinterp_dict[1].to_numpy()
    Q2=Qinterp_dict[2].to_numpy()
    Q3=Qinterp_dict[3].to_numpy()
    Strmpowcri1=Strmpowcri_dict[1].to_numpy()
    Strmpowcri2=Strmpowcri_dict[2].to_numpy()
    Strmpowcri3=Strmpowcri_dict[3].to_numpy()
    slope1=np.float64(0.002566667)
    slope2=np.float64(0.00153)
    slope3=np.float64(0.0017)
    return (h1,h2,h3,q1,q2,q3,Q1,Q2,Q3,Strmpowcri1,Strmpowcri2,Strmpowcri3,slope1,slope2,slope3)
