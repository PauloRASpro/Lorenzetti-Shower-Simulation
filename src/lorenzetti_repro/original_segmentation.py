"""Original segmentation and integration functions used to archive article figures.

The two function bodies were extracted directly from notebook cells 250 and 246.
The limits below are the Lorenzetti limits from source notebook cell 221.
"""
import pandas as pd
import numpy as np
from scipy import stats

BINS_ET_LORENZETTI = [0, 44701.0, 71508.0, 103481.0, 164279.0, 499965.0]
BINS_ABS_ETA_LORENZETTI = [0, 0.43, 0.83, 1.46]

def convert_noSEG_to_SEG_DF(BDT_noSEG_Full_DF, bins_et, bins_eta):
    
    method_list = list(BDT_noSEG_Full_DF["method"].unique())
    print(method_list)
    useRings_list = list(BDT_noSEG_Full_DF["useRings"].unique())
    print(useRings_list)
    results_DF_BDT_ETA_ET = pd.DataFrame()
    for m in method_list:
        for ur in useRings_list:
            
            for i in range(1,len(bins_et)):
                for j in range(1,len(bins_eta)): 
                    
                    segment  = BDT_noSEG_Full_DF[(BDT_noSEG_Full_DF["method"]==m) &
                                                 (BDT_noSEG_Full_DF["useRings"]==ur) &
                                                 (BDT_noSEG_Full_DF["cluster_truthet"]<bins_et[i]) & 
                                                 (BDT_noSEG_Full_DF["cluster_truthet"]>=bins_et[i-1]) & 
                                                 (BDT_noSEG_Full_DF["cluster_trutheta"]<bins_eta[j]) &
                                                 (BDT_noSEG_Full_DF["cluster_trutheta"]>=bins_eta[j-1])]
                    
                    yte_in_r = list(segment["yte"])
                    ypred_in_r = list(segment["ypred"])
                    
                    if len(ypred_in_r)>10:
    
                        #Compute IQR (Interquartile Range)
                        IQRraw=stats.iqr(yte_in_r, axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')
                        IQRpred=stats.iqr(np.divide(yte_in_r, ypred_in_r), axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')
        
                        #Compute MD (Mean Shift)
                        MDraw=np.median(yte_in_r)
                        MDpred=np.median(np.divide(yte_in_r,ypred_in_r))
        
                        #Compute MAE and RMSE (Errors)
                        e_mae =    (1/len(ypred_in_r))*np.sum( abs( np.array(ypred_in_r) - np.array(yte_in_r) )      )
                        e_rmse = ( (1/len(ypred_in_r))*np.sum(    ( np.array(ypred_in_r) - np.array(yte_in_r) )**2 ) )**(1/2)
            
                        e_mae_calib_n = (1/len(ypred_in_r))*np.sum( abs( np.ones(len(ypred_in_r)) - np.divide( np.ones(len(ypred_in_r)) , np.array(ypred_in_r) ) ) )
                        e_mae_fast_n =  (1/len(ypred_in_r))*np.sum( abs( np.ones(len(ypred_in_r)) - np.divide( np.ones(len(ypred_in_r)) , np.array(yte_in_r)   ) ) ) 
    
                        #saving variables into a dataframe
                        results_dict = {"method": [m],
                                "useRings": [ur],
                                "et_range": [(bins_et[i-1],bins_et[i])], 
                                "eta_range": [(bins_eta[j-1],bins_eta[j])],
                                "yte": [yte_in_r],
                                "ypred": [ypred_in_r],
                                "IQRraw": [IQRraw],
                                "IQRpred": [IQRpred],
                                "MDraw": [MDraw],
                                "MDpred": [MDpred],
                                "e_mae": [e_mae],
                                "e_rmse": [e_rmse]}
                
                                
                        results_DF = pd.DataFrame(results_dict)                
                        frame = [results_DF_BDT_ETA_ET, results_DF]                
                        results_DF_BDT_ETA_ET = pd.concat(frame, ignore_index=True)
    
    return results_DF_BDT_ETA_ET

def integrate_results_ETA_ET(results_DF_BDT_ETA_ET):
    
    # Ranges in ET, ETA and userRings
    valid_eta_ranges = list(set(results_DF_BDT_ETA_ET["eta_range"]))
    valid_et_ranges = list(set(results_DF_BDT_ETA_ET["et_range"]))
    valid_useRings = list(set(results_DF_BDT_ETA_ET["useRings"]))
    method = results_DF_BDT_ETA_ET["method"][0]
    
    print("valid_eta_ranges: ", valid_eta_ranges)
    print("valid_et_ranges: ", valid_et_ranges)
    print("valid_useRings: ", valid_useRings)

    # Output DataFrame Definition
    integrated_results_DF_BDT_ETA_ET = pd.DataFrame()

    # Integrate Results for ETA ranges
    print("############# integrate in ETA ###############") 
    for ur in valid_useRings:
        for eta in valid_eta_ranges:
            print(f"=== userRings: {ur} | eta: {eta} ===")
        
            parcial = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==ur) & (results_DF_BDT_ETA_ET["eta_range"]==eta)]
             
            ypred_in_r = []
            yte_in_r = []
        
            len_parcial = len(parcial)        
        
            parcial_ypred_np = parcial["ypred"].to_numpy()
            parcial_yte_np = parcial["yte"].to_numpy()        
        
            for n in range(len_parcial):            
                #print(f"=== {n} / {len_parcial-1} ===")
                new_line_ypred = list(parcial_ypred_np[n])
                new_line_yte = list(parcial_yte_np[n])
            
                #print(new_line_ypred)
                #print(new_line_yte)
            
                if ypred_in_r == [] and yte_in_r ==[]:
                    ypred_in_r = new_line_ypred
                    yte_in_r = new_line_yte
                else:
                    ypred_in_r.extend(new_line_ypred)
                    yte_in_r.extend(new_line_yte)
            
            #len_ypred_in_r = len(ypred_in_r)        
            #eta_in_r = [eta] * len_ypred_in_r

            #Compute IQR (Interquartile Range)
            IQRraw=stats.iqr(yte_in_r, axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')
            IQRpred=stats.iqr(np.divide(yte_in_r,ypred_in_r), axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')
        
            #Compute MD (Mean Shift)
            MDraw=np.median(yte_in_r)
            MDpred=np.median(np.divide(yte_in_r,ypred_in_r))
        
            #Compute MAE and RMSE (Errors)
            e_mae = (1/len(ypred_in_r))*np.sum( abs( np.array(ypred_in_r) - np.array(yte_in_r) ) )
            e_rmse = ( (1/len(ypred_in_r))*np.sum( ( np.array(ypred_in_r) - np.array(yte_in_r) )**2 ) )**(1/2)
            
            #e_mae_calib_n = (1/len(ypred_in_r))*np.sum( abs( np.ones(len(yte_in_r)) - np.divide( np.ones(len(yte_in_r)) , np.array(ypred_in_r) ) ) )
            #e_mae_fast_n =  (1/len(ypred_in_r))*np.sum( abs( np.ones(len(yte_in_r)) - np.divide( np.ones(len(yte_in_r)) , np.array(yte_in_r) ) ) )
            
            e_mae_calib_n = (1/len(ypred_in_r))*np.sum( 1 - np.array(ypred_in_r) / np.array(yte_in_r) )  # Erro normalizado
            e_mae_fast_n =  (1/len(ypred_in_r))*np.sum( 1 - 1 / np.array(yte_in_r) ) # Erro normalizado
            
            #Agregate results in a dataframe
            results_dict = {"method": [method],
                        "useRings": [ur],
                        "et_range": ["All"], 
                        "eta_range": [eta],
                        "yte": [yte_in_r],
                        "ypred": [ypred_in_r],
                        "IQRraw": [IQRraw],
                        "IQRpred": [IQRpred],
                        "MDraw": [MDraw],
                        "MDpred": [MDpred],
                        "e_mae": [e_mae],
                        "e_rmse": [e_rmse],
                        "e_mae_calib_n": [e_mae_calib_n],
                        "e_mae_fast_n": [e_mae_fast_n]}
        
            results_DF = pd.DataFrame(results_dict)                
            frame = [integrated_results_DF_BDT_ETA_ET, results_DF]                
            integrated_results_DF_BDT_ETA_ET = pd.concat(frame, ignore_index=True)

    print("############# integrate in ET ###############")    
    # Integrate Results for ET ranges
    for ur in valid_useRings:
        for et in valid_et_ranges:
            print(f"=== userRings: {ur} | ET: {et} ===")
        
            parcial = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==ur) & (results_DF_BDT_ETA_ET["et_range"]==et)]
             
            ypred_in_r = []
            yte_in_r = []
        
            len_parcial = len(parcial)        
        
            parcial_ypred_np = parcial["ypred"].to_numpy()
            parcial_yte_np = parcial["yte"].to_numpy()        
        
            for n in range(len_parcial):            
                #print(f"=== {n} / {len_parcial-1} ===")
                new_line_ypred = list(parcial_ypred_np[n])
                new_line_yte = list(parcial_yte_np[n])
            
                #print(new_line_ypred)
                #print(new_line_yte)
            
                if ypred_in_r == [] and yte_in_r ==[]:
                    ypred_in_r = new_line_ypred
                    yte_in_r = new_line_yte
                else:
                    ypred_in_r.extend(new_line_ypred)
                    yte_in_r.extend(new_line_yte)
            
            #len_ypred_in_r = len(ypred_in_r)        
            #et_in_r = [et] * len_ypred_in_r

            #Compute IQR (Interquartile Range)
            IQRraw=stats.iqr(yte_in_r, axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')
            IQRpred=stats.iqr(np.divide(yte_in_r,ypred_in_r), axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')
        
            #Compute MD (Mean Shift)
            MDraw=np.median(yte_in_r)
            MDpred=np.median(np.divide(yte_in_r,ypred_in_r))
        
            #Compute MAE and RMSE (Errors)
            e_mae = (1/len(ypred_in_r))*np.sum( abs( np.array(ypred_in_r) - np.array(yte_in_r) ) )
            e_rmse = ( (1/len(ypred_in_r))*np.sum( ( np.array(ypred_in_r) - np.array(yte_in_r) )**2 ) )**(1/2)            
            
            #e_mae_calib_n = (1/len(ypred_in_r))*np.sum( abs( np.ones(len(yte_in_r)) - np.divide( np.ones(len(yte_in_r)) , np.array(ypred_in_r) ) ) )
            #e_mae_fast_n =  (1/len(ypred_in_r))*np.sum( abs( np.ones(len(yte_in_r)) - np.divide( np.ones(len(yte_in_r)) , np.array(yte_in_r) ) ) )
            
            e_mae_calib_n = (1/len(ypred_in_r))*np.sum( 1 - np.array(ypred_in_r) / np.array(yte_in_r) )  # Erro normalizado
            e_mae_fast_n =  (1/len(ypred_in_r))*np.sum( 1 - 1 / np.array(yte_in_r) ) # Erro normalizado
        
            #Agregate results in a dataframe
            results_dict = {"method": [method],
                            "useRings": [ur],
                        "et_range": [et], 
                        "eta_range": ["All"],
                        "yte": [yte_in_r],
                        "ypred": [ypred_in_r],
                        "IQRraw": [IQRraw],
                        "IQRpred": [IQRpred],
                        "MDraw": [MDraw],
                        "MDpred": [MDpred],
                        "e_mae": [e_mae],
                        "e_rmse": [e_rmse],
                        "e_mae_calib_n": [e_mae_calib_n],
                        "e_mae_fast_n": [e_mae_fast_n]}
        
            results_DF = pd.DataFrame(results_dict)                
            frame = [integrated_results_DF_BDT_ETA_ET, results_DF]                
            integrated_results_DF_BDT_ETA_ET = pd.concat(frame, ignore_index=True)

    return integrated_results_DF_BDT_ETA_ET

def build_segmented_tables(frame):
    """Use the source functions and Lorenzetti segmentation, without ATLAS bins."""
    if frame.empty:
        return pd.DataFrame(),pd.DataFrame()
    cols=['method','useRings','ypred','yte','cluster_truthet','cluster_trutheta']
    missing=[x for x in cols if x not in frame]
    if missing:raise ValueError(f"Event archive missing segmentation columns: {missing}")
    local=frame.copy()
    # The theorem and manuscript use |eta|; never bin signed eta by accident.
    local['cluster_trutheta']=local['cluster_trutheta'].abs()
    result=convert_noSEG_to_SEG_DF(local,BINS_ET_LORENZETTI,BINS_ABS_ETA_LORENZETTI)
    if result.empty:
        return result,pd.DataFrame()
    integrated=integrate_results_ETA_ET(result)
    return result,integrated
