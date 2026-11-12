"""Original plotting, resolution and statistical functions from the experimental notebook.

Function bodies are preserved verbatim. Notebook cell provenance is recorded in
ORIGINAL_FUNCTION_MAP.json. The adapter that supplies arrays is in original_replay.py.
"""
from __future__ import annotations
import math
import re
from statistics import mean, median, stdev
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel, norm, f, wilcoxon, shapiro, normaltest, anderson, probplot, t
COLORS = ["C0", "orange", "green", "red", "lightblue", "lightgreen", "pink", "purple", "brown", "grey", "olive", "indigo", "paleturquoise", "darkslategray", "fuchsia", "darkred", "darkgreen", "darkblue", "darkgoldenrod", "darkcyan", "darkolivegreen", "lightcoral", "peru", "orangered", "royalblue", "navy"]
full_data_text_label = ''
useRings_cenarios = []

# Original notebook cell 11; function plota_data_label
def plota_data_label(plt, full_data_text_label):
    ax = plt.gca()                      
    
                                  
    ax.set_title(label="")   
    ax.tick_params(which='both', top=True, right=True, left=True, bottom=True)     
    ax.xaxis.set_tick_params(direction='in', which='both')
    ax.yaxis.set_tick_params(direction='in', which='both')    
    ax.minorticks_on()
    
    if full_data_text_label:
                                           
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        scale_y = ax.get_yscale()
        scale_x = ax.get_xscale()
        
                                       
        if scale_y == 'log':
                                                             
            log_range = math.log10(ymax) - math.log10(ymin)
            ymax_new = 10**(math.log10(ymax) + 0.3 * log_range)
            y_text = 10**(math.log10(ymax_new) - 0.05 * log_range)                          
        else:
                                                  
            delta = 0.3 * (ymax - ymin)
            ymax_new = ymax + delta
            y_text = ymax_new - 0.05 * (ymax_new - ymin)                          
        
                              
        if scale_x == 'log':
            x_text = 10**(math.log10(xmin) + 0.05 * (math.log10(xmax) - math.log10(xmin)))                  
        else:
            x_text = xmin + 0.025 * (xmax - xmin)                    
        
                                 
        ax.set_ylim(top=ymax_new)
        
                          
        ax.text(
            x=x_text,
            y=y_text,
            s=full_data_text_label,
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='left'
        )


# Original notebook cell 12; function plota_data_label_ax
def plota_data_label_ax(ax, full_data_text_label):
                                  
    ax.set_title(label="")     
    ax.tick_params(which='both', top=True, right=True, left=True, bottom=True)     
    ax.xaxis.set_tick_params(direction='in', which='both')
    ax.yaxis.set_tick_params(direction='in', which='both')    
    ax.minorticks_on()
    
    if full_data_text_label:
                                                                 
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        scale_y = ax.get_yscale()
        scale_x = ax.get_xscale()
        
                                                         
        if scale_y == 'log':
                                                   
            ymax_new = ymax * 10**(0.15 * math.log10(ymax/ymin))                     
            y_text_position = ymax_new * 10**(-0.05 * math.log10(ymax_new/ymin))                      
        else:
                                       
            delta = 0.15 * (ymax - ymin)                         
            ymax_new = ymax + delta
            y_text_position = ymax_new - 0.05 * (ymax_new - ymin)                     
        
                              
        if scale_x == 'log':
            x_text_position = xmin * 10**(0.05 * math.log10(xmax/xmin))                    
        else:
            x_text_position = xmin + 0.025 * (xmax - xmin)                        
        
                                 
        ax.set_ylim(top=ymax_new)
        
                                                  
        ax.text(
            x=x_text_position,
            y=y_text_position,
            s=full_data_text_label,
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='left'
        )


# Original notebook cell 83; function _build_color_dict
def _build_color_dict(labels, colors=COLORS):
    """Assigns a stable color to each distinct label using the palette in first-occurrence order."""
    ordered_unique = list(dict.fromkeys(labels))                            
    return {lab: colors[i % len(colors)] for i, lab in enumerate(ordered_unique)}


# Original notebook cell 83; function comparison_hist_boxplot_erros
def comparison_hist_boxplot_erros(
    data, 
    labels, 
    full_data_text_label="", 
    xtextlabel=""
):
    """Displays the original paired residual histogram and horizontal boxplot. Inputs are residual arrays, labels, the detector label, and the horizontal-axis caption."""

                                                                 
    df = pd.DataFrame({"erros": data, "labels": labels})
    iqr_list, q3_list, med_list, rmse_list = [], [], [], []

    for e in df["erros"]:
        e = np.asarray(e)
        q1, med, q3 = np.quantile(e, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        rmse = float(np.sqrt(mean(e**2)))                          

        iqr_list.append(iqr)
        q3_list.append(q3)
        med_list.append(med)
        rmse_list.append(rmse)

    df["iqr"] = iqr_list
    df["q3"] = q3_list
    df["median"] = med_list
    df["rmse"] = rmse_list

                                                           
    data = df["erros"].tolist()
    labels = df["labels"].tolist()
    iqr = df["iqr"].tolist()
    q3 = df["q3"].tolist()
    mediana = df["median"].tolist()
    rmse = df["rmse"].tolist()

                                  
    if len(data) / 3 > 11:
        fig_y = int(len(data) / 3)
        ncol = 2
    else:
        fig_y = 11
        ncol = 1

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, fig_y))
    fig.subplots_adjust(hspace=0)

                                                                     
    color_dict = _build_color_dict(labels, COLORS)

                                        
    bins = 100
                                                                               
    deviations = [float(np.std(np.asarray(d), ddof=1)) if len(d) > 1 else 0.0 for d in data]
    deviation = max(deviations) if deviations else 1.0
    if not np.isfinite(deviation) or deviation == 0.0:
        deviation = 1.0                          

    for i in range(len(data)):
        cor = color_dict[labels[i]]
        axs[0].hist(
            data[i], bins=bins, range=(-deviation * 4, +deviation * 4),
            density=True, histtype="step", label=str(labels[i]),
            linewidth=2, color=cor
        )

    axs[0].set_title(
        r"Normalized Error Histogram: Standard FastCalo x MLR (Test Data without Segmentation)",
        fontsize=16
    )
    axs[0].set_ylabel(r"Probability Density", fontsize=16)
    axs[0].legend(loc="upper right", frameon=False, fontsize=10, ncol=ncol)
    axs[0].axvline(x=0, color="grey", linestyle="--")

                                                                                        
    try:
        plota_data_label_ax(axs[0], full_data_text_label)
    except NameError:
        pass

                                     
    vertical = False
    bp = axs[1].boxplot(
        data, showfliers=False, vert=vertical, patch_artist=True, labels=labels
    )

                                                          
    for i, patch in enumerate(bp["boxes"]):
        lab = labels[i]
        c = color_dict[lab]
        patch.set_facecolor(c)
                               
                                 

                                                                          
                                                 
                                  
                              
                                    

                                     
    for med_line in bp["medians"]:
        med_line.set_color("black")
        med_line.set_linewidth(2.0)

                                                                
                                              
                                          
                                 
                                  
                   
                                     
                                          
                                                       
                                                         
               
                                   
                                        
                                                           
                                                             

                                 
    xmin, xmax, ymin, ymax = axs[1].axis()
    for i in range(len(data)):
        axs[1].text(
            xmax - (xmax - xmin) * .30, axs[1].get_yticks()[i],
            f'IQR: {iqr[i]:.2e}, RMSE: {rmse[i]:.2e}',
            va='center', fontsize=10
        )
        axs[1].plot(rmse[i], axs[1].get_yticks()[i], 'o', color='black')

    try:
        plota_data_label_ax(axs[1], "")
    except NameError:
        pass

                    
    if vertical:
        axs[1].set_xlabel("Evaluation", fontsize=16)
        axs[1].axhline(y=0, color='grey', linestyle='--')
        axs[1].set_ylabel(xtextlabel + " (MeV)" if xtextlabel else "Normalized Errors", fontsize=16)
    else:
        axs[1].set_ylabel("Evaluation", fontsize=16)
        axs[1].axvline(x=0, color='grey', linestyle='--')
        axs[1].set_xlabel(xtextlabel if xtextlabel else "Normalized Errors", fontsize=16)

                        
    for l in axs[1].get_yticklabels():
        l.set_horizontalalignment('left')
    axs[1].tick_params(axis="y", direction="in", pad=-10, labelsize=12)
    xmin, xmax, ymin, ymax = axs[1].axis()
    ydiff = ymax - ymin
    axs[1].set_ylim([ymin - ydiff * 0.05, ymax + ydiff * 0.05])
    axs[1].minorticks_off()

                    
                                                                                                             
    plt.show()


# Original notebook cell 88; function comparison_metric_all
def comparison_metric_all(df,full_data_text_label, rank='D0', show='D0'):    
       

    if rank == 'D0' and show == 'D0': 
        df = df.sort_values(by=[rank])
        name = df['Method-Selection']
        std_label = 'Standard FastCalo:0'
        std_show = rank
        times = df[std_show]
        x_title = r'D0=$\sqrt{\text{IQR}_{(N)}^{2} + \text{RMSE}_{(N)}^{2} + \text{T}_{(N)}^{2} + \text{F}_{(N)}^{2}}$'
    elif rank == 'D0' and show == 'd0': 
        df = df.sort_values(by=[rank])
        name = df['Method-Selection']
        std_label = 'Standard FastCalo:0'
        std_show = 'd0 (GeV)'
        times = df[std_show]
        x_title = r'd0=$\sqrt{\text{IQR}^{2} + \text{RMSE}^{2} } $ (MeV)'
    elif rank == 'd0' and show == 'd0':
        df = df.sort_values(by=[rank])
        name = df['data_labels']
        std_label = 'Standard FastCalo'
        std_show = rank
        times = df[std_show]
        x_title = r'd0=$\sqrt{\text{IQR}^{2} + \text{RMSE}^{2} } $ (MeV)'
    elif rank == 'd0' and show == 'D0':
        df = df.sort_values(by=['d0 (GeV)'])
        name = df['Method-Selection']
        std_label = 'Standard FastCalo:0'
        std_show = show
        times = df[std_show]
        x_title = r'D0=$\sqrt{\text{IQR}_{(N)}^{2} + \text{RMSE}_{(N)}^{2} + \text{T}_{(N)}^{2} + \text{F}_{(N)}^{2}}$'
    else:
        print('ERROR: rank not defined')
    
        
    xmax = max(times)

    
    stardard_d0 = 1
    for i in range(len(name)):
        if name[i]==std_label:
            standard_d0 = times[i]
            print(f"Encontrou o Standard FastCalo, {std_show} =", standard_d0)
            break
    
                                 
    n_bars = 10
    name = name[0:n_bars]
    times = times[0:n_bars]

                               
    if len(name)/3 > 7: 
        fig_y = int(len(name)/3)
    else: 
        fig_y = 7
                 
                                           
    fig, ax = plt.subplots(figsize=(8, fig_y))
                             
   
 
                         
    ax.barh(name, times, color='w', edgecolor='b', linewidth=2)
 
                         
                                                 
                                         
 
                       
                                        
                                        
                                        
                                        

     
 
                                         
    ax.xaxis.set_tick_params(pad=5, labelsize=13, direction='in')
    ax.yaxis.set_tick_params(pad=3, labelsize=13, direction='in')

    
    
    for l in ax.get_yticklabels():
        l.set_fontweight('bold')
        l.set_horizontalalignment('left')
                                              
                                                                               
    ax.tick_params(axis="y",direction="in", pad=-10, labelsize=12, colors="k")

    
      
 
                        
                                                                                       
 
                     
    ax.invert_yaxis()
 
                            
    for i in ax.patches:
        metric = i.get_width()
                                              
                                                              
        reduction = 100*(metric-standard_d0) / standard_d0
                                                                
                                                                                             
        str_reduction =" (" + str( round(reduction, 2) ) + "\% Std)"                     
        
        plt.text(i.get_width()+xmax*.01, i.get_y()+0.5, str(round(metric, 4))+str_reduction, fontsize=12, fontweight='bold', color='black')
        
 
                    
                                                                                                          
                                                                                   
    ax.set_xlabel(x_title, fontsize=16)
    ax.set_ylabel(f'Top 10 Ranked by {rank}', fontsize=16, labelpad=30)
 
               
                                                   
    xmin, xmax, ymin, ymax = ax.axis()
    ax.set_xlim([0, xmax + (xmax-xmin)*.35])
    ax.set_ylim([ymin, ymax + (ymax-ymin)*.10])
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(xmin + (xmax-xmin)*.05, ymax - (ymax-ymin)*.05, full_data_text_label, fontsize=13)
                                                                                                
    plt.show()


# Original notebook cell 247; function integrated_plots_in_ETA_ET_geral_with_ranges
def integrated_plots_in_ETA_ET_geral_with_ranges(integrated_results_DF_BDT_ETA_ET, useRings_cenarios, graph_prefix=''):
        
    
                
    fmt_mark = ['v', 's', 'p', '*', 'P', '^', '<', '>', '1', '2', '3', '4', '8', 'h', '+','x','d','.']
    
    print()
    print("################## Plots in ETA ##################")  
    print()
                                     
    results_DF_BDT_ETA_ET = integrated_results_DF_BDT_ETA_ET[integrated_results_DF_BDT_ETA_ET["et_range"]=="All"]
    
                                                      
    eta_range = list(set(results_DF_BDT_ETA_ET["eta_range"])) 
    print('eta_range:', eta_range)
    print()
        
                
    useRings = list(set(results_DF_BDT_ETA_ET["useRings"]))   
    print('useRings', useRings)
    print()
    
                        
    m_eta = list(results_DF_BDT_ETA_ET["method"])
    r_eta = list(results_DF_BDT_ETA_ET["useRings"])
    par_m_r_eta = [ (m_eta[i], r_eta[i]) for i in range(len(m_eta))]    
    set_par_m_r_eta = set(par_m_r_eta)
    list_set_par_m_r_eta = list(set_par_m_r_eta)
    
    print('par_m_r_eta:', par_m_r_eta)
    print('set_par_m_r_eta:', set_par_m_r_eta)
    print('list_set_par_m_r_eta:', list_set_par_m_r_eta) 
    print()
    
                  
    methods_list = list(set(results_DF_BDT_ETA_ET["method"]))
    print('methods_list:', methods_list)
    print()
    
    
                  
    etaXPos_ETA = []
    etaXDelta_ETA = []
    for r in eta_range:       
        pto_med = (r[1]+r[0])/2
        diff = (r[1]-r[0])/2
        etaXPos_ETA.append(pto_med)
        etaXDelta_ETA.append(diff)  
    
    def eta_pos_delta(eta_range):
        etaXPos_ETA = []
        etaXDelta_ETA = []
        for r in eta_range:       
            pto_med = (r[1]+r[0])/2
            diff = (r[1]-r[0])/2
            etaXPos_ETA.append(pto_med)
            etaXDelta_ETA.append(diff)   
        return etaXPos_ETA, etaXDelta_ETA
        
    print('etaXPos_ETA:', len(etaXPos_ETA), etaXPos_ETA)
    print('etaXDelta_ETA:', len(etaXDelta_ETA), etaXDelta_ETA)
    print()

    std_method_eta = list_set_par_m_r_eta[0][0]
    std_useRings_eta = list_set_par_m_r_eta[0][1]
    
                 
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (8, 7))  
    fig.subplots_adjust(hspace=0)
    
    IQRraw = list(results_DF_BDT_ETA_ET[ (results_DF_BDT_ETA_ET["useRings"]==std_useRings_eta) & 
                                         (results_DF_BDT_ETA_ET["method"]==std_method_eta) ]["IQRraw"] )
    
    print('std_method_eta:', std_method_eta)
    print('std_useRings_eta:', std_useRings_eta)
    print('IQRraw:', len(IQRraw), IQRraw)
    print()
    
    a0.errorbar(etaXPos_ETA, IQRraw, xerr=etaXDelta_ETA, fmt='o', label=f"Standard FastCalo")    
    a1.errorbar(etaXPos_ETA,np.ones(len(IQRraw)), xerr=etaXDelta_ETA, fmt='o', label=f"Standard FastCalo")
    
    j=0
    for lmr in range(len(list_set_par_m_r_eta)):
        method = list_set_par_m_r_eta[lmr][0]
        i = list_set_par_m_r_eta[lmr][1]
        IQRpred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["IQRpred"]
        eta_ranges_m = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["eta_range"]        
        
        etaXPos_ETA, etaXDelta_ETA = eta_pos_delta(eta_ranges_m)
        print("=> method:",method, ", scenario:", i, ", len(IQRpred):", len(IQRpred),", len(etaXPos_ETA):", len(etaXPos_ETA))    
                                        
                                                           
                                           
                                               
        
        if len(IQRpred) > 0:                
            a0.errorbar(etaXPos_ETA, IQRpred, xerr=etaXDelta_ETA, fmt=fmt_mark[j], label=f"{method}: {useRings_cenarios[i]}") 
            a1.errorbar(etaXPos_ETA, np.divide(np.array(IQRpred),np.array(IQRraw)), xerr=etaXDelta_ETA, fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            j+=1
        else:
            print("ERROR: len(IQRpred) <= 0")              
   
    
    """
    for method in methods_list:    
        j=0
        for i in useRings: 
            IQRpred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["IQRpred"]
            print("method:", method, ", scenario:", i, ", len(IQRpred):", len(IQRpred),", len(etaXPos_ETA):", len(etaXPos_ETA))
            if len(IQRpred) > 0:                
                a0.errorbar(etaXPos_ETA, IQRpred, xerr=etaXDelta_ETA, fmt=fmt_mark[j], label=f"{method}: {useRings_cenarios[i]}") 
                a1.errorbar(etaXPos_ETA, np.divide(np.array(IQRpred),np.array(IQRraw)), xerr=etaXDelta_ETA, fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                j+=1
    """
    
    a0.set_title('IQR Integrated in all $E_t$ bins')
    
    ymin, ymax = a0.get_ylim()        
    a0.set_ylim([ymin, ymax+(ymax-ymin)*.35])
    
    a0.legend(loc='upper right', frameon=False, fontsize=12, ncol=1)
                                                                   
    a0.set_ylabel(r'IQR$(\alpha_{calib} = E_{true} / E_{calib})$', fontsize=16)
                                                                         
    
                                                             
    a1.set_ylabel('Ratio w.r.t.', fontsize=16)
                                                                   
    a1.set_xlabel(r'Bins : $|\eta|$', fontsize=16)
                                                                         
                                                                                   
                                                
    plota_data_label_ax(a0, full_data_text_label)
    plota_data_label_ax(a1, '')
                                                                           
    plt.show()
    
    
    
    
                       
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (8, 7))
    fig.subplots_adjust(hspace=0)
    MDraw = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==std_useRings_eta) & (results_DF_BDT_ETA_ET["method"]==std_method_eta)]["MDraw"]
    a0.errorbar(etaXPos_ETA,1-np.array(MDraw),xerr=etaXDelta_ETA,fmt='o',label=f"Standard FastCalo")
    a1.errorbar(etaXPos_ETA,np.divide(1-np.array(MDraw),1-np.array(MDraw)),xerr=etaXDelta_ETA,fmt='o',label=f"Standard FastCalo")
    
    
    j=0
    for lmr in range(len(list_set_par_m_r_eta)):
        method = list_set_par_m_r_eta[lmr][0]
        i = list_set_par_m_r_eta[lmr][1]
        MDpred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                       (results_DF_BDT_ETA_ET["method"]==method)]["MDpred"]
        
        eta_ranges_m = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["eta_range"]        
        
        etaXPos_ETA, etaXDelta_ETA = eta_pos_delta(eta_ranges_m)
        print("=> method:",method, ", scenario:", i, ", len(MDpred):", len(MDpred),", len(etaXPos_ETA):", len(etaXPos_ETA))    
                                        
                                                          
                                           
                                               
        
        if len(MDpred) > 0:                
            a0.errorbar(etaXPos_ETA,1-np.array(MDpred),xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            a1.errorbar(etaXPos_ETA,np.divide(1-np.array(MDpred),1-np.array(MDraw)),xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            j+=1
        else:
            print("ERROR: len(MDpred) <= 0")
    
    
    
    """
    for method in methods_list:
        j=0
        for i in useRings: 
            MDpred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["MDpred"]
            print("method:", method, ", scenario:", i, ", len(MDpred):", len(MDpred),", len(etaXPos_ETA):", len(etaXPos_ETA))
            if len(MDpred) > 0:                
                a0.errorbar(etaXPos_ETA,1-np.array(MDpred),xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                a1.errorbar(etaXPos_ETA,np.divide(1-np.array(MDpred),1-np.array(MDraw)),xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                j+=1
    """
    
    
    a0.set_title('Mean Shift Integrated in all $E_t$ bins')

    ymin, ymax = a0.get_ylim()        
    a0.set_ylim([ymin, ymax+(ymax-ymin)*.35])
    
    a0.legend(loc='upper right', frameon=False, fontsize=12, ncol=1)
                                                                  
    a0.set_ylabel(r'Mean Shift $(\alpha_{calib} = E_{true} / E_{calib})$', fontsize=16)

                                                             
                                                               
    a1.set_ylabel('Ratio w.r.t.', fontsize=16)    
    a1.set_xlabel(r'Bins : $|\eta|$', fontsize=16)
                                                                                 
                                                
    plota_data_label_ax(a0, full_data_text_label)
    plota_data_label_ax(a1, '')
                                                                                  
    plt.show()
    
    
    
                                 
    
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize = (8, 7))
    fig.subplots_adjust(hspace=0)
    a0.errorbar(etaXPos_ETA,np.zeros(len(etaXPos_ETA)),xerr=etaXDelta_ETA,fmt='o',label="Zero Error Reference")
    a1.errorbar(etaXPos_ETA,np.zeros(len(etaXPos_ETA)),xerr=etaXDelta_ETA,fmt='o',label="Zero Error Reference")
    
    
    j=0
    for lmr in range(len(list_set_par_m_r_eta)):
        method = list_set_par_m_r_eta[lmr][0]
        i = list_set_par_m_r_eta[lmr][1]
        
        e_mae = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                      (results_DF_BDT_ETA_ET["method"]==method)]["e_mae"]
        
        e_rmse = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                       (results_DF_BDT_ETA_ET["method"]==method)]["e_rmse"]
        
        eta_ranges_m = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["eta_range"]        
        
        etaXPos_ETA, etaXDelta_ETA = eta_pos_delta(eta_ranges_m)
        print("=> method:",method, ", scenario:", i, ", len(MDpred):", len(MDpred),", len(etaXPos_ETA):", len(etaXPos_ETA))    
                                        
                                                          
                                           
                                               
        
        if len(e_mae) > 0:
            a0.errorbar(etaXPos_ETA,e_mae,xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            a1.errorbar(etaXPos_ETA,e_rmse,xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            j+=1
        else:
            print("ERROR: len(e_mae) <= 0")
    
    
    
    """
    for method in methods_list:
        j=0
        for i in useRings: 
            e_mae = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["e_mae"]
            e_rmse = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["e_rmse"]
            print("method:", method, ", scenario:", i, ", len(e_mae):", len(e_mae),", len(etaXPos_ETA):", len(etaXPos_ETA))
            if len(e_mae) > 0:
                a0.errorbar(etaXPos_ETA,e_mae,xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                a1.errorbar(etaXPos_ETA,e_rmse,xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                j+=1
    """
    
    
    a0.set_title(r'$\alpha$ Estimation Error Integrated in all $E_t$ bins')

    ymin, ymax = a0.get_ylim()        
    a0.set_ylim([ymin, ymax+(ymax-ymin)*.65])
    
    a0.legend(loc='upper right', frameon=False, fontsize=12, ncol=1)
                                                                  
    a0.set_ylabel(r'MAE$(\alpha_{calib})$', fontsize=16)   

                                                                         
                                                             
                                                                   
    a1.set_ylabel(r'RMSE$(\alpha_{calib})$', fontsize=16)
    a1.set_xlabel(r'$|\eta|$', fontsize=16)
                                                 
                                                                           
                                                
    plota_data_label_ax(a0, full_data_text_label)
    plota_data_label_ax(a1, '')
                                                                                      
    plt.show()
    
    
    
                                 
    
    fig, a0 = plt.subplots(figsize =(8, 7)) 
    e_mae_fast_n = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==std_useRings_eta) & (results_DF_BDT_ETA_ET["method"]==std_method_eta)]["e_mae_fast_n"]
    a0.errorbar(etaXPos_ETA,abs(e_mae_fast_n),xerr=etaXDelta_ETA,fmt='o',label="Standard FastCalo")
                                                                                                        
    
    
    j=0
    for lmr in range(len(list_set_par_m_r_eta)):
        method = list_set_par_m_r_eta[lmr][0]
        i = list_set_par_m_r_eta[lmr][1]
        
        e_mae_calib_n = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                              (results_DF_BDT_ETA_ET["method"]==method)]["e_mae_calib_n"]
        
        eta_ranges_m = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["eta_range"]        
        
        etaXPos_ETA, etaXDelta_ETA = eta_pos_delta(eta_ranges_m)
        print("=> method:",method, ", scenario:", i, ", len(MDpred):", len(MDpred),", len(etaXPos_ETA):", len(etaXPos_ETA))    
                                        
                                                          
                                           
                                               
        
        if len(e_mae_calib_n) > 0:
            a0.errorbar(etaXPos_ETA,e_mae,xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            a1.errorbar(etaXPos_ETA,e_rmse,xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            j+=1
        else:
            print("ERROR: len(e_mae_calib_n) <= 0")
    
    
    
    
    """
    for method in methods_list:
        j=0
        for i in useRings: 
            e_mae_calib_n = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["e_mae_calib_n"]
            print("method:", method, ", scenario:", i, ", len(e_mae_calib_n):", len(e_mae_calib_n),", len(etaXPos_ETA):", len(etaXPos_ETA))
            if len(e_mae) > 0:
                a0.errorbar(etaXPos_ETA,abs(e_mae_calib_n),xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                #a0.errorbar(etaXPos_ETA,e_rmse,xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                j+=1
    """
    
    
    a0.set_title('Normalized Error Integrated in all $E_t$ bins')

    ymin, ymax = a0.get_ylim()        
    a0.set_ylim([ymin, ymax+(ymax-ymin)*.35])
    
    a0.legend(loc='upper right', frameon=False, fontsize=12, ncol=1)
                                                                   
    a0.set_ylabel(r'$ (E_{true} - E_{calib})/E_{true} $', fontsize=16) 
    a0.set_xlabel(r'$|\eta|$', fontsize=16)

                                                                         
                                                             
                                                    
                                        
                                            
                                                 
                                                                            
    plota_data_label(plt, full_data_text_label)    
                                                                                        
    plt.show()
    
    
    
                         
    
    for r in eta_range:
        print(f"range: {r}")
        cont = 0
        fig, a0 = plt.subplots(figsize = (8, 7)) 
        
                                                                                           
                                       
                                            
                                                                                                          
                                                  
                                         
                                        
                  
                                                                                                                                                         
        
                                                
                                
                
        for lmr in range(len(list_set_par_m_r_eta)):
            method = list_set_par_m_r_eta[lmr][0]
            i = list_set_par_m_r_eta[lmr][1]
            if 1 == 1:
                
                alfa_true = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["eta_range"]==r) & 
                                                  (results_DF_BDT_ETA_ET["useRings"]==i) & 
                                                  (results_DF_BDT_ETA_ET["method"]==method)]["yte"]
                alfa_true = list(alfa_true)[0]
                                                    
                alfa_esti = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["eta_range"]==r) & 
                                                  (results_DF_BDT_ETA_ET["useRings"]==i) & 
                                                  (results_DF_BDT_ETA_ET["method"]==method)]["ypred"]
                alfa_esti = list(alfa_esti)[0]
                                                     
                print("method:", method, ", scenario:", i, ", len(alfa_esti):", len(alfa_esti),", len(alfa_true):", len(alfa_true))
                
                                                                                    
                erro_calib = 1 - np.array(alfa_esti) / np.array(alfa_true)           
                                                          
                
                                                 
                                                
                
                bins = 100
                if cont == 0:
                    erro_fast =  1 - 1 / np.array(alfa_true)
                    erro_fast_m = np.mean(erro_fast)
                    erro_fast_s = np.std(erro_fast)
                                                                                                                                                                  
                    a0.hist(erro_fast, bins = bins, range=(-erro_fast_s*3,+erro_fast_s*3), label=f"Standard FastCalo", histtype='step', density=True)      
                    cont += 1
                    
                if len(erro_calib) > 0:
                                                                                                                                                                             
                    a0.hist(erro_calib , bins = bins, range=(-erro_fast_s*3,+erro_fast_s*3), label=f"{method}: {useRings_cenarios[i]}", histtype='step', density=True)
                               
                                                       
        a0.set_title(f"Normalized Error Histogram: $|\eta|$ in {r} range")
        a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                        
        a0.set_xlabel(r'$(E_{true} - E_{calib})/E_{true}$', fontsize=16) 
        a0.set_ylabel(f'Probability Density ($|\eta|$ in {r} range)', fontsize=16)   
                                                   
        plt.axvline(x=0, color='lightgrey', linestyle='--')
        plota_data_label(plt, full_data_text_label)    
                                                                                                         
        plt.show()
    
    
    
                 
    
    
    
    
    
    print()
    print("################## Plots in ET ##################") 
    print()
                                     
    results_DF_BDT_ETA_ET = integrated_results_DF_BDT_ETA_ET[integrated_results_DF_BDT_ETA_ET["eta_range"]=="All"]
    
                                                      
    et_range = list(set(results_DF_BDT_ETA_ET["et_range"])) 
    print('et_range:', et_range)
    
                
    useRings = list(set(results_DF_BDT_ETA_ET["useRings"]))
    print('useRings:', useRings)    
    
                        
    m_et = list(results_DF_BDT_ETA_ET["method"])
    r_et = list(results_DF_BDT_ETA_ET["useRings"])
    par_m_r_et = [ (m_et[i], r_et[i]) for i in range(len(m_et))]    
    set_par_m_r_et = set(par_m_r_et)
    list_set_par_m_r_et = list(set_par_m_r_et)
    print('par_m_r_et:',par_m_r_et)
    print('set_par_m_r_et:',set_par_m_r_et)
    print('list_set_par_m_r_et:',list_set_par_m_r_et)
    
    
                 
    etXPos_ET = []
    etXDelta_ET = []
    for r in et_range:       
        pto_med = (r[1]+r[0])/2
        diff = (r[1]-r[0])/2
        etXPos_ET.append(pto_med)
        etXDelta_ET.append(diff)
        
    
    def et_pos_delta(et_range):
        etXPos_ET = []
        etXDelta_ET = []
        for r in et_range:       
            pto_med = (r[1]+r[0])/2
            diff = (r[1]-r[0])/2
            etXPos_ET.append(pto_med)
            etXDelta_ET.append(diff)   
        return etXPos_ET, etXDelta_ET
        
    print('etXPos_ET:',etXPos_ET)
    print('etXDelta_ET:',etXDelta_ET)
    

    std_method_et = list_set_par_m_r_et[0][0]
    std_useRings_et = list_set_par_m_r_et[0][1]
    
                 
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (8, 7)) 
    fig.subplots_adjust(hspace=0)
    IQRraw = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==std_useRings_et) & (results_DF_BDT_ETA_ET["method"]==std_method_et)]["IQRraw"]
    a0.errorbar(etXPos_ET, IQRraw, xerr=etXDelta_ET, fmt='o', label=f"Standard FastCalo")
    a1.errorbar(etXPos_ET,np.ones(len(IQRraw)), xerr=etXDelta_ET, fmt='o', label=f"Standard FastCalo")
    
    
    j=0
    for lmr in range(len(list_set_par_m_r_eta)):
        method = list_set_par_m_r_eta[lmr][0]
        i = list_set_par_m_r_eta[lmr][1]
        
        IQRpred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["IQRpred"]
        
        et_ranges_m = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["et_range"]        
        
        etXPos_ET, etXDelta_ET = et_pos_delta(et_ranges_m)
        print("=> method:",method, ", scenario:", i, ", len(IQRpred):", len(IQRpred),", len(etXPos_ET):", len(etXPos_ET))    
                                        
                                                         
                                        
                                            
        
        if len(IQRpred) > 0:                
            a0.errorbar(etXPos_ET, IQRpred, xerr=etXDelta_ET, fmt=fmt_mark[j], label=f"{method}: {useRings_cenarios[i]}") 
            a1.errorbar(etXPos_ET, np.divide(np.array(IQRpred),np.array(IQRraw)), xerr=etXDelta_ET, fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            j+=1
        else:
            print("ERROR: len(IQRpred) <= 0")   
    
    
    
    """
    for method in methods_list:
        j=0
        for i in useRings: 
            IQRpred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["IQRpred"]
            print("method:", method, ", scenario:", i, ", len(IQRpred):", len(IQRpred),", len(etaXPos_ET):", len(etaXPos_ET))
            if len(IQRpred) > 0:
                a0.errorbar(etaXPos_ET, IQRpred, xerr=etaXDelta_ET, fmt=fmt_mark[j], label=f"{method}: {useRings_cenarios[i]}") 
                a1.errorbar(etaXPos_ET, np.divide(np.array(IQRpred),np.array(IQRraw)), xerr=etaXDelta_ET, fmt=fmt_mark[j],label=f"{method}:{useRings_cenarios[i]}")
                j+=1
    
    
    """
    
    a0.set_title('IQR Integrated in all $|\eta|$ bins')
    
    ymin, ymax = a0.get_ylim()        
    a0.set_ylim([ymin, ymax+(ymax-ymin)*.35])
    
    a0.legend(loc='upper right', frameon=False, fontsize=12, ncol=1)
                                                                
                                                                              
    a0.set_ylabel(r'IQR$(\alpha = E_{true} / E_{calib})$', fontsize=16)
    a0.set_xscale('log')
                                                                     
    
                                                             
    a1.set_ylabel('Ratio w.r.t.', fontsize=16)
                                                                
                                                                              
    a1.set_xlabel(r'Bins : $E_t$ (MeV)', fontsize=16)
    a1.set_xscale('log')
                                                                         
                                                                                  
                                                
    plota_data_label_ax(a0, full_data_text_label)
    plota_data_label_ax(a1, '')
                                                                          
    plt.show()
    
    
    
    
                       
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (8, 7))
    fig.subplots_adjust(hspace=0)
    MDraw = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==std_useRings_et) & (results_DF_BDT_ETA_ET["method"]==std_method_et)]["MDraw"]
    a0.errorbar(etXPos_ET,1-np.array(MDraw),xerr=etXDelta_ET,fmt='o',label=f"Standard FastCalo")
    a1.errorbar(etXPos_ET,np.divide(1-np.array(MDraw),1-np.array(MDraw)),xerr=etXDelta_ET,fmt='o',label=f"Standard FastCalo")
    
    
    j=0
    for lmr in range(len(list_set_par_m_r_eta)):
        method = list_set_par_m_r_eta[lmr][0]
        i = list_set_par_m_r_eta[lmr][1]
        
        MDpred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["MDpred"]
        
        et_ranges_m = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["et_range"]        
        
        etXPos_ET, etXDelta_ET = et_pos_delta(et_ranges_m)
        print("=> method:",method, ", scenario:", i, ", len(MDpred):", len(MDpred),", len(etXPos_ET):", len(etXPos_ET))    
                                        
                                                           
                                           
                                               
        
        if len(IQRpred) > 0:                
            a0.errorbar(etXPos_ET,1-np.array(MDpred),xerr=etXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            a1.errorbar(etXPos_ET,np.divide(1-np.array(MDpred),1-np.array(MDraw)),xerr=etXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            j+=1
        else:
            print("ERROR: len(IQRpred) <= 0") 
    
    
    
    
    
    """
    for method in methods_list:
        j=0
        for i in useRings: 
            MDpred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["MDpred"]
            print("method:", method, ", scenario:", i, ", len(MDpred):", len(MDpred),", len(etaXPos_ET):", len(etaXPos_ET))
            if len(MDpred) > 0:
                a0.errorbar(etaXPos_ET,1-np.array(MDpred),xerr=etaXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                a1.errorbar(etaXPos_ET,np.divide(1-np.array(MDpred),1-np.array(MDraw)),xerr=etaXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                j+=1
    """
    
    
    
    a0.set_title('Mean Shift Integrated in all $|\eta|$ bins')
    
    ymin, ymax = a0.get_ylim()        
    a0.set_ylim([ymin, ymax+(ymax-ymin)*.35])
    
    a0.legend(loc='upper right', frameon=False, fontsize=12, ncol=1)
                                                                
                                                                              
    a0.set_ylabel(r'Mean Shift $(E_{true} / E_{calib})$', fontsize=16)
    a0.set_xscale('log')

                                                             
                                                                
                                                                              
    a1.set_ylabel('Ratio w.r.t.', fontsize=16)    
    a1.set_xlabel(r'Bins : $E_t$ (MeV)', fontsize=16)
    a1.set_xscale('log')
                                                                                 
                                                
    plota_data_label_ax(a0, full_data_text_label)
    plota_data_label_ax(a1, '')
                                                                                 
    plt.show()
    
    
    
                                 
    
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize = (8, 7)) 
    fig.subplots_adjust(hspace=0)
    a0.errorbar(etXPos_ET,np.zeros(len(etXPos_ET)),xerr=etXDelta_ET,fmt='o',label="Zero Error Reference")
    a1.errorbar(etXPos_ET,np.zeros(len(etXPos_ET)),xerr=etXDelta_ET,fmt='o',label="Zero Error Reference")
    
    
    j=0
    for lmr in range(len(list_set_par_m_r_eta)):
        method = list_set_par_m_r_eta[lmr][0]
        i = list_set_par_m_r_eta[lmr][1]
        
        e_mae = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                      (results_DF_BDT_ETA_ET["method"]==method)]["e_mae"]
        
        e_rmse = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                       (results_DF_BDT_ETA_ET["method"]==method)]["e_rmse"]
            
        
        et_ranges_m = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["et_range"]        
        
        etXPos_ET, etXDelta_ET = et_pos_delta(et_ranges_m)
        print("=> method:",method, ", scenario:", i, ", len(e_mae):", len(e_mae),", len(etXPos_ET):", len(etXPos_ET))    
                                        
                                                           
                                           
                                               
        
        if len(e_mae) > 0:                
            a0.errorbar(etXPos_ET,e_mae,xerr=etXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            a1.errorbar(etXPos_ET,e_rmse,xerr=etXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            j+=1
        else:
            print("ERROR: len(e_mae) <= 0") 
    
    
    
    """
    for method in methods_list:
        j=0
        for i in useRings: 
            e_mae = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["e_mae"]
            e_rmse = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["e_rmse"]
            print("method:", method, ", scenario:", i, ", len(e_mae):", len(e_mae),", len(etaXPos_ET):", len(etaXPos_ET))
            if len(e_mae) > 0:    
                a0.errorbar(etaXPos_ET,e_mae,xerr=etaXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                a1.errorbar(etaXPos_ET,e_rmse,xerr=etaXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                j+=1
    
    """
    
    
    a0.set_title(r'$\alpha$ Estimation Error Integrated in all $|\eta|$ bins')
    
    ymin, ymax = a0.get_ylim()        
    a0.set_ylim([ymin, ymax+(ymax-ymin)*.45])
    
    a0.legend(loc='upper right', frameon=False, fontsize=12, ncol=1)
                                                                   
                                                                              
    a0.set_ylabel(r'MAE$(\alpha)$', fontsize=16)  
    a0.set_xscale('log')

                                                                         
                                                             
                                                                 
                                                                              
    a1.set_ylabel(r'RMSE$(\alpha)$', fontsize=16)
    a1.set_xlabel(r'$E_t$ (MeV)', fontsize=16)
    a1.set_xscale('log')
                                                 
                                                                           
                                                
    plota_data_label_ax(a0, full_data_text_label)
    plota_data_label_ax(a1, '')
                                                                                     
    plt.show()
    
    
    
                                 
    
    fig, a0 = plt.subplots(figsize =(8, 7)) 
    e_mae_fast_n = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==std_useRings_et) & (results_DF_BDT_ETA_ET["method"]==std_method_et)]["e_mae_fast_n"]
    a0.errorbar(etXPos_ET,abs(e_mae_fast_n),xerr=etXDelta_ET,fmt='o',label="Standard FastCalo")
                                                                                                        
    
    
    j=0
    for lmr in range(len(list_set_par_m_r_eta)):
        method = list_set_par_m_r_eta[lmr][0]
        i = list_set_par_m_r_eta[lmr][1]
        
        e_mae_calib_n = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                             (results_DF_BDT_ETA_ET["method"]==method)]["e_mae_calib_n"]
            
            
        
        et_ranges_m = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & 
                                        (results_DF_BDT_ETA_ET["method"]==method)]["et_range"]        
        
        etXPos_ET, etXDelta_ET = et_pos_delta(et_ranges_m)
        print("=> method:",method, ", scenario:", i, ", len(e_mae_calib_n):", len(e_mae_calib_n),", len(etXPos_ET):", len(etXPos_ET))    
                                        
                                                           
                                           
                                               
        
        if len(e_mae) > 0:                
            a0.errorbar(etXPos_ET,abs(e_mae_calib_n),xerr=etXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
            j+=1
        else:
            print("ERROR: len(e_mae) <= 0") 
    
    
    
    
    """
    for method in methods_list:
        j=0
        for i in useRings: 
            e_mae_calib_n = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["e_mae_calib_n"]
            print("method:", method, ", scenario:", i, ", len(e_mae_calib_n):", len(e_mae_calib_n),", len(etaXPos_ETA):", len(etaXPos_ETA))
            if len(e_mae) > 0:
                a0.errorbar(etaXPos_ET,abs(e_mae_calib_n),xerr=etaXDelta_ET,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                #a0.errorbar(etaXPos_ETA,e_rmse,xerr=etaXDelta_ETA,fmt=fmt_mark[j],label=f"{method}: {useRings_cenarios[i]}")
                j+=1
    """
    
    
    
    a0.set_title('Normalized Error Integrated in all $|\eta|$ bins')
    
    ymin, ymax = a0.get_ylim()        
    a0.set_ylim([ymin, ymax+(ymax-ymin)*.35])
    
    a0.legend(loc='upper right', frameon=False, fontsize=12, ncol=1)
                                                                  
                                                                              
    a0.set_ylabel(r'$ (E_{true} - E_{calib})/E_{true} $', fontsize=16) 
    a0.set_xlabel(r'$E_t$ (MeV)', fontsize=16)
    a0.set_xscale('log')

                                                                         
                                                             
                                                    
                                        
                                            
                                                 
                                                                            
    plota_data_label(plt, full_data_text_label)
                                                                                       
    plt.show()
    
    
        
                           
    for r in et_range:
        print(f"range: {r}")
        cont = 0
        fig, a0 = plt.subplots(figsize = (8, 7)) 
        
                                                                                          
                                       
                                            
                                                                                                          
                                                  
                                         
                                        
                  
                                                                                                                                                         
        
                                    
                                
                
        for lmr in range(len(list_set_par_m_r_et)):
            method = list_set_par_m_r_et[lmr][0]
            i = list_set_par_m_r_et[lmr][1]            
            if 1 == 1:
                
                alfa_true = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["et_range"]==r) & 
                                                  (results_DF_BDT_ETA_ET["useRings"]==i) & 
                                                  (results_DF_BDT_ETA_ET["method"]==method) ]["yte"]
                alfa_true = list(alfa_true)[0]
                                                    
                alfa_esti = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["et_range"]==r) & 
                                                  (results_DF_BDT_ETA_ET["useRings"]==i) & 
                                                  (results_DF_BDT_ETA_ET["method"]==method)]["ypred"]
                alfa_esti = list(alfa_esti)[0]
                                                     
                print("method:", method, ", scenario:", i, ", len(alfa_esti):", len(alfa_esti),", len(alfa_true):", len(alfa_true))
                
                                                                    
                erro_calib = 1 - np.array(alfa_esti) / np.array(alfa_true)          
                                                          
                                                 
                                                                
                
                bins = 100                
                if cont == 0:
                    erro_fast =  1 - 1 / np.array(alfa_true) 
                    erro_fast_m = np.mean(erro_fast)
                    erro_fast_s = np.std(erro_fast) 
                                                                                                                                                                     
                    a0.hist(erro_fast, bins = bins, range=(-erro_fast_s*3,+erro_fast_s*3), label=f"Standard FastCalo", histtype='step', density=True)         
                    cont += 1
                
                if len(erro_calib) > 0:
                                                                                                                                                                             
                    a0.hist(erro_calib , bins = bins, range=(-erro_fast_s*3,+erro_fast_s*3), label=f"{method}: {useRings_cenarios[i]}", histtype='step', density=True)
                     
        a0.set_title("Prediction Error Histogram")    
        a0.set_title(f"Normalized Error Histogram: $E_t$ in {(int(r[0]/1000), int(r[1]/1000))} GeV range")
        a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                        
        a0.set_xlabel(r'$(E_{true} - E_{calib})/E_{true}$', fontsize=16) 
        a0.set_ylabel(f'Probability Density $E_t$ in {(int(r[0]/1000), int(r[1]/1000))} GeV range', fontsize=16)   
                                                   
        plt.axvline(x=0, color='lightgrey', linestyle='--')
        plota_data_label(plt, full_data_text_label)
                                                                                                        
        plt.show()
    
    
    
                                 
    
    def list_of_list_to_extended_list(x):
        y=[]
        for i in range(len(x)):
            y.extend(x[i])
        return y
    
    
                                        
    fig, a0 = plt.subplots(figsize = (8, 7))
    standtard = 0
    for method in methods_list:        
        for i in useRings:

            print(f"#### General Normalized Error Histogram: {method}-{i} ####")
            
            ypred = results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["ypred"]           
            
            alfa_pred = list(ypred)
            print(f"=>Número de listas de alpha_pred  = {len(alfa_pred)}")
            alfa_pred = list_of_list_to_extended_list(alfa_pred)
            print(f"=>Tamanho de alpha_pred estendido = {len(alfa_pred)}")            
            print()

            if len(alfa_pred) > 0:
            
                alfa_true = list(results_DF_BDT_ETA_ET[(results_DF_BDT_ETA_ET["useRings"]==i) & (results_DF_BDT_ETA_ET["method"]==method)]["yte"])
                alfa_true = list_of_list_to_extended_list(alfa_true)
                
                print("alfa_pred: ", alfa_pred[0:5])
                print("method:", method, ", scenario:", i, ", len(alfa_pred):", len(alfa_pred),", len(alfa_true):", len(alfa_true))
                
                                                                    
                erro_calib = 1 - np.array(alfa_pred) / np.array(alfa_true)          
                erro_fast = 1 - 1 / np.array(alfa_true)
                
                erro_fast_s = max(mean(erro_calib)+stdev(erro_calib), mean(erro_fast)+stdev(erro_fast))
                
                bins = 100
                
                if len(erro_calib) > 0:                
                    if standtard == 0:
                                                                                                                                                                                          
                        a0.hist(erro_fast , bins = bins, range=(-erro_fast_s*3,+erro_fast_s*3), label=f"Standard Fast Calo: {useRings_cenarios[i]}", histtype='step', density=True)
                        standtard = 1
                        data_erros = [erro_fast]
                        data_labels = [f"Standard FastCalo"]
                    
                                                                                                                                                                             
                    a0.hist(erro_calib , bins = bins, range=(-erro_fast_s*3,+erro_fast_s*3), label=f"{method}: {useRings_cenarios[i]}", histtype='step', density=True)
                    data_erros.append(erro_calib)
                    data_labels.append(f"{method}: {useRings_cenarios[i]}")        
                
                print()            
                print(f"General Results: {method}, {useRings_cenarios[i]}")
                print("###########################")
                print()
                                                                               
                                                                
                                                               
            else:
                print("len alfa_pred = 0")
                print("###########################")
                print()
    
    a0.set_title("General Normalized Error Histogram")   
    a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                    
    a0.set_xlabel(r'$(E_{true} - E_{calib})/E_{true}$', fontsize=16) 
    a0.set_ylabel(r'Probability Density', fontsize=16)   
                                               
    plt.axvline(x=0, color='lightgrey', linestyle='--')
    plota_data_label(plt, full_data_text_label)        
                                                                                                       
    plt.show()           
            
    
    comparison_boxplot(data_erros, data_labels, full_data_text_label)


# Original notebook cell 248; function box_plot_comparison_geral
def box_plot_comparison_geral(data_ranges, data_alphas, data_algorithms, segment="ET", graph_prefix=''):
    
    
    
                                          
                                                            
                                      
    if segment=="ET":        
        
        
                                          
                                                     
                                                                                             
        
                                                            
                           
                                       
                             
                
        x_label = "$E_t$ (MeV)"
    
    if segment =="eta":
        
                                            
                                                
        
                                        
        
        x_label = "$|\eta|$"
    
    
    
    
    
                                         
    positions = [ [] for i in range(len(data_ranges)) ]
                                      
                                    
    
    for i in range(len(data_ranges)):
        for j in range(len(data_ranges[0])):
            positions[i].append(j*len(data_ranges) + i)  
    
    no_labels = []
    for j in range(len(data_ranges[0])):
        no_labels.append("")

             
   
    
                                  
    facecolor_list = ["C0", "orange", "green", "red", "lightblue", "lightgreen", "pink", "purple", "brown", "grey", "olive", "indigo", "paleturquoise", "darkslategray", "fuchsia", "darkred", "darkgreen", "darkblue", "darkgoldenrod", "darkcyan", "darkolivegreen", "lightcoral", "peru", "orangered", "royalblue", "navy"]
    fig, ax = plt.subplots(figsize = (8, 7)) 
    bp = [ [] for i in range(len(data_ranges)) ]
    boxes = []    
    data_algorithms_final = []
    posi = False
    for i in range(len(data_ranges)): 
    
                                                                                        
        
        if i>0: 
            list_labels = no_labels
        else:
            if segment=="eta": list_labels = data_ranges[0] 
            if segment=="ET":                                                   
                list_labels = []
                for j in data_ranges[0]:
                    list_labels.append( ( round(j[0]/1000, 1 ), round( j[1]/1000, 1) ) )                
        
                                                                                                                                                                         
        print("i: ", i, f", len(data_alphas[{i}]: ", len(data_alphas[i]), ", len(data_alphas[{i}][0]): ", len(data_alphas[i][0]))        
         
        
                                             
        if len(data_alphas[i])>0:
                                                                                                      
                                                                                                    
            
            bp[i] =  ax.boxplot(data_alphas[i], positions=positions[i], labels=list_labels,  
                 patch_artist=True, boxprops=dict(facecolor=facecolor_list[i]), showfliers=False,)        
            
            boxes.append(bp[i]["boxes"][0])

                                                          
            if posi==False:
                vertical_positions = positions[i]
                posi=True           
            
            data_algorithms_final.append(data_algorithms[i])
    
    print("data_algorithms_final:", data_algorithms_final)
    
    ax.legend(boxes, data_algorithms_final, loc='upper center', ncol=1, fontsize=10, frameon=False)    
                                                 
    ax.set_ylabel(r"$\alpha_{calib} = E_{true}/E_{calib}$", fontsize=16)    
    ax.set_xlabel(f"Bins : {x_label}", fontsize=16)
    ax.tick_params(axis='x', which='major', labelsize=12, rotation=10)
                                                                     
    
    ax.axhline(y=1, color='lightgrey', linestyle='--')    

                           
    for p in vertical_positions:
         ax.axvline(x=p-.5, color='lightgrey', linestyle='--')   

                              
                       

    vp = np.array(vertical_positions)
    vp_tick = vp+vp[1]/2-.5
    
    ax.set_xticks(vp_tick, labels=data_ranges[0])
    
    plota_data_label(plt, full_data_text_label)
                                                                                           
    plt.show()


# Original notebook cell 249; function boxplot_comparison_improved
def boxplot_comparison_improved(integrated_results_DF_BDT_ETA_ET, graph_prefix=''):
    
    
                        
    m = list(integrated_results_DF_BDT_ETA_ET["method"])
    r = list(integrated_results_DF_BDT_ETA_ET["useRings"])
    par_m_r = [ (m[i], r[i]) for i in range(len(m))]    
    set_par_m_r = set(par_m_r)
    # Determine model order using archived per-bin calibration-factor RMSE.
    # Keep the plotting geometry, fonts, colors, labels, and legend settings
    # unchanged; only the data order is made deterministic.
    def _archived_rmse(group_key):
        method,selection=group_key
        rows=integrated_results_DF_BDT_ETA_ET[
            (integrated_results_DF_BDT_ETA_ET['method']==method) &
            (integrated_results_DF_BDT_ETA_ET['useRings']==selection) &
            (integrated_results_DF_BDT_ETA_ET['eta_range']=='All')]
        if rows.empty:return float('-inf')
        yte=np.concatenate([np.asarray(a,dtype=float).reshape(-1) for a in rows.yte])
        yp=np.concatenate([np.asarray(a,dtype=float).reshape(-1) for a in rows.ypred])
        return float(np.sqrt(np.mean((yte-yp)**2))) if len(yte)==len(yp) and len(yte) else float('-inf')
    list_set_par_m_r = sorted(set_par_m_r,key=lambda key:(-_archived_rmse(key),str(key)))
                              
                                      
                                                
            

                                                         
        
    range_ETA = list(integrated_results_DF_BDT_ETA_ET[ integrated_results_DF_BDT_ETA_ET["eta_range"]!="All" ]["eta_range"].unique())
    range_ETA.sort()
                                                     
            
    
    alfa_ETA = []
    for eta in range_ETA:
        alfa_ETA_0 = list( integrated_results_DF_BDT_ETA_ET[ integrated_results_DF_BDT_ETA_ET["eta_range"]==eta ]["yte"] ) 
        alfa_ETA_1 = []
        for a in alfa_ETA_0:
            alfa_ETA_1.extend(a)
        alfa_ETA.append(alfa_ETA_1)    
    
                                              
    
                      
    data_algorithms = ["Standard FastCalo"]
    data_ranges = [range_ETA]
    data_alphas = [[list(alfa_ETA[i]) for i in range(len(alfa_ETA))]]
    
    
                                           
                                                      
    
    
    new_algorithms = []
    new_ranges = []
    new_alphas = []   
    
                                
                                     
        
    for lmr in range(len(list_set_par_m_r)):
        method = list_set_par_m_r[lmr][0]
        i = list_set_par_m_r[lmr][1]
        
                    
        new_ranges.append(range_ETA)
        
                    
        new_algorithms.append(f"{method}: {useRings_cenarios[i]}") 
        
        new_alphas_1 =[]
        for eta in range_ETA:                     
            
                    
            ypred = list (integrated_results_DF_BDT_ETA_ET[(integrated_results_DF_BDT_ETA_ET["useRings"]==i) & 
                                                     (integrated_results_DF_BDT_ETA_ET["eta_range"]==eta) & 
                                                     (integrated_results_DF_BDT_ETA_ET["method"]==method)]["ypred"] )[0]
            new_alphas_1.append( ypred )
                                              
                                            
                                                        
                                        
                                              
        
        new_alphas.append(new_alphas_1)       
  
    
                                                     
                                                             
                                                         
                                                          
    
    
    data_algorithms.extend(new_algorithms)
                           
    data_ranges.extend(new_ranges)
                       
    data_alphas.extend(new_alphas)
                            
    
                                                                                
                                                                        
                                                       
            
    

                          
    box_plot_comparison_geral(data_ranges, data_alphas, data_algorithms, segment="eta", graph_prefix=graph_prefix)
    
                                                             
    
    range_ET = list(integrated_results_DF_BDT_ETA_ET[ integrated_results_DF_BDT_ETA_ET["et_range"]!="All" ]["et_range"].unique())
    range_ET.sort()
                                
            
    
    alfa_ET = []
    for et in range_ET:
        alfa_ET_0 = list( integrated_results_DF_BDT_ETA_ET[ integrated_results_DF_BDT_ETA_ET["et_range"]==et ]["yte"] ) 
        alfa_ET_1 = []
        for a in alfa_ET_0:
            alfa_ET_1.extend(a)
        alfa_ET.append(alfa_ET_1)

                    
    data_algorithms = ["Standard FastCalo"]
    data_ranges = [range_ET]
    data_alphas = [[list(alfa_ET[i]) for i in range(len(alfa_ET))]]    
    
                                          
    
    new_algorithms = []
    new_ranges = []
    new_alphas = [] 
    
                                 
                                   
        
    for lmr in range(len(list_set_par_m_r)):
        method = list_set_par_m_r[lmr][0]
        i = list_set_par_m_r[lmr][1]
        
                    
        new_ranges.append(range_ET)
        
                    
        new_algorithms.append(f"{method}: {useRings_cenarios[i]}") 
        
        new_alphas_1 =[]
        for et in range_ET:                     
            
                    
            ypred = list (integrated_results_DF_BDT_ETA_ET[(integrated_results_DF_BDT_ETA_ET["useRings"]==i) & 
                                                     (integrated_results_DF_BDT_ETA_ET["et_range"]==et) & 
                                                     (integrated_results_DF_BDT_ETA_ET["method"]==method)]["ypred"] )[0]
            new_alphas_1.append( ypred )
                                              
                                            
                                                        
                                        
                                              
        
        new_alphas.append(new_alphas_1) 
        
       
    
    data_algorithms.extend(new_algorithms)
                           
    data_ranges.extend(new_ranges)
                       
    data_alphas.extend(new_alphas)
                            
     
                                                                                
                                                                        
                                                       
            

                          
    box_plot_comparison_geral(data_ranges, data_alphas, data_algorithms, segment="ET", graph_prefix=graph_prefix)


# Original notebook cell 273; function plota_energy_alpha_scatter_por_metodo
def plota_energy_alpha_scatter_por_metodo(MLR_noSEG_Full_DF, full_data_text_label):
    
    metodos = MLR_noSEG_Full_DF['method-selection'].unique()

    for m in metodos:   
                                                                                 
        

        metodo_df = MLR_noSEG_Full_DF[ MLR_noSEG_Full_DF['method-selection'] == m ]
        E_true = metodo_df['mc_match_et']
        E_fast = metodo_df['cluster_truthet']
        E_calib = metodo_df['calib_et']

        alpha_calib = metodo_df['ypred']
        alpha_true = metodo_df['yte']

                             
        fig = plt.figure(figsize = (8, 7))
        
        plt.scatter(E_true, E_fast, marker='.', label = r"$E_\text{fast}$")        
        plt.scatter(E_true, E_calib, marker='.', label = r"$E_\text{calib}$"+f": {m}")
        plt.ylabel(r"$E_\text{calib}$ (MeV)", fontsize=16)
        plt.xlabel(r"$E_\text{true}$ Monte Carlo (MeV)", fontsize=16)
        plt.xticks(rotation=0, fontsize=12)        
        
        ax = plt.subplot()    
        xmin, xmax, ymin, ymax = plt.axis()
        dx = (xmax-xmin)/20
        dy = (ymax-ymin)/20
        plt.text(xmin+dx, ymax-dy, full_data_text_label)
        
        line_x = [i for i in range(0,int(xmax)) ]
        plt.plot(line_x, '--', color='black')
        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.85), frameon=False, fontsize=13) 
        
        plt.show()

        
                                    
        fig = plt.figure(figsize = (8, 7))
        
        plt.scatter(alpha_true, alpha_calib, marker='.', label = r"$\alpha_\text{calib}$"+f": {m}")        
                                                                                       
        plt.ylabel(r"$\alpha_\text{calib}$"+f": {m}", fontsize=16)
        plt.xlabel(r"$\alpha_\text{true}$ Monte Carlo", fontsize=16)
        plt.xticks(rotation=0, fontsize=12)        
        
        ax = plt.subplot()    
        xmin, xmax, ymin, ymax = plt.axis()
        dx = (xmax-xmin)/20
        dy = (ymax-ymin)/20
                                                         
        
        line_x = [i for i in range(0,int(xmax)) ]
        plt.plot(line_x, '--', color='black')
                                                                                            
        
        plt.show()


                                    
        fig = plt.figure(figsize = (8, 7))
        bins=50
        deviation = np.std(alpha_true)
        a_m = np.mean(alpha_true)
        plt.hist(alpha_true, bins=bins, range=(a_m-deviation*3, a_m+deviation*3) ,density=True, histtype='step', label=r"$\alpha_\text{true}$", linewidth=2)
        plt.hist(alpha_calib, bins=bins, range=(a_m-deviation*3, a_m+deviation*3) ,density=True, histtype='step', label=r"$\alpha_\text{calib}$"+f": {m}", linewidth=2)
                                                                                                                        
        plt.ylabel(r"Probability Density", fontsize=16)
        plt.xlabel(r"$\alpha$ (Calibration Factor)", fontsize=16)
        plt.xticks(rotation=0, fontsize=12)
        
        plt.legend(loc='upper right', frameon=False, fontsize=13) 
                                                       

        ax = plt.subplot()    
        xmin, xmax, ymin, ymax = plt.axis()
        dx = (xmax-xmin)/20
        dy = (ymax-ymin)/20
        plt.text(xmin+dx, ymax-dy, full_data_text_label)
         
        plt.show()


        

                                         
        fig = plt.figure(figsize = (8, 7))
        bins=50
        erro = np.array(alpha_true)-np.array(alpha_calib)
        deviation = np.std(erro)
        plt.hist(erro, bins=bins, range=(-deviation*3, +deviation*3) ,density=True, histtype='step', label=r"$\varepsilon_\alpha$"+f": {m}", linewidth=2)
        
                                                                                                                        
        plt.ylabel(r"Probability Density", fontsize=16)
        plt.xlabel(r"$\varepsilon_\alpha = \alpha_\text{true} - \alpha_\text{calib}$", fontsize=16)
        plt.xticks(rotation=0, fontsize=12)
        
        plt.legend(loc='upper right', frameon=False, fontsize=13) 
        plt.axvline(x=0, color='grey', linestyle='--')

        ax = plt.subplot()    
        xmin, xmax, ymin, ymax = plt.axis()
        dx = (xmax-xmin)/20
        dy = (ymax-ymin)/20
        plt.text(xmin+dx, ymax-dy, full_data_text_label)
         
        plt.show()


                                              
        fig = plt.figure(figsize = (8, 7))
        
        plt.scatter(E_true, alpha_true, marker='.', label = r"$\alpha_\text{true}$") 
        plt.scatter(E_true, alpha_calib, marker='.', label = r"$\alpha_\text{calib}$"+f": {m}") 
                                                                                       
        plt.ylabel(r"$\alpha_\text{calib}$"+f": {m}", fontsize=16)
        plt.xlabel(r"$E_\text{true}$ Monte Carlo", fontsize=16)
        plt.xticks(rotation=0, fontsize=12)        
        
        ax = plt.subplot()    
        xmin, xmax, ymin, ymax = plt.axis()
        dx = (xmax-xmin)/20
        dy = (ymax-ymin)/20
        plt.text(xmin+dx, ymax-dy, full_data_text_label)
        
        line_x = [i for i in range(0,int(xmax)) ]
                                              
        plt.legend(loc='upper right', frameon=False, fontsize=13) 
        
        plt.show()


    

    return None


# Original notebook cell 627; function integrated_plots_in_ETA_geral
def integrated_plots_in_ETA_geral(results_BDT_ETA, useRings_cenarios):
    
    ur = list(results_BDT_ETA.keys())[0]
    eta_range = results_BDT_ETA[ur]["eta_range"]
                                                                                  
    fmt_mark = ['v', 's', 'p', '*', 'P', '^', '<', '>', '1', '2', '3', '4', '8', 'h', '+','x','d','.']
        
    uRs = list( results_BDT_ETA.keys() )                          
    
                  
    etaXPos_ETA = []
    etaXDelta_ETA = []
    for r in eta_range:       
        pto_med = (r[1]+r[0])/2
        diff = (r[1]-r[0])/2
        etaXPos_ETA.append(pto_med)
        etaXDelta_ETA.append(diff)
    

                 
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (8, 7))  
    
    IQRraw = results_BDT_ETA[ur]["IQRraw"]
    a0.errorbar(etaXPos_ETA, IQRraw, xerr=etaXDelta_ETA, fmt='o', label=f"Standard FastCalo")
    a1.errorbar(etaXPos_ETA,np.ones(len(IQRraw)), xerr=etaXDelta_ETA, fmt='o', label=f"Standard FastCalo")
    
    for i in uRs: 
        IQRpred = results_BDT_ETA[i]["IQRpred"]
        a0.errorbar(etaXPos_ETA, IQRpred, xerr=etaXDelta_ETA, fmt=fmt_mark[i], label=f"{useRings_cenarios[i]}") 
        a1.errorbar(etaXPos_ETA, np.divide(np.array(IQRpred),np.array(IQRraw)), xerr=etaXDelta_ETA, fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
    
    a0.set_title('IQR Integrated in all $E_t$ bins')
    a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                   
    a0.set_ylabel(r'IQR $(E_{true} / )$', fontsize=16)
                                                                     
    
                                                             
    a1.set_ylabel('Ratio w.r.t. No BDT')
                                                   
    a1.set_xlabel(r'Bins : $|\eta|$')
                                                                         
                                                                                     
    plota_data_label(plt, full_data_text_label)
    plt.show()
    
    
    
    
                       
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (8, 7))
    MDraw = results_BDT_ETA[ur]["MDraw"]
    a0.errorbar(etaXPos_ETA,1-np.array(MDraw),xerr=etaXDelta_ETA,fmt='o',label=f"Standard FastCalo")
    a1.errorbar(etaXPos_ETA,np.divide(1-np.array(MDraw),1-np.array(MDraw)),xerr=etaXDelta_ETA,fmt='o',label=f"Standard FastCalo")
    
    for i in uRs: 
        MDpred = results_BDT_ETA[i]["MDpred"]
        a0.errorbar(etaXPos_ETA,1-np.array(MDpred),xerr=etaXDelta_ETA,fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
        a1.errorbar(etaXPos_ETA,np.divide(1-np.array(MDpred),1-np.array(MDraw)),xerr=etaXDelta_ETA,fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
    
    a0.set_title('Mean Shift Integrated in all $E_t$ bins')
    a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                   
    a0.set_ylabel(r'Mean Shift $(E_{true} / )$', fontsize=16)

                                                             
                                               
    a1.set_ylabel('Ratio w.r.t. No BDT')    
    a1.set_xlabel(r'Bins : $|\eta|$', fontsize=16)
                                                                                 
    plota_data_label(plt, full_data_text_label)
    plt.show()
    
    
    
                                 
    
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize = (8, 7))    
    a0.errorbar(etaXPos_ETA,np.zeros(len(etaXPos_ETA)),xerr=etaXDelta_ETA,fmt='o',label="Standard FastCalo")
    a1.errorbar(etaXPos_ETA,np.zeros(len(etaXPos_ETA)),xerr=etaXDelta_ETA,fmt='o',label="Standard FastCalo")
    
    for i in uRs: 
        e_mae = results_BDT_ETA[i]["e_mae"]
        e_rmse = results_BDT_ETA[i]["e_rmse"]
        a0.errorbar(etaXPos_ETA,e_mae,xerr=etaXDelta_ETA,fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
        a1.errorbar(etaXPos_ETA,e_rmse,xerr=etaXDelta_ETA,fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
    
    a0.set_title('Error Integrated in all $E_t$ bins')
    a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                   
    a0.set_ylabel(r'MAE', fontsize=16)   

                                                                         
                                                             
                                                   
    a1.set_ylabel(r'RMSE', fontsize=16)
    a1.set_xlabel(r'$|\eta|$', fontsize=16)
                                                 
                                                                              
    plota_data_label(plt, full_data_text_label)
    plt.show()


# Original notebook cell 631; function integrated_plots_in_ET_geral
def integrated_plots_in_ET_geral(results_BDT_ET, useRings_cenarios):
    
    ur = list(results_BDT_ET.keys())[0]
    et_range = results_BDT_ET[ur]["eta_range"]
                                                                                  
    fmt_mark = ['v', 's', 'p', '*', 'P', '^', '<', '>', '1', '2', '3', '4', '8', 'h', '+','x','d','.']
    
    uRs = list(results_BDT_ET.keys())
    
                  
    etaXPos_ET = []
    etaXDelta_ET = []
    for r in et_range:       
        pto_med = (r[1]+r[0])/2
        diff = (r[1]-r[0])/2
        etaXPos_ET.append(pto_med)
        etaXDelta_ET.append(diff)
    

                 
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (8, 7))  
    
    IQRraw = results_BDT_ET[ur]["IQRraw"]
    a0.errorbar(etaXPos_ET, IQRraw, xerr=etaXDelta_ET, fmt='o', label=f"Standard FastCalo")
    a1.errorbar(etaXPos_ET,np.ones(len(IQRraw)), xerr=etaXDelta_ET, fmt='o', label=f"Standard FastCalo")
    
    for i in uRs: 
        IQRpred = results_BDT_ET[i]["IQRpred"]
        a0.errorbar(etaXPos_ET, IQRpred, xerr=etaXDelta_ET, fmt=fmt_mark[i], label=f"{useRings_cenarios[i]}") 
        a1.errorbar(etaXPos_ET, np.divide(np.array(IQRpred),np.array(IQRraw)), xerr=etaXDelta_ET, fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
    
    a0.set_title('IQR Integrated in all $|\eta|$ bins')
    a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                   
    a0.set_ylabel(r'IQR $(E_{true} / )$', fontsize=16)
    a0.set_xscale('log')
                                                                     
    
                                                             
    a1.set_ylabel('Ratio w.r.t. No BDT')
                                                   
    a1.set_xlabel(r'Bins : $E_t$')
    a1.set_xscale('log')
                                                                         
                                                                                   
    plota_data_label(plt, full_data_text_label)
    plt.show()
    
    
    
    
                       
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize = (8, 7))
    MDraw = results_BDT_ET[ur]["MDraw"]
    a0.errorbar(etaXPos_ET,1-np.array(MDraw),xerr=etaXDelta_ET,fmt='o',label=f"Standard FastCalo")
    a1.errorbar(etaXPos_ET,np.divide(1-np.array(MDraw),1-np.array(MDraw)),xerr=etaXDelta_ET,fmt='o',label=f"Standard FastCalo")
    
    for i in uRs: 
        MDpred = results_BDT_ET[i]["MDpred"]
        a0.errorbar(etaXPos_ET,1-np.array(MDpred),xerr=etaXDelta_ET,fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
        a1.errorbar(etaXPos_ET,np.divide(1-np.array(MDpred),1-np.array(MDraw)),xerr=etaXDelta_ET,fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
    
    a0.set_title('Mean Shift Integrated in all $|\eta|$ bins')
    a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                   
    a0.set_ylabel(r'Mean Shift $(E_{true} / )$', fontsize=16)
    a0.set_xscale('log')

                                                             
                                               
    a1.set_ylabel('Ratio w.r.t. No BDT')    
    a1.set_xlabel(r'Bins : $E_t$', fontsize=16)
    a1.set_xscale('log')
                                                                                 
    plota_data_label(plt, full_data_text_label)
    plt.show()
    
    
    
                                 
    
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize = (8, 7))    
    a0.errorbar(etaXPos_ET,np.zeros(len(etaXPos_ET)),xerr=etaXDelta_ET,fmt='o',label="Standard FastCalo")
    a1.errorbar(etaXPos_ET,np.zeros(len(etaXPos_ET)),xerr=etaXDelta_ET,fmt='o',label="Standard FastCalo")
    
    for i in uRs: 
        e_mae = results_BDT_ET[i]["e_mae"]
        e_rmse = results_BDT_ET[i]["e_rmse"]
        a0.errorbar(etaXPos_ET,e_mae,xerr=etaXDelta_ET,fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
        a1.errorbar(etaXPos_ET,e_rmse,xerr=etaXDelta_ET,fmt=fmt_mark[i],label=f"{useRings_cenarios[i]}")
    
    a0.set_title('Error Integrated in all $|\eta|$ bins')
    a0.legend(loc='upper right', frameon=False, fontsize=12)
                                                   
    a0.set_ylabel(r'MAE', fontsize=16)  
    a0.set_xscale('log')

                                                                         
                                                             
                                                   
    a1.set_ylabel(r'RMSE', fontsize=16)
    a1.set_xlabel(r'$E_t$', fontsize=16)
    a1.set_xscale('log')
                                                 
                                                                              
    plota_data_label(plt, full_data_text_label)
    plt.show()


# Original notebook cell 649; function boxplot_comparison
def boxplot_comparison(integrated_results_DF_BDT_ETA_ET):

                                                         
               
    list_algorithms = list(set(integrated_results_DF_BDT_ETA_ET["useRings"]))    
                                

    data_algorithms = ["Standard FastCalo"]    
    new_algorithms = [ useRings_cenarios[i] for i in list_algorithms]
    data_algorithms.extend(new_algorithms)
                           

            
    data_ranges = [range_ETA]
    new_ranges = [ list(set(integrated_results_DF_BDT_ETA_ET[ (integrated_results_DF_BDT_ETA_ET["useRings"]==i) & (integrated_results_DF_BDT_ETA_ET["eta_range"]!="All") ]["eta_range"])) for i in list_algorithms ]
    data_ranges.extend(new_ranges)
                       

            
    data_alphas = [[list(alfa_ETA[i]) for i in range(len(alfa_ETA))]]
    new_alphas = [ list(integrated_results_DF_BDT_ETA_ET[ (integrated_results_DF_BDT_ETA_ET["useRings"]==i) & (integrated_results_DF_BDT_ETA_ET["eta_range"]!="All") ]["ypred"]) for i in list_algorithms ]
    data_alphas.extend(new_alphas)
                            

                          
    box_plot_comparison_geral(data_ranges, data_alphas, data_algorithms, segment="eta")
    
                                                             

            
    data_ranges = [range_ET]
    new_ranges = [ list(set(integrated_results_DF_BDT_ETA_ET[ (integrated_results_DF_BDT_ETA_ET["useRings"]==i) & (integrated_results_DF_BDT_ETA_ET["et_range"]!="All") ]["et_range"])) for i in list_algorithms ]
    data_ranges.extend(new_ranges)
                       

            
    data_alphas = [[list(alfa_ET[i]) for i in range(len(alfa_ET))]]
    new_alphas = [ list(integrated_results_DF_BDT_ETA_ET[ (integrated_results_DF_BDT_ETA_ET["useRings"]==i) & (integrated_results_DF_BDT_ETA_ET["et_range"]!="All") ]["ypred"]) for i in list_algorithms ]
    data_alphas.extend(new_alphas)
                            

                          
    box_plot_comparison_geral(data_ranges, data_alphas, data_algorithms, segment="ET")


# Original notebook cell 755; function ordenar_dataframe_por_lista
def ordenar_dataframe_por_lista(df, lista_ordenada, coluna='data_labels'):
    df_ordenado = df.copy()
    
    df_ordenado[coluna] = pd.Categorical(
        df_ordenado[coluna],
        categories=lista_ordenada,
        ordered=True
    )
    
    df_ordenado = df_ordenado.sort_values(by=coluna).reset_index(drop=True)
    
    return df_ordenado


# Original notebook cell 755; function plota_histograma_erros
def plota_histograma_erros(All_NoSeg_erros_data, n=10):

    numero_de_resultados = n
    
    m_data_erros = list(All_NoSeg_erros_data['data_erros'])[0:numero_de_resultados]
    m_data_labels = list(All_NoSeg_erros_data['data_labels'])[0:numero_de_resultados]
        
    sdf = All_NoSeg_erros_data[ All_NoSeg_erros_data['useRings'] == 0 ]
    s_data_erros = list(sdf['data_erros'])
    s_data_labels = list(sdf['data_labels'])
        
    m_data_erros.extend(s_data_erros)
    m_data_labels.extend(s_data_labels)
    
    
    comparison_hist_boxplot_erros(m_data_erros, 
                                  m_data_labels, 
                                  full_data_text_label, xtextlabel=r"$E_{true} - E_{calib}$ (MeV)")


# Original notebook cell 755; function ordena_e_plota_histogramas
def ordena_e_plota_histogramas(ordenador, df1_ordenado, df1_coluna_ordenada, df2_a_ordenar, df2_coluna, n = 10):
    
    lista_ordenada_por_d0 = list(df1_ordenado.sort_values(ordenador, ascending=True)[df1_coluna_ordenada])
    lista_ordenada_por_d0_ = []
    for i in range(len(lista_ordenada_por_d0)):    
        if lista_ordenada_por_d0[i] != 'Standard FastCalo:0':
            lista_ordenada_por_d0_.append( lista_ordenada_por_d0[i].replace('no-Bin:','no-Bin: ') )
        else:
            lista_ordenada_por_d0_.append( lista_ordenada_por_d0[i].replace('FastCalo:0','FastCalo') )
    lista_ordenada_por_d0 = lista_ordenada_por_d0_
    
                                 
    df_ordenado_por_d0 = ordenar_dataframe_por_lista(df2_a_ordenar, lista_ordenada_por_d0, coluna=df2_coluna)
    plota_histograma_erros(df_ordenado_por_d0, n=10)
    return df_ordenado_por_d0


# Original notebook cell 782; function plotar_resolucao
def plotar_resolucao(Et, Ef, Ec, Ef_label, Ec_label, n_bins=20):
    """
    Plota a resolução relativa σ/E para Ef (sem calibração) e Ec (com calibração)
    em função da energia verdadeira Et.

    Parâmetros:
    - Et: array-like, energia verdadeira
    - Ef: array-like, energia medida sem calibração
    - Ec: array-like, energia calibrada
    - n_bins: número de divisões da energia (default: 20) 

    Saída:
    - Gráfico matplotlib com curva de resolução relativa
    """

    Et = np.array(Et)
    Ef = np.array(Ef)
    Ec = np.array(Ec) 

    bins = np.linspace(np.min(Et), np.max(Et), n_bins + 1)
    bin_centers = []
    resolucao_ef = []
    resolucao_ec = [] 

    for i in range(n_bins):
                                          
        mask = (Et >= bins[i]) & (Et < bins[i + 1])
        if np.sum(mask) > 10:                                     
            Et_bin = Et[mask]
            Ef_bin = Ef[mask]
            Ec_bin = Ec[mask] 

                                                                            
            sigma_ef = np.std(Ef_bin - Et_bin, ddof=1)
            sigma_ec = np.std(Ec_bin - Et_bin, ddof=1)
            mean_Et = np.mean(Et_bin)

            resolucao_ef.append(sigma_ef / mean_Et)
            resolucao_ec.append(sigma_ec / mean_Et)
            bin_centers.append(mean_Et) 

          
    plt.figure(figsize=(8, 7))
    plt.plot(bin_centers, resolucao_ef, label=Ef_label, marker='o', linewidth=2)
    plt.plot(bin_centers, resolucao_ec, label=Ec_label, marker='s', linewidth=2)
    plt.xlabel(r'True Transverse Energy, $E_\text{true}$ (MeV)', fontsize=16)
    plt.ylabel(r'Energy Resolution ($\sigma_E / E$)', fontsize=16)
                                                             
    plt.legend(loc='upper right', frameon=False, fontsize=12)
                                                                           
                       
    plt.show()


# Original notebook cell 782; function plotar_resolucao_multipla
def plotar_resolucao_multipla(Et, Ef, Ec, Ef_label, Ec_label, n_bins=20):
    """
    Plota a resolução relativa σ/E para Ef (sem calibração) e Ec (com calibração)
    em função da energia verdadeira Et.

    Parâmetros:
    - Et: array-like, energia verdadeira
    - Ef: array-like, energia medida sem calibração
    - Ec: array-like, energia calibrada
    - n_bins: número de divisões da energia (default: 20) 

    Saída:
    - Gráfico matplotlib com curva de resolução relativa
    """

    Et = np.array(Et)
    Ef = np.array(Ef)
                               
                       

    bins = np.linspace(np.min(Et), np.max(Et), n_bins + 1)
    bin_centers = []
    resolucao_ef = []
    

                     
    for i in range(n_bins):
                                          
        mask = (Et >= bins[i]) & (Et < bins[i + 1])
        if np.sum(mask) > 10:                                     
            Et_bin = Et[mask]
            Ef_bin = Ef[mask]            

                                                                            
            sigma_ef = np.std(Ef_bin - Et_bin)            
            mean_Et = np.mean(Et_bin)

            resolucao_ef.append(sigma_ef / mean_Et)            
            bin_centers.append(mean_Et) 
   
                                                                                      
    
                                                         
    plt.figure(figsize=(8, 7))
    plt.plot(bin_centers, resolucao_ef, label=Ef_label, marker='o', linewidth=2)    
                                                                     

    markers_cal = ['s', '^', 'v', 'D', 'P', 'X', '*', '<', '>', 'h', 'H', 'd', 'p', '+', 'x']

    
                                
    for j in range(len(Ec)):
        
        ec = np.array(Ec[j])
                                   
        ecl = Ec_label[j]        

                         
        resolucao_ec = [] 
        for i in range(n_bins):
                                              
            mask = (Et >= bins[i]) & (Et < bins[i + 1])
            if np.sum(mask) > 10:                                     
                Et_bin = Et[mask]            
                ec_bin = ec[mask] 
    
                                                                                            
                sigma_ec = np.std(ec_bin - Et_bin)
                mean_Et = np.mean(Et_bin)
                
                resolucao_ec.append(sigma_ec / mean_Et)
                                             
        
        
                                                                                          
        
        plt.plot(bin_centers, resolucao_ec, label=ecl, marker=markers_cal[j], linewidth=2)
                                                                    
    
    plt.xlabel(r'True Transverse Energy, $E_\text{true}$ (MeV)', fontsize=16)
    plt.ylabel(r'Energy Resolution ($\sigma_E / E$)', fontsize=16)
                                                             
    plt.legend(loc='upper right', frameon=False, fontsize=12)
                                                                           
                       
    plt.show()


# Original notebook cell 782; function calcular_resolucao_com_incerteza
def calcular_resolucao_com_incerteza(E_medida, E_true):
    erro = E_medida - E_true
    sigma = np.std(erro, ddof=1)
    mu = np.mean(E_true)
    n = len(E_true) 

                  
    delta_sigma = sigma / np.sqrt(2 * (n - 1))
    delta_mu = sigma / np.sqrt(n)
    R = sigma / mu
    delta_R = R * np.sqrt((delta_sigma / sigma) ** 2 + (delta_mu / mu) ** 2)
    return R, delta_R


# Original notebook cell 782; function calcular_ganho_com_incerteza
def calcular_ganho_com_incerteza(Ef, Ec, Et):
    R_ef, dR_ef = calcular_resolucao_com_incerteza(Ef, Et)
    R_ec, dR_ec = calcular_resolucao_com_incerteza(Ec, Et)

    G = (R_ef - R_ec) / R_ef

                        
    dG = np.sqrt((R_ec / R_ef**2 * dR_ef) ** 2 + (dR_ec / R_ef) ** 2)

    return G, dG, R_ef, dR_ef, R_ec, dR_ec


# Original notebook cell 783; function _iqr
def _iqr(x: np.ndarray) -> float:
    q75 = np.nanpercentile(x, 75)
    q25 = np.nanpercentile(x, 25)
    return q75 - q25


# Original notebook cell 783; function _make_bin_edges
def _make_bin_edges(Et: np.ndarray, n_bins: int, strategy: str = "quantile") -> np.ndarray:
    Et = np.asarray(Et)
    Et = Et[~np.isnan(Et)]
    if strategy == "quantile":
        edges = np.quantile(Et, np.linspace(0, 1, n_bins + 1))
                                                                                   
        edges = np.unique(edges)
        if len(edges) < 2:
            raise ValueError("Binning por quantil falhou (edges repetidos). Tente 'width'.")
    elif strategy == "width":
        edges = np.linspace(np.nanmin(Et), np.nanmax(Et), n_bins + 1)
    else:
        raise ValueError("strategy deve ser 'quantile' ou 'width'.")
    return edges


# Original notebook cell 783; function compute_resolution_by_energy
def compute_resolution_by_energy(
    Et_true: np.ndarray,
    methods: Dict[str, np.ndarray],
    n_bins: int = 20,
    bin_strategy: str = "quantile",
    min_count: int = 30,
    robust: bool = False,
    use_full_SE_for_R: bool = True,
    bootstrap_B: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Calculates per-bin relative energy resolution and uncertainty using the original experimental method."""
    Et_true = np.asarray(Et_true)

                                                  
    mask_valid = ~np.isnan(Et_true)
    for m in methods.values():
        mask_valid &= ~np.isnan(m)
    Et_true = Et_true[mask_valid]
    methods = {k: np.asarray(v)[mask_valid] for k, v in methods.items()}

    edges = _make_bin_edges(Et_true, n_bins, bin_strategy)
    K = len(edges) - 1

    out_rows = []
    for k in range(K):
        left, right = edges[k], edges[k + 1]
                                           
        if k < K - 1:
            sel = (Et_true >= left) & (Et_true < right)
        else:
            sel = (Et_true >= left) & (Et_true <= right)

        n = int(np.sum(sel))
        if n < min_count:
            continue

        Et_bin = Et_true[sel]
        mean_E = float(np.mean(Et_bin))
        var_E = float(np.var(Et_bin, ddof=1)) if n > 1 else np.nan

        for name, E_meas in methods.items():
            E_bin = E_meas[sel]
            err = E_bin - Et_bin

            if robust:
                sigma = _iqr(err) / 1.349
                sigma_se = np.nan                                                             
            else:
                                      
                sigma = float(np.std(err, ddof=1))
                sigma_se = sigma / math.sqrt(2 * (n - 1)) if n > 1 else np.nan

            R = sigma / mean_E if mean_E != 0 else np.nan

            if use_full_SE_for_R and (not robust):
                                                                                             
                                                                  
                                                                            
                term_sigma = (sigma_se / mean_E) ** 2 if (mean_E and not np.isnan(sigma_se)) else 0.0
                term_E = (sigma**2 / (mean_E**4)) * (var_E / n) if (mean_E and not np.isnan(var_E)) else 0.0
                R_se = math.sqrt(term_sigma + term_E)
            else:
                                                  
                R_se = R / math.sqrt(2 * (n - 1)) if (n > 1 and not robust) else np.nan

                                                              
            if bootstrap_B is not None and bootstrap_B > 0:
                rng = np.random.default_rng(12345 + k)
                R_star = []
                idx = np.arange(n)
                for _ in range(bootstrap_B):
                    b = rng.choice(idx, size=n, replace=True)
                    err_b = (E_bin[b] - Et_bin[b])
                    mean_E_b = np.mean(Et_bin[b])
                    if robust:
                        sigma_b = _iqr(err_b) / 1.349
                    else:
                        sigma_b = np.std(err_b, ddof=1) if n > 1 else np.nan
                    R_b = sigma_b / mean_E_b if mean_E_b != 0 else np.nan
                    R_star.append(R_b)
                R_star = np.asarray(R_star)
                R_lo, R_hi = np.nanpercentile(R_star, [2.5, 97.5])
            else:
                R_lo, R_hi = np.nan, np.nan

            out_rows.append({
                "method": name,
                "bin_left": left,
                "bin_right": right,
                "bin_center": mean_E,
                "n": n,
                "sigma": sigma,
                "sigma_se": sigma_se,
                "R": R,
                "R_se": R_se,
                "R_lo": R_lo,
                "R_hi": R_hi,
            })

                                                 
    out = {k: np.array([row[k] for row in out_rows], dtype=object)
           for k in out_rows[0].keys()} if out_rows else {}
    return out


# Original notebook cell 783; function plot_resolution_curves
def plot_resolution_curves(
    res_dict: Dict[str, np.ndarray],
    methods_order: Optional[List[str]] = None,
    show_errorbars: bool = True,
    use_bootstrap_ci: bool = False,
    ax=None,
    xlabel: str = r"True Transverse Energy, $E_{\rm true}$ (MeV)",
    ylabel: str = r"Energy Resolution ($\sigma_E / E$)"    
):
    """Plots relative-energy-resolution curves and uncertainty bars using the original figure layout."""
    if not res_dict:
        raise ValueError("res_dict vazio. Rode compute_resolution_by_energy primeiro.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    methods = np.unique(res_dict["method"])
    if methods_order:
        methods = [m for m in methods_order if m in methods]
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "H", "d", "p", "+", "x"]

    for idx_m, m in enumerate(methods):
        mask_m = (res_dict["method"] == m)
        x = res_dict["bin_center"][mask_m].astype(float)
        y = res_dict["R"][mask_m].astype(float)

        if show_errorbars:
            if use_bootstrap_ci and not np.isnan(res_dict["R_lo"][mask_m]).all():
                yerr = np.vstack([
                    y - res_dict["R_lo"][mask_m].astype(float),
                    res_dict["R_hi"][mask_m].astype(float) - y
                ])
            else:
                yerr = res_dict["R_se"][mask_m].astype(float)
            ax.errorbar(x, y, yerr=yerr, linestyle='-', fmt=markers[idx_m % len(markers)],
                        capsize=3, linewidth=2, label=m)
        else:
            ax.plot(x, y, linestyle='-', marker=markers[idx_m % len(markers)], linewidth=2, label=m)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    ax.legend(loc="upper right", frameon=False, fontsize=12)    
                                                           
    plt.show()
    return ax


# Original notebook cell 783; function global_resolution_with_uncertainty
def global_resolution_with_uncertainty(E_meas: np.ndarray, E_true: np.ndarray) -> Tuple[float, float]:
    """
    R = sigma(e) / mean(E_true), com SE por propagação aproximada.
    """
    E_true = np.asarray(E_true)
    err = np.asarray(E_meas) - E_true
    n = np.sum(~np.isnan(err) & ~np.isnan(E_true))
    if n < 2:
        return np.nan, np.nan
    sigma = np.std(err, ddof=1)
    meanE = np.mean(E_true)
    varE = np.var(E_true, ddof=1)
    R = sigma / meanE
    sigma_se = sigma / math.sqrt(2 * (n - 1))
                     
    term_sigma = (sigma_se / meanE) ** 2
    term_E = (sigma**2 / (meanE**4)) * (varE / n)
    R_se = math.sqrt(term_sigma + term_E)
    return R, R_se


# Original notebook cell 783; function resolution_gain_with_uncertainty
def resolution_gain_with_uncertainty(Ef: np.ndarray, Ec: np.ndarray, Et: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Computes the relative resolution gain and its propagated uncertainty."""
    R_fast, dR_fast = global_resolution_with_uncertainty(Ef, Et)
    R_cal, dR_cal = global_resolution_with_uncertainty(Ec, Et)
    G = 1.0 - (R_cal / R_fast)
                                                                    
                                                                                  
    if np.any(np.isnan([R_fast, dR_fast, R_cal, dR_cal])) or R_fast == 0:
        dG = np.nan
    else:
        dG = math.sqrt((R_cal / (R_fast**2) * dR_fast) ** 2 + ((-1.0 / R_fast) * dR_cal) ** 2)
    return G, dG, R_fast, dR_fast, R_cal, dR_cal


# Original notebook cell 788; function testar_normalidade
def testar_normalidade(amostra, nome="Amostra"):
    print(f"\n=== Teste de normalidade para {nome} ===")

                  
    if len(amostra) <= 5000:
        stat_sw, p_sw = shapiro(amostra)
        print(f"Shapiro-Wilk:     W = {stat_sw:.4f}, p = {p_sw:.4e}  {'✅ normal' if p_sw > 0.05 else '❌ não normal'}")
    else:
        print("Shapiro-Wilk:     ⚠️ não aplicável (n > 5000)") 

                          
    stat_dp, p_dp = normaltest(amostra)
    print(f"D’Agostino-Pearson: K² = {stat_dp:.4f}, p = {p_dp:.4e}  {'✅ normal' if p_dp > 0.05 else '❌ não normal'}") 

                      
    result_ad = anderson(amostra)
    print(f"Anderson-Darling: A² = {result_ad.statistic:.4f}")
    for i, (cv, sig) in enumerate(zip(result_ad.critical_values, result_ad.significance_level)):
        status = '✅' if result_ad.statistic < cv else '❌'
        print(f"  Nível {sig}%: crítica = {cv:.4f} {status}")


# Original notebook cell 788; function qq_plot
def qq_plot(amostra, nome="Amostra"):
    plt.figure(figsize=(8, 7))
    probplot(amostra, dist="norm", plot=plt)
    plt.title(f"QQ-plot: {nome}", fontsize=16)
    plt.ylabel("Quantiles (Ordered Values from Sample)", fontsize=16)
    plt.xlabel(r"Quantiles (Theoretical from Normal Distribution $\mathcal{N}(0,1)$)", fontsize=16)
                                                                           

                                 
                                
                                
                                
                                   
                                
                                
                                
                                
                                
                     

    nome_limpo = re.sub(r'[^a-zA-Z0-9_-]', '_', nome)
    
                                                                          
    plt.show()


# Original notebook cell 788; function teste_t_pareado
def teste_t_pareado(amostra_a, amostra_b, alpha=0.05):
    amostra_a = np.array(amostra_a)
    amostra_b = np.array(amostra_b)
    if len(amostra_a) != len(amostra_b):
        raise ValueError("As amostras devem ter o mesmo tamanho.")

    t_stat, p_val_bilateral = ttest_rel(amostra_a, amostra_b)
    p_val_unilateral = p_val_bilateral / 2 if t_stat < 0 else 1 - p_val_bilateral / 2
    rejeita = p_val_unilateral < alpha
    decisao = "Rejeita H0 (A < B)" if rejeita else "Não rejeita H0"

    return {
        't': t_stat,
        'p_valor_unilateral': p_val_unilateral,
        'nível_significância': alpha,
        'decisão': decisao
    }


# Original notebook cell 788; function teste_z_pareado
def teste_z_pareado(amostra_a, amostra_b, alpha=0.05):
    amostra_a = np.array(amostra_a)
    amostra_b = np.array(amostra_b)
    if len(amostra_a) != len(amostra_b):
        raise ValueError("As amostras devem ter o mesmo tamanho.")
    d = amostra_a - amostra_b
    media_d = np.mean(d)
    std_d = np.std(d, ddof=1)
    n = len(d)
    z = media_d / (std_d / np.sqrt(n))
    p_valor = norm.cdf(z)
    rejeita = p_valor < alpha
    decisao = "Rejeita H0 (A < B)" if rejeita else "Não rejeita H0"

    return {
        'z': z,
        'p_valor_unilateral': p_valor,
        'nível_significância': alpha,
        'decisão': decisao
    }


# Original notebook cell 788; function teste_f_pareado
def teste_f_pareado(amostra_a, amostra_b, alpha=0.05):
    amostra_a = np.array(amostra_a)
    amostra_b = np.array(amostra_b)
    if len(amostra_a) != len(amostra_b):
        raise ValueError("As amostras devem ter o mesmo tamanho.")
    n = len(amostra_a)
    var_a = np.var(amostra_a, ddof=1)
    var_b = np.var(amostra_b, ddof=1)
    F = var_a / var_b
    df1 = n - 1
    df2 = n - 1
    p_valor = f.cdf(F, df1, df2)
    rejeita = p_valor < alpha
    decisao = "Rejeita H0 (var(A) < var(B))" if rejeita else "Não rejeita H0"

    return {
        'F': F,
        'p_valor_unilateral': p_valor,
        'nível_significância': alpha,
        'decisão': decisao
    }


# Original notebook cell 788; function teste_wilcoxon_pareado
def teste_wilcoxon_pareado(amostra_a, amostra_b, alpha=0.05):
    amostra_a = np.array(amostra_a)
    amostra_b = np.array(amostra_b)
    if len(amostra_a) != len(amostra_b):
        raise ValueError("As amostras devem ter o mesmo tamanho.")

    stat, p_valor = wilcoxon(amostra_a, amostra_b, alternative='less')
    rejeita = p_valor < alpha
    decisao = "Rejeita H0 (A < B)" if rejeita else "Não rejeita H0"

    return {
        'estatística': stat,
        'p_valor_unilateral': p_valor,
        'nível_significância': alpha,
        'decisão': decisao
    }


# Original notebook cell 788; function teste_t_pareado_less
def teste_t_pareado_less(Ec, Ef, alpha=0.05):
    Ec = np.asarray(Ec); Ef = np.asarray(Ef)
    if len(Ec) != len(Ef): raise ValueError("Amostras devem ter o mesmo tamanho.")
    d = Ec - Ef
    n = len(d); df = n - 1
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    se = sd_d / np.sqrt(n)
    t_stat = mean_d / se
    p_val = t.cdf(t_stat, df)                                                     
                                                                  
    q = t.ppf(1 - alpha, df)
    ci = (-np.inf, mean_d + q*se)
    return {
        "t": t_stat, "df": df, "p_valor": p_val,
        "media_diff": mean_d, "IC_media_diff": ci,
        "cohens_d_pareado": mean_d / sd_d,
        "decisao": "Reject H0" if p_val < alpha else "Fail to Reject H0"
    }


# Original notebook cell 788; function pitman_morgan_less
def pitman_morgan_less(Ec, Ef, alpha=0.05, n_boot=5000, seed=42):
    """One-sided Pitman–Morgan test for correlated variance comparison under the bivariate-normal assumption."""
    Ec = np.asarray(Ec); Ef = np.asarray(Ef)
    if len(Ec) != len(Ef): raise ValueError("Amostras devem ter o mesmo tamanho.")
    n = len(Ec); df = n - 2

    S = Ec + Ef
    D = Ec - Ef
    r = np.corrcoef(S, D)[0,1]
                     
    t_stat = r * np.sqrt(df / max(1e-15, (1 - r*r)))
                                                          
    p_val = t.cdf(t_stat, df)

                          
    var_ec = np.var(Ec, ddof=1)
    var_ef = np.var(Ef, ddof=1)
    ratio = var_ec / var_ef

                                                                             
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    ratios = []
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)                                    
        v_ec = np.var(Ec[i], ddof=1)
        v_ef = np.var(Ef[i], ddof=1)
        ratios.append(v_ec / v_ef)
    ratios = np.array(ratios)
                                                                                                         
    ci_bilateral = (np.quantile(ratios, 0.025), np.quantile(ratios, 0.975))
    ci_sup_95 = ( 0.0, float(np.quantile(ratios, 1-alpha)) )                         

    return {
        "t": t_stat, "df": df, "p_valor": p_val,
        "r_SD": r,
        "var_ec": var_ec, "var_ef": var_ef, "ratio_var": ratio,
        "IC_ratio_boot_95": ci_bilateral,
        "IC_unilateral_sup_95": ci_sup_95,
        "decisao": "Reject H0" if p_val < alpha else "Fail to Reject H0"
    }


# Original notebook cell 788; function teste_z_pareado_less
def teste_z_pareado_less(Ec, Ef, alpha=0.05, sigma_diff=None):
    """One-sided paired Z test of the mean difference between two sets of matched residuals."""
    Ec = np.asarray(Ec); Ef = np.asarray(Ef)
    if len(Ec) != len(Ef):
        raise ValueError("Amostras devem ter o mesmo tamanho.")
    d = Ec - Ef
    n = len(d)
    mean_d = d.mean()

                                                                        
    sd_used = float(sigma_diff) if sigma_diff is not None else np.std(d, ddof=1)
    se = sd_used / np.sqrt(n)

    z = mean_d / se
    p_val = norm.cdf(z)                         

                                                 
    q = norm.ppf(1 - alpha)
    ci = (-np.inf, mean_d + q*se)

    return {
        "z": z,
        "p_valor": p_val,
        "media_diff": mean_d,
        "se_diff": se,
        "sigma_usada": sd_used,
        "n": n,
        "IC_media_diff": ci,
        "nivel_significancia": alpha,
        "decisao": "Reject H0" if p_val < alpha else "Fail to Reject H0"
    }


# Original notebook cell 788; function teste_f_indep_less
def teste_f_indep_less(Ec, Ef, alpha=0.05):
    """Classical one-sided F test for independent approximately normal samples."""
    Ec = np.asarray(Ec); Ef = np.asarray(Ef)
    n1 = len(Ec); n2 = len(Ef)
    if n1 != n2:
                                                                                    
                                                           
        pass

                                   
    s1_sq = np.var(Ec, ddof=1)
    s2_sq = np.var(Ef, ddof=1)

                                     
    F_stat = s1_sq / s2_sq
    df1 = n1 - 1
    df2 = n2 - 1

                                                                  
    p_val = f.cdf(F_stat, df1, df2)

                                      
                                                           
    low_bi = F_stat / f.ppf(1 - alpha/2, df1, df2)
    up_bi  = F_stat / f.ppf(alpha/2,     df1, df2)
    ci_bilateral = (low_bi, up_bi)

                                                      
    ci_unilateral_sup = (0.0, F_stat / f.ppf(alpha, df1, df2))

    return {
        "F": F_stat,
        "df1": df1, "df2": df2,
        "p_valor_unilateral": p_val,
        "var_ec": s1_sq, "var_ef": s2_sq,
        "ratio_var": F_stat,
        "IC_ratio_bilateral_1-alpha": ci_bilateral,
        "IC_ratio_unilateral_sup_1-alpha": ci_unilateral_sup,
        "decisao": "Reject H0" if p_val < alpha else "Fail to Reject H0"        
    }


# Original notebook cell 788; function wilcoxon_pareado_less
def wilcoxon_pareado_less(Ec, Ef, alpha=0.05, n_boot=2000, seed=42, ci_type="one-sided"):
    import numpy as np
    from scipy.stats import wilcoxon

    Ec = np.asarray(Ec); Ef = np.asarray(Ef)
    if Ec.shape != Ef.shape:
        raise ValueError("Amostras devem ter o mesmo tamanho.")
    d = Ec - Ef
    n = d.size

    res = wilcoxon(d, zero_method='pratt', alternative='less', correction=True, mode='auto')
    T = res.statistic
    p_val = res.pvalue

    med_d = float(np.median(d))

    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        i = rng.integers(0, n, size=n)                     
        meds[b] = np.median(d[i])

    if ci_type == "one-sided":
                                                           
        ci_med = (-np.inf, float(np.quantile(meds, 1 - alpha)))
    elif ci_type == "two-sided":
        ci_med = (float(np.quantile(meds, alpha/2)), float(np.quantile(meds, 1 - alpha/2)))
    else:
        raise ValueError("ci_type must be 'one-sided' or 'two-sided'")

    decision = "Reject H0" if p_val < alpha else "Fail to Reject H0"

    return {
        "test": "Wilcoxon signed-rank (paired, one-sided less)",
        "T": T,
        "p_value": p_val,
        "n": n,
        "median_diff": med_d,                                              
        "CI_median_bootstrap": ci_med,                                                    
        "ci_type": ci_type,
        "alpha": alpha,
        "decision": decision
    }


# Original notebook cell 788; function var_ratio_pareado_boot_perm_less
def var_ratio_pareado_boot_perm_less(Ec, Ef, alpha=0.05, n_boot=2000, n_perm=5000, seed=42):
    """Paired residual-variance comparison using bootstrap and permutation-based inference."""
    Ec = np.asarray(Ec); Ef = np.asarray(Ef)
    if Ec.shape != Ef.shape:
        raise ValueError("Amostras devem ter o mesmo tamanho.")
    n = Ec.size

                     
    var_ec = np.var(Ec, ddof=1)
    var_ef = np.var(Ef, ddof=1)
    ratio_obs = var_ec / var_ef

                                         
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, size=n)
        v_ec = np.var(Ec[i], ddof=1)
        v_ef = np.var(Ef[i], ddof=1)
        ratios[b] = v_ec / v_ef

    ci_bi = (np.quantile(ratios, alpha/2), np.quantile(ratios, 1 - alpha/2))
    ci_sup = (0.0, np.quantile(ratios, 1 - alpha))                            

                                                      
                                                                                     
                                                                         
    swap_mask = np.ones(n, dtype=bool)
    ratios_perm = np.empty(n_perm)
    for k in range(n_perm):
                                           
        swap_mask = rng.random(n) < 0.5
        Ec_perm = np.where(swap_mask, Ef, Ec)
        Ef_perm = np.where(swap_mask, Ec, Ef)
        ratios_perm[k] = np.var(Ec_perm, ddof=1) / np.var(Ef_perm, ddof=1)

                                                                                
    p_val = (np.sum(ratios_perm <= ratio_obs) + 1.0) / (n_perm + 1.0)

    decision = "Reject H0" if p_val < alpha else "Fail to Reject H0"

    return {
        "test": "Paired variance ratio: bootstrap CIs + paired permutation p-value (one-sided less)",
        "n": n,
        "var_Ec": var_ec,
        "var_Ef": var_ef,
        "ratio_var": ratio_obs,
        "CI_ratio_bootstrap": ci_bi,
        "CI_ratio_unilateral_sup": ci_sup,
        "p_value_permutation": p_val,
        "alpha": alpha,
        "decision": decision
    }


# Original notebook cell 831; function trans_reg_to_cls
def trans_reg_to_cls(erro1, lambda_dif):
    xn1_ptp_cls = []
    for e in erro1:
        if e > -lambda_dif and e < lambda_dif: 
            xn1_ptp_cls.append(1)
        else:
            xn1_ptp_cls.append(0)
    
    return xn1_ptp_cls


# Original notebook cell 831; function count_true_false
def count_true_false(vector):
    true_count = 0
    false_count = 0
    for v in vector:
        if v == 1: true_count += 1
        if v == 0: false_count += 1
    
    return true_count, false_count, true_count/len(vector), false_count/len(vector)


# Original notebook cell 831; function lambda_analysis
def lambda_analysis(erro1, erro2, label_erro1, label_erro2, full_data_text_label):
    
    lambda_max=max(np.std(erro1)*6,np.std(erro2)*6)                              
    lbd = np.linspace(0,lambda_max,num=1000)
    true1 = []
    true2 = []
    false1 = []
    false2 = []
    for l in lbd:
        true1.append(count_true_false(trans_reg_to_cls(erro1, l))[2])
        true2.append(count_true_false(trans_reg_to_cls(erro2, l))[2])
    
        false1.append(count_true_false(trans_reg_to_cls(erro1, l))[3])
        false2.append(count_true_false(trans_reg_to_cls(erro2, l))[3])

                                                            
    fig = plt.figure(figsize=(8,7))
    plt.plot(lbd, true1, label=label_erro1)
    plt.plot(lbd, true2, label=label_erro2)
    
    diff = list(np.array(true1) - np.array(true2))
    ymax = max(diff)
    ixmax = diff.index(ymax)
    xmax = lbd[ixmax]

    plt.plot(lbd, diff, label=f'ABS(({label_erro1})-({label_erro2}))')
    plt.plot(xmax, ymax, 'o', color='green')
                                                                                       
    plt.text(xmax, ymax, f'  ({float(xmax):.2f}, {float(ymax):.2f})', fontsize=12)
    plt.axvline(x = xmax, linestyle = '--', color = 'grey')
    plt.axhline(y = .9, linestyle = '--', color = 'grey')

                               
    for i in range(len(true1)):
        if true1[i] >=.9: 
            true1_y90 = true1[i]
            true1_x90 = lbd[i]
            break

    plt.plot(true1_x90, true1_y90, 'o', color='blue', label=f'{label_erro1}: EAR (90\%), $\Delta\lambda$={round(true1_x90,2)} (MeV)')
                                                                                                              
    plt.text(true1_x90, true1_y90, f'  ({float(true1_x90):.2f}, {float(true1_y90):.2f})', fontsize=12) 

    for i in range(len(true2)):
        if true2[i] >=.9: 
            true2_y90 = true2[i]
            true2_x90 = lbd[i]
            break

    plt.plot(true2_x90, true2_y90, 'o', color='orange', label=f'{label_erro2}: EAR (90\%), $\Delta\lambda$={round(true2_x90,2)} (MeV)')
                                                                                                            
    plt.text(true2_x90, true2_y90, f'  ({float(true2_x90):.2f}, {float(true2_y90):.2f})', fontsize=12)

                                           
    true1_ymaxdif = true1[ixmax]
    true2_ymaxdif = true2[ixmax]

                                                                                                      
                                                                    

    plt.plot(xmax, true1_ymaxdif, '^', color='blue', markersize=10, label=f'{label_erro1}: EAR ({round(true1_ymaxdif*100,0)}\%), $\Delta\lambda$={round(xmax,2)} (MeV)')
    plt.text(xmax, true1_ymaxdif, f'  ({float(xmax):.2f}, {float(true1_ymaxdif):.2f})', fontsize=12) 

    plt.plot(xmax, true2_ymaxdif, '^', color='orange', markersize=10, label=f'{label_erro2}: EAR ({round(true2_ymaxdif*100,0)}\%), $\Delta\lambda$={round(xmax,2)} (MeV)')
    plt.text(xmax, true2_ymaxdif, f'  ({float(xmax):.2f}, {float(true2_ymaxdif):.2f})', fontsize=12) 


    plt.ylabel('Event Acceptance Rate (EAR)', fontsize=16)
    plt.xlabel(r'$\Delta\lambda$ Threshold (MeV)', fontsize=16)

    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim([ymin, ymax + (ymax-ymin)*.5])
    
    plt.legend(frameon=False, fontsize=11, loc='upper right')
                                                
    
                                                                                                
    plt.show()
    
    if true2_x90 >= true1_x90:
        print(f"Ganho na EAR de 90%: Janela de aceitação do {label_erro1} é {round(true2_x90/true1_x90,2)} vezes menor que a do {label_erro2}.")
        print(f"Isso indica um aumento na precisão das estimativas, menor variância, do {label_erro1} em relação ao {label_erro2}.")
        print(f"{label_erro1} atinge {round(true1_ymaxdif*100,0)}% de TA com Lambda={round(xmax,2)}, enquanto que {label_erro2} atinge somente {round(true2_ymaxdif*100,0)}% para o mesmo Lambda.")
    else:
        print(f"Ganho na EAR de 90%: Janela de aceitação do {label_erro2} é {round(true1_x90/true2_x90,2)} vezes menor que a do {label_erro1}.")
        print(f"Isso indica um aumento na precisão das estimativas, menor variância, do {label_erro2} em relação ao {label_erro1}.")
        print(f"{label_erro2} atinge {round(true2_ymaxdif*100,0)}% de TA com Lambda={round(xmax,2)}, enquanto que {label_erro1} atinge somente {round(true1_ymaxdif*100,0)}% para o mesmo Lambda.")
    
   
       
              
                                    
                                
                                
                            
                                 
                               
                   
                  

                                    
                            
                                 
                               
                   
                  

                  
                            
                                                                                                         

                                    
                                    

                           

                                     
                                     
   
                                                  
                                                  

                                    
                                    

                                                                         
                                                                          
                                        
    
    
                                           
                                            
                                           
                                                
                  

    return round(true1_x90,2), round(true2_x90,2)

# Original notebook cell 77
def comparison_boxplot(data, labels, full_data_text_label, xtextlabel=''):
    
                    
                                                                        
                                                                                 
                                                                        

                                         
                                            
    
    
    fig, ax = plt.subplots(figsize=(8, 7))    
    
                            
                                    
 
                   
    vertical = True
    bp = ax.boxplot(data, showfliers=False, vert = vertical, patch_artist=True)    
    plota_data_label(plt, full_data_text_label) 
    
                      
    colors = ["C0", "orange", "green", "red", "lightblue", "lightgreen", "pink", "purple", "brown", "grey", "olive", "indigo", "paleturquoise", "darkslategray", "fuchsia", "darkred", "darkgreen", "darkblue", "darkgoldenrod", "darkcyan", "darkolivegreen", "lightcoral", "peru", "orangered", "royalblue", "navy"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
       
                     
                                            
    if vertical == True:
        ax.set_xlabel("Evaluation", fontsize=16)
        plt.axhline(y=0, color='grey', linestyle='--')
        
        if xtextlabel=='': 
            ax.set_ylabel("Normalized Errors", fontsize=16)  
        else:
            ax.set_ylabel(xtextlabel, fontsize=16)
    else:
        ax.set_ylabel("Evaluation", fontsize=16) 
        plt.axvline(x=0, color='grey', linestyle='--')
        
        if xtextlabel=='': 
            ax.set_xlabel("Normalized Errors", fontsize=16)  
        else:
            ax.set_xlabel(xtextlabel, fontsize=16)
    
    
              
    ax.legend(bp["boxes"], labels, loc='upper right', frameon=False, fontsize=16, ncol=1)  
    
                                                                                                   
    
     
                  
    plt.show()

# Original notebook cell 80
def comparison_times(df,full_data_text_label):
    
    df = df.sort_values(by=['Processing-time'])   
     
    name = df['Method-Selection']
    times = df['Processing-time']*1e6                                                                                    
    xmax = max(times)
 
    if len(name)/3 > 7: 
        fig_y = int(len(name)/3)
    else: 
        fig_y = 7
                 
                                           
    fig, ax = plt.subplots(figsize=(8, fig_y))
                             
 
                         
    ax.barh(name, times)
 
                         
                                                                         
                                         
 
                       
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
 
                                         
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
 
                        
                                                                                       
 
                     
    ax.invert_yaxis()
 
                            
    for i in ax.patches:
        plt.text(i.get_width()+xmax*.01, i.get_y()+0.5, str(round((i.get_width()), 4)), fontsize=12, fontweight='bold', color='grey')
 
                    
    ax.set_title(r'Methods x Prediction-times per event')
    ax.set_xlabel(r'Prediction-times per event (micro-seconds)')
 
               
                                                      
    ax.set_xlim([0,xmax*1.15])
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(xmin + (xmax-xmin)*.025, ymax - (ymax-ymin)*.025, full_data_text_label)
    plt.show()

# Original notebook cell 85
def comparison_metric(df,full_data_text_label):
    
    df = df.sort_values(by=['d0'])   
     
    name = df['data_labels']
    times = df['d0']    
    xmax = max(times)
    
    stardard_d0 = 1
    for i in range(len(name)):
        if name[i]=='Standard FastCalo':
            standard_d0 = times[i]
            print("Encontrou o Standard FastCalo", standard_d0)
            break
 
    if len(name)/3 > 7: 
        fig_y = int(len(name)/3)
    else: 
        fig_y = 7
                 
                                           
    fig, ax = plt.subplots(figsize=(8, fig_y))
                             
 
                         
                         
    ax.barh(name, times, color='w', edgecolor='b', linewidth=2)
 
                         
                                                 
                                         
 
                       
                                        
    ax.yaxis.set_ticks_position('none')
 
                                         
    ax.xaxis.set_tick_params(pad=5)
                                    


    
    for l in ax.get_yticklabels():
        l.set_fontweight('bold')
        l.set_horizontalalignment('left')
                                              
                                                                                   
    ax.tick_params(axis="y",direction="in", pad=-5, labelsize=10, colors="k")
    
 
                        
                                                                                       
 
                     
    ax.invert_yaxis()
 
                            
    for i in ax.patches:
        metric = i.get_width()
                                              
                                                              
        reduction = 100*(metric-standard_d0) / standard_d0
                                                                
        str_reduction =" (" + str( round(reduction, 2) ) + "\% Std)"                     
        
        plt.text(i.get_width()+xmax*.01, i.get_y()+0.5, str(round(metric, 2))+str_reduction, fontsize=10, fontweight='bold', color='black')
 
                    
                                                              
    ax.set_xlabel(r'$\sqrt{\text{IQR}^2 + \text{RMSE}^2}$ (MeV)')
 
               
                                                      
    ax.set_xlim([0,xmax*1.3])
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(xmin + (xmax-xmin)*.025, ymax - (ymax-ymin)*.025, full_data_text_label)
    plt.show()

# Original notebook cell 86
def comparison_metric_new(df,full_data_text_label):
    
    df = df.sort_values(by=['d0'])   
     
    name = df['data_labels']
    times = df['d0']    
    xmax = max(times)
    
    stardard_d0 = 1
    for i in range(len(name)):
        if name[i]=='Standard FastCalo':
            standard_d0 = times[i]
            print("Encontrou o Standard FastCalo", standard_d0)
            break
    
                                 
    n_bars = 10
    name = name[0:n_bars]
    times = times[0:n_bars]

                               
    if len(name)/3 > 7: 
        fig_y = int(len(name)/3)
    else: 
        fig_y = 7
                 
                                           
    fig, ax = plt.subplots(figsize=(8, fig_y))
                             
   
 
                         
    ax.barh(name, times, color='w', edgecolor='b', linewidth=2)
 
                         
                                                 
                                         
 
                       
                                        
                                        
                                        
                                        

     
 
                                         
    ax.xaxis.set_tick_params(pad=5, labelsize=13, direction='in')
    ax.yaxis.set_tick_params(pad=3, labelsize=13, direction='in')

    
    
    for l in ax.get_yticklabels():
        l.set_fontweight('bold')
        l.set_horizontalalignment('left')
                                              
                                                                               
    ax.tick_params(axis="y",direction="in", pad=-10, labelsize=12, colors="k")

    
      
 
                        
                                                                                       
 
                     
    ax.invert_yaxis()
 
                            
    for i in ax.patches:
        metric = i.get_width()
                                              
                                                              
        reduction = 100*(metric-standard_d0) / standard_d0
                                                                
                                                                                             
        str_reduction =" (" + str( round(reduction, 2) ) + "\% Std)"                     
        
        plt.text(i.get_width()+xmax*.01, i.get_y()+0.5, str(round(metric, 2))+str_reduction, fontsize=12, fontweight='bold', color='black')
        
 
                    
                                                                                                          
    ax.set_xlabel(r'$\sqrt{\text{IQR}^{2} + \text{RMSE}^{2}}$ (MeV)', fontsize=16)
    ax.set_ylabel(r'Ranked 10 methods', fontsize=16, labelpad=30)
 
               
                                                   
    xmin, xmax, ymin, ymax = ax.axis()
    ax.set_xlim([0, xmax + (xmax-xmin)*.35])
    ax.set_ylim([ymin, ymax + (ymax-ymin)*.10])
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(xmin + (xmax-xmin)*.05, ymax - (ymax-ymin)*.05, full_data_text_label, fontsize=13)
                                                                                               
    plt.show()

# Original notebook cell 87
def comparison_metric_global(df,full_data_text_label):
    
    df = df.sort_values(by=['D0'])   
     
    name = df['Method-Selection']
    times = df['D0']    
    xmax = max(times)
    
                    
                               
                                         
                                   
                                                                
                  
 
    if len(name)/3 > 7: 
        fig_y = int(len(name)/3)
    else: 
        fig_y = 7
                 
                                           
    fig, ax = plt.subplots(figsize=(8, fig_y))
                             
 
                         
    ax.barh(name, times, color='w', edgecolor='b', linewidth=2)
 
                         
                                                 
                                         
 
                       
                                        
                                        
                                        
                                        

     
 
                                         
    ax.xaxis.set_tick_params(pad=5, labelsize=13, direction='in')
    ax.yaxis.set_tick_params(pad=3, labelsize=13, direction='in')

    
    
    for l in ax.get_yticklabels():
        l.set_fontweight('bold')
        l.set_horizontalalignment('left')
                                              
                                                                               
    ax.tick_params(axis="y",direction="in", pad=-5, labelsize=12, colors="k")

    
      
 
                        
                                                                                       
 
                     
    ax.invert_yaxis()
 
                            
    for i in ax.patches:
        metric = i.get_width()
                                              
                                                              
                                                           
                                                                
        str_reduction =""                                                                   
        
        plt.text(i.get_width()+xmax*.01, i.get_y()+0.5, str(round(metric, 4))+str_reduction, fontsize=13, fontweight='bold', color='black')
 
                    
                                                                                                          
    ax.set_xlabel(r'$\sqrt{\text{IQR}_{(N)}^{2} + \text{RMSE}_{(N)}^{2} + \text{T}_{(N)}^{2} + \text{F}_{(N)}^{2}}$', fontsize=16)
                                                                                                               
    ax.set_ylabel(r'Ranked 10 methods', fontsize=16, labelpad=30)
 
               
                                                   
    xmin, xmax, ymin, ymax = ax.axis()
    ax.set_xlim([0, xmax + (xmax-xmin)*.02])
    ax.set_ylim([ymin, ymax + (ymax-ymin)*.10])
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(xmin + (xmax-xmin)*.05, ymax - (ymax-ymin)*.05, full_data_text_label, fontsize=13)
                                                                                       
    plt.show()

# Original notebook cell 135
def correlation_matrix(X):
    full_data_atual = X
    full_data_atual.to_csv("full_data_atual.csv", sep=',', index=False, encoding='utf-8')
    return full_data_atual

# Original notebook cell 135
def corr_decomposition(X, n=0.9):
    corr = X.corr()                                                          
    
    corr = corr.drop(labels=['cluster_trutheta','cluster_truthe237','cluster_truthe277','cluster_truthet','cluster_truthphi','el_truth_pt','alfa'], axis='index')                            
    
    corr_n = corr[corr['mc_match_et']>=n]                                             
    
    corr_n = corr_n.drop(labels=['mc_match_et'], axis='index')                            
    
    valid_indexes = corr_n.index.values.tolist()                                          
    
                         
    
    X_corr = X[valid_indexes]
    
    return X_corr, corr_n

# Original notebook cell 136
def MI_matrix(full_data):   

    features = full_data.columns
    print("Features: ", features)
    l_features = len(features)
    print()
    print("Number of Features: ", l_features)
    X = full_data.to_numpy()

    mi_matrix = {}
    for i in range(l_features):
        print(f"=== {i}/{l_features-1} ===")
        y = full_data.iloc[:, i].to_numpy()
                      
        mi = mutual_info_regression(X, y)
        mi /= np.max(mi)
        mi_matrix[features[i]] = mi    
    
    df_mi_matrix = pd.DataFrame(mi_matrix, index = features)  
                        
    
    return df_mi_matrix

# Original notebook cell 136
def imutua_decomposition(X, df_mi_matrix, n=0.9):   
    
                                                         
                                                           
                       
                       
    
    print(df_mi_matrix.shape)
    
    df_mi_matrix = df_mi_matrix.drop(labels=[100, 101, 102, 103, 104, 105, 106, 107], axis='index')                                       
    
    print(df_mi_matrix.shape)
    
    df_mi_matrix = (df_mi_matrix - df_mi_matrix.min()) / (df_mi_matrix.max() - df_mi_matrix.min())              
    
    df_mi_matrix_n = df_mi_matrix[df_mi_matrix['mc_match_et']>=n]                                                 
    
                                                                                                           
    
    valid_indexes = df_mi_matrix_n.index.values.tolist()                                          
    
                                            
    
                                                       
    
                                     
    
    valid_indexes_s = [ str(i) for i in valid_indexes]
    
    X_mutua = X.loc[:, valid_indexes_s]
    
    return X_mutua, df_mi_matrix_n

# Original notebook cell 141
def pca_decomposition(X, n=0):
    
    indexes_X = X.index
    
    X = np.array(X)
    X = (X - np.mean(X)) / np.std(X)
    
    if n==0: 
        n = X.shape[1]
                                                                          
          
                                                     
    
    pca = PCA(n_components=n, random_state=0)
    
    X_pca = pca.fit_transform(X)    
                                                                      
                 
    
    X_pca_Df = pd.DataFrame(data = X_pca, columns = [ f'pc{i}' for i in range(X_pca.shape[1]) ], index=indexes_X )
    pca_var_ratio = pca.explained_variance_ratio_    
    
    print("length pca_var_ratio:", len(pca_var_ratio))
                                           
    
                                                                                         
    
                                                       
    
    index99p = -1
    cumul = 0
    while (cumul < .99) and (index99p < len(pca_var_ratio) - 1):       
        index99p += 1
        cumul += pca_var_ratio[index99p]                   
    
    print("99% of representation:", index99p) 
    print(f"Variance {index99p}-component:", cumul) 
    print()
    
    return X_pca_Df, pca_var_ratio

# Original notebook cell 146
def ica_decomposition(X, n=7):
    
    indexes_X = X.index

    transformer = FastICA(n_components=n, random_state=0, whiten='unit-variance')

    X_ica_n = transformer.fit_transform(X)

                         
    
    X_ica = pd.DataFrame(X_ica_n, columns = [ f'ic{i}' for i in range(X_ica_n.shape[1]) ], index=indexes_X)
    
    return X_ica

# Original notebook cell 286
def plota_stats_tests_intervals(data_metrics, full_data_text_label):
    
    data_m = data_metrics[ data_metrics['Feature-Selection-Method'] != 'Standard FastCalo' ] 
    data_m['labels'] = data_m['Evaluation-Method'] + ': '+ data_m['Feature-Selection-Method']
    
    method = list(data_m['Evaluation-Method'])[::-1]
    selection = list(data_m['Feature-Selection-Method'])[::-1]
    f_value = list(data_m['F-H0: f-value (0.05)'])[::-1]
    f_critical = list(data_m['F-H0: f-critical (0.05)'])[::-1]
    t_value = list(data_m['T-H0: t-value (0.05)'])[::-1]
    t_critical = list(data_m['T-H0: t-critical (0.05)'])[::-1]
    labels = list(data_m['labels'])[::-1]  
    f_IC = list(data_m['F-H1: IC[Errcalib_var/Errfast_var; 0.95]'])[::-1]
    t_IC = list(data_m['T-H1: IC[Errfast_mean - Errcalib_mean; 0.95]'])[::-1]
    
    f_min = min(1/max(f_critical), min(f_value))
    f_max = max(max(f_critical), max(f_value))
    f_delta = f_max-f_min
    
    t_min = min(-min(t_critical), min(t_value))
    t_max = max(max(t_critical), max(t_value))   
    t_delta = t_max-t_min
    
    bar_colors = ["C0", "orange", "green", "red", "lightblue", "lightgreen", "pink", "purple", "brown", "grey", "olive", "indigo", "paleturquoise", "darkslategray", "fuchsia", "darkred", "darkgreen", "darkblue", "darkgoldenrod", "darkcyan", "darkolivegreen", "lightcoral", "peru", "orangered", "royalblue", "navy"]
    bar_colors = bar_colors[0:len(method)][::-1]   
    
                             
    if len(method)/3 > 9: 
        fig_y = int(len(method)/3)
    else: 
        fig_y = 9 
        
                                                                       
    fig, ax = plt.subplots(1,2, figsize =(10, fig_y))
                                  
    fig.subplots_adjust(wspace=0)    
                             
    tv=[]   
    ax[0].barh(labels, width=2*np.array(t_critical), height=.6, left=-np.array(t_critical), align='center', color=bar_colors)
    for i in range(len(method)): 
        
                                                                                                                         
        
        if t_value[i]>-t_critical[i] and t_value[i]<t_critical[i]: 
            H0 = ' (H0: Accepted)'
        else:
            H0 = ' (H0: Rejected)'
        
                  
        ax[0].plot(t_value[i], i, '.', color = 'black', label=f'{labels[i]}') 
        ax[0].text(t_value[i]-50, i+.15, 'T-Value: '+'{:.2e}'.format(t_value[i])+H0, fontsize=12)
                                                        
        
               
                                                                                                                   
        ax[0].text(t_critical[i]+.02, i-.4, 'T-ACR: ['+'{:.2e}'.format(-t_critical[i])+';'+'{:.2e}'.format(t_critical[i])+']', fontsize=12)
            
                            
                                                                                                   
                                                                    
        
        tv.append(max(t_value[i], t_critical[i]))        
    
    
    
    ax[0].xaxis.set_tick_params(direction='in', which='both')
    ax[0].yaxis.set_tick_params(direction='in', which='both')    
    
    ax[0].set_xlim([t_min-t_delta/10, t_max+5*t_delta/10])    
    ax[0].set_xlabel(r'T-Test H0: $\mu_{Ecalib-Etrue} = \mu_{Efast-Etrue}$', fontsize=16)
        
    xmin, xmax, ymin, ymax = ax[0].axis()
    ax[0].set_ylim([ymin, ymax*1.1])
    ax[0].set_xlim([xmin*.8, xmax*1.1])
    xmin, xmax, ymin, ymax = ax[0].axis()
    ax[0].text(xmin+(xmax-xmin)/10, ymax-(ymax-ymin)/20, full_data_text_label)
    
                   
                                      
                                          
                                                                     
                                              
                                           
               
    
    
                                               
    t_IC_index = []
    for i in range(len(t_IC)):
        if t_IC[i] == '[]':
            t_IC[i]='[0,0.0000001]'
            t_IC_index.append(i)
    
    t_IC_g = [ [float(t_IC[i].split(',')[0].replace('[','').replace(' ','')), float(t_IC[i].split(',')[1].replace(']','').replace(' ',''))] for i in range(len(t_IC)) ]
    
                         
               
                             
    
    t_width = [ t_IC_g[i][1]-t_IC_g[i][0] for i in range(len(t_IC_g)) ]
    t_left = [ t_IC_g[i][0] for i in range(len(t_IC_g)) ]
    
                                                                                                
    for i in range(len(method)):        
        
        ax[1].barh(i, width=t_width[i], height=.6, left=t_left[i], align='center', color=bar_colors[i])
        
        if t_IC_g[i][0]>0: 
            H1 = 'H1: Accepted'
        else:
            H1 = 'H1: Rejected'
        
                    
        if i not in t_IC_index:
            text = 'T-IC [{:.2e}'.format(t_IC_g[i][0])+';'+'{:.2e}]'.format(t_IC_g[i][1])        
            ax[1].text(t_IC_g[i][1]+50, i+.15, text, color='black', fontsize=12)
            ax[1].text(t_IC_g[i][1]+60, i-.2, H1, fontsize=12)
        else:
            text = 'H0 Accepted, then H1 was not tested'        
            ax[1].text(t_IC_g[i][1]+50, i+.15, text, color='black', fontsize=12)
                                                               
            
    
    
    ax[1].axvline(x=0, color='grey', linestyle='--')
                                                             
    
    ax[1].set_xlabel(r'T-Test H1: $\mu_{Efast-Etrue} - \mu_{Ecalib-Etrue} > 0 $', fontsize=16)
    
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].set_xlim([xmin-1*(xmax-xmin)/10, xmax+6*(xmax-xmin)/10])  
    
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].set_ylim([ymin, ymax*1.1])
    ax[1].set_xlim([xmin*.8, xmax*1.1])
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].text(xmin+(xmax-xmin)/10, ymax-(ymax-ymin)/20, '')
    ax[1].text(0, ymax-(ymax-ymin)/20, r'$\rightarrow$ H1: Decision Threshold', fontsize=12)
            
    ax[1].xaxis.set_tick_params(direction='in', which='both')
    ax[1].yaxis.set_tick_params(direction='in', which='both') 
    ax[1].tick_params(which='both', left=False)
                                                            
    ax[1].set_yticklabels([])
    
    
    
    plt.show()

    
                             
    fig, ax = plt.subplots(1,2, figsize =(10, fig_y))
                                  
    fig.subplots_adjust(wspace=0)        
    
    
    fv=[]
    ax[0].barh(labels, width=np.array(f_critical)-1/np.array(f_critical), height=.6, left=1/np.array(f_critical), align='center', color=bar_colors)          
    for i in range(len(method)):      
        
                                                                                                                                                        
        
        if f_value[i]>1/f_critical[i] and f_value[i]<f_critical[i]: 
            H0 = ' (H0: Accepted)'
        else:
            H0 = ' (H0: Rejected)'
        
                        
        ax[0].plot(f_value[i], i, '.', color = 'black', label=f'{labels[i]}') 
                                                                           
        ax[0].text(f_value[i]-.2, i+.15, 'F-Value: '+'{:.2e}'.format(f_value[i])+H0, fontsize=12)
                                                          
                
               
                                                                                                                   
        ax[0].text(f_critical[i]+.02, i-.4, 'F-ACR: ['+'{:.2e}'.format(1/f_critical[i])+';'+'{:.2e}'.format(f_critical[i])+']', fontsize=12)
                    
                            
                                                                                                   
                                                                    
            
        fv.append(max(f_value[i], f_critical[i]))        
    
    
      
    
    ax[0].xaxis.set_tick_params(direction='in', which='both')   
    ax[0].yaxis.set_tick_params(direction='in', which='both')    
    
    ax[0].set_xlim([f_min-4*f_delta/10, f_max+10*f_delta/10])    
    ax[0].set_xlabel(r'F-Test H0: $\sigma_{Ecalib-Etrue}^{2} = \sigma_{Efast-Etrue}^{2}$', fontsize=16)    
    
    xmin, xmax, ymin, ymax = ax[0].axis()
    ax[0].set_ylim([ymin, ymax*1.1])
    ax[0].set_xlim([xmin*.8, xmax*1.1])
    xmin, xmax, ymin, ymax = ax[0].axis()
    ax[0].text(xmin+(xmax-xmin)/10, ymax-(ymax-ymin)/20, full_data_text_label)
        
                   
                                      
                                            
                                                                     
                                              
                                       
        
                                               
    f_IC_index = []
    for i in range(len(f_IC)):
        if f_IC[i] == '[]':
            f_IC[i]='[0,0.0000001]'
            f_IC_index.append(i)
    
    
                         
    
    f_IC_g = [ [float(f_IC[i].split(',')[0].replace('[','').replace(' ','')), float(f_IC[i].split(',')[1].replace(']','').replace(' ',''))] for i in range(len(f_IC)) ]
    
               
                             
    
    f_width = [ f_IC_g[i][1]-f_IC_g[i][0] for i in range(len(f_IC_g)) ]
    f_left = [ f_IC_g[i][0] for i in range(len(f_IC_g)) ]
    
                                                                                                
    for i in range(len(method)):
        
                                                                                                                         
        ax[1].barh(i, width=f_width[i], height=.6, left=f_left[i], align='center', color=bar_colors[i])
        
        if f_IC_g[i][0]>1: 
            H1 = 'H1: Accepted'
        else:
            H1 = 'H1: Rejected'
        
                     
        if i not in f_IC_index:
            text = 'F-IC [{:.2e}'.format(f_IC_g[i][0])+';'+'{:.2e}]'.format(f_IC_g[i][1])        
            ax[1].text(f_IC_g[i][1]+.02, i+.15, text, color='black', fontsize=12)
            ax[1].text(f_IC_g[i][1]+.02, i-.2, H1, fontsize=12)
        else:
            text = 'H0 Accepted, then H1 was not tested'        
            ax[1].text(f_IC_g[i][1]+.02, i+.15, text, color='black', fontsize=12)
                                                                
    
    
    ax[1].axvline(x=1, color='grey', linestyle='--')    
    
    ax[1].set_xlabel(r'F-Test H1: $ \sigma_{Ecalib-Etrue}^{2} / \sigma_{Efast-Etrue}^{2} > 1 $', fontsize=16)
    
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].set_xlim([xmin-1*(xmax-xmin)/10, xmax+6*(xmax-xmin)/10]) 
    
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].set_ylim([ymin, ymax*1.1])
    ax[1].set_xlim([xmin*.8, xmax*1.1])
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].text(xmin+(xmax-xmin)/10, ymax-(ymax-ymin)/20, '')
    ax[1].text(1, ymax-(ymax-ymin)/20, r'$\rightarrow$ H1: Decision Threshold', fontsize=12)
            
    ax[1].xaxis.set_tick_params(direction='in', which='both')
    ax[1].yaxis.set_tick_params(direction='in', which='both') 
    ax[1].tick_params(which='both', left=False)
                                                            
    ax[1].set_yticklabels([])
        
    plt.show()

# Original notebook cell 287
def plota_stats_tests_intervals_h(data_metrics, full_data_text_label):
    
    data_m = data_metrics[ data_metrics['Feature-Selection-Method'] != 'Standard FastCalo' ] 
    data_m['labels'] = data_m['Evaluation-Method'] + ': '+ data_m['Feature-Selection-Method']
    
    method = list(data_m['Evaluation-Method'])[::-1]
    selection = list(data_m['Feature-Selection-Method'])[::-1]
    f_value = list(data_m['F-H0: f-value (0.05)'])[::-1]
    f_critical = list(data_m['F-H0: f-critical (0.05)'])[::-1]
    t_value = list(data_m['T-H0: t-value (0.05)'])[::-1]
    t_critical = list(data_m['T-H0: t-critical (0.05)'])[::-1]
    labels = list(data_m['labels'])[::-1]  
    f_IC = list(data_m['F-H1: IC[Errcalib_var/Errfast_var; 0.95]'])[::-1]
    t_IC = list(data_m['T-H1: IC[Errfast_mean - Errcalib_mean; 0.95]'])[::-1]
    
    f_min = min(1/max(f_critical), min(f_value))
    f_max = max(max(f_critical), max(f_value))
    f_delta = f_max-f_min
    
    t_min = min(-min(t_critical), min(t_value))
    t_max = max(max(t_critical), max(t_value))   
    t_delta = t_max-t_min
    
    bar_colors = ["C0", "orange", "green", "red", "lightblue", "lightgreen", "pink", "purple", "brown", "grey", "olive", "indigo", "paleturquoise", "darkslategray", "fuchsia", "darkred", "darkgreen", "darkblue", "darkgoldenrod", "darkcyan", "darkolivegreen", "lightcoral", "peru", "orangered", "royalblue", "navy"]
    bar_colors = bar_colors[0:len(method)][::-1]   
    
                             
    if len(method)/3 > 9: 
        fig_y = int(len(method)/3)
    else: 
        fig_y = 9 
        
                                                                       
    fig, ax = plt.subplots(2,1, figsize =(10, fig_y))
                                  
    fig.subplots_adjust(hspace=0)    
                             
    tv=[]   
                                                                                                                              
                                                                                                                               
    for i in range(len(labels)): 
        
                                                                                                                         
        ax[0].bar(i, height=2*np.array(t_critical[i]), width=.6, bottom=-np.array(t_critical[i]), align='center', color=bar_colors[i])
        
        if t_value[i]>-t_critical[i] and t_value[i]<t_critical[i]: 
            H0 = ' (H0: Accepted)'
        else:
            H0 = ' (H0: Rejected)'
        
                  
                                                                               
                                                                                                  
        
        ax[0].plot(i, t_value[i],  '.', color = 'black', label=f'{labels[i]}') 
        ax[0].text(i+.15, t_value[i],  'T-Value: '+'{:.2e}'.format(t_value[i])+H0, fontsize=12, rotation=90)        
        
        
               
                                                                                                                   
        ax[0].text(i-.4, t_critical[i]+.02,  'T-ACR: ['+'{:.2e}'.format(-t_critical[i])+';'+'{:.2e}'.format(t_critical[i])+']', fontsize=12, rotation=90)
            
                            
                                                                                                   
                                                                    
        
        tv.append(max(t_value[i], t_critical[i]))        
    
    
    ax[0].xaxis.set_tick_params(direction='in', which='both')
    ax[0].yaxis.set_tick_params(direction='in', which='both')    
        
    ax[0].set_ylim([t_min-t_delta/10, t_max+5*t_delta/10])    
    ax[0].set_ylabel(r'T-Test H0: $\mu_{E_{calib}-E_{true}} = \mu_{E_{fast}-E_{true}}$', fontsize=16)
        
    xmin, xmax, ymin, ymax = ax[0].axis()
    ax[0].set_ylim([ymin, ymax*2.2])
    xmin, xmax, ymin, ymax = ax[0].axis()
    ax[0].text(xmin+(xmax-xmin)/10, ymax-(ymax-ymin)/20, full_data_text_label)
    ax[0].set_xticks(np.arange(len(labels)))
    ax[0].set_xticklabels(labels, rotation=90)
        
               
    
    
                                               
    t_IC_index = []
    for i in range(len(t_IC)):
        if t_IC[i] == '[]':
            t_IC[i]='[0,0.0000001]'
            t_IC_index.append(i)
    
    t_IC_g = [ [float(t_IC[i].split(',')[0].replace('[','').replace(' ','')), float(t_IC[i].split(',')[1].replace(']','').replace(' ',''))] for i in range(len(t_IC)) ]
    
                         
               
                             
    
    t_width = [ t_IC_g[i][1]-t_IC_g[i][0] for i in range(len(t_IC_g)) ]
    t_left = [ t_IC_g[i][0] for i in range(len(t_IC_g)) ]
    
    
    ax[1].bar(labels, height=t_width, width=.6, bottom=t_left, align='center', color=bar_colors)
    for i in range(len(labels)):         
        
        if t_IC_g[i][0]>0: 
            H1 = 'H1: Accepted'
        else:
            H1 = 'H1: Rejected'
        
                    
        if i not in t_IC_index:
            text = 'T-IC [{:.2e}'.format(t_IC_g[i][0])+';'+'{:.2e}]'.format(t_IC_g[i][1])        
            ax[1].text(i+.15, t_IC_g[i][1]+50,  text, color='black', fontsize=12, rotation=90)
            ax[1].text(i-.2, t_IC_g[i][1]+60,  H1, fontsize=12, rotation=90)
        else:
            text = 'H0 Accepted, then H1 was not tested'        
            ax[1].text(i+.15, t_IC_g[i][1]+50,  text, color='black', fontsize=12, rotation=90)
                                                               
            
    
    
    ax[1].axhline(y=0, color='grey', linestyle='--')   
    ax[1].set_ylabel(r'T-Test H1: $\mu_{E_{fast}-E_{true}} - \mu_{E_{calib}-E_{true}} > 0 $', fontsize=16)
    
    xmin, xmax, ymin, ymax = ax[1].axis()      
    ax[1].set_ylim([ymin, ymax+5*(ymax-ymin)/10])  
    
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].set_ylim([ymin, ymax*1.2])
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].text(xmin+(xmax-xmin)/10, ymax-(ymax-ymin)/20, '')
    ax[1].text(xmin+(xmax-xmin)/10, 0,  r'H1: Decision Threshold', fontsize=12)
            
    ax[1].xaxis.set_tick_params(direction='in', which='both')
    ax[1].yaxis.set_tick_params(direction='in', which='both') 
                                                
                                                            
    ax[1].set_xticklabels(labels, rotation=90)
    
       
    plt.show()

    
                             
    fig, ax = plt.subplots(2,1, figsize =(10, fig_y))
                                  
    fig.subplots_adjust(hspace=0)        
    
    
    fv=[]
                                                                                                                                                              
    for i in range(len(labels)):      
        
                                                                                                                                                        
        ax[0].bar(i, height=np.array(f_critical[i])-1/np.array(f_critical[i]), width=.6,  bottom=1/np.array(f_critical[i]), align='center', color=bar_colors[i]) 
        
        if f_value[i]>1/f_critical[i] and f_value[i]<f_critical[i]: 
            H0 = ' (H0: Accepted)'
        else:
            H0 = ' (H0: Rejected)'
        
                        
        ax[0].plot(i, f_value[i],  '.', color = 'black', label=f'{labels[i]}')        
        ax[0].text(i+.15, f_value[i],  'F-Value: '+'{:.2e}'.format(f_value[i])+H0, fontsize=12, rotation=90)
                        
               
                                                                                                                   
        ax[0].text(i-.4, f_critical[i]+.02, 'F-ACR: ['+'{:.2e}'.format(1/f_critical[i])+';'+'{:.2e}'.format(f_critical[i])+']', fontsize=12, rotation=90)
                    
                            
                                                                                                   
                                                                    
            
        fv.append(max(f_value[i], f_critical[i]))        
    
    
    ax[0].xaxis.set_tick_params(direction='in', which='both')    
    ax[0].yaxis.set_tick_params(direction='in', which='both')       
    
    ax[0].set_ylim([f_min-4*f_delta/10, f_max+10*f_delta/10])    
    ax[0].set_ylabel(r'F-Test H0: $\sigma_{Ecalib-Etrue}^{2} = \sigma_{Efast-Etrue}^{2}$', fontsize=16)
        
    xmin, xmax, ymin, ymax = ax[0].axis()
    ax[0].set_ylim([ymin, ymax*1.8])
    xmin, xmax, ymin, ymax = ax[0].axis()
    ax[0].text(xmin+(xmax-xmin)/10, ymax-(ymax-ymin)/20, full_data_text_label)    
    ax[0].set_xticks(np.arange(len(labels)))
    ax[0].set_xticklabels(labels, rotation=90)
    
        
                                               
    f_IC_index = []
    for i in range(len(f_IC)):
        if f_IC[i] == '[]':
            f_IC[i]='[0,0.0000001]'
            f_IC_index.append(i)
    
    
                         
    
    f_IC_g = [ [float(f_IC[i].split(',')[0].replace('[','').replace(' ','')), float(f_IC[i].split(',')[1].replace(']','').replace(' ',''))] for i in range(len(f_IC)) ]
    
               
                             
    
    f_width = [ f_IC_g[i][1]-f_IC_g[i][0] for i in range(len(f_IC_g)) ]
    f_left = [ f_IC_g[i][0] for i in range(len(f_IC_g)) ]    
    
    ax[1].bar(labels, height=f_width[i], width=.6,  bottom=f_left, align='center', color=bar_colors)
    for i in range(len(labels)):
        
                                                                                                                         
                                                                                                        
        
        if f_IC_g[i][0]>1: 
            H1 = 'H1: Accepted'
        else:
            H1 = 'H1: Rejected'
        
                     
        if i not in f_IC_index:
            text = 'F-IC [{:.2e}'.format(f_IC_g[i][0])+';'+'{:.2e}]'.format(f_IC_g[i][1])        
            ax[1].text(i+.15, f_IC_g[i][1]+.02,  text, color='black', fontsize=12, rotation=90)
            ax[1].text(i-.2, f_IC_g[i][1]+.02, H1, fontsize=12, rotation=90)
        else:
            text = 'H0 Accepted, then H1 was not tested'        
            ax[1].text(i+.15, f_IC_g[i][1]+.02,  text, color='black', fontsize=12, rotation=90)
            
    
    
    ax[1].axhline(y=1, color='grey', linestyle='--')   
    ax[1].set_ylabel(r'F-Test H1: $ \sigma_{Ecalib-Etrue}^{2} / \sigma_{Efast-Etrue}^{2} > 1 $', fontsize=16)
    
    xmin, xmax, ymin, ymax = ax[1].axis()      
    ax[1].set_ylim([ymin, ymax+5*(ymax-ymin)/10])  
    
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].set_ylim([ymin, ymax*1.2])
    xmin, xmax, ymin, ymax = ax[1].axis()
    ax[1].text(xmin+(xmax-xmin)/10, ymax-(ymax-ymin)/20, '')
    ax[1].text(xmin+(xmax-xmin)/10, 1,  r'H1: Decision Threshold', fontsize=12)
            
    ax[1].xaxis.set_tick_params(direction='in', which='both')
    ax[1].yaxis.set_tick_params(direction='in', which='both')   
    ax[1].set_xticklabels(labels, rotation=90)
       
        
    plt.show()
