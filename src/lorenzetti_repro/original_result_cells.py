"""Author-supplied result plotting cells, preserved verbatim from the experimental notebook.

These strings are executed with recorded experimental arrays, not manuscript values.
"""
ORIGINAL_RESULT_CELLS = {'713': '# Métrica x Tempo de processamento\n'
        '\n'
        "d0_method = list(All_NoSeg_erros_data['method'])\n"
        "d0_selection = list(All_NoSeg_erros_data['useRings'])\n"
        "d0_metric = list(All_NoSeg_erros_data['d0'])\n"
        '\n'
        "time_method = list(All_noSeg_processing_times['Method'])\n"
        "time_selection = list(All_noSeg_processing_times['Selection'])\n"
        "time_metric = list(All_noSeg_processing_times['Processing-time'])\n"
        '\n'
        'result_str = []\n'
        'result_d0 = []\n'
        'result_time = []\n'
        'cont = 0\n'
        'for i in range(len(d0_method)):\n'
        '    for j in range(len(time_method)):\n'
        '        d0_str = str(d0_method[i]) +":"+ str(d0_selection[i]).replace(\' \',\'\')\n'
        '        time_str = str(time_method[j]) +":"+ str(time_selection[j]).replace(\' \',\'\')\n'
        '        #print(d0_str)\n'
        '        #print(time_str)\n'
        '        if d0_str == time_str:\n'
        '            cont+=1\n'
        '            #print("IGUAL", cont, len(d0_method))\n'
        '            result_str.append(d0_str)\n'
        '            result_d0.append(d0_metric[i])\n'
        '            result_time.append(time_metric[j])\n'
        '            \n'
        '\n'
        '#print(result_str)\n'
        '#print(result_d0)\n'
        '#print(result_time)\n'
        '\n'
        'metrics_result_Df_d0_time = pd.DataFrame({"Method-Selection":result_str, "d0 (GeV)": result_d0, '
        '"Time (ms)":result_time})\n'
        '\n'
        '\n'
        '#figure = plt.figure(figsize=(12,12))\n'
        'figure = plt.figure(figsize=(8,7))\n'
        '\n'
        '#n = len(result_str)\n'
        'n = 10\n'
        '\n'
        'ymax_r = max(result_d0)\n'
        'delta_ymax_r = ymax_r/100\n'
        '\n'
        'xmax_r = max(result_time)\n'
        'delta_xmax_r = xmax_r/100\n'
        '\n'
        'for i in range(n):\n'
        '    \n'
        '    plt.plot(result_time[i],result_d0[i],\'o\', label=f"{i}: {result_str[i]}")\n'
        '    #plt.text(result_time[i]+20,result_d0[i]-12, f"{i}", fontsize=12)\n'
        '    plt.text(result_time[i]+delta_xmax_r,result_d0[i]-delta_ymax_r, f"{i}", fontsize=12)\n'
        '\n'
        '#plt.scatter(result_time, result_d0)\n'
        "plt.xlabel(r'Prediction-Times (ms)', fontsize=16)\n"
        "plt.ylabel(r'$\\sqrt{IQR^2 + RMSE^2}$', fontsize=16)\n"
        '\n'
        'plt.ylim([0,ymax_r])\n'
        'plt.xlim([0,xmax_r])\n'
        '\n'
        '\n'
        '# Texto com informação da base de dados\n'
        'xmin, xmax, ymin, ymax = plt.axis()\n'
        'plt.ylim([ymin,ymax*1.2])\n'
        '#plt.xlim([xmin,xmax*1.2])\n'
        'xmin, xmax, ymin, ymax = plt.axis()\n'
        'plt.text(xmin + (xmax-xmin)*.025, ymax - (ymax-ymin)*.05, full_data_text_label, fontsize=13) \n'
        'limiar_y = (ymax+ymin)/2\n'
        'limiar_x = (xmax+xmin)/2\n'
        '\n'
        '\n'
        '# Linhas de Limiares\n'
        "#plt.axvline(x = limiar_x, linestyle = '--', color = 'lightgrey')\n"
        "#plt.axhline(y = limiar_y, linestyle = '--', color = 'lightgrey')\n"
        '\n'
        '#plt.axvline(x = xmin+(xmax-xmin)*.1, linestyle = \'--\', color = \'lightgreen\', label="Metric '
        '10\\% Threshold")\n'
        '#plt.axhline(y = ymin+(ymax-ymin)*.1, linestyle = \'--\', color = \'lightgreen\', label="Time 10\\% '
        'Threshold")\n'
        '\n'
        '#plt.text(xmin+(xmax-xmin)*.28, ymin+(ymax-ymin)*.105, "10\\% Metric Threshold", fontsize=12) \n'
        '#plt.text(xmin+(xmax-xmin)*.102, ymin+(ymax-ymin)*.28, "10\\% Time Threshold", rotation=270,  '
        'fontsize=12) \n'
        '\n'
        '\n'
        '# Legenda\n'
        "#plt.legend(loc='upper right', frameon=False, fontsize=12, ncol=2) \n"
        "plt.legend(bbox_to_anchor = (.98, .92), loc='upper right', frameon=False, fontsize=12, ncol=2) \n"
        "#plt.legend(bbox_to_anchor = (1, 1), loc='upper left', fontsize=12, ncol=2, frameon=False)  \n"
        "#plt.legend(loc='upper right', fontsize=12, ncol=2)\n"
        '\n'
        '#ax = plt.subplot()\n'
        '\n'
        "#ax.xaxis.set_tick_params(direction='in', which='both',labelsize=12)\n"
        "#ax.yaxis.set_tick_params(direction='in', which='both',labelsize=12)\n"
        '\n'
        '##plt.savefig("IEEE_2_revised_images/2d_d0_Times.png", bbox_inches=\'tight\', dpi=300)\n'
        '\n'
        'plt.show()\n'
        '\n'
        '   ',
 '722': '# Metrica Number of Features\n'
        '\n'
        "atrib_method = list(table_results_sorted_features['Method'])\n"
        "atrib_selection = list(table_results_sorted_features['Selection'])\n"
        "atrib_num = list(table_results_sorted_features['Features'])\n"
        '\n'
        'for i in range(len(atrib_num)):\n'
        "    #if atrib_num[i] == '[pc0:pc99]' or atrib_num[i] == '[ic0:ic99]' or atrib_num[i] == "
        "'Rings[0:99]':\n"
        "    if atrib_num[i] == '[pc0:pc99]' or atrib_num[i] == '[ic0:ic99]' or atrib_num[i] == "
        "'Rings[1:100]':\n"
        '        atrib_num[i] = 100\n'
        '    else:\n'
        '        #print(atrib_num[i], type(atrib_num[i]))\n'
        '        atrib_num[i] = len(atrib_num[i])\n'
        '    \n'
        'result_atrib_str = []\n'
        'result_atrib_d0 = []\n'
        'result_atrib_num = []\n'
        'cont = 0\n'
        'for i in range(len(d0_method)):\n'
        '    for j in range(len(atrib_method)):\n'
        '        d0_str = str(d0_method[i]) +":"+ str(d0_selection[i]).replace(\' \',\'\')\n'
        '        atrib_str = str(atrib_method[j]) +":"+ str(atrib_selection[j]).replace(\' \',\'\')\n'
        '        #print(d0_str)\n'
        '        #print(time_str)\n'
        '        if d0_str == atrib_str:\n'
        '            cont+=1\n'
        '            #print("IGUAL", cont, len(d0_method))\n'
        '            result_atrib_str.append(d0_str)\n'
        '            result_atrib_d0.append(d0_metric[i])\n'
        '            result_atrib_num.append(atrib_num[j])\n'
        '\n'
        '            \n'
        'metrics_result_Df_d0_features = pd.DataFrame({"Method-Selection":result_atrib_str, "d0 (GeV)": '
        'result_atrib_d0, "Number of Features":result_atrib_num})\n'
        '\n'
        '\n'
        '#figure = plt.figure(figsize=(12,12))\n'
        'figure = plt.figure(figsize=(8,7))\n'
        '\n'
        'n = len(result_atrib_str)\n'
        'n = 10\n'
        '\n'
        'for i in range(n):\n'
        '    \n'
        '    plt.plot(result_atrib_num[i],result_atrib_d0[i],\'o\', label=f"{i}: {result_atrib_str[i]}")\n'
        '    plt.text(result_atrib_num[i]+1.5,result_atrib_d0[i]-12, f"{i}", fontsize=12)\n'
        '\n'
        '#plt.scatter(result_time, result_d0)\n'
        "plt.xlabel('Number of Features', fontsize=16, labelpad=24)\n"
        "plt.ylabel(r'$\\sqrt{IQR^2 + RMSE^2}$', fontsize=16)\n"
        '\n'
        'xmin, xmax, ymin, ymax = plt.axis()\n'
        'plt.ylim([ymin,ymax*1.2])\n'
        'plt.xlim([xmin,xmax*1.2])\n'
        'xmin, xmax, ymin, ymax = plt.axis()\n'
        'plt.text(xmin + (xmax-xmin)*.025, ymax - (ymax-ymin)*.05, full_data_text_label, fontsize=13) \n'
        'limiar_y = (ymax+ymin)/2\n'
        'limiar_x = (xmax+xmin)/2\n'
        "#plt.axvline(x = limiar_x, linestyle = '--', color = 'lightgrey')\n"
        "#plt.axhline(y = limiar_y, linestyle = '--', color = 'lightgrey')\n"
        '\n'
        '#plt.axvline(x = xmin+(xmax-xmin)*.1, linestyle = \'--\', color = \'lightgreen\', label="Metric '
        '10\\% Threshold")\n'
        '#plt.axhline(y = ymin+(ymax-ymin)*.1, linestyle = \'--\', color = \'lightgreen\', label="Number of '
        'Features 10\\% Threshold")\n'
        '\n'
        '#plt.text(xmin+(xmax-xmin)*.28, ymin+(ymax-ymin)*.105, "10\\% Metric Threshold") \n'
        '#plt.text(xmin+(xmax-xmin)*.102, ymin+(ymax-ymin)*.28, "10\\% Number of Features Threshold", '
        'rotation=270) \n'
        '\n'
        "plt.legend(bbox_to_anchor = (.98, .92), loc='upper right', frameon=False, fontsize=12, ncol=2) \n"
        "#plt.legend(bbox_to_anchor = (1, 1), loc='upper left', fontsize=12, ncol=2, frameon=False)  \n"
        "#plt.legend(loc='upper right', fontsize=12, ncol=2)\n"
        '\n'
        'ax = plt.subplot()\n'
        "ax.xaxis.set_tick_params(direction='in', which='both',labelsize=12)\n"
        "ax.yaxis.set_tick_params(direction='in', which='both',labelsize=12)\n"
        '#plt.savefig("IEEE_1_revised_images/2d_d0_Features.png", bbox_inches=\'tight\', dpi=300)\n'
        'plt.show()\n'
        '\n'
        'metrics_result_Df_d0_features.head(10)',
 '767': '# Quadrantes RMSE x IQR\n'
        "n=len(All_NoSeg_erros_data['data_labels'])\n"
        'n = 10\n'
        '\n'
        "Q_labels = list(All_NoSeg_erros_data['data_labels'])[0:n]\n"
        "Q_iqr = list(All_NoSeg_erros_data['iqr'])[0:n]\n"
        "Q_rmse = list(All_NoSeg_erros_data['rmse'])[0:n]\n"
        '\n'
        '#fig = plt.figure(figsize=(12,12))\n'
        'fig = plt.figure(figsize=(8,7))\n'
        '\n'
        'delta_iqr = max(Q_iqr)/500\n'
        'delta_rmse = max(Q_rmse)/500\n'
        '\n'
        'for i in range(len(Q_labels)):\n'
        "    if i<n or Q_labels[i]=='Standard FastCalo':\n"
        '        plt.plot(Q_iqr[i],Q_rmse[i],\'o\', label=f"{i}: {Q_labels[i]}")\n'
        '        #plt.text(Q_iqr[i]+10,Q_rmse[i]-10, f"{i}", fontsize=12)\n'
        '        plt.text(Q_iqr[i]+delta_iqr,Q_rmse[i]-delta_rmse, f"{i}", fontsize=12)\n'
        '    \n'
        '        #plt.plot(Q_iqr[i],Q_rmse[i],\'o\', label=f"{i}: {Q_labels[i]}")\n'
        '        #plt.text(Q_iqr[i]+10,Q_rmse[i]-10, f"{Q_labels[i]}", fontsize=12)\n'
        '    \n'
        '        \n'
        "plt.ylabel('RMSE($E_{calib},E_{true}$) MeV', fontsize=16)\n"
        "plt.xlabel('IQR($E_{calib}-E_{true}$) MeV', fontsize=16) \n"
        '    \n'
        'xmin, xmax, ymin, ymax = plt.axis()\n'
        'plt.ylim([ymin,ymax*1.2])\n'
        'plt.xlim([xmin,xmax*1.05])\n'
        'xmin, xmax, ymin, ymax = plt.axis()\n'
        'plt.text(xmin + (xmax-xmin)*.025, ymax - (ymax-ymin)*.05, full_data_text_label, fontsize=13) \n'
        '\n'
        '#limiar_y = (ymax+ymin)/2\n'
        '#limiar_x = (xmax+xmin)/2\n'
        "#plt.axvline(x = limiar_x, linestyle = '--', color = 'lightgrey')\n"
        "#plt.axhline(y = limiar_y, linestyle = '--', color = 'lightgrey')\n"
        '\n'
        '#plt.axvline(x = xmin+(xmax-xmin)*.1, linestyle = \'--\', color = \'lightgreen\', label="RMSE 10\\% '
        'Threshold")\n'
        '#plt.axhline(y = ymin+(ymax-ymin)*.1, linestyle = \'--\', color = \'lightgreen\', label="IQR 10\\% '
        'Threshold")\n'
        '\n'
        '#plt.text(xmin+(xmax-xmin)*.18, ymin+(ymax-ymin)*.105, "10\\% RMSE  Threshold") \n'
        '#plt.text(xmin+(xmax-xmin)*.102, ymin+(ymax-ymin)*.18, "10\\% IQR Threshold", rotation=270) \n'
        '\n'
        "plt.legend(bbox_to_anchor = (.99, .92), loc='upper right', frameon=False, fontsize=12, ncol=2) \n"
        "#plt.legend(bbox_to_anchor = (1, 1), loc='upper left', fontsize=12, ncol=2, frameon=False)  \n"
        "#plt.legend(loc='upper right', fontsize=12, ncol=2)\n"
        '\n'
        'ax = plt.subplot()\n'
        "ax.xaxis.set_tick_params(direction='in', which='both', labelsize=12)\n"
        "ax.yaxis.set_tick_params(direction='in', which='both', labelsize=12)\n"
        '#plt.savefig("IEEE_2_revised_images/2d_RMSE_IQR.png", bbox_inches=\'tight\', dpi=300)\n'
        'plt.show()\n'
        '\n'
        'threshold_iqr = xmin+(xmax-xmin)*.1\n'
        'threshold_rmse = ymin+(ymax-ymin)*.1\n'
        '\n'
        'print(f"Considerar resultados com IQR <= {threshold_iqr} e RMSE <= {threshold_rmse}. Incluir '
        'Standard FastCalo na Avaliação Final")'}
