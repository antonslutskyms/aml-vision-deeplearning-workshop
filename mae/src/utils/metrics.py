import torch

def compute_thresholds(outputs, label_name, epoch,metric="youden", suffix="VAL"):

    #compute
    tensor_y_hat = [] 
    tensor_y_true =[]
    for out in outputs:
        tensor_y_hat.append(out['y_hat'])
        tensor_y_true.append(out['y'])

    tensor_y_hat = torch.sigmoid(torch.cat(tensor_y_hat))
    tensor_y_hat = tensor_y_hat.cpu().numpy()

    tensor_y_true =torch.cat(tensor_y_true)
    tensor_y_true = tensor_y_true.int()
    tensor_y_true = tensor_y_true.cpu().numpy()

    num_classes = len(label_name)

    # calculate cutoff values

    if metric=="youden":
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        # Compute False Positive and True Positive Rates for each class
        for i in range(num_classes):
            fpr[i], tpr[i], thresholds[i] = roc_curve(tensor_y_true[:, i], tensor_y_hat[:, i], drop_intermediate=False, pos_label=1)
            roc_auc[i] = auc(fpr[i], tpr[i])

        J_stats = [None]*num_classes
        opt_thresholds = [None]*num_classes

        # Compute Youden's J Statistic for each class
        #with open(f'cutoff_evaluation_{epoch}') as cut_f:
        for i in range(num_classes):
            J_stats[i] = tpr[i] - fpr[i]
            try:
                opt_thresholds[i] = thresholds[i][np.nanargmax(J_stats[i])]
                if opt_thresholds[i] > 1:
                    opt_thresholds[i] = np.nan
                print(f'Optimum threshold {label_name[i]}: {opt_thresholds[i]}')
            except:
                ValueError
                opt_thresholds[i] = np.nan
                print(f'Optimum threshold {label_name[i]}: {opt_thresholds[i]}')
        
        pd.DataFrame({'label':label_name,'threshold':opt_thresholds}).to_csv(f'visualizations/{suffix}/{epoch}_youden_opt_thresh.csv')
