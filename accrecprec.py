def acc_calc(y_pred, y_test, smoothing=0):
    tp = sum([y_pred[i] == 1 and y_pred[i] == y_test[i] for i in range(len(y_pred))]) + smoothing
    tn = sum([y_pred[i] == -1 and y_pred[i] == y_test[i] for i in range(len(y_pred))]) + smoothing
    fp = sum([y_pred[i] == 1 and y_pred[i] != y_test[i] for i in range(len(y_pred))]) + smoothing
    fn = sum([y_pred[i] == -1 and y_pred[i] != y_test[i] for i in range(len(y_pred))]) + smoothing
    acc = (tp + tn) / (len(y_pred))
    prec_c1 = (tp) / (tp + fp) if (tp + fp) else 0
    rec_c1 = (tp) / (tp + fn) if (tp + fn) else 0
    prec_cm1 = (tn) / (tn + fn) if (tn + fn) else 0
    rec_cm1 = (tn) / (tn + fp) if (tn + fp) else 0
    f1_c1 = 2 * prec_c1 * rec_c1 / (prec_c1 + rec_c1) if (prec_c1 + rec_c1) else 0
    f1_cm1 = 2 * prec_cm1 * rec_cm1 / (prec_cm1 + rec_cm1) if (prec_cm1 + rec_cm1) else 0
    print("--------------")
    print('accuracy is', acc)
    print('if view=1 is considered to be the positive class: precision>', prec_c1, '- recall>', rec_c1, '- f1>', f1_c1)
    print('if view=-1 is considered to be the positive class: precision>', prec_cm1, '- recall>', rec_cm1, '- f1>',
          f1_cm1)
    print("--------------")
    return {'acc': acc, 'precision': {1: prec_c1, -1: prec_cm1}, 'recall': {1: rec_c1, -1: rec_cm1},
            'f1': {1: f1_c1, -1: f1_cm1}}
