"""
A helper for training & testing
models

"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.preprocessing import scale, robust_scale

def load_splits(splits_dir, full_df, f, k):
#     """
# 	full_df:	Complete dataframe
# 	f:			Name of feature dataset (csv)
# 	k:			Split number
# 	"""
#     f_split = "".join([splits_dir, str(k), '/', os.path.basename(f)[:-4], ".txt"])

#     splits_data = np.loadtxt(f_split, delimiter='=', dtype=str)
#     # print('f_split ', f_split)
#     # print('splits_data ', splits_data)
    
#     train_recs = eval(splits_data[0, 1])
#     test_recs = eval(splits_data[1, 1])

#     train_df = full_df[full_df.Study_Num.isin(train_recs)]
#     test_df = full_df[full_df.Study_Num.isin(test_recs)]

#     return train_df, test_df
    """
	full_df:	Complete dataframe
	f:			Name of feature dataset (csv)
	k:			Split number
	"""
    f_split = "".join([splits_dir, str(k), '/', os.path.basename(f)[:-4], ".txt"])

    splits_data = np.loadtxt(f_split, delimiter='=', dtype=str)

    train_recs = eval(splits_data[0, 1])
    test_recs = eval(splits_data[1, 1])
    val_recs = eval(splits_data[2, 1])
    
    train_df = full_df[full_df.Study_Num.isin(train_recs)]
    test_df = full_df[full_df.Study_Num.isin(test_recs)]
    val_df = full_df[full_df.Study_Num.isin(val_recs)]
    
    return train_df, test_df, val_df


def cv_param_estimation(df, model, params, n_folds, feat_names, type=None):

    def make_ps(df, n_folds=4):
        """
        :param df: full dataframe
        :param n_folds: number of folds

        :return: list indicating which samples
                 belong to which fold
        """

        # Get the list of recordings and labels
        recs = np.array(df.Study_Num.unique())

        labels = np.array([df[df.Study_Num == rec].TB_status.values[0] for rec in recs])

        skf = StratifiedKFold(n_splits=n_folds)

        y_ps = np.zeros(shape=(len(df.index)),)

        fold = 0
        for train_idx, test_idx in skf.split(labels, labels):
            test_recs = recs[test_idx]

            ps_test_idx = np.array(df[df['Study_Num'].isin(test_recs)].index)

            y_ps[ps_test_idx] = fold

            fold += 1

        return PredefinedSplit(y_ps)

    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)

    # Make the PredefinedSplit label
    ps = make_ps(df_, n_folds=n_folds)

    # labels
    y = df_.TB_status.values
    # data
    X = df_[feat_names]

    if type == 'LR':
        from sklearn.linear_model import LogisticRegressionCV

        lr_gs = LogisticRegressionCV(Cs = params['C'], cv=ps, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = params['l1_ratios'], penalty='elasticnet', n_jobs = 20)
        X_r_scaled = robust_scale(X)
        lr_gs.fit(X_r_scaled, y)
        opt_params_C = {'C':lr_gs.C_[0]}
        opt_params_l1_ratio = {'l1_ratio': lr_gs.l1_ratio_[0]}

    else:
        grid_search = GridSearchCV(model, params, scoring='roc_auc', cv=ps, verbose=0, max_iter = 1000000)
        X_r_scaled = robust_scale(X)
        grid_search.fit(X_r_scaled, y)

        opt_params = grid_search.best_params_

    model.set_params(**opt_params_C)
    model.set_params(**opt_params_l1_ratio)
    
    return model


def calc_kappa(y_ref, y_pred):
    """
    Calculate the Cohen's Kappa coeff
    to evaluate the observed acc vs the
    expected acc.
    """

    CM = confusion_matrix(y_ref, y_pred)
    observed_acc = (CM[1, 1] + CM[0, 0]) / float(len(y_ref))
    expected_acc = (y_ref.count(0) * CM[0, 0] / float(len(y_ref)) + y_ref.count(1) * CM[1, 1] / float(
        len(y_ref))) / float(len(y_ref))
    kappa = float(observed_acc - expected_acc) / float(1 - expected_acc)

    return kappa


def validation(opt_model, dev_df, feat_names):
    """
    Combine train_df and val_df to create
    dev_df.

    With dev_df, using k-fold validation, get
    validation results.

    Use the validation results to select equal-error
    rate threshold GAMMA.

    """

    dev_recs = np.array((dev_df.Study_Num.unique()))
    dev_labels = np.array([dev_df[dev_df.Study_Num == rec].TB_status.values[0] for rec in dev_recs])

    LRM = opt_model

    probs = []  # Probabilities during validation
    preds = []  # Predictions made
    y_ref = []  # Labels as they were used in validation

    skf = StratifiedKFold(n_splits=2)

    for train_idx, test_idx in skf.split(dev_labels, dev_labels):
        train_recs = dev_recs[train_idx]
        test_recs = dev_recs[test_idx]

        train_df = dev_df[dev_df.Study_Num.isin(train_recs)]
        test_df = dev_df[dev_df.Study_Num.isin(test_recs)]

        y_train = list(train_df.TB_status.values)
        y_test = list(test_df.TB_status.values)

        X_train = train_df[feat_names]
        X_test = test_df[feat_names]

        # Train the LRM
        X_r_scaled = robust_scale(X_train)
        LRM.fit(X_r_scaled, y_train)
        
        """
        Do sample-based testing
        """
        # Save this LRM performance
        probs.extend(list(LRM.predict_proba(X_test)[:, 1]))
        preds.extend(list(LRM.predict(X_test)))
        y_ref.extend(y_test)

    val_ = zip(probs, y_ref)
    return val_


def test_model(model, test_data, feat_names, threshold):

    def TBI_eval(probs, y_ref, test_px, cough_nums, GAMMA):
        """

        :param probs:
        :param y_ref:
        :param test_px:
        :param cough_nums:
        :param GAMMA:
        :return:
        """

        # Get the different TBI probs and refs
        i = np.arange(len(test_px))

        df = pd.DataFrame({"Recording": pd.Series(test_px, index=i),
                           "Reference": pd.Series(y_ref, index=i),
                           "Probabilities": pd.Series(probs, index=i),
                           'Cough_No': pd.Series(cough_nums, index=i)
                           })

        tbi_ref = []  # Recording based labels
        TBI_a_list = []  # Arithmetic mean of all probabilities of all cough in one recording
        TBI_s_list = []  # Ratio of pos_coughs/all_coughs in recording

        for name, group in df.groupby("Recording"):
            l = group.Reference.iloc[0]
            tbi_ref.append(l)

            # TBI_A
            TB_a_prob = sum(group.Probabilities.values) / float(len(group.Probabilities))
            TBI_a_list.append(TB_a_prob)

            # TBI_S
            pos_coughs_count = 0
            all_coughs_count = 0
            # Count all coughs in recording
            for cough_no, cough_df in group.groupby('Cough_No'):
                # Probability that this cough is TB+
                cough_prob = sum(cough_df.Probabilities.values) / float(len(cough_df.Probabilities))
                if cough_prob >= GAMMA:
                    pos_coughs_count += 1
                all_coughs_count += 1
            # Save ratio of TB+ coughs/all coughs
            TBI_s_list.append(float(pos_coughs_count) / all_coughs_count)

        return [TBI_a_list, TBI_s_list], tbi_ref

    test_recs = list(test_data.Study_Num.values)
    coughs = list(test_data.Cough_No.values)
    test_ref = list(test_data.TB_status.values)
    X_test = test_data[feat_names]

    # P(y=1|X)
    test_probs = model.predict_proba(X_test)[:, 1]

    """
    Convert per-frame probs to per-recording probs
    TBI_probs = [TBI_A_probs, TBI_G_probs, TBI_S_probs]
    """
    TBI_probs, TBI_ref = TBI_eval(probs=test_probs,
                                  y_ref=test_ref,
                                  test_px=test_recs,
                                  cough_nums=coughs,
                                  GAMMA=threshold)


    return TBI_probs, TBI_ref


def test_model_b_TBI(model, test_data, feat_names, threshold):

    def TBI_eval(probs, y_ref, test_px, cough_nums, GAMMA):
        """

        :param probs:
        :param y_ref:
        :param test_px:
        :param cough_nums:
        :param GAMMA:
        :return:
        """

        # Get the different TBI probs and refs
        i = np.arange(len(test_px))

        df = pd.DataFrame({"Recording": pd.Series(test_px, index=i),
                           "Reference": pd.Series(y_ref, index=i),
                           "Probabilities": pd.Series(probs, index=i),
                           'Cough_No': pd.Series(cough_nums, index=i)
                           })

        tbi_ref = []  # Recording based labels
        tbi_recs = []
        TBI_a_list = []  # Arithmetic mean of all probabilities of all cough in one recording
        TBI_s_list = []  # Ratio of pos_coughs/all_coughs in recording

        for name, group in df.groupby("Recording"):
            tbi_recs.append(name)

            l = group.Reference.iloc[0]
            tbi_ref.append(l)

            # TBI_A
            TB_a_prob = sum(group.Probabilities.values) / float(len(group.Probabilities))
            TBI_a_list.append(TB_a_prob)

            # TBI_S
            pos_coughs_count = 0
            all_coughs_count = 0
            # Count all coughs in recording
            for cough_no, cough_df in group.groupby('Cough_No'):
                # Probability that this cough is TB+
                cough_prob = sum(cough_df.Probabilities.values) / float(len(cough_df.Probabilities))
                if cough_prob >= GAMMA:
                    pos_coughs_count += 1
                all_coughs_count += 1
            # Save ratio of TB+ coughs/all coughs
            TBI_s_list.append(float(pos_coughs_count) / all_coughs_count)

        return [TBI_a_list, TBI_s_list], tbi_ref, tbi_recs

    test_recs = list(test_data.Study_Num.values)
    coughs = list(test_data.Cough_No.values)
    test_ref = list(test_data.TB_status.values)
    X_test = test_data[feat_names]

    # P(y=1|X)
    test_probs = model.predict_proba(X_test)[:, 1]

    """
    Convert per-frame probs to per-recording probs
    TBI_probs = [TBI_A_probs, TBI_G_probs, TBI_S_probs]
    """
    TBI_probs, TBI_ref, TBI_recs = TBI_eval(probs=test_probs,
                                  y_ref=test_ref,
                                  test_px=test_recs,
                                  cough_nums=coughs,
                                  GAMMA=threshold)


    return TBI_probs, TBI_ref, TBI_recs



def test_model_b_ADS(model, test_data, feat_names, threshold):

    def ADS_eval(probs, y_ref, test_px, cough_nums, GAMMA):

        # Get the different TBI probs and refs
        i = np.arange(len(test_px))

        df = pd.DataFrame({"Recording": pd.Series(test_px, index=i),
                           "Reference": pd.Series(y_ref, index=i),
                           "Probabilities": pd.Series(probs, index=i),
                           'Cough_No': pd.Series(cough_nums, index=i)
                           })

        ADS_ref = []  # Recording based labels
        ads_recs = []
        ADS_list = []  # Arithmetic mean of all probabilities of all cough in one recording
        
        for name, group in df.groupby("Recording"):
            ads_recs.append(name)
            l = group.Reference.iloc[0]
            ADS_ref.append(l)
            ADS_prob = sum(group.Probabilities.values) / float(len(group.Probabilities))
            ADS_list.append(ADS_prob)

        return ADS_list, ADS_ref, ads_recs
    
    test_recs = list(test_data.Study_Num.values)
    cough_nums = list(test_data.Cough_No.values)
    test_ref = list(test_data.TB_status.values)
    X_test = test_data[feat_names]

    # P(y=1|X)
    test_probs = model.predict_proba(X_test)[:, 1]

    """
    Convert per-frame probs to per-recording probs
    TBI_probs = [TBI_A_probs, TBI_G_probs, TBI_S_probs]
    """
    ADS_probs, ADS_ref, ADS_recs = ADS_eval(probs=test_probs,
                                  y_ref=test_ref,
                                  test_px=test_recs,
                                  cough_nums=cough_nums,
                                  GAMMA=threshold)

    return ADS_probs, ADS_ref, ADS_recs



def evaluate_model(y_ref, preds):
    # convert to list
    y_ref = list(y_ref)
    preds = list(preds)

    CM = confusion_matrix(y_ref, preds)

    TP = CM[1, 1]
    TN = CM[0, 0]
    FP = CM[0, 1]
    FN = CM[1, 0]

    ACC = (TP + TN) / float(TP + TN + FP + FN)
    SENS = TP / float(TP + FN)
    SPEC = TN / float(TN + FP)
    F = 2 * (SPEC * SENS) / (SPEC + SENS)

    KAPPA = calc_kappa(y_ref, preds)

    return [SENS, SPEC, ACC, KAPPA]


def get_gamma_ee(prob_ref_list):

    """
    :param prob_ref_list: list of tuples (prob, ref)
    :return: threshold where tpr ~= 1-fpr
    """

    # prob_arr = np.array(prob_ref_list)
    # probs = prob_arr[:, 0]
    # y_true = prob_arr[:, 1]
    
    [probs, y_true] = zip(*prob_ref_list)
    fpr, tpr, thresholds = roc_curve(y_true, probs, drop_intermediate=False)
    auc_acc = auc(fpr, tpr)

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i),
                        'tpr': pd.Series(tpr, index=i),
                        '1-fpr': pd.Series(1 - fpr, index=i),
                        'tf': pd.Series(tpr - (1 - fpr), index=i),
                        'thresholds': pd.Series(thresholds, index=i)
                        })
    idx = roc.tf.abs().argmin()
    threshold = roc.thresholds.iloc[idx]

    return threshold

