from sklearn import metrics
import pandas as pd
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, Union, List


## Evaluation metrics
def eval_metrics(model, X, y, task_type:Literal['classification', 'regression']='regression', 
                 transform_type=None, average_type:Literal['weighted','micro', 'macro']='macro'):
    if task_type == 'regression':
        scores = regression_eval_metrics(model, X, y, transform_type)
    elif task_type == 'classification':
        scores = classification_eval_metrics(model, X, y, average_type)
    
    return scores

def classification_eval_metrics(model, X, y, average_type:Literal['weighted','micro', 'macro']='macro'):
    preds = model.predict(X).ravel()
    probs = model.predict_proba(X)
    y = y.values if isinstance(y, pd.Series) else y
    unique_class = sorted(np.unique(y))
    if len(unique_class) == 2:
        auc = metrics.roc_auc_score(y, probs[:,1])
        rec = metrics.recall_score(y, preds)
        prec = metrics.precision_score(y, preds)
        ap_score = metrics.average_precision_score(y, probs[:,1], average=average_type)
        f1 = metrics.f1_score(y, preds)
        spec = specificity(preds, y, unique_class, average_type='binary')
        
    elif len(unique_class) > 2:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y, classes=unique_class)
        auc = metrics.roc_auc_score(y, probs, average=average_type, multi_class='ovr')
        rec = metrics.recall_score(y, preds, average=average_type)
        prec = metrics.precision_score(y, preds, average=average_type)
        ap_score = metrics.average_precision_score(y_bin, probs, average=average_type)
        f1 = metrics.f1_score(y, preds, average=average_type)
        spec = specificity(preds, y, unique_class, average_type=average_type)

    acc = np.mean(y == preds)
    lloss = metrics.log_loss(y, probs)
    

    scores = pd.DataFrame([acc, auc, rec, prec, ap_score, f1, spec, lloss], 
                          columns=['scores'], 
                          index=['Acc', 'AUC', 'Rec', 'Prec', 'AP', 'F1', 'Spec', 'LLoss'])
    return scores.T



def regression_eval_metrics(model, X, y, transform_type=None):
    preds = model.predict(X)
    if transform_type == 'log1p':
        y = np.expm1(y)
        preds = np.expm1(preds)
    elif transform_type == 'sqrt':
        y = np.square(y)
        preds = np.square(preds)

    
    mae = metrics.mean_absolute_error(y, preds)
    mse = metrics.mean_squared_error(y, preds)
    rsq = metrics.r2_score(y, preds)
    rmse = metrics.root_mean_squared_error(y, preds)
    scores = pd.DataFrame([rmse, mae, mse, rsq], 
                          columns=['scores'], 
                          index=['RMSE', 'MAE', 'MSE', 'Rsq'])
    return scores.T


def specificity(preds, actual, unique_class=None, positive_val = 1,
                average_type:Literal['weighted','micro', 'macro', 'binary']='weighted'):
        preds = preds.ravel() # convert to 1D
        # class specificity tn / (tn+fp)
        if unique_class is None:
             unique_class = sorted(np.unique(actual))

        if len(unique_class) == 2:
            average_type = 'binary'
            return np.sum((preds != positive_val) & (actual != positive_val)) / np.sum(actual != positive_val)
        elif len(unique_class) > 2:
            tn = np.array([np.sum((preds != i) & (actual != i)) for i in sorted(unique_class)])
            fp = np.array([np.sum((preds != i) & (actual == i)) for i in sorted(unique_class)])
            score = tn / (tn+fp)
            class_counts = np.array([np.sum(actual==i) for i in sorted(unique_class)])
        
            if average_type == 'weighted':
                return np.average(score, weights=class_counts)
            elif average_type == 'macro':
                return np.mean(score)
            elif average_type == 'micro':
                return np.sum(tn) / np.sum(tn+fp)
            else:
                print(f'{average_type} not valid')
                exit(1)


# Model evaluation visualisation
def print_classification_report(model, X, y, display_names=None):
    preds = model.predict(X)
    print(metrics.classification_report(y, preds, target_names=display_names))


def regression_performance_chart_report(model, X, y):
    predictions = model.predict(X)
    df = pd.DataFrame({'Actual':y, 'Predictions':predictions, 'Errors': y-predictions})
    cor_result = np.corrcoef(df['Actual'], df['Predictions'])[0][1]
    fig, ax = plt.subplots(1,2,figsize=(9,5))

    sns.scatterplot(df, x='Actual', y='Predictions', color='indianred', alpha=0.8, ax=ax[0])
    sns.scatterplot(df, x='Predictions', y='Errors', color='steelblue', alpha=0.8, ax=ax[1])
    ax[1].axhline(0, color='#2C2D2D', linestyle='--', lw=0.8)
    ax[0].set(xlabel='Actual values', ylabel='Predicted values')
    ax[1].set(xlabel='Predicted values', ylabel='Errors')
    ax[0].set_title('Actual vs Predicted values', fontsize=10)
    ax[1].set_title('Residual Plot', fontsize=10)
    ax[0].legend([f'r = {cor_result:.3f}'], loc='lower right')
    plt.suptitle(model.__class__.__name__, x=0.12, y=.96, fontweight='bold')
    plt.tight_layout()


def classification_performance_chart_report(model, X, y, display_names=None):
    display_labs = np.unique(y)

    if len(display_labs) > 2:
        uniq_class = list(display_labs)
        probas = model.predict_proba(X)
        preds = model.predict(X).ravel()
        cm = pd.DataFrame(metrics.confusion_matrix(y, preds), index=uniq_class, columns=uniq_class)
        fig, ax = plt.subplots(1, 3, figsize=(11,4.3))
        for i in range(len(uniq_class)):
            y_clas = np.where(y == uniq_class[i], 1, 0)
            auc = metrics.roc_auc_score(y_clas, probas[:,i])
            pr_score = metrics.average_precision_score(y_clas, probas[:,i])
            precision, recall, _ = metrics.precision_recall_curve(y_clas, probas[:,i])
            fpr, tpr, _ = metrics.roc_curve(y_clas, probas[:,i])
            ax[0].plot(fpr, tpr, label=f'{uniq_class[i]} (AUC={auc:.2f})' if display_names is None else f'{display_names[i]} (AUC={auc:.2f})')
            ax[1].plot(recall, precision, label=f'{uniq_class[i]} (PR={pr_score:.2f})' if display_names is None else f'{display_names[i]} (PR={pr_score:.2f})')
            ax[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
            ax[0].set_title('ROC curve', fontsize=10)
            ax[1].set(xlabel='Recall', ylabel='Precision')
            ax[1].set_title('Precision-Recall curve', fontsize=10)
        
        sns.heatmap(cm, annot=True, cmap='Greens', cbar=False, ax=ax[2], linecolor='k', square=True, fmt='d')
        ax[2].set_yticklabels(uniq_class if display_names is None else display_names, rotation=0)
        ax[2].set_xticklabels(uniq_class if display_names is None else display_names, rotation=0)
        ax[2].set(xlabel='Predicted Class', ylabel='Actual Class')
        ax[2].set_title('Confusion matrix', fontsize=10)
        
        fig.suptitle(model.__class__.__name__, fontweight='bold', x=0.12)
        fig.tight_layout()
        
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='lower left')
    else:
        fig, ax = plt.subplots(1,3, figsize=(12,5))
        _ = metrics.RocCurveDisplay.from_estimator(model, X, y, ax=ax[0])
        _ = metrics.PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax[1])
        _ = metrics.ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=display_labs if display_names is None else display_names, 
                                                          ax=ax[2], cmap ='Greens', colorbar=False, values_format='d')
        ax[0].set_title('ROC curve', fontsize=10)
        ax[1].set_title('Precision-Recall curve', fontsize=10)
        ax[2].set_title('Confusion Matrix', fontsize=10)
        plt.suptitle(model.__class__.__name__, x=0.15, y=0.90, fontweight='bold')
        plt.tight_layout()


# feature importance
def get_feature_importance_scores(model, columns=None, sort=False):
    """
    Returns feature importance scores of fitted model

    :param model: Fitted Model
    :param columns: None|Model input variables (optional).
    """
    if hasattr(model, 'coef_'):
        varimp = model.coef_.squeeze()
    elif hasattr(model, 'named_steps'):
        model_name = list(model.named_steps.keys())[-1] # get model name
        model = model.named_steps[model_name] # reassign model and check for coef_ or feature_importances_ attributes
        if hasattr(model, 'coef_'):
            varimp = model.coef_.squeeze()
        elif hasattr(model, 'feature_importances_'):
            varimp = model.feature_importances_/model.feature_importances_.sum()    
    elif hasattr(model, 'feature_importances_'):
        varimp = model.feature_importances_/model.feature_importances_.sum()
    else:
        raise TypeError (f'{model.__class__.__name__} not fitted')
    
    if columns is None:
        if hasattr(model, 'feature_names_in_'):
            columns = model.feature_names_in_ 
        elif hasattr(model, 'feature_names_'):
            columns = model.feature_names_
    imp_scores = pd.Series(varimp, columns)
    return imp_scores.sort_values(ascending=False) if sort else imp_scores