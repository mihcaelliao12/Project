import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from sklearn.metrics import precision_recall_curve
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def set_index(df, col='index'):
    df = (
        df
        .withColumn(col, F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))-1)
    )
    return df

def get_except_cols(df, cols):
    if isinstance(cols, str):
        cols = [cols]
    except_cols = [col for col in df.columns if col not in cols]
    return except_cols

def get_cat_cols(df, feature_cols):
    return [f.name for f in df.schema.fields if isinstance(f.dataType, StringType) and f.name in feature_cols]
    
def is_zero(df, cols, drop=False):
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        df = (
            df
            .withColumn(f'{col}_isZero', F.when(F.col(col)==0, 1).otherwise(0))
        )
    if drop:
        df = df.drop(*cols)
    return df
    
def log1p(df, cols, drop=False):
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        df = (
            df
            .withColumn(f'{col}_log', F.log1p(F.col(col)))
        )
    if drop:
        df = df.drop(*cols)
    return df
    
def first_str(df, cols, drop=False):
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        df = (
            df
            .withColumn(f'{col}_firstStr', F.substring(F.col(col), 1, 1))
        )
    if drop:
        df = df.drop(*cols)
    return df
    
def concat_cols(df, col1, col2, join_char=None ,new_col=None, drop=False):
    if join_char is None:
        join_char = '_'
    if new_col is None:
        new_col = f'{col1}_{col2}'
    df = (
        df
        .withColumn(new_col, F.concat_ws(join_char, F.col(col1), F.col(col2)))
    )
    if drop:
        df = df.drop(col1, col2)
    return df

def lag_mean_encode(df, cols, target_col, time_col='index', lag=1, prefix='lag_mean', drop=True):
    if lag < 1:
        raise ValueError("lag must be >= 1")

    if isinstance(cols, str):
        cols = [cols]
        
    for col in cols:
        df = (
            df
            .withColumn(
                f'{prefix}_{col}',
                F.mean(target_col).over(
                    Window
                    .partitionBy(col)
                    .orderBy(time_col)
                    .rowsBetween(Window.unboundedPreceding, -lag))
            )
            .orderBy(time_col)
        )
    if drop:
        df = df.drop(*cols)
    return df

# 时间序列交叉验证
def tscv(train, pipeline, evaluator, k=3, index_col='index'):
    n_folds = k
    max_idx = train.agg(F.max(index_col)).first()[0]
    N = max_idx + 1
    
    for t in range(1, n_folds + 1):
        # 开始计时
        t0 = time.time()
        
        # 切分时间窗口
        train_end = int(N * (t/(n_folds+1)))
        valid_end = int(N * ((t+1)/(n_folds+1)))
        
        train_part = train.filter(F.col(index_col) < train_end)
        valid_part = train.filter(
            (F.col(index_col) >= train_end) & (F.col(index_col) < valid_end)
        )
        
        print(f'Fold {t}: train [0, {train_end}), valid [{train_end}, {valid_end})')
        
        # 拟合和验证
        cvModel = pipeline.fit(train_part)
        pred = cvModel.transform(valid_part)
        score = round(evaluator.evaluate(pred) * 100, 2)
        
        # 结束计时
        t1 = time.time()
        elapsed = t1 - t0
        print(f'  Time used: {int(elapsed//60)} min {elapsed%60:.2f} s')
        
        # 每折结果
        print(f'Score: {score:.2f}%')
        

# 时间序列滚动交叉验证
def tscv_rolling(train, pipeline, evaluator, k=3, index_col='index'):
    n_folds = k
    max_idx = train.agg(F.max(index_col)).first()[0]
    N = max_idx + 1
    metrics = []
    
    for t in range(1, n_folds + 1):
        # 开始计时
        t0 = time.time()
        
        # 切分时间窗口
        train_start = int(N * ((t-1)/(n_folds+1)))
        train_end = int(N * (t/(n_folds+1)))
        valid_end = int(N * ((t+1)/(n_folds+1)))
        
        train_part = train.filter(
            (F.col(index_col) >= train_start) & (F.col(index_col) < train_end)
        )
        valid_part = train.filter(
            (F.col(index_col) >= train_end) & (F.col(index_col) < valid_end)
        )
        
        print(f'Fold {t}: train [{train_start}, {train_end}), valid [{train_end}, {valid_end})')
        
        # 拟合和验证
        cvModel = pipeline.fit(train_part)
        pred = cvModel.transform(valid_part)
        score = round(evaluator.evaluate(pred) * 100, 2)
        
        # 结束计时
        t1 = time.time()
        elapsed = t1 - t0
        print(f'  Time used: {int(elapsed//60)} min {elapsed%60:.2f} s')
        
        # 每窗口分数
        print(f'Score: {score:.2f}%')
        metrics.append(score)
    
    # 平均分数
    print(f'\nMean score: {np.mean(metrics)}%±{round(np.std(metrics), 2)}')
        
        
def hold_out(train, pipeline, evaluator, test_size=0.2, index_col='index'):
    max_idx = train.agg(F.max(index_col)).first()[0]
    N = max_idx + 1
    
    # 切分数据集
    train_end = int(N * (1-test_size))
    train_part = train.filter(F.col(index_col) < train_end)
    valid_part = train.filter(F.col(index_col) >= train_end)
    
    # 拟合和验证
    hold_out_model = pipeline.fit(train_part)
    pred = hold_out_model.transform(valid_part)
    
    # 记录分数
    score = round(evaluator.evaluate(pred) * 100, 2)
    
    return score, hold_out_model


# hold out选阈值
def hold_out_threshold(
    train,
    model,
    test_size=0.2,
    index_col='index',
    positive_label=1,
    tp=None,
    tr=None,
    priority='f1',
    plot=False,
    ):
    max_idx = train.agg(F.max(index_col)).first()[0]
    N = max_idx + 1
    
    # 切分数据集
    train_end = int(N * (1-test_size))
    valid_part = train.filter(F.col(index_col) >= train_end)
    
    predict = model.transform(valid_part)
    predict_df = predict.select('probability', 'label').toPandas()
    y_true = predict_df['label'].values
    y_proba = predict_df['probability'].apply(lambda x: float(x[1])).values
    
    # 计算precision-recall曲线
    precision, recall, threshold = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]
    
    # 计算最佳阈值
    # ================= priority == 'accuracy' 分支 =================
    if priority =='accuracy':
        y_proba_unique = np.sort(np.unique(y_proba))
        mid_points = (y_proba_unique[:-1]+y_proba_unique[1:])/2
        candidates = np.r_[0.0, mid_points, 1.0]
        threshold_best = 0
        acc_best = -1
        for thre in candidates:
            y_pred = (y_proba >= thre).astype(int)
            acc = accuracy_score(y_true, y_pred)
            if acc > acc_best:
                acc_best, threshold_best = acc, thre
        print(f"Best threshold: {threshold_best:}")
        print(f"Accuracy at best threshold: {accuracy_score(y_true, (y_proba >= threshold_best).astype(int)):}")
        return threshold_best
    
    # ================= 其他 priority 分支（基于 PR 曲线） =================
    else:
        # 判断priority
        if priority == 'f1':
            target_series = 2 * precision * recall / (precision + recall + 1e-12)
        elif priority == 'precision':
            target_series = precision
        elif priority =='recall':
            target_series = recall
        elif priority == 'closest':
            if (tp is not None) and (tr is not None):
                target_series = -((precision - tp)**2 + (recall - tr)**2)
            elif tp is not None:
                target_series = -(precision - tp)**2 
            elif tr is not None:
                target_series = -(recall - tr)**2
            else:
                print('No target value is given. Use default value 1 for target precision and 1 for target recall.')
                target_series = -((precision - 1)**2 + (recall - 1)**2)
        # elif priority == 'accuracy':
        #     target_series = (y_train == y_proba.round()).astype(int)
        else:
            raise ValueError("priority only support 'f1'|'recall'|'precision'|'closest'。")
        
        
        
        # ========== tr & tp 都有 ==========
        if (tr is not None) and (tp is not None):
            mask = (recall >= tr) & (precision >= tp)
            if mask.any():
                idx = np.argmax(target_series[mask])
                thre_best = threshold[mask][idx]
                prec_best = precision[mask][idx]
                rec_best = recall[mask][idx]
            else:
                print(f'No any threshold make precision above {tp} and recall above {tr}')
                return None
        
        # ========== 只有 tr ==========
        elif tr is not None:
            mask = recall >= tr
            if mask.any():
                idx = np.argmax(target_series[mask])
                thre_best = threshold[mask][idx]
                prec_best = precision[mask][idx]
                rec_best = recall[mask][idx]
            else:
                print(f'No any threshold make recall above {tr}')
                return None
        
        # ========== 只有 tp ==========
        elif tp is not None:
            mask = precision >= tp
            if mask.any():
                idx = np.argmax(target_series[mask])
                thre_best = threshold[mask][idx]
                prec_best = precision[mask][idx]
                rec_best = recall[mask][idx]
            else:
                print(f'No any threshold make precision above {tp}')
                return None
        
        # ========== tp 和 tr 都 None，纯粹最大化 target ==========
        elif (tp is None) and (tr is None):
            thre_best = threshold[np.argmax(target_series)]
            prec_best = precision[np.argmax(target_series)]
            rec_best = recall[np.argmax(target_series)]
    
    if plot:
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, label='Precision-Recall curve')
        
        if thre_best is not None:
            plt.scatter(rec_best, prec_best, s=60)
            plt.annotate(
                f"thr={thre_best:.3f}\nP={prec_best:.2f}, R={rec_best:.2f}",
                xy=(rec_best, prec_best),
                xytext=(10, -10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.5)
            )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Selected Threshold')
        plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        plt.grid(which='major', linestyle='-', linewidth=0.75, alpha=0.6)
        plt.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.4)
        plt.legend()
        plt.show()
    
    print(f"Best threshold: {thre_best}")
    print(f"Precision at best threshold: {prec_best}")
    print(f"Recall at best threshold: {rec_best}")
    
    return thre_best