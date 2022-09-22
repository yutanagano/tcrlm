import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp


def parse_command_line_arguments() -> str:
    parser = argparse.ArgumentParser(
        description='Generate a visualisation of the model\'s post-training'
            'analysis.'
    )
    parser.add_argument(
        'pretrain_id',
        help='Specify the pretrain run ID for the model to analyse.'
    )
    args = parser.parse_args()

    return args.pretrain_id


def main(pretrain_id):
    fig = plt.figure(figsize=(10,10))

    analysis_dir = Path('pretrain_runs')/pretrain_id/'analysis'

    metrics = pd.read_csv(analysis_dir/'clustering_metrics.csv', index_col=0)
    v_control = pd.read_csv(analysis_dir/'clustering_metrics_control_V.csv')
    j_control = pd.read_csv(analysis_dir/'clustering_metrics_control_J.csv')
    mhca_control = pd.read_csv(analysis_dir/'clustering_metrics_control_MHC_A.csv')
    mhc_control = pd.read_csv(analysis_dir/'clustering_metrics_control_MHC_class.csv')
    epitope_control = pd.read_csv(analysis_dir/'clustering_metrics_control_Epitope.csv')

    for i, metric, ymin, ymax in ((0, 'Silhouette', -1, 1), (1, 'CH', 0, None), (2, 'DB', 0, None)):
        def make_subplot(index, control_df, rowname, title=None):
            subplot = plt.subplot(5,3,index)
            subplot.set_xticks([0,1],['C','M'])

            controls = control_df[metric]
            model_metric = metrics[metric].loc[rowname]

            subplot.scatter(x=[0]*5,y=controls)
            subplot.scatter(x=[1],y=model_metric)

            if ymin is not None:
                subplot.set_ylim(bottom=ymin)
            if ymax is not None:
                subplot.set_ylim(top=ymax)

            rymin, rymax = subplot.get_ylim()
            yrange = rymax-rymin
            ymean = (rymax+rymin)/2

            alt='less'
            if metric == 'DB':
                alt='greater'

            pval = ttest_1samp(controls, popmean=model_metric, alternative=alt).pvalue
            colour = 'k'
            if pval < 0.01:
                colour = 'r'
            if pval < 0.000001:
                colour = 'm'

            subplot.text(x=0.5,y=ymean,s=f'{model_metric: .3f}')
            subplot.text(x=0.5,y=ymean-yrange*0.2,s=f'{pval: .3E}',c=colour)

            subplot.margins(0.5,0.5)

            if title is not None:
                subplot.set_title(title)

            if i == 0:
                subplot.set_ylabel(rowname)

            return subplot

        v_plot = make_subplot(i+1, v_control, 'V regions', title=metric)
        j_plot = make_subplot(i+4, j_control, 'J regions')
        mhca_plot = make_subplot(i+7, mhca_control, 'MHC A')
        mhc_plot = make_subplot(i+10, mhc_control, 'MHC class')
        epitope_plot = make_subplot(i+13, epitope_control, 'Epitope')

    plt.tight_layout()
    fig.savefig(analysis_dir/'post_training_analysis.png')
    plt.show()


if __name__ == '__main__':
    plt.style.use('seaborn')
    PRETRAIN_ID = parse_command_line_arguments()
    main(PRETRAIN_ID)