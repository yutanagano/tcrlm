import matplotlib.pyplot as plt
import os
import pandas as pd


RUN_ID = 'training_run_220117'


stats = pd.read_csv(os.path.join('training_runs',RUN_ID,'train_stats.csv'),
                    index_col='epoch')

fig = plt.figure(figsize=(8,8))

loss = plt.subplot(5,1,(1,2))
loss.tick_params(
    axis='x',
    bottom=False,
    labelbottom=False
)
loss.set_title('Loss')

loss.plot(stats['train_loss'],label='training loss')
loss.plot(stats['valid_loss'],label='validation loss')
loss.legend(loc='upper right')

acc = plt.subplot(5,1,(3,4))
acc.tick_params(
    axis='x',
    bottom=False,
    labelbottom=False
)
acc.set_title('Accuracy')

acc.plot(stats['train_acc'],label='training accuracy')
acc.plot(stats['valid_acc'],label='validation accuracy')
acc.legend(loc='lower right')

lr = plt.subplot(5,1,5)
lr.set_xticks(
    ticks=range(1,21)
)
lr.set_title('Average Learning Rate')

lr.plot(stats['avg_lr'],c='g')

fig.tight_layout()

plt.savefig(os.path.join('training_runs',RUN_ID,'train_stats.png'))