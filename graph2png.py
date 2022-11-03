import matplotlib.pyplot as plt
import json

num_epochs = 100
loss_hist =json.load(open('result/loss_hist.json', 'r'))
metric_hist =json.load(open('result/metric_hist.json', 'r'))

# xAxis = [key for key, value in dictionary.items()]
# yAxis = [value for key, value in dictionary.items()]

# plt.grid(True)

# Plot train-val loss
plt.title('Train-Val Loss')
plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()
# plt.show()
plt.savefig('result/loss.png')

# plot train-val accuracy
plt.title('Train-Val Accuracy')
plt.plot(range(1, num_epochs+1), metric_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), metric_hist['val'], label='val')
plt.ylabel('Accuracy')
plt.xlabel('Training Epochs')
plt.legend()
# plt.show()
plt.savefig('result/acc.png')
