import Adaline

PATH = "iris.csv"

adaline = Adaline.Adaline(path=PATH, number_features=2, col1=0, col2=1, learning_rate=0.001, number_epoch=10000)

adaline.start_train()

taxa = adaline.show_accuracy()
print('Taxa de Acertos: %.2f%%' % (taxa*100))

adaline.plot()
